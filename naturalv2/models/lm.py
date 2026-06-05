"""Classes for interacting with language models."""

import asyncio
import logging
import os
from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Optional, Union

from dotenv import load_dotenv
from litellm import model_cost
from litellm.cost_calculator import completion_cost
from litellm.types.utils import ModelResponse as ChatCompletionResponse
from litellm.types.utils import TextCompletionResponse
from pydantic import BaseModel

from naturalv2.models.rate_limiter.rate_limiter import RateLimiter
from naturalv2.models.types import (
    BatchResponse,
    EndpointType,
    LogProbs,
    ModelInput,
    ModelResponse,
    TokenUsage,
    ToolCall,
)
from naturalv2.models.utils import (
    _extract_output_and_parsed,
    estimate_token_count,
    extract_token_usage,
    get_message_content,
    parse_model_output_with_format,
    validate_endpoint,
)


if TYPE_CHECKING:
    import litellm
    from litellm.types.llms.openai import ResponsesAPIResponse
    from litellm.types.router import LiteLLMParamsTypedDict
    from vllm.outputs import RequestOutput
    from vllm.tokenizers import TokenizerLike

load_dotenv()
logger = logging.getLogger(__name__)

is_weave_available = os.getenv("USE_WEAVE", "false").lower() == "true"
if is_weave_available:
    import weave

    weave_op = weave.op
else:
    # Fallback decorator: does nothing
    def weave_op(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

# ruff: noqa: PLC0415 # Ignore "import outside top-level" for imports


class Model(ABC):
    """Abstract base class for language model wrappers.

    Parameters
    ----------
    model_id : str
        The identifier of the model.
    endpoint : EndpointType, optional, default="chat_completion"
        The default endpoint type to use. The options are:
        - "chat_completion": "/v1/chat/completions" endpoint
        - "text_completion": "/v1/completions" endpoint
        - "responses": "/v1/responses" endpoint
    **kwargs
        Additional keyword arguments for the model.
    """

    def __init__(
        self, model_id: str, endpoint: EndpointType = "chat_completion", **kwargs
    ) -> None:
        """Initialize the Model base class."""
        self.model_id = model_id
        self.endpoint = endpoint
        self.kwargs = kwargs

        self._cost: float = 0.0

    @property
    def cost(self) -> float:
        """Return the accumulated cost of all requests.

        Returns
        -------
        float
            The running cost.
        """
        return self._cost

    @abstractmethod
    def invoke(
        self, input_data: ModelInput, *args, endpoint: EndpointType = None, **kwargs
    ) -> ModelResponse | BatchResponse:
        """Synchronously invoke the model.

        Parameters
        ----------
        input_data : ModelInput
            The input data for the model.
        endpoint : EndpointType, optional, default=None
            The endpoint to use for this request (overrides default class-level endpoint).
        **kwargs
            Additional keyword arguments for the request.

        Returns
        -------
        ModelResponse or BatchResponse
            The model's response.
        """
        pass

    async def ainvoke(
        self, input_data: ModelInput, *args, endpoint: EndpointType = None, **kwargs
    ) -> ModelResponse | BatchResponse:
        """Asynchronously invoke the model.

        Parameters
        ----------
        input_data : ModelInput
            The input data for the model.
        endpoint : EndpointType, optional, default=None
            The endpoint to use for this request (overrides default class-level endpoint).
        **kwargs
            Additional keyword arguments for the request.

        Returns
        -------
        ModelResponse or BatchResponse
            The model's response.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.invoke, input_data, *args, endpoint=endpoint, **kwargs
        )

    def _validate_and_normalize_input(
        self, input_data: ModelInput, endpoint: EndpointType
    ) -> tuple[Union[list[str], list[dict[str, str]], dict[str, str]], bool]:
        """Validate and normalize the input data based on the endpoint.

        Returns the normalized input and flag indicating if it's a batch.
        """
        is_batch = False
        if endpoint == "text_completion":
            if isinstance(input_data, str):
                return [input_data], is_batch

            if isinstance(input_data, list) and all(
                isinstance(item, str) for item in input_data
            ):
                is_batch = True
                return input_data, is_batch

            raise ValueError(
                "For the `'text_completion'` endpoint, `input_data` must be a "
                "string or a list of strings."
            )

        if endpoint == "chat_completion":
            if isinstance(input_data, str):
                # Convert string to chat message format
                return [{"role": "user", "content": input_data}], is_batch

            if isinstance(input_data, list) and all(
                isinstance(item, dict) and "role" in item and "content" in item
                for item in input_data
            ):
                return input_data, is_batch

            raise ValueError(
                "For the `'chat_completion'` endpoint, `input_data` must be a string or a list of "
                "dictionaries with 'role' and 'content' keys."
            )

        if endpoint == "responses":
            if isinstance(input_data, str):
                # Convert string to responses format
                return {"input": input_data}, is_batch

            if isinstance(input_data, dict) and "input" in input_data:
                return input_data, is_batch

            raise ValueError(
                "For the `'responses'` endpoint, `input_data` must be a string or a dictionary "
                "with an 'input' key and optionally an 'instructions' key."
            )

        raise ValueError(f"Unsupported endpoint: {endpoint}")


class VLLMModel(Model):
    """vLLM model wrapper.

    Parameters
    ----------
    model_id : str
        The identifier/name of the model.
    model_kwargs : dict, optional, default=None
        Keyword arguments for vLLM model initialization i.e. the arguments to pass to
        ``vllm.LLM()``.
    endpoint : EndpointType, optional, default="chat_completion"
        The default endpoint type to use. This class supports:
        - "chat_completion": "/v1/chat/completions" endpoint for chat-based models.
        - "text_completion": "/v1/completions" endpoint for text-based models.
    **kwargs
        Additional keyword arguments for the model.
    """

    def __init__(
        self,
        model_id: str,
        model_kwargs: dict[str, Any] | None = None,
        endpoint: EndpointType = "chat_completion",
        **kwargs,
    ) -> None:
        """Initialize the VLLMModel."""
        super().__init__(model_id=model_id, endpoint=endpoint, **kwargs)

        try:
            import msgspec
            from vllm import LLM
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Please install 'vllm' to use VLLMModel: `pip install vllm`"
            ) from e

        self.model_kwargs = model_kwargs or {}
        self.model = LLM(model=model_id, **self.model_kwargs)
        self.tokenizer: TokenizerLike = self.model.get_tokenizer()
        self._default_sampling_params = msgspec.structs.asdict(
            self.model.get_default_sampling_params()
        )

    def cleanup(self) -> None:
        """Clean up the model and free resources."""
        import gc

        import torch
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        destroy_model_parallel()
        if self.model is not None:
            # taken from https://github.com/vllm-project/vllm/issues/1908#issuecomment-2975218097
            self.model.llm_engine.engine_core.shutdown()
        gc.collect()
        destroy_distributed_environment()
        torch.cuda.empty_cache()

    async def ainvoke(self, *args, **kwargs) -> ModelResponse:
        """Asynchronous invocation is not supported for VLLMModel.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError("VLLMModel does not support async calls.")

    def invoke(
        self,
        input_data: ModelInput,
        endpoint: EndpointType = None,
        response_format: type[BaseModel] | dict | None = None,
        parse_output: bool = False,
        **kwargs,
    ) -> ModelResponse | BatchResponse:
        """Make a request to the vLLM model.

        Parameters
        ----------
        input_data : ModelInput
            The input data for the model.
        endpoint : EndpointType, optional, default=None
            The endpoint to use for this request (overrides default class-level endpoint).
        response_format : type[BaseModel] or dict, optional, default=None
            The response format for guided decoding.
        parse_output : bool, optional. Default is False.
            Whether to parse the output using the response format. If ``True``,
            the output ``ModelResponse.output_parsed`` will be populated.
        **kwargs
            Additional keyword arguments for the request.

        Returns
        -------
        ModelResponse or BatchResponse
            The model's response.
        """
        from vllm import SamplingParams

        endpoint = endpoint or self.endpoint
        validate_endpoint(
            endpoint, supported_endpoints=["chat_completion", "text_completion"]
        )

        inputs, is_batch = self._validate_and_normalize_input(input_data, endpoint)

        # Prepare sampling params
        # - Sampling params come from 3 places, in order of precedence:
        #   1. kwargs to this method
        #   2. self.kwargs (set at model init)
        #   3. self._default_sampling_params (from vLLM model)
        # - self.kwargs and kwargs may contain non-sampling params, so we filter them out
        combined_kwargs = {**self.kwargs, **kwargs}

        # self._default_sampling_params may not contain all possible sampling params,
        # so we use it as a base and update it with self.kwargs and kwargs
        sampling_params_dict = self._default_sampling_params.copy()
        sampling_param_keys = SamplingParams.__annotations__.keys()
        sampling_kwargs = {}
        remaining_kwargs = {}
        for k, v in combined_kwargs.items():
            if k in sampling_param_keys and v is not None:
                sampling_kwargs[k] = v
            else:
                remaining_kwargs[k] = v

        sampling_params_dict.update(sampling_kwargs)

        # Handle response_format for guided decoding
        if response_format is not None and (
            "guided_decoding" not in sampling_params_dict
            or (sampling_params_dict["guided_decoding"] is None)
        ):
            from vllm.sampling_params import GuidedDecodingParams

            sampling_params_dict["guided_decoding"] = GuidedDecodingParams(
                json=response_format.model_json_schema()
            )

        sampling_params = SamplingParams(**sampling_params_dict)

        if endpoint == "chat_completion":
            tools = remaining_kwargs.pop("tools", None)
            inputs = self.tokenizer.apply_chat_template(
                inputs,
                tools=tools,
                add_generation_prompt=True,
                tokenize=False,
            )

        response = self.model.generate(
            inputs, sampling_params=sampling_params, **remaining_kwargs
        )

        return self._parse_response(
            response,
            is_batch=is_batch,
            response_format=response_format,
            parse_output=parse_output,
        )

    def _parse_response(
        self,
        response: list["RequestOutput"],
        is_batch: bool = False,
        response_format: type[BaseModel] | dict | None = None,
        parse_output: bool = False,
    ) -> ModelResponse | BatchResponse:
        """Parse the vLLM response into ``ModelResponse`` or ``BatchResponse``."""

        def parse_single_output(output: "RequestOutput") -> ModelResponse:
            """Parse a single vLLM output into a ``ModelResponse.``"""
            # Get output text
            texts: list[str] = []
            parsed: list[Any] = []
            sample_logprobs = []
            for completion_output in output.outputs:
                text = completion_output.text
                texts.append(text)
                if parse_output and response_format is not None:
                    parsed.append(parse_model_output_with_format(text, response_format))
                if completion_output.logprobs is not None:
                    sample_logprobs.append(completion_output.logprobs)
            output_text = "\n".join(texts)

            def _parse_vllm_logprob_dict(
                logprob_dict: dict[int, Any],
            ) -> LogProbs:
                token_id = next(iter(logprob_dict))
                logprob_obj = logprob_dict.get(token_id)
                return token_id, logprob_obj.logprob

            # Get logprobs
            logprobs: LogProbs | None = None
            for logprob_list in sample_logprobs:
                logprobs = []
                token_ids = []
                for item in logprob_list:
                    token_id, logprob = _parse_vllm_logprob_dict(item)
                    token_ids.append(token_id)
                    logprobs.append(logprob)
                logprobs = LogProbs(logprobs=logprobs, tokens=token_ids)

            # Get prompt logprobs
            prompt_logprobs = output.prompt_logprobs
            if prompt_logprobs is not None:
                logprobs = []
                token_ids = []
                for item in prompt_logprobs:
                    if item is None:
                        continue

                    token_id, logprob = _parse_vllm_logprob_dict(item)
                    token_ids.append(token_id)
                    logprobs.append(logprob)

                prompt_logprobs = LogProbs(logprobs=logprobs, tokens=token_ids)

            # Get token usage
            prompt_tokens = (
                len(output.prompt_token_ids) if output.prompt_token_ids else 0
            )
            completion_tokens = 0.0
            for completion_output in output.outputs:
                completion_tokens += len(completion_output.token_ids or [])

            token_usage = TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )

            return ModelResponse(
                output_text=output_text,
                output_parsed=parsed[0]
                if parsed and len(parsed) == 1
                else parsed or None,
                model_id=self.model_id,
                logprobs=logprobs,
                prompt_logprobs=prompt_logprobs,
                token_usage=token_usage,
                finish_reason=output.outputs[-1].finish_reason,
                raw_response=output,
                request_id=output.request_id,
            )

        if is_batch:
            responses = [parse_single_output(output) for output in response]
            return BatchResponse(responses=responses)

        return parse_single_output(response[0])


class APIModel(Model):
    """Base class for models that use an API.

    Parameters
    ----------
    model_id : str
        The identifier of the model.
    client : Any, optional, default=None
        The API client instance.
    endpoint : EndpointType, optional, default="chat_completion"
        The default endpoint type to use. The options are:
        - "chat_completion": "/v1/chat/completions" endpoint for conversations.
        - "text_completion": "/v1/completions" endpoint for text continuation.
        - "responses": "/v1/responses" endpoint.
    **kwargs
        Additional keyword arguments for the model.
    """

    def __init__(
        self,
        model_id: str,
        client: Any | None = None,
        endpoint: EndpointType = "chat_completion",
        **kwargs,
    ) -> None:
        """Initialize the APIModel."""
        super().__init__(model_id=model_id, endpoint=endpoint, **kwargs)
        self.client = client or self.create_client()

    def create_client(self) -> Any:
        """Create the API client.

        Returns
        -------
        Any
            The API client instance.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError(
            "Subclasses must implement this method to create a client."
        )


class LiteLLMModel(APIModel):
    """Provider-agnostic API model wrapper using LiteLLM.

    Parameters
    ----------
    model_id : str
        The identifier of the model.
    api_base : str, optional, default=None
        The API base URL.
    api_key : str, optional, default=None
        The API key.
    rpm : int, optional, default=None
        Requests per minute limit.
    tpm : int, optional, default=None
        Tokens per minute limit.
    rpd : int, optional, default=None
        Requests per day limit.
    tpd : int, optional, default=None
        Tokens per day limit.
    max_request_burst : int, optional, default=None
        Maximum request burst.
    max_token_burst : int, optional, default=None
        Maximum token burst.
    max_parallel_requests : int, optional, default=None
        Maximum parallel requests.
    endpoint : EndpointType, optional, default="chat_completion"
        The default endpoint type to use. The options are:
        - "chat_completion": "/v1/chat/completions" endpoint
        - "text_completion": "/v1/completions" endpoint
        - "responses": "/v1/responses" endpoint
    **kwargs
        Additional keyword arguments for the model.
    """

    def __init__(
        self,
        model_id: str,
        api_base: str | None = None,
        api_key: str | None = None,
        rpm: int | None = None,
        tpm: int | None = None,
        rpd: int | None = None,
        tpd: int | None = None,
        max_request_burst: int | None = None,
        max_token_burst: int | None = None,
        max_parallel_requests: int | None = None,
        endpoint: EndpointType = "chat_completion",
        **kwargs,
    ):
        """Initialize the LiteLLMModel."""
        self.api_base = api_base
        self.api_key = api_key
        self.max_parallel_requests = max_parallel_requests

        self._semaphore = (
            asyncio.Semaphore(max_parallel_requests)
            if max_parallel_requests is not None
            else nullcontext()
        )

        self._rate_limiter: RateLimiter | None = None
        if rpm or tpm:
            self._rate_limiter = RateLimiter(
                requests_per_minute=rpm if rpm is not None else tpm // 1000 // 6,
                tokens_per_minute=tpm,
                requests_per_day=rpd,
                tokens_per_day=tpd,
                max_request_burst=max_request_burst,
                max_token_burst=max_token_burst,
            )

        super().__init__(model_id=model_id, endpoint=endpoint, **kwargs)

    def create_client(self) -> "litellm":
        """Create the LiteLLM client.

        Returns
        -------
        litellm
            The litellm module.

        Raises
        ------
        ModuleNotFoundError
        """
        try:
            import litellm
            import litellm._logging

            litellm._logging.verbose_logger.setLevel(logging.WARNING)
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Please install 'litellm' to use LiteLLMModel: `pip install litellm`"
            ) from e

        return litellm

    def invoke(
        self,
        input_data: ModelInput,
        endpoint: EndpointType = None,
        response_format: type[BaseModel] | dict | None = None,
        parse_output: bool = False,
        **kwargs,
    ) -> ModelResponse | BatchResponse:
        """Synchronously invoke the LiteLLM model.

        Parameters
        ----------
        input_data : ModelInput
            The input data for the model.
        endpoint : EndpointType, optional, default=None
            The endpoint to use for this request (overrides default class-level endpoint).
        response_format : type[BaseModel] or dict, optional, default=None
            The response format for structured output.
        parse_output : bool, optional
            Whether to parse the output using the response format. If ``True``,
            the output ``ModelResponse.output_parsed`` will be populated after
            successful parsing.
        **kwargs
            Additional keyword arguments for the request.

        Returns
        -------
        ModelResponse or BatchResponse
            The model's response.
        """
        endpoint = endpoint or self.endpoint
        inputs, is_batch, request_kwargs = self._prepare_request(
            input_data, endpoint, kwargs
        )

        if endpoint == "responses":
            request_kwargs = self._handle_responses_response_format(
                request_kwargs, response_format
            )
            response = self.client.responses(
                model=self.model_id,
                input=inputs["input"],
                instructions=inputs.get("instructions", None),
                **request_kwargs,
            )
        else:
            if is_batch:
                # Use asyncio to parallelize the requests
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    # No event loop in this thread, create one
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                return loop.run_until_complete(
                    asyncio.gather(
                        *[
                            self.ainvoke(
                                item,
                                endpoint=endpoint,
                                response_format=response_format,
                                parse_output=parse_output,
                                **request_kwargs,
                            )
                            for item in inputs
                        ]
                    )
                )

            response = self.client.completion(
                model=self.model_id,
                messages=inputs,
                text_completion=(endpoint == "text_completion"),
                response_format=response_format,
                **request_kwargs,
            )
        return self._parse_model_response(
            response,
            endpoint,
            response_format=response_format,
            parse_output=parse_output,
        )

    async def ainvoke(
        self,
        input_data: ModelInput,
        endpoint: EndpointType = None,
        response_format: type[BaseModel] | dict | None = None,
        parse_output: bool = False,
        **kwargs,
    ) -> ModelResponse | BatchResponse:
        """Asynchronously invoke the LiteLLM model.

        Parameters
        ----------
        input_data : ModelInput
            The input data for the model.
        endpoint : EndpointType, optional, default=None
            The endpoint to use for this request (overrides default class-level endpoint).
        response_format : type[BaseModel] or dict, optional, default=None
            The response format for structured output.
        parse_output : bool, optional
            Whether to parse the output using the response format. If ``True``,
            the output ``ModelResponse.output_parsed`` will be populated after
            successful parsing.
        **kwargs
            Additional keyword arguments for the request.

        Returns
        -------
        ModelResponse or BatchResponse
            The model's response.
        """
        endpoint = endpoint or self.endpoint
        inputs, is_batch, request_kwargs = self._prepare_request(
            input_data, endpoint, kwargs
        )

        async def _single_request(
            input_item: list[dict[str, str]] | dict[str, str],
            request_kwargs: dict[str, Any],
        ) -> Any:
            async with self._semaphore:
                if self._rate_limiter:
                    token_count_estimate = estimate_token_count(
                        self.model_id,
                        request_kwargs.get("max_tokens", 16),
                        messages=input_item
                        if endpoint != "responses"
                        else [
                            {
                                "role": "system",
                                "content": input_item.get("instructions", ""),
                            },
                            {"role": "user", "content": input_item["input"]},
                        ],
                    )
                    await self._rate_limiter.acquire(token_count_estimate)

                if endpoint == "responses":
                    request_kwargs = self._handle_responses_response_format(
                        request_kwargs, response_format
                    )
                    resp = await self.client.aresponses(
                        input=input_item["input"],
                        model=self.model_id,
                        instructions=input_item.get("instructions", None),
                        **request_kwargs,
                    )
                else:
                    resp = await self.client.acompletion(
                        model=self.model_id,
                        messages=input_item,
                        atext_completion=(endpoint == "text_completion"),
                        response_format=response_format,
                        **request_kwargs,
                    )
                return resp

        if is_batch:
            responses = await asyncio.gather(
                *[_single_request(item, request_kwargs) for item in inputs]
            )
            result = self._parse_model_response(
                responses,
                endpoint,
                response_format=response_format,
                parse_output=parse_output,
            )
        else:
            response = await _single_request(inputs, request_kwargs)
            result = self._parse_model_response(
                response,
                endpoint,
                response_format=response_format,
                parse_output=parse_output,
            )
        return result

    def _validate_and_normalize_input(
        self, input_data: ModelInput, endpoint: EndpointType
    ) -> tuple[
        Union[list[dict[str, str]], list[list[dict[str, str]]], dict[str, str]], bool
    ]:
        """Validate and normalize the input data for ``LiteLLMModel``.

        Return the normalized input and a flag indicating if it's a batch.
        """
        normalized_data, is_batch = super()._validate_and_normalize_input(
            input_data, endpoint
        )
        if endpoint == "text_completion" and not is_batch:
            normalized_data = [{"role": "user", "content": normalized_data[0]}]

        return normalized_data, is_batch

    def _prepare_request(
        self, input_data: ModelInput, endpoint: EndpointType, kwargs: Any
    ) -> tuple[
        Union[list[dict[str, str]], list[list[dict[str, str]]], dict[str, str]],
        bool,
        dict[str, Any],
    ]:
        """Prepare the request for LiteLLMModel.

        Returns a tuple containing: Inputs, batch flag, and request keyword arguments.
        """
        validate_endpoint(endpoint)
        inputs, is_batch = self._validate_and_normalize_input(input_data, endpoint)
        request_kwargs = {**self.kwargs, **kwargs}
        if request_kwargs.get("stream"):
            logger.warning(
                "Streaming response is not supported for the LM class. "
                "This parameter will be ignored.",
                stacklevel=2,
            )
            kwargs.pop("stream")

        if self.api_base:
            request_kwargs["api_base"] = self.api_base
        if self.api_key:
            request_kwargs["api_key"] = self.api_key

        return inputs, is_batch, request_kwargs

    def _handle_responses_response_format(
        self,
        request_kwargs: dict[str, Any],
        response_format: type[BaseModel] | dict | None = None,
    ) -> dict[str, Any]:
        """Handle the response format for the 'responses' endpoint.

        Returns Updated request keyword arguments.
        """
        if (
            "text" in request_kwargs
            and isinstance(request_kwargs["text"], dict)
            and "format" in request_kwargs["text"]
            and request_kwargs["text"]["format"]["type"] == "json_schema"
        ):
            raise ValueError(
                "When using the 'responses' endpoint with a JSON schema format, "
                "please use the 'response_format' parameter instead of setting "
                "'text.format'."
            )

        # Convert response_format to text.format if provided
        if response_format is not None:
            request_kwargs["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": response_format.__name__
                    if hasattr(response_format, "__name__")
                    else "ResponseFormat",
                    "schema": response_format.model_json_schema(),
                }
            }
        return request_kwargs

    def _parse_model_response(
        self,
        response: Union[
            ChatCompletionResponse,
            TextCompletionResponse,
            "ResponsesAPIResponse",
            list[ChatCompletionResponse],
            list[TextCompletionResponse],
            list["ResponsesAPIResponse"],
        ],
        endpoint: EndpointType,
        response_format: type[BaseModel] | dict | None = None,
        parse_output: bool = False,
    ) -> ModelResponse | BatchResponse:
        """Parse the LiteLLM response into ``ModelResponse`` or ``BatchResponse``."""

        def parse_single_response(
            resp: Union[
                ChatCompletionResponse,
                TextCompletionResponse,
                "ResponsesAPIResponse",
            ],
            cost: float = 0.0,
        ) -> ModelResponse:
            if endpoint == "responses":
                return self._parse_responses_api(
                    resp, response_format, parse_output, cost=cost
                )
            return self._parse_completions(
                resp, response_format, parse_output, cost=cost
            )

        if isinstance(response, list):
            responses: list[ModelResponse] = []
            for resp in response:
                cost = self._update_cost(resp)
                responses.append(parse_single_response(resp, cost=cost))
            return BatchResponse(responses=responses)

        cost = self._update_cost(response)
        return parse_single_response(response, cost=cost)

    def _parse_responses_api(
        self,
        response: "ResponsesAPIResponse",
        response_format: type[BaseModel] | dict | None = None,
        parse_output: bool = False,
        cost: float = 0.0,
    ) -> ModelResponse:
        """Parse a 'responses' endpoint API response.

        Parameters
        ----------
        response : ResponsesAPIResponse
            The raw API response.
        response_format : type[BaseModel] or dict, optional
            The response format for parsing.
        parse_output : bool, optional
            Whether to parse the output.
        cost : float, optional
            Monetary cost associated with this response.

        Returns
        -------
        ModelResponse
            The parsed response.
        """
        texts: list[str] = []
        tool_calls: list[ToolCall] = []
        for output in response.output:
            output_type = output.type if hasattr(output, "type") else output["type"]
            if output_type == "message":
                output_contents = (
                    output.content if hasattr(output, "content") else output["content"]
                )
                for content in output_contents or []:
                    if content.type == "output_text":
                        texts.append(content.text)
            if output_type == "function_call":
                output_id = output.id if hasattr(output, "id") else output["id"]
                output_name = output.name if hasattr(output, "name") else output["name"]
                output_arguments = (
                    output.arguments
                    if hasattr(output, "arguments")
                    else output["arguments"]
                )
                tool_calls.append(
                    ToolCall(
                        id=output_id,
                        name=output_name,
                        arguments=output_arguments,
                    )
                )

        output_text, output_parsed = _extract_output_and_parsed(
            texts, response_format, parse_output
        )
        usage = response.usage if hasattr(response, "usage") else response["usage"]
        token_usage = extract_token_usage(usage, is_responses=True)

        return ModelResponse(
            request_id=response.id,
            output_text=output_text,
            output_parsed=output_parsed,
            model_id=self.model_id,
            tool_calls=tool_calls if tool_calls else None,
            token_usage=token_usage,
            raw_response=response,
            cost=cost,
        )

    def _parse_completions(
        self,
        response: ChatCompletionResponse | TextCompletionResponse,
        response_format: type[BaseModel] | dict | None = None,
        parse_output: bool = False,
        cost: float = 0.0,
    ) -> ModelResponse:
        """Parse a completions endpoint response.

        Parameters
        ----------
        response : ChatCompletionResponse or TextCompletionResponse
            The raw completion response.
        response_format : type[BaseModel] or dict, optional
            The response format for parsing.
        parse_output : bool, optional
            Whether to parse the output.
        cost : float, optional
            Monetary cost associated with this response.

        Returns
        -------
        ModelResponse
            The parsed response.
        """
        contents = get_message_content(response)
        output_text, output_parsed = _extract_output_and_parsed(
            contents, response_format, parse_output
        )
        usage = response.usage if hasattr(response, "usage") else response["usage"]
        token_usage = extract_token_usage(usage)

        return ModelResponse(
            request_id=response.id,
            output_text=output_text,
            output_parsed=output_parsed,
            model_id=self.model_id,
            finish_reason=response.choices[-1].finish_reason,
            logprobs=self._get_logprobs(response),
            prompt_logprobs=self._get_prompt_logprobs(response),
            token_usage=token_usage,
            raw_response=response,
            cost=cost,
        )

    @staticmethod
    def _get_logprobs(
        response: ChatCompletionResponse | TextCompletionResponse,
    ) -> LogProbs | None:
        """Get prompt log probabilities from a completion response."""
        if not all(hasattr(c, "logprobs") for c in response.choices):
            return None

        tokens: list[str] = []
        logprobs: list[float] = []
        for c in response.choices:
            if isinstance(response, ChatCompletionResponse):
                for obj in c.logprobs.content:
                    if obj is None:
                        continue

                    tokens.append(obj.token)
                    logprobs.append(obj.logprob)
            else:
                tokens.extend(c.logprobs.tokens)
                logprobs.extend(c.logprobs.token_logprobs)
        return LogProbs(tokens=tokens, logprobs=logprobs)

    @staticmethod
    def _get_prompt_logprobs(
        response: ChatCompletionResponse | TextCompletionResponse,
    ) -> LogProbs | None:
        """Get log probabilities from a completion response."""
        if isinstance(response, ChatCompletionResponse) and (
            not hasattr(response, "prompt_logprobs")
            and "prompt_logprobs" not in response
        ):
            return None

        if isinstance(response, TextCompletionResponse) and (
            not all("prompt_logprobs" in c for c in response.choices)
        ):
            return None

        logprobs: list[float] = []
        token_ids: list[int] = []
        decoded_tokens: list[str] = []

        def _parse_prompt_logprob_dicts(prompt_logprobs: list[Optional[dict]]) -> None:
            for item in prompt_logprobs:
                if item is None:
                    continue

                key = next(iter(item))
                values = item.get(key)

                token_ids.append(key)
                logprobs.append(values["logprob"])
                decoded_tokens.append(values["decoded_token"])

        if isinstance(response, ChatCompletionResponse):
            if (
                not hasattr(response, "prompt_logprobs")
                or response.prompt_logprobs is None
            ):
                return None
            _parse_prompt_logprob_dicts(response["prompt_logprobs"])
        else:
            for c in response.choices:
                if not hasattr(c, "prompt_logprobs"):
                    continue

                prompt_logprobs: list[Optional[dict]] = c["prompt_logprobs"]
                if prompt_logprobs is not None:
                    _parse_prompt_logprob_dicts(prompt_logprobs)

        return LogProbs(logprobs=logprobs, tokens=decoded_tokens)

    def _update_cost(self, response: Any) -> float:
        """Update the running cost of the LLM requests."""
        cost = 0.0
        if self.model_id in model_cost:
            try:
                computed = completion_cost(completion_response=response)
                cost = float(computed or 0.0)
                self._cost += cost
            except Exception as e:
                logger.error(f"Failed to calculate cost: {e}")
            else:
                logger.debug(f"Running cost: ${float(self._cost):.10f}")
        return cost


class LiteLLMRouterModel(LiteLLMModel):
    """Class for interacting with multiple instances of a model via LiteLLM Router.

    Parameters
    ----------
    deployment_params : dict[str, LiteLLMParamsTypedDict]
        Deployment parameters for router. The keys should follow the format
        "<unique_prefix>---<model_id>" where <model_id> is common across all deployments.
    client_kwargs : dict, optional, default=None
        Additional client keyword arguments. These are passed to the ``litellm.Router``
        class.
    endpoint : EndpointType, optional, default="chat_completion"
        The default endpoint type to use for this model. The options are:
        - "chat_completion": "/v1/chat/completions" endpoint
        - "text_completion": "/v1/completions" endpoint
        - "responses": "/v1/responses" endpoint
    **kwargs
        Additional keyword arguments for the model.

    Examples
    --------
    >>> from naturalv2.models.lm import LiteLLMRouterModel

    >>> lm = LiteLLMRouterModel(
    ...     deployment_params={
    ...         "local---llama-3.3-70b": {
    ...             "model": "hosted_vllm/Llama-3.3-70B-Instruct",
    ...             "api_base": "http://localhost:8080/v1",
    ...         },
    ...         "together-ai---llama-3.3-70b: {
    ...             "model": "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo",
    ...             "api_key": "your_api_key_here",
    ...         },
    ...     ],
    ...     client_kwargs={
    ...         "routing_strategy": "simple-shuffle",
    ...     },
    ...     endpoint="chat_completion",
    ...     seed=42,
    ...     temperature=0.7,
    ... )

    >>> response = await lm.ainvoke(
    ...     "What is the significance of the Magna Carta?", max_tokens=256
    ... )
    """

    def __init__(
        self,
        deployment_params: dict[str, "LiteLLMParamsTypedDict"],
        client_kwargs: dict[str, Any] | None = None,
        endpoint: EndpointType = "chat_completion",
        **kwargs,
    ):
        """Initialize the LiteLLMRouterModel."""
        model_list, model_id = self._build_model_list(
            deployment_params=deployment_params
        )
        self.client_kwargs = {
            "model_list": model_list,
            **(client_kwargs or {}),
        }
        super().__init__(
            model_id=model_id,
            endpoint=endpoint,
            **kwargs,
        )

    def create_client(self) -> "litellm.Router":
        """Create the LiteLLM Router client.

        Returns
        -------
        litellm.Router
            An instance of the `litellm.Router` class.

        Raises
        ------
        ModuleNotFoundError
        """
        try:
            import litellm._logging
            from litellm.router import Router

            litellm._logging.verbose_logger.setLevel(logging.WARNING)
            litellm._logging.verbose_router_logger.setLevel(logging.WARNING)
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Please install 'litellm' to use LiteLLMRouterModel: `pip install litellm`"
            ) from e
        return Router(**self.client_kwargs)

    def _build_model_list(
        self, deployment_params: dict[str, "LiteLLMParamsTypedDict"]
    ) -> tuple[list[dict[str, Any]], str]:
        """Build the model list from deployment parameters.

        Parameters
        ----------
        deployment_params : dict[str, LiteLLMParamsTypedDict]
            Deployment parameters.

        Returns
        -------
        list[dict[str, Any]], str
            A tuple containing:
            - The model list for the router.
            - The common model_id.

        Raises
        ------
        ValueError
            If model names are not in the correct format or if model_ids do not match.

        """
        model_list = []
        model_ids: set[str] = set()
        for model_name, litellm_params in deployment_params.items():
            # Common model_id should be a suffix after '---'
            model_id = model_name.split("---")[-1]
            if not model_id:
                raise ValueError(
                    f"Invalid model name '{model_name}'. Must be in the format '<unique_prefix>---<model_id>'."
                )
            model_ids.add(model_id)

            deployment_dict = {"model_name": model_id}
            deployment_dict["litellm_params"] = litellm_params
            model_list.append(deployment_dict)

        if len(model_ids) != 1:
            raise ValueError(
                f"All model names must share the same base model_id. Found: {model_ids}"
            )

        return model_list, model_ids.pop()
