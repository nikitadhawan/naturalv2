import asyncio
import logging
import warnings
from dataclasses import dataclass
from typing import Any, Literal, Optional, Union

import httpx
from litellm import Router, model_cost, token_counter
from litellm.cost_calculator import completion_cost
from litellm.types.router import AllowedFailsPolicy, RetryPolicy, RouterGeneralSettings
from litellm.types.utils import ModelResponse, TextCompletionResponse
from omegaconf import DictConfig, OmegaConf
from typing_extensions import TypedDict

from naturalv2.utils import ListResponse


ResponseType = Union[ModelResponse, TextCompletionResponse]

logger = logging.getLogger(__name__)


class LLMParams(TypedDict, total=False):
    """Parameters for the LLM deployment."""

    #: The model name to use for the LLM. This is the model name used by the LLM provider.
    model: Optional[str]

    #: The provider name to use for the LLM.
    custom_llm_provider: Optional[str]

    #: The API key for the accessing the LLM.
    api_key: Optional[str]

    #: The API base URL for the LLM.
    api_base: Optional[str]

    #: The API version for the LLM.
    api_version: Optional[str]

    #: The organization ID for the LLM, if applicable (typically for OpenAI orgs).
    organization: Optional[Union[list, str]]

    #: The tokens per minute limit for the LLM requests.
    tpm: Optional[int]

    #: The requests per minute limit for the LLM requests.
    rpm: Optional[int]

    #: The maximum number of concurrent requests to the LLM.
    #: If tpm/rpm is set, and no max parallel request limit given, we use the
    # RPM or calculated RPM (tpm/1000/6) as the max parallel request limit.
    max_parallel_requests: Optional[int]

    #: The order of the LLM in the routing process.
    order: Optional[int]

    #: The weight of the LLM in the routing process. This sets how often the LLM is used.
    #: The higher the weight, the more often the LLM is used.
    weight: Optional[int]

    #: The number of seconds to timeout the LLM request if it takes too long.
    timeout: Optional[Union[float, str, httpx.Timeout]]

    #: The maximum number of times to retry the LLM request if it fails.
    max_retries: Optional[int]
    num_retries: Optional[int]

    #: The maximum budget for LLM requests. This only works for LLMs with known costs.
    max_budget: Optional[float]

    #: A mock response to return instead of making a real request.
    #: This is useful for testing and debugging.
    mock_response: Optional[Union[str, ModelResponse, Exception]]


class LM:
    """An interface for OpenAI-compatible LLM providers.

    This class supports routing requests to multiple deployments of the same model
    and provides caching and retrying capabilities.

    Parameters
    ----------
    model_name : str
        The name of the model to use. This should be a valid model name for the LLM provider.
    deployment_params : list[LLMParams]
        A list of dictionaries containing the deployment parameters for the LLM.
        Each dictionary should contain the following keys:
            - model: The model name to use for the LLM. This is the model name used by the LLM provider.
            - custom_llm_provider: The provider name to use for the LLM.
            - api_key: The API key for accessing the LLM.
            - api_base: The API base URL for the LLM.
            - api_version: The API version for the LLM.
            - organization: The organization ID for the LLM, if applicable (typically for OpenAI orgs).
            - tpm: The tokens per minute limit for the LLM requests.
            - rpm: The requests per minute limit for the LLM requests.
            - max_parallel_requests: The maximum number of concurrent requests to the LLM.
            - order: The order of the LLM in the routing process.
            - weight: The weight of the LLM in the routing process. This sets how often the LLM is used.
            - timeout: The number of seconds to timeout the LLM request if it takes too long.
            - max_retries: The maximum number of times to retry the LLM request if it fails.
            - num_retries: The maximum number of times to retry the LLM request if it fails.
            - max_budget: The maximum budget for LLM requests. This only works for LLMs with known costs.
            - mock_response: A mock response to return instead of making a real request.
    completion_type : Literal["chat", "text"], default="chat"
        The type of completion to use. This should be either "chat" or "text".
    routing_strategy : Literal[
        "simple-shuffle",
        "least-busy",
        "cost-based-routing",
        "usage-based-routing-v2"
    ], default="simple-shuffle"
        The routing strategy to use for the LLM. This should be one of the following:
            - simple-shuffle: randomly picks a deployment unless TPM, RPM or weight is set.
            - least-busy: picks the deployment with the least number of ongoing requests.
            - cost-based-routing: Picks deployment based on the lowest cost.
            - usage-based-routing-v2: routes to deployment with lowest TPM usage.
    cache_responses : Optional[bool], default=None
        Whether to cache the responses from the LLM. This should be either True or False.
    redis_host : Optional[str], default=None
        The host name of the Redis server to use for caching.
    redis_port : Optional[int], default=None
        The port number of the Redis server to use for caching.
    redis_password : Optional[str], default=None
        The password for the Redis server to use for caching.
    redis_client_kwargs : Optional[dict[str, Any]], default=None
        Additional keyword arguments to pass to the Redis client.
    cache_ttl : int, default=3600
        The time-to-live (TTL) for the cached responses, in seconds. Cache TTL is the
        duration for which the cached responses will be stored in the cache.
    allowed_failures : Optional[int], default=None
        The maximum number of allowed failures for the LLM requests.
    allowed_failures_policy : Optional[AllowedFailsPolicy], default=None
        The policy to use for allowed failures.
    cooldown_time : Optional[float], default=None
        The cooldown time to use for the LLM requests, in seconds.
    retry_after : int, default=2
        The number of seconds to wait before retrying the request if it fails.
    num_retries : int, default=4
        The maximum number of times to retry the request if it fails.
    retry_policy : Optional[RetryPolicy], default=None
        The policy to use for retries.
    seed : Optional[int], default=None
        The random seed to use for the LLM requests.
    extra_headers : Optional[dict[str, str]], default=None
        Additional headers to include in the LLM requests.
    default_request_level_params : dict[str, Any], default={}
        Additional request-level parameters to include in the LLM requests.
        This should be a dictionary of key-value pairs, where the keys are the parameter names
        and the values are the parameter values.

    Examples
    --------
    >>> from naturalv2.models.lm import LM

    >>> lm = LM(
    ...     model_name="Llama-3.3-70B-Instruct",
    ...     deployment_params=[
    ...         {
    ...             "model": "hosted_vllm/Llama-3.3-70B-Instruct",
    ...             "api_key": "EMPTY",
    ...             "api_base": "http://gpu054:8080/v1",
    ...         },
    ...         {
    ...             "model": "hosted_vllm/Llama-3.3-70B-Instruct",
    ...             "api_key": "EMPTY",
    ...             "api_base": "http://gpu051:8080/v1",
    ...         },
    ...     ],
    ...     completion_type="text",
    ... )

    >>> response = await lm(
    ...     "What is the significance of the Magna Carta?", max_tokens=256
    ... )
    """

    def __init__(
        self,
        model_name: str,
        deployment_params: list[LLMParams],
        completion_type: Literal["chat", "text"] = "chat",
        routing_strategy: Literal[
            "simple-shuffle",
            "least-busy",
            "cost-based-routing",
            "usage-based-routing-v2",
        ] = "simple-shuffle",
        # Caching
        cache_responses: Optional[bool] = None,
        redis_host: Optional[str] = None,
        redis_port: Optional[int] = None,
        redis_password: Optional[str] = None,
        redis_client_kwargs: Optional[dict[str, Any]] = None,
        cache_ttl: int = 3600,
        # Reliability
        allowed_failures: Optional[int] = None,
        allowed_failures_policy: Optional[AllowedFailsPolicy] = None,
        cooldown_time: Optional[float] = None,
        retry_after: int = 2,
        num_retries: int = 4,
        retry_policy: Optional[RetryPolicy] = None,
        # Request-level parameters (e.g. temperature, top_p, etc.)
        seed: Optional[int] = None,
        extra_headers: Optional[dict[str, str]] = None,
        **default_request_level_params: dict[str, Any],
    ) -> None:
        """Initialize the LM class."""
        if completion_type not in ["chat", "text"]:
            raise ValueError(
                "Expected ``completion_type`` to be one of ['chat', 'text'] but "
                f"got {completion_type}"
            )

        self._model_name = model_name
        self._model_list = self._build_model_list(
            model_name=model_name, deployment_params=deployment_params
        )
        self.completion_type = completion_type
        self.cache_responses = cache_responses

        self._num_retries = num_retries

        self._router = Router(
            model_list=self._model_list,
            cache_responses=cache_responses,
            caching_groups=[(model_name,)] if cache_responses is True else None,
            redis_host=redis_host,
            redis_port=redis_port,
            redis_password=redis_password,
            cache_kwargs=redis_client_kwargs or {},
            client_ttl=cache_ttl,
            routing_strategy=routing_strategy,  # rely on Router class to validate the strategy,
            allowed_fails=allowed_failures,
            allowed_fails_policy=allowed_failures_policy,
            cooldown_time=cooldown_time,
            retry_after=retry_after,
            num_retries=num_retries,
            model_group_retry_policy={model_name: retry_policy} if retry_policy else {},
            router_general_settings=RouterGeneralSettings(async_mode_only=True),
        )

        self._request_params = dict(
            seed=seed, extra_headers=extra_headers, **default_request_level_params
        )
        self._cost = 0.0

    @property
    def cost(self) -> float:
        return self._cost

    async def __call__(  # noqa: PLR0912, PLR0915
        self, prompt: Optional[str] = None, messages: Optional[list] = None, **kwargs
    ) -> ResponseType:
        """Make a request to the LLM.

        Parameters
        ----------
        prompt : Optional[str], default=None
            The prompt to use for the LLM request.
        messages : Optional[list], default=None
            The messages to use for the LLM request. This should be a list of dictionaries
            containing the role and content of each message. This is typically used for
            chat-based models.
        **kwargs : dict[str, Any]
            Additional keyword arguments to pass to the LLM request.

        Returns
        -------
        ResponseType
            The response from the LLM. This will be an instance of ``ModelResponse``
            if the completion type is "chat", or an instance of ``TextCompletionResponse``
            if the completion type is "text". The response will contain the generated
            text, the token usage, and other relevant information. The type of response will

        """
        request_params = self._prepare_request_params(
            prompt=prompt, messages=messages, **kwargs
        )
        response: ResponseType = await self._router.acompletion(
            model=self._model_name,
            **request_params,
            text_completion=self.completion_type == "text",
        )

        logger.debug("Token usage: %s", response.usage)
        self._update_cost(response)

        return response

    def call_sync(
        self, prompt: Optional[str] = None, messages: Optional[list] = None, **kwargs
    ) -> ResponseType:
        """Make a request to the LLM and return the response.

        This is a synchronous version of the __call__ method. It blocks until the
        response is received.

        Parameters
        ----------
        prompt : Optional[str], default=None
            The prompt to use for the LLM request.
        messages : Optional[list], default=None
            The messages to use for the LLM request. This should be a list of dictionaries
            containing the role and content of each message. This is typically used for
            chat-based models.
        **kwargs : dict[str, Any]
            Additional keyword arguments to pass to the LLM request.

        Returns
        -------
        ResponseType
            The response from the LLM.
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():  # run the async function in a new event loop
            # NOTE: This is a workaround for running async code in a Jupyter notebook
            # or other environments where the event loop is already running.
            import nest_asyncio

            nest_asyncio.apply()
            response = loop.run_until_complete(
                self.__call__(prompt=prompt, messages=messages, **kwargs)
            )
        else:  # run the async function directly
            response = asyncio.run(
                self.__call__(prompt=prompt, messages=messages, **kwargs)
            )

        return response

    def _build_model_list(
        self, model_name: str, deployment_params: list[LLMParams]
    ) -> list[dict[str, Any]]:
        """Get the model list from the deployment params."""
        model_list = []
        for deployment in deployment_params:
            model_params = {"model_name": model_name}
            if deployment.get("model") is None:
                deployment["model"] = model_name
            model_params["litellm_params"] = deployment

            model_list.append(model_params)
        return model_list

    def _prepare_request_params(
        self, prompt: str, messages: Optional[list] = None, **kwargs
    ) -> dict[str, Union[str, list]]:
        """Prepare the request parameters for the LLM request."""
        cache = kwargs.pop("cache", self.cache_responses)
        messages = messages or [{"role": "user", "content": prompt}]
        request_params = {**self._request_params, **kwargs}

        if request_params.get("stream"):
            logger.warning(
                "Streaming response is not supported for the LM class. "
                "This parameter will be ignored.",
                stacklevel=2,
            )
            kwargs.pop("stream")

        request_params.update(
            {
                "messages": messages,
                "cache": {"no-cache": not cache, "no-store": not cache},
            }
        )

        return request_params

    def _update_cost(self, response: ResponseType) -> None:
        """Update the running cost of the LLM requests."""
        if self._model_name in model_cost:
            try:
                self._cost += completion_cost(completion_response=response)
            except Exception as e:
                logger.error(f"Failed to calculate cost: {e}")

            logger.debug(f"Running cost: ${float(self._cost):.10f}")


# ---------------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------------- #
@dataclass
class LogprobsOutput:
    tokens: list[str]
    logprobs: list[float]


@dataclass
class PromptLogprobsOutput:
    logprobs: list[float]
    token_ids: list[int]
    decoded_tokens: list[str]


def get_message_content(response: ResponseType) -> list[Optional[str]]:
    return [
        c.message.content if hasattr(c, "message") else c["text"]
        for c in response.choices
    ]


def extract_list_response(response: ResponseType) -> Optional[list[list[Any]]]:
    response_strs = get_message_content(response)
    if not response_strs:
        return None

    response_list_objs = []
    for response_str in response_strs:
        try:
            response_list_obj = ListResponse.model_validate_json(response_str)
            response_list_objs.append(response_list_obj.output)
        except Exception as e:
            logger.error(f"Failed to parse response: {e}")
            continue

    return response_list_objs


def get_logprobs(response: ResponseType) -> Optional[list[LogprobsOutput]]:
    assert isinstance(response, (ModelResponse, TextCompletionResponse)), (
        f"Expected response to be an instance of ModelResponse or TextCompletionResponse "
        f"but got {type(response)}"
    )
    if not all(hasattr(c, "logprobs") for c in response.choices):
        return None

    outputs: list[LogprobsOutput] = []

    for c in response.choices:
        tokens: list[str] = []
        logprobs: list[float] = []

        if isinstance(response, ModelResponse):
            for obj in c.logprobs.content:
                if obj is None:
                    continue

                tokens.append(obj.token)
                logprobs.append(obj.logprob)
        else:
            tokens = c.logprobs.tokens
            logprobs = c.logprobs.token_logprobs

        outputs.append(LogprobsOutput(tokens=tokens, logprobs=logprobs))

    return outputs


def get_prompt_logprobs(response: ResponseType) -> Optional[PromptLogprobsOutput]:
    if isinstance(response, ModelResponse) and (
        not hasattr(response, "prompt_logprobs") and "prompt_logprobs" not in response
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

    if isinstance(response, ModelResponse):
        _parse_prompt_logprob_dicts(response["prompt_logprobs"])
    else:
        for c in response.choices:
            prompt_logprobs: list[Optional[dict]] = c["prompt_logprobs"]
            if prompt_logprobs is not None:
                _parse_prompt_logprob_dicts(prompt_logprobs)

    return PromptLogprobsOutput(
        logprobs=logprobs, token_ids=token_ids, decoded_tokens=decoded_tokens
    )


def estimate_token_count(
    model: str,
    max_tokens: int,
    n: int = 1,
    text: Optional[Union[str, list[str]]] = None,
    messages: Optional[list] = None,
    count_response_tokens: Optional[bool] = False,
) -> int:
    assert n > 0, f"Expected `n` to be greater than 0 but got {n}"
    assert max_tokens > 0, (
        f"Expected `max_tokens` to be greater than 0 but got {max_tokens}"
    )

    if "claude-3" in model:
        warnings.warn(
            "The model claude-3 is not supported for token counting. "
            "OpenAI tokenizer will be used, so the token count may not be accurate.",
            stacklevel=2,
        )

    # convert the text to messages since the LM class always uses message format
    messages = messages or [{"role": "user", "content": text}]

    input_token_count = token_counter(
        model=model,
        messages=messages,
        count_response_tokens=count_response_tokens,
        default_token_count=0,
    )
    output_token_count = max_tokens * n

    return input_token_count + output_token_count


def build_lm_instance_from_cfg(cfg: DictConfig):
    """Get LM instance from configuration."""
    # make a deep copy of the config
    cfg_copy = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    # change 'deployment' params from dict to list, ignore keys.
    cfg_copy["deployment_params"] = [
        value for _, value in cfg_copy["deployment_params"].items()
    ]

    # remove keys that are not needed for LM initialization
    cfg_copy.pop("local", None)
    cfg_copy.pop("get_response", None)

    return LM(**cfg_copy)
