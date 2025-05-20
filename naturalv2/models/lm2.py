import asyncio
import logging
import os
import re
import warnings
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Literal, Optional, Union

import httpx
import litellm
import tenacity
from litellm import acompletion, atext_completion, model_cost, token_counter
from litellm.cost_calculator import completion_cost
from litellm.types.utils import (
    ModelResponse,
    TextCompletionResponse,
)

from naturalv2.models.rate_limiter import RateLimiter
from naturalv2.models.rate_limiter.rate_limiter import RateLimiterAcquisitionHandle


ResponseType = Union[ModelResponse, TextCompletionResponse]

DEFAULT_MAX_TOKENS = 128


class LM:
    def __init__(
        self,
        model: str,
        completion_type: Literal["chat", "text"] = "chat",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        seed: Optional[int] = None,
        max_concurrent_requests: Optional[int] = None,
        requests_per_minute_limit: Optional[int] = None,
        tokens_per_minute_limit: Optional[int] = None,
        requests_per_day_limit: Optional[int] = None,
        tokens_per_day_limit: Optional[int] = None,
        max_request_burst: Optional[int] = None,
        max_token_burst: Optional[int] = None,
        cache: bool = False,
        cache_ttl: Optional[float] = None,
        disk_cache_dir: Optional[str] = None,
        num_retries: int = 8,
        retry_wait_multiplier: int = 1,
        retry_wait_min: int = 1,
        retry_wait_max: int = 60,
        retry_if_exception_func: Optional[callable] = None,
        **kwargs,
    ) -> None:
        assert completion_type in ["chat", "text"], (
            f"Expected ``completion_type`` to be one of ['chat', 'text'] but got {completion_type}"
        )

        self.model = model
        self.completion_type = completion_type
        self.cache = cache
        self._num_retries = num_retries
        self._retry_wait_multiplier = retry_wait_multiplier
        self._retry_wait_min = retry_wait_min
        self._retry_wait_max = retry_wait_max

        self._cost = 0.0
        self._request_params = dict(
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            **kwargs,
        )

        if cache:
            if cache_ttl is None:  # set ttl to 2 days if not provided
                cache_ttl = float(2 * 24 * 60 * 60)

            if not isinstance(cache_ttl, float) and cache_ttl <= 0.0:
                raise ValueError(
                    f"Expected ``cache_ttl`` to be a positive float but got {cache_ttl}"
                )

            func = (
                litellm.enable_cache if litellm.cache is None else litellm.update_cache
            )
            func(
                type="disk",
                disk_cache_dir=os.getenv("LITELLM_CACHE_DIR") or disk_cache_dir,
                ttl=cache_ttl,
            )

        self._concurrency_limiter = (
            asyncio.Semaphore(max_concurrent_requests)
            if max_concurrent_requests
            else None
        )
        self._limiter: Optional[RateLimiter] = None
        if requests_per_minute_limit:
            self._limiter = RateLimiter(
                requests_per_minute=requests_per_minute_limit,
                tokens_per_minute=tokens_per_minute_limit,
                requests_per_day=requests_per_day_limit,
                tokens_per_day=tokens_per_day_limit,
                max_request_burst=max_request_burst,
                max_token_burst=max_token_burst,
            )

        self._retry_stop = tenacity.stop_after_attempt(num_retries)
        self._retry_wait = tenacity.wait_exponential(
            multiplier=retry_wait_multiplier, min=retry_wait_min, max=retry_wait_max
        )
        self._retry_if = tenacity.retry_if_exception(
            retry_if_exception_func or is_retryable_exception
        )
        # configure logging during retries
        self._retry_before_sleep = tenacity.before_sleep_log(
            logging.getLogger(__name__), logging.INFO
        )

    @property
    def cost(self) -> float:
        return self._cost

    async def __call__(  # noqa: PLR0912, PLR0915
        self, prompt: Optional[str] = None, messages: Optional[list] = None, **kwargs
    ) -> ResponseType:
        handle: Optional[RateLimiterAcquisitionHandle] = None
        response: Optional[ResponseType] = None
        request_params = self._prepare_request_params(
            prompt=prompt, messages=messages, **kwargs
        )
        estimated_tokens = estimate_token_count(
            model=self.model,
            max_tokens=request_params.get("max_tokens", DEFAULT_MAX_TOKENS),
            n=request_params.get("n", 1),
            text=request_params.get("prompt")
            if self.completion_type == "text"
            else None,
            messages=request_params.get("messages")
            if self.completion_type == "chat"
            else None,
            count_response_tokens=True,
        )
        actual_tokens: int = estimated_tokens

        num_rate_limit_retries = 0

        while True:  # retry rate limit errors until success or max retries
            try:
                retrying = tenacity.AsyncRetrying(
                    stop=self._retry_stop,
                    wait=self._retry_wait,
                    retry=self._retry_if,
                    before_sleep=self._retry_before_sleep,
                    reraise=True,  # if all retries fail
                )

                if self._limiter is not None:
                    handle = await self._limiter.acquire(estimated_tokens)
                    if handle is None:
                        raise asyncio.CancelledError(
                            "Rate limit acquisition failed or was cancelled."
                        )

                should_break = False

                try:
                    response = await retrying(
                        self._dispatch_concurrent_request, request_params
                    )
                    should_break = True
                except tenacity.RetryError as e:
                    # API call failed for retryable reasons
                    logging.error(
                        f"API call failed after maximum retries: {e.last_attempt}",
                        exc_info=True,
                    )
                    raise e
                except litellm.RateLimitError as e:
                    # handle RateLimitError from litellm
                    logging.error(f"Rate limit error: {e}", exc_info=True)
                    num_rate_limit_retries += 1

                    error_response_obj = getattr(e, "response", None)  # httpx.Response
                    response_headers = None
                    if error_response_obj and hasattr(error_response_obj, "headers"):
                        response_headers = error_response_obj.headers

                    if self._limiter is not None and handle is not None:
                        logging.debug(
                            f"Releasing {estimated_tokens} estimated tokens due to rate limit."
                        )

                        await self._limiter.adjust(
                            handle,
                            actual_tokens=0,  # release without usage
                            response_headers=response_headers,
                        )
                        handle = None  # release the handle

                    if num_rate_limit_retries >= self._num_retries:
                        logging.error("Maximum rate limit retries exceeded.")
                        raise e

                    # determine how long to wait before retrying
                    retry_after = self._retry_wait_min * (
                        self._retry_wait_multiplier ** (num_rate_limit_retries - 1)
                    )
                    retry_after = min(retry_after, self._retry_wait_max)

                    # try getting wait time from error response
                    wait_match = re.search(r"try again in ([\d\.]+)ms", str(e))
                    if wait_match:
                        server_wait_ms = float(wait_match.group(1))
                        server_wait_sec = server_wait_ms / 1000.0

                        # use server wait time if it's longer than the exponential backoff
                        retry_after = max(retry_after, server_wait_sec + 0.1)
                        logging.debug(
                            f"Using server-suggested wait time of {server_wait_sec} seconds."
                        )
                    else:
                        server_retry_after = (
                            response_headers.get("retry-after")
                            if response_headers
                            else None
                        )

                        if server_retry_after:
                            try:
                                server_wait_sec = float(server_retry_after)
                                retry_after = max(retry_after, server_wait_sec + 0.1)
                                logging.debug(
                                    f"Using Retry-After header wait time: {retry_after:.2f}s"
                                )
                            except ValueError:
                                logging.warning(
                                    f"Could not parse Retry-After header: {retry_after}"
                                )

                    logging.debug(f"Waiting {retry_after:.2f} seconds before retrying.")
                    await asyncio.sleep(retry_after)
                    continue

            except Exception as e:
                logging.error(f"Operation failed with error: {e}", exc_info=True)

                if self._limiter is not None and handle is not None:
                    # Release consumed tokens as we don't know actual usage
                    await self._limiter.adjust(handle, actual_tokens=0)
                    handle = None
                raise e
            finally:  # adjust the rate limiter regardless of success or failure
                if (
                    response is not None
                    and self._limiter is not None
                    and handle is not None
                ):
                    try:
                        actual_tokens = response.usage.total_tokens
                        logging.debug(
                            f"Adjusting rate limiter. Estimate={estimated_tokens}, Actual={actual_tokens}"
                        )
                        await self._limiter.adjust(
                            handle,
                            actual_tokens,
                            response._response_headers if response else None,
                        )
                    except Exception as e:
                        logging.error(
                            f"Failed to adjust rate limiter: {e}", exc_info=True
                        )

            if should_break:
                break

        if response is None:
            raise RuntimeError(
                "Internal logic error: response is ``None`` after successful API call."
            )

        logging.debug("Token usage: %s", response.usage)
        self._update_cost(response)  # update the cost after a successful call

        return response

    async def _dispatch_concurrent_request(self, request_params: dict) -> ResponseType:
        async with self._concurrency_limiter or nullcontext():
            logging.debug("Calling API with request params: %s", request_params)
            completion = (
                atext_completion if self.completion_type == "text" else acompletion
            )
            return await completion(**request_params)

    def _prepare_request_params(
        self, prompt: str, messages: Optional[list] = None, **kwargs
    ) -> dict[str, Union[str, list]]:
        cache = kwargs.pop("cache", self.cache)
        messages = messages or [{"role": "user", "content": prompt}]
        request_params = {**self._request_params, **kwargs}

        if request_params.get("stream"):
            logging.warning(
                "Streaming response is not supported for the LM class. "
                "This parameter will be ignored.",
                stacklevel=2,
            )
            kwargs.pop("stream")

        add_text_completion_prefix = request_params.pop("get_response", True)

        if self.completion_type == "text":
            # Use the API key and base from the request, or from the environment variables
            model, provider = self._get_model_name_and_provider()

            api_key = request_params.pop("api_key", None) or os.getenv(
                f"{provider.upper()}_API_KEY" if provider else ""
            )
            api_base = request_params.pop("api_base", None) or os.getenv(
                f"{provider.upper()}_API_BASE" if provider else ""
            )

            # Build the prompt from the messages.
            prompt = "\n\n".join([x["content"] for x in messages])
            if add_text_completion_prefix:
                prompt += "\n\nBEGIN RESPONSE:"

            request_params.update(
                {
                    "model": f"text-completion-openai/{model}",
                    "prompt": prompt,
                    "api_key": api_key,
                    "api_base": api_base,
                    "num_retries": 0,  # the LM class handles retries, not litellm
                    "cache": {"no-cache": not cache, "no-store": not cache},
                }
            )
        else:
            request_params.update(
                {
                    "model": self.model,
                    "messages": messages,
                    "num_retries": 0,  # the LM class handles retries, not litellm
                    "cache": {"no-cache": not cache, "no-store": not cache},
                }
            )

        return request_params

    def _get_model_name_and_provider(self) -> tuple[str, Optional[str]]:
        model_names = self.model.split("/", 1)
        provider, model = (
            model_names[0] if len(model_names) > 1 else None,
            model_names[-1],
        )

        return model, provider

    def _update_cost(self, response: ResponseType) -> None:
        if self.model in model_cost:
            try:
                self._cost += completion_cost(completion_response=response)
            except Exception as e:
                logging.error(f"Failed to calculate cost: {e}")

            logging.debug(f"Running cost: ${float(self._cost):.10f}")


# ---------------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------------- #
class RateLimitAcquisitionError(Exception):
    """Raised when rate limit acquisition fails."""


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


def is_retryable_exception(e: BaseException) -> bool:
    if isinstance(e, httpx.TimeoutException):
        return True
    if isinstance(e, httpx.NetworkError):
        return True
    return bool(isinstance(e, httpx.HTTPStatusError) and e.response.status_code >= 500)
