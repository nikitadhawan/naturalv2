import logging
import os
import warnings
from dataclasses import dataclass
from typing import Literal, Optional, Union

import litellm
from litellm import (
    completion,
    model_cost,
    text_completion,
    token_counter,
)
from litellm.cost_calculator import completion_cost
from litellm.types.utils import (
    ModelResponse,
    TextCompletionResponse,
)


ResponseType = Union[ModelResponse, TextCompletionResponse]


class LM:
    def __init__(
        self,
        model: str,
        completion_type: Literal["chat", "text"] = "chat",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 64,
        num_retries: int = 3,
        seed: Optional[int] = None,
        cache: bool = False,
        disk_cache_dir: Optional[str] = None,
        **kwargs,
    ) -> None:
        assert completion_type in ["chat", "text"], (
            f"Expected ``completion_type`` to be one of ['chat', 'text] but got {completion_type}"
        )

        self.model = model
        self.completion_type = completion_type
        self.cache = cache
        self.num_retries = num_retries

        self._cost = 0.0
        self._request_params = dict(
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            drop_params=True,  # drop unsupported OpenAI params for other providers/models
            **kwargs,
        )

        if cache:
            func = (
                litellm.enable_cache if litellm.cache is None else litellm.update_cache
            )
            func(
                type="disk",
                disk_cache_dir=os.getenv("LITELLM_CACHE_DIR") or disk_cache_dir,
            )

    @property
    def cost(self) -> float:
        return self._cost

    def __call__(
        self, prompt: Optional[str] = None, messages: Optional[list] = None, **kwargs
    ) -> ResponseType:
        cache = kwargs.pop("cache", self.cache)
        messages = messages or [{"role": "user", "content": prompt}]
        request_params = {**self._request_params, **kwargs}

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

            response = text_completion(
                model=f"text-completion-openai/{model}",
                prompt=prompt,
                api_key=api_key,
                api_base=api_base,
                num_retries=self.num_retries,
                cache={"no-cache": not cache, "no-store": not cache},
                **request_params,
            )
        else:
            response = completion(
                model=self.model,
                messages=messages,
                num_retries=self.num_retries,
                cache={"no-cache": not cache, "no-store": not cache},
                **request_params,
            )

        logging.debug("Token usage: %s", response.usage)
        self._update_cost(response)

        return response

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


def get_token_usage(response: ResponseType) -> tuple[int, int, int]:
    if not hasattr(response, "usage") or (
        hasattr(response, "usage") and response.usage is None
    ):
        return 0, 0, 0

    num_prompt_tokens: int = response.usage.prompt_tokens
    num_completion_tokens: int = response.usage.completion_tokens
    num_total_tokens: int = response.usage.total_tokens

    return num_prompt_tokens, num_completion_tokens, num_total_tokens
