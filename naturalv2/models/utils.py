import json
import logging
import warnings
from typing import TYPE_CHECKING, Any, Optional, Union

from litellm import token_counter
from pydantic import BaseModel, create_model

from naturalv2.models.types import (
    BatchResponse,
    EndpointType,
    ModelResponse,
    TokenUsage,
)


if TYPE_CHECKING:
    from litellm.types.utils import ModelResponse as ChatCompletionResponse
    from litellm.types.utils import TextCompletionResponse

logger = logging.getLogger(__name__)


def parse_model_output_with_format(
    output_text: str, response_format: type[BaseModel] | dict | None
) -> Any | None:
    """Parse model output using the provided response format.

    Parameters
    ----------
    output_text : str
        The raw output text from the model.
    response_format : type[BaseModel] | dict | None
        The response format to parse the output. Can be a pydantic model class
        or a dictionary with 'schema' and optional 'name' keys.

    Returns
    -------
    Any | None
        The parsed output as an instance of the response format, or None if parsing
        fails.

    """
    if not response_format:
        return None

    try:
        if hasattr(response_format, "model_validate_json"):
            return response_format.model_validate_json(output_text)

        if isinstance(response_format, dict):
            schema = response_format.get("schema")
            name = response_format.get("name", "ParsedOutput")

            # Dynamically create a pydantic model from schema
            # If schema is a dict, use it directly; if str, parse it
            if isinstance(schema, str):
                schema = json.loads(schema)
            model_cls = create_model(name, __base__=BaseModel)
            model_cls.model_json_schema = lambda: schema
            return model_cls.model_validate_json(output_text)
    except Exception as e:
        logger.error(f"Failed to parse output with format: {e}")
    return None


def estimate_token_count(
    model: str,
    max_tokens: int,
    n: int = 1,
    text: Union[str, list[str]] | None = None,
    messages: list[dict[str, str]] | None = None,
    count_response_tokens: bool | None = False,
) -> int:
    """Estimate the total token count for a request to a language model.

    Parameters
    ----------
    model : str
        The model name.
    max_tokens : int
        The maximum number of tokens to generate in the response.
    n : int, default=1
        The number of completions to generate.
    text : str or list of str, optional, default=None
        The input text prompt(s). Either `text` or `messages` must be provided.
    messages : list of dict, optional, default=None
        The input messages in chat format. Either `text` or `messages` must be provided.
    count_response_tokens : bool, optional, default=False
        Whether to include the response tokens in the count.

    Returns
    -------
    int
        The estimated total token count.

    Raises
    ------
    ValueError
        If neither `text` nor `messages` is provided.
    """

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


def validate_endpoint(
    endpoint: str, supported_endpoints: list[str] | None = None
) -> None:
    """Validate that the endpoint is supported.

    Parameters
    ----------
    endpoint : str
        The endpoint to validate.
    supported_endpoints : list of str, optional
        The list of supported endpoints. If None, all EndpointType values are supported.

    Raises
    ------
    ValueError
        If the endpoint is not supported.
    """

    if supported_endpoints is None:
        supported_endpoints = EndpointType.__args__

    if endpoint not in supported_endpoints:
        raise ValueError(
            f"Expected ``endpoint`` to be one of {supported_endpoints}, "
            f"but got {endpoint}"
        )


def extract_token_usage(
    usage_obj: Any, is_responses: bool = False
) -> Optional[TokenUsage]:
    """Extract token usage from a usage object.

    Parameters
    ----------
    usage_obj : Any
        The usage object from the model response.
    is_responses : bool, default=False
        Whether the usage object is from a responses endpoint.

    Returns
    -------
    TokenUsage | None
        The extracted token usage, or None if usage_obj is None.
    """
    if not usage_obj:
        return None
    if is_responses:
        return TokenUsage(
            prompt_tokens=usage_obj.input_tokens,
            completion_tokens=usage_obj.output_tokens,
            reasoning_tokens=getattr(
                usage_obj.output_tokens_details, "reasoning_tokens", None
            )
            if getattr(usage_obj, "output_tokens_details", None)
            else None,
            total_tokens=usage_obj.total_tokens,
        )
    return TokenUsage(
        prompt_tokens=usage_obj.prompt_tokens,
        completion_tokens=usage_obj.completion_tokens,
        reasoning_tokens=getattr(
            usage_obj.completion_tokens_details, "reasoning_tokens", None
        )
        if getattr(usage_obj, "completion_tokens_details", None)
        else None,
        total_tokens=usage_obj.total_tokens,
    )


def get_message_content(
    response: Union["ChatCompletionResponse", "TextCompletionResponse"],
) -> list[Optional[str]]:
    """Get message content from a completion response.

    Parameters
    ----------
    response : ChatCompletionResponse or TextCompletionResponse
        The response object from the model.

    Returns
    -------
    list of str or None
        The list of message contents from the response choices.
    """
    # Handle both dict and object responses
    choices = response.choices if hasattr(response, "choices") else response["choices"]

    result = []
    for c in choices:
        if isinstance(c, dict):
            # Dict-based choice (from LiteLLM)
            if "message" in c:
                result.append(c["message"]["content"])
            else:
                result.append(c["text"])
        # Object-based choice
        elif hasattr(c, "message") and c.message is not None:
            result.append(c.message.content)
        elif hasattr(c, "text"):
            result.append(c.text)
        else:
            result.append(None)

    return result


def _extract_output_and_parsed(
    texts: list[str],
    response_format: type[BaseModel] | dict | None,
    parse_output: bool,
) -> tuple[str, Any]:
    """Extract output text and parsed output from a list of texts."""
    parsed: list[Any] = []
    for text in texts:
        if parse_output and response_format is not None:
            parsed.append(parse_model_output_with_format(text, response_format))
    output_text = texts[0] if len(texts) == 1 else "".join(texts)
    output_parsed = parsed[0] if parsed and len(parsed) == 1 else parsed or None
    return output_text, output_parsed


class TokenTracker:
    """
    Tracks token usage across pipeline stages and for the entire pipeline.

    Examples
    --------
    >>> tracker = TokenTracker()
    >>> tracker.add("stage1", response)
    >>> tracker.add("stage2", response2)
    >>> tracker.get_stage_stats("stage1")
    >>> tracker.get_total_stats()
    """

    def __init__(self) -> None:
        """Initialize the TokenTracker."""
        self._stage_tokens: dict[str, dict[str, int]] = {}
        self._total_tokens: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "reasoning_tokens": 0,
        }

    def add(self, stage: str, response: ModelResponse | BatchResponse) -> None:
        """
        Add token usage from a ModelResponse or BatchResponse to a stage.

        Parameters
        ----------
        stage : str
            The name of the pipeline stage.
        response : ModelResponse or BatchResponse
            The response object(s) to accumulate tokens from.
        """
        if (
            isinstance(response, list)
            or hasattr(response, "__iter__")
            and not isinstance(response, ModelResponse)
        ):
            responses = list(response)
        else:
            responses = [response]

        stage_stats = self._stage_tokens.setdefault(
            stage,
            {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "reasoning_tokens": 0,
            },
        )

        for resp in responses:
            usage: TokenUsage = getattr(resp, "token_usage", None)
            if usage:
                stage_stats["prompt_tokens"] += usage.prompt_tokens or 0
                stage_stats["completion_tokens"] += usage.completion_tokens or 0
                stage_stats["total_tokens"] += usage.total_tokens or 0
                stage_stats["reasoning_tokens"] += usage.reasoning_tokens or 0

                self._total_tokens["prompt_tokens"] += usage.prompt_tokens or 0
                self._total_tokens["completion_tokens"] += usage.completion_tokens or 0
                self._total_tokens["total_tokens"] += usage.total_tokens or 0
                self._total_tokens["reasoning_tokens"] += usage.reasoning_tokens or 0

    def get_stage_stats(self, stage: str) -> dict[str, int]:
        """Get token usage stats for a specific stage.

        Parameters
        ----------
        stage : str
            The name of the pipeline stage.

        Returns
        -------
        dict[str, int]
            Token usage stats for the stage.
        """
        return self._stage_tokens.get(
            stage,
            {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "reasoning_tokens": 0,
            },
        )

    def get_total_stats(self) -> dict[str, int]:
        """Get total token usage stats across all stages.

        Returns
        -------
        dict[str, int]
            Total token usage stats.
        """
        return dict(self._total_tokens)

    def reset(self) -> None:
        """Reset all tracked token stats."""
        self._stage_tokens.clear()
        self._total_tokens = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "reasoning_tokens": 0,
        }
