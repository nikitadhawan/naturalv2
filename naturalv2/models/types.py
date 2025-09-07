from dataclasses import dataclass
from typing import Any, Iterator, Literal, Optional, Union

import pandas as pd


# Input types
PromptInput = Union[
    str,  # Single prompt
    list[str],  # Batch prompts
]
ChatInput = list[dict[str, str]]  # {"role": "user/assistant/system", "content": "..."}
ResponsesInput = dict[str, str]  # {"instructions": "...", "input": "..."}

ModelInput = Union[PromptInput, ChatInput, ResponsesInput]


# Endpoint types
EndpointType = Literal["text_completion", "chat_completion", "responses"]


# Output types
@dataclass
class ToolCall:
    """Represents a tool call."""

    id: str
    name: str
    arguments: Optional[str]


@dataclass
class TokenUsage:
    """Token usage information."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    reasoning_tokens: Optional[int] = None  # reasoning models, if applicable

    @property
    def output_tokens(self) -> int:
        """Alias for completion_tokens for consistency."""
        return self.completion_tokens


@dataclass
class LogProbs:
    tokens: list[str]
    logprobs: list[float]


@dataclass
class ModelResponse:
    model_id: str
    """The identifier of the model used."""

    output_text: str
    """The generated text from the model."""

    reasoning: str | None = None  # For models that show reasoning
    """The reasoning text from the model, if applicable."""

    logprobs: LogProbs | None = None  # For completion tokens
    """Log probabilities for the generated tokens, if available."""

    prompt_logprobs: LogProbs | None = None  # For prompt tokens
    """Log probabilities for the prompt tokens, if available."""

    tool_calls: list[ToolCall] | None = None
    """List of tool calls made by the model, if any."""

    # Metadata
    token_usage: TokenUsage | None = None
    """Token usage information, if available."""

    finish_reason: str | None = None
    """The reason why the model finished generating text."""

    raw_response: Any | None = None
    """The raw response from the model for debugging purposes."""

    request_id: str | None = None
    """The unique identifier for the request, if available."""

    output_parsed: Any | None = None  # Parsed output if applicable

    def __str__(self) -> str:
        return self.output_text


@dataclass
class BatchResponse:
    """Response for batch operations."""

    responses: list[ModelResponse]

    def __iter__(self) -> Iterator[ModelResponse]:
        return iter(self.responses)

    def __getitem__(self, index) -> ModelResponse:
        return self.responses[index]

    def __len__(self) -> int:
        return len(self.responses)


@dataclass
class CausalData:
    """Data container for causal estimation."""

    #: Covariates (features) matrix.
    X: pd.DataFrame

    #: Treatment assignment.
    T: pd.Series

    #: Observed Outcome.
    Y: pd.Series

    def __post_init__(self) -> None:
        """Validate data after initialization."""
        self.validate()

    def validate(self) -> None:
        """Validate data consistency."""
        if len(self.X) != len(self.T) or len(self.T) != len(self.Y):
            raise ValueError("X, T, and Y must have the same length")

        if self.X.isnull().any().any():
            raise ValueError("X contains missing values")

        if self.T.isnull().any() or self.Y.isnull().any():
            raise ValueError("T or Y contains missing values")
