"""Validated metadata for continuous clinical outcomes."""

from typing import Self

from pydantic import BaseModel, ConfigDict, FiniteFloat, TypeAdapter, model_validator


class OutcomeBounds(BaseModel):
    """Inclusive numeric bounds for one continuous outcome."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum: FiniteFloat
    maximum: FiniteFloat

    @model_validator(mode="after")
    def validate_order(self) -> Self:
        """Require a non-empty interval."""
        if self.minimum >= self.maximum:
            raise ValueError("minimum must be less than maximum")
        return self


OUTCOME_BOUNDS_ADAPTER = TypeAdapter(dict[str, OutcomeBounds])
