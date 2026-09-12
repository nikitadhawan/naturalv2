import pytest
from pydantic import ValidationError

from naturalv2.pipeline import SampleValidationConfig


def test_sample_validation_requires_configured_threshold():
    with pytest.raises(ValidationError):
        SampleValidationConfig()


@pytest.mark.parametrize(
    "high_rejection_rate",
    [0.0, -0.1, 1.01, float("nan"), float("inf")],
)
def test_sample_validation_rejects_invalid_thresholds(high_rejection_rate):
    with pytest.raises(ValidationError):
        SampleValidationConfig(high_rejection_rate=high_rejection_rate)


def test_sample_validation_rejects_unknown_settings():
    with pytest.raises(ValidationError):
        SampleValidationConfig.model_validate(
            {"high_rejection_rate": 0.10, "unknown_setting": True}
        )
