import pytest
from pydantic import ValidationError

from naturalv2.outcome_metadata import OutcomeBounds


def test_outcome_bounds_require_an_increasing_interval():
    with pytest.raises(ValidationError):
        OutcomeBounds(minimum=1, maximum=1)
