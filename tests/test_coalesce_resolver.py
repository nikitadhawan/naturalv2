import pytest
from omegaconf import OmegaConf

import naturalv2.hydra_setup  # noqa: F401 # Ensure custom resolvers are registered


@pytest.mark.parametrize(
    "description, config_data, lookup_key, expected_value",
    [
        # --- Basic Success and Fallback ---
        (
            "SUCCESS: Should return the value of the first key when it exists",
            {
                "model": {"max_parallel": 16},
                "default_max": 8,
                "result": "${coalesce:model.max_parallel, default_max}",
            },
            "result",
            16,
        ),
        (
            "FALLBACK: Should skip the first key if it is explicitly null",
            {
                "model": {"max_parallel": None},
                "default_max": 8,
                "result": "${coalesce:model.max_parallel, default_max}",
            },
            "result",
            8,
        ),
        (
            "FALLBACK: Should skip the first key if it is missing entirely",
            {
                "model": {},  # max_parallel is missing
                "default_max": 8,
                "result": "${coalesce:model.max_parallel, default_max}",
            },
            "result",
            8,
        ),
        (
            "FALLBACK: Should handle multiple fallbacks correctly",
            {
                "model": {"max_parallel": None},
                "client": {"default_max": None},
                "global_default": 8,
                "result": "${coalesce:model.max_parallel, client.default_max, global_default}",
            },
            "result",
            8,
        ),
        # --- Data Type Tests ---
        (
            "DATATYPE: Should work correctly with strings",
            {
                "model_id": "gpt-4",
                "default_id": "gpt-3.5",
                "result": "${coalesce:model_id, default_id}",
            },
            "result",
            "gpt-4",
        ),
        (
            "DATATYPE: Should work correctly with booleans (True)",
            {
                "use_gpu": True,
                "default_gpu": False,
                "result": "${coalesce:use_gpu, default_gpu}",
            },
            "result",
            True,
        ),
        (
            "DATATYPE: Should work correctly with booleans (False)",
            {
                "use_gpu": None,
                "default_gpu": False,
                "result": "${coalesce:use_gpu, default_gpu}",
            },
            "result",
            False,
        ),
        (
            "DATATYPE: Should work correctly with floats",
            {
                "learning_rate": 0.001,
                "default_lr": 0.01,
                "result": "${coalesce:learning_rate, default_lr}",
            },
            "result",
            0.001,
        ),
        (
            "DATATYPE: Should correctly handle zero as a valid value",
            {
                "retries": None,
                "default_retries": 0,
                "result": "${coalesce:retries, default_retries}",
            },
            "result",
            0,
        ),
        # --- Edge Cases ---
        (
            "EDGE CASE: Should return None if all candidate keys are missing or null",
            {
                "model": {"max_parallel": None},
                "client": {},
                "result": "${coalesce:model.max_parallel, client.default_max}",
            },
            "result",
            None,
        ),
        (
            "EDGE CASE: Should work with deeply nested keys",
            {
                "pipeline": {"model": {"client": {"kwargs": {"max_req": 32}}}},
                "default_max": 8,
                "result": "${coalesce:pipeline.model.client.kwargs.max_req, default_max}",
            },
            "result",
            32,
        ),
        (
            "EDGE CASE: The final fallback can be another interpolation",
            {
                "defaults": {"workers": 4},
                "model": {},
                "result": "${coalesce:model.workers, defaults.workers}",
            },
            "result",
            4,
        ),
    ],
)
def test_coalesce_scenarios(description, config_data, lookup_key, expected_value):
    """
    Tests various scenarios for the 'coalesce' resolver using parameterization.
    """
    # Arrange
    conf = OmegaConf.create(config_data)

    # Act
    resolved_value = conf[lookup_key]

    # Assert
    assert resolved_value == expected_value, f"Failed test: {description}"


def test_coalesce_with_no_arguments():
    """
    Tests that calling coalesce with no arguments returns None.
    """
    # Arrange
    conf = OmegaConf.create({"result": "${coalesce:}"})

    # Act
    resolved_value = conf.result

    # Assert
    assert resolved_value is None


def test_coalesce_inside_a_list():
    """
    Ensures the resolver works correctly when used inside a list structure.
    """
    # Arrange
    config_data = {
        "params": [
            {"val": "${coalesce:a,b}"},
        ],
        "a": None,
        "b": 100,
    }
    conf = OmegaConf.create(config_data)

    # Act
    resolved_value = conf.params[0].val

    # Assert
    assert resolved_value == 100
