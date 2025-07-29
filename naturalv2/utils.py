import asyncio
import logging
import os
import re
from itertools import product
from typing import Any, Coroutine

from pydantic import BaseModel, create_model

from naturalv2.evals.clinical_trial import (
    ArmGroup,
    ArmGroupType,
    ClinicalTrial,
    DesignAllocation,
    Outcome,
)


logger = logging.getLogger(__name__)


class ListResponse(BaseModel):
    output: list[str] | None


def create_response_format(
    name: str, keys: list[str], types: dict[str, Any] | None = None
) -> BaseModel:
    "Generate a Pydantic model with fields specified by the given keys."

    if types is None:
        types = dict.fromkeys(keys, Any)

    fields = {key: (types.get(key, Any)) for key in keys}

    return create_model(name, **fields)


async def concurrency_limited(coro: Coroutine, semaphore: asyncio.Semaphore) -> Any:
    """Run a coroutine with a concurrency limit using a semaphore."""

    async with semaphore:
        return await coro


def sanitize_filename(filename: str) -> str:
    """Sanitize filename by replacing disallowed characters with underscores."""

    return re.sub(r"[^\w\-.]", "_", filename)


def get_save_path(
    base_path: str,
    nct_id: str,
    model_name: str,
    extract_type: str,
    outcome: str | None = None,
) -> str:
    """Generate save path for extracted data."""
    return os.path.join(
        base_path,
        "results",
        f"{nct_id}",
        sanitize_filename(f"{model_name}_{extract_type}") + ".csv"
        if outcome is None
        else sanitize_filename(f"{model_name}_{extract_type}_{outcome}") + ".csv",
    )


def check_nonplacebo(intervention_names: list[str] | None) -> bool:
    nonplacebo_interventions = [
        name for name in (intervention_names or []) if "placebo" not in name.lower()
    ]
    return len(nonplacebo_interventions) > 0


def check_noncontrol(intervention_type: ArmGroupType | None) -> bool:
    return intervention_type != ArmGroupType.NO_INTERVENTION


def check_binary_endpoint(text: str) -> bool:
    binary_patterns = [
        r"""
    \b(                  # Word boundary to ensure full-word match
        number           | # "number of ..."
        count            | # "count of ..."
        proportion       | # "proportion of ..."
        percentage       | # "percentage of ..."
        percent          | # "percent of ..."
        rate             | # "rate of ..."
        fraction           # "fraction of ..."
    )\s+of\s+              # Required "of" phrase with spaces
    (                      # Second group: Who the proportion applies to
        participants     | # "participants"
        subjects         | # "subjects"
        patients         | # "patients"
        individuals      | # "individuals"
        people           | # "people"
        volunteers       | # "volunteers"
        enrollees          # "enrollees"
    )\b                   # Ensure we match full words
    """
    ]
    return any(
        re.search(pattern, text, re.IGNORECASE | re.VERBOSE)
        for pattern in binary_patterns
    )


def check_trial(trial: ClinicalTrial) -> tuple[dict[str, int], bool]:
    stats = {
        "total": 1,
        "randomized": 0,
        "multiple_noncontrol": 0,
        "nonhealthy": 0,
        "binary_endpoint": 0,
    }

    if (
        get_nested_value(trial, "protocolSection.designModule.designInfo.allocation")
        == DesignAllocation.RANDOMIZED
    ):
        stats["randomized"] = 1
        arm_groups: list[ArmGroup] | None = get_nested_value(
            trial, "protocolSection.armsInterventionsModule.armGroups"
        )
        noncontrol_arms = [
            arm for arm in (arm_groups or []) if check_noncontrol(arm.type)
        ]
        nonplacebo_arms = [
            arm for arm in noncontrol_arms if check_nonplacebo(arm.interventionNames)
        ]
        if len(nonplacebo_arms) >= 2:
            stats["multiple_noncontrol"] = 1

            inclusion_criteria = trial.protocolSection.eligibilityModule
            if inclusion_criteria and not inclusion_criteria.healthyVolunteers:
                stats["nonhealthy"] = 1
                binary = False

                endpoints: list[Outcome] | None = get_nested_value(
                    trial, "protocolSection.outcomesModule.primaryOutcomes"
                )
                for endpoint in endpoints or []:
                    if check_binary_endpoint(endpoint.measure):
                        stats["binary_endpoint"] = 1
                        binary = True
                        break
                if binary:
                    return stats, True
    return stats, False


def get_nested_value(data: Any, path: str) -> Any | None:
    """
    Gets a value from a deeply nested data structure using a path string.

    Parameters
    ----------
    data : Any
        The nested data structure (e.g., dictionary, list, or Pydantic BaseModel).
    path : str
        The path to the desired value (e.g., 'a.b.c.2.d.1.[1]'). Uses dot notation
        for dictionary keys and attribute access, and square brackets for list/tuple indices.

    Returns
    -------
    Any
        The value at the specified path, or ``None`` if any intermediate key/index
        is not found or if a value in the path is ``None``.
    """
    if data is None:
        return None

    if not path:
        return data

    parts = re.findall(r"([^.\[\]]+)|\[(\d+)\]", path)
    current = data

    for key_part, index_part in parts:
        try:
            if index_part:  # list/tuple index
                index = int(index_part)
                if isinstance(current, (list, tuple)):
                    current = current[index]
                else:
                    return None  # Not indexable
            elif key_part:  # dictionary key or attribute
                if isinstance(current, dict):
                    current = current[key_part]
                else:
                    # handle objects with attribute access (e.g., Pydantic models)
                    current = getattr(current, key_part)

            if current is None:
                return None  # early return if None is encountered

        except (KeyError, IndexError, AttributeError, TypeError):
            return None  # path doesn't exist

    return current


def qa_interleaved_enum(q_dct, options_dct, a_enum, to_enum):
    all_interleaved_options = []
    alph = ["a) ", "b) ", "c) ", "d) "]
    for option in a_enum:
        interleaved_enum = " \n\nMultiple Choice Questions"
        for num in range(len(to_enum)):
            key = to_enum[num]
            interleaved_enum += " \n\nQ: " + q_dct[key]
            interleaved_enum += " \nOptions: "
            for i in range(len(options_dct[key])):
                interleaved_enum += alph[i] + options_dct[key][i] + " "
            split_option = [i.split(":") for i in option.split(",")]
            interleaved_enum += " \nA: " + split_option[num][1][1:]
        all_interleaved_options.append(interleaved_enum)
    return all_interleaved_options


def concatenate_q(dct):
    keys = list(dct.keys())
    num = 1
    all_qs = " \nAnswer the following questions."
    for key in keys:
        all_qs += " \nQ" + str(num) + ": " + dct[key]
        num += 1
    all_qs += "\n"
    return all_qs


def enumerate_strings(string_map: dict[str, list[str]]) -> list[str]:
    combinations = product(*list(string_map.values()))
    result = []
    for combo in combinations:
        labeled = [f"A{i + 1}: {v}" for i, v in enumerate(combo)]
        result.append(", ".join(labeled))
    return result


def convert_enum_to_dicts(
    enumerated: list[str], enum_keys: list[str]
) -> list[dict[str, str]]:
    return_dcts = []
    for elem in enumerated:
        separate = _parse_key_value_pairs(elem)
        dct = {}
        for field in range(len(enum_keys)):
            dct[enum_keys[field]] = separate[field][1]
        return_dcts.append(dct)
    return return_dcts


def _parse_key_value_pairs(text: str) -> list[list[str]]:
    """Parse a string containing key-value pairs formatted as "A<digit>: value."""
    # Split on pattern A<digit>: but keep the delimiter
    parts = re.split(r"(A\d+):", text)

    # Remove empty strings and strip whitespace
    parts = [part.strip() for part in parts if part.strip()]

    # Group into pairs
    result = []
    for i in range(0, len(parts), 2):
        if i + 1 < len(parts):
            key = parts[i]
            value = parts[i + 1].rstrip(",")  # Remove trailing comma
            result.append([key, value])

    return result
