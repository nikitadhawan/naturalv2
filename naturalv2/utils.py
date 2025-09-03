"""Utility functions."""

import asyncio
import logging
import os
import re
import string
from itertools import product
from typing import Any, Coroutine, Literal, get_args, get_origin

from pydantic import BaseModel, create_model, model_validator

from naturalv2.clinical_trial import (
    ArmGroup,
    ArmGroupType,
    ClinicalTrial,
    DesignAllocation,
    Outcome,
)


logger = logging.getLogger(__name__)


class ListResponse(BaseModel):
    """Response model for list outputs."""

    output: list[str] | None


def create_response_format(
    name: str, keys: list[str], types: dict[str, Any] | None = None
) -> type[BaseModel]:
    """Generate a Pydantic model with fields specified by the given keys.

    If `types` is provided, it should be a dictionary mapping each key to its
    corresponding type. If `types` is None, all fields will default to Any type.
    Literal types will be validated to match the provided options, allowing
    case-insensitive matching.

    Parameters
    ----------
    name : str
        The name of the model to be created.
    keys : list[str]
        A list of keys that will be used as field names in the model.
    types : dict[str, Any] | None
        A dictionary mapping keys to their respective types. If None, defaults
        to Any type for all keys.

    Returns
    -------
    type[BaseModel]
        A Pydantic model class with fields corresponding to the provided keys.
        If any field is a Literal type, it will be validated to allow case-insensitive
        matching of its values.
    """
    if types is None:
        types = dict.fromkeys(keys, str)

    fields = {}
    literal_fields = {}

    for key in keys:
        field_type = types.get(key, str)
        # OpenAI API errors with Any; using str
        fields[key] = (field_type, ...)

        if get_origin(field_type) is Literal:
            literal_fields[key] = get_args(field_type)

    # Create base model
    BaseModelClass = create_model(name, **fields)  # noqa: N806

    if not literal_fields:
        return BaseModelClass

    # Create class with model validator
    class ValidatedModel(BaseModelClass):
        @model_validator(mode="before")
        @classmethod
        def normalize_literals(cls, values):
            if isinstance(values, dict):
                for field_name, literal_options in literal_fields.items():
                    if field_name in values:
                        value = values[field_name]
                        if isinstance(value, str):
                            value_lower = value.lower()
                            for option in literal_options:
                                if str(option).lower() == value_lower:
                                    values[field_name] = option
                                    break
            return values

    ValidatedModel.__name__ = name
    ValidatedModel.__qualname__ = name
    return ValidatedModel


async def concurrency_limited(coro: Coroutine, semaphore: asyncio.Semaphore) -> Any:
    """Run a coroutine with a concurrency limit using a semaphore.

    Parameters
    ----------
    coro : Coroutine
        The coroutine to be executed.
    semaphore : asyncio.Semaphore
        The semaphore to limit concurrency.

    Returns
    -------
    Any
        The result of the coroutine execution.
    """

    async with semaphore:
        return await coro


def sanitize_filename(filename: str) -> str:
    """Sanitize filename by replacing disallowed characters with underscores.

    This function replaces characters that are not alphanumeric, hyphens, or
    underscores with underscores. It is useful for ensuring that filenames are
    valid across different operating systems and filesystems.

    Parameters
    ----------
    filename : str
        The original filename to be sanitized.

    Returns
    -------
    str
        The sanitized filename with disallowed characters replaced by underscores.

    Examples
    --------
    >>> sanitize_filename("example file.txt")
    'example_file.txt'
    >>> sanitize_filename("invalid/file:name*?.txt")
    'invalid_file_name_.txt'
    >>> sanitize_filename("data@2023#report.csv")
    'data_2023_report.csv'
    """

    return re.sub(r"[^\w\-.]", "_", filename)


def get_save_path(
    base_path: str,
    nct_id: str,
    exp_name: str,
    model_name: str,
    extract_type: str,
    outcome: str | None = None,
) -> str:
    """Generate save path for extracted data.

    Parameters
    ----------
    base_path : str
        The base directory where results will be saved.
    nct_id : str
        The National Clinical Trial ID (NCT ID) of the clinical trial.
    model_name : str
        The name of the model used for extraction.
    extract_type : str
        The type of extraction performed (e.g., "relevance", "imputation").
    outcome : str | None
        The specific outcome of interest, if applicable. If None, the path will
        not include an outcome.
    """
    return os.path.join(
        base_path,
        "results",
        f"{nct_id}_{exp_name}",
        sanitize_filename(f"{model_name}_{extract_type}") + ".csv"
        if outcome is None
        else sanitize_filename(f"{model_name}_{extract_type}_{outcome.lower()}")
        + ".csv",
    )


def check_nonplacebo(intervention_names: list[str] | None) -> bool:
    """Check if there are any non-placebo interventions.

    Parameters
    ----------
    intervention_names : list[str] | None
        A list of intervention names to check for non-placebo interventions.

    Returns
    -------
    bool
        True if there are non-placebo interventions, False otherwise.
    """
    nonplacebo_interventions = [
        name for name in (intervention_names or []) if "placebo" not in name.lower()
    ]
    return len(nonplacebo_interventions) > 0


def check_noncontrol(intervention_type: ArmGroupType | None) -> bool:
    """Check if the intervention type is not a control group.

    Parameters
    ----------
    intervention_type : ArmGroupType | None
        The type of the arm group to check.

    Returns
    -------
    bool
        True if the intervention type is not a control group, False otherwise.
    """
    return intervention_type != ArmGroupType.NO_INTERVENTION


def check_binary_endpoint(text: str) -> bool:
    """Check if the text contains a binary endpoint pattern.

    Parameters
    ----------
    text : str
        The text to check for binary endpoint patterns.

    Returns
    -------
    bool
        True if the text matches a binary endpoint pattern, False otherwise.
    """
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
    """Check if the trial meets specific criteria.

    Parameters
    ----------
    trial : ClinicalTrial
        The clinical trial object to check.

    Returns
    -------
    tuple[dict[str, int], bool]
        A dictionary with statistics about the trial and a boolean indicating
        whether the trial meets the criteria for further processing.
    """
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
    """Gets a value from a deeply nested data structure using a path string.

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


def get_answer_dicts(answer_map: dict[str, list[str]]) -> list[dict[str, str]]:
    """Generate all combinations of strings from a dictionary of lists.

    Parameters
    ----------
    answer_map : dict[str, list[str]]
        A dictionary where keys are variables (e.g., "treatment", "outcome")
        and values are lists of possible string values they take.

    Returns
    -------
    list[dict[str, str]]
        A list of dictionaries where each dictionary maps the keys of answer_map
        to a possible value it takes.
    """
    combinations = product(*list(answer_map.values()))
    result = []
    for combo in combinations:
        answer_dict = {}
        for i, key in enumerate(answer_map.keys()):
            answer_dict[key] = combo[i]
        result.append(answer_dict)
    return result


def get_alphabet_labels(n: int) -> list[str]:
    """Generate alphabet labels for multiple choice options.

    For a given number n, generate labels like a), b), ..., z), aa), ab), etc.
    """
    labels = []
    alphabet = string.ascii_lowercase
    for i in range(n):
        label = ""
        idx = i
        while True:  # allow for more than 26 labels
            label = alphabet[idx % 26] + label
            idx = idx // 26 - 1
            if idx < 0:
                break
        labels.append(f"{label}) ")
    return labels
