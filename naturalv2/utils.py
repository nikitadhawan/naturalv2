import json
import logging
import os
import re
from itertools import product
from string import Template
from typing import Any, Literal

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


def load_prompt(
    base_dir: str,
    prompt_type: str,
    return_format: Literal["messages", "prompt"] | None = None,
    **user_prompt_format_kwargs: Any,
) -> str | list[dict[str, str]]:
    if return_format not in ["messages", "prompt", None]:
        raise ValueError("return_format must be either 'messages', 'prompt', or None.")

    filepath = os.path.join(base_dir, f"{prompt_type}.json")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Prompt file {filepath} not found.")

    # load json file
    with open(filepath, "r") as f:
        prompt_data: dict[str, Any] = json.load(f)

    # 'user_prompt_template' must be in the json file
    if "user_prompt_template" not in prompt_data:
        raise KeyError(
            f"'user_prompt_template' key not found in {prompt_type}.json file."
            "Please check the file format."
        )

    # 'system_prompt_template' is optional, but if it exists, construct and return
    # a list of dictionaries with 'role' and 'content' keys
    system_prompt: str | None = prompt_data.get("system_prompt_template")
    if system_prompt:
        logger.debug(f"System prompt loaded from {prompt_type}.json: {system_prompt}")

        system_role_dict = {"role": "system", "content": system_prompt}

    user_prompt_template: str = prompt_data["user_prompt_template"]
    if user_prompt_format_kwargs:
        user_prompt_template = Template(user_prompt_template).safe_substitute(
            **user_prompt_format_kwargs
        )

    logger.debug(
        f"User prompt template loaded from {prompt_type}.json: {user_prompt_template}"
    )

    if "examples_intro" in prompt_data and (
        "examples" in prompt_data and len(prompt_data["examples"]) > 0
    ):
        examples_intro = prompt_data["examples_intro"]
        examples: list[dict[str, str]] = prompt_data["examples"]

        # append examples intro and examples to user_prompt_template
        user_prompt_template += f"\n\n{examples_intro}"
        for example in examples:
            user_prompt_template += f"\n\n{example['input']}\n\n{example['output']}"

    if return_format in ["messages", None]:
        user_role_dict = {"role": "user", "content": user_prompt_template}

        if system_prompt:
            return [system_role_dict, user_role_dict]
        if return_format == "messages":
            return [user_role_dict]

    # concatenate system_prompt and user_prompt_template
    # and return as a single string
    if system_prompt:
        return f"{system_prompt}\n\n{user_prompt_template}"

    return user_prompt_template


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
        f"{model_name.replace('/', '-')}_{extract_type}.csv"
        if outcome is None
        else f"{model_name.replace('/', '-')}_{extract_type}_{outcome}.csv",
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
