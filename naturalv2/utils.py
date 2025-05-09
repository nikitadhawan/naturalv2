import re
from typing import Any, Dict, List, Literal, Optional, Type, Union

from pydantic import BaseModel, create_model

from naturalv2.evals.clinical_trial import (
    ArmGroup,
    ArmGroupType,
    ClinicalTrial,
    DesignAllocation,
    Outcome,
)


class ListResponse(BaseModel):
    output: Optional[list[str]]


def create_response_format(
    name: str, keys: List[str], types: Optional[Dict[str, Type]] = None
) -> BaseModel:
    "Generate a Pydantic model with fields specified by the given keys."

    fields = {key: (types.get(key, Any), ...) for key in keys}
    return create_model(name, **fields)


class TYFilterResponse(BaseModel):
    """Response format for treatment-outcome filter stage"""

    start_weight: Union[float, Literal["Unknown"]]
    end_weight: Union[float, Literal["Unknown"]]
    weight_unit: Literal["kg", "lb", "Unknown"]
    weight_change: Union[float, Literal["Unknown"]]
    percentage_weight_change: Union[float, Literal["Unknown"]]
    treatment: Literal["Semaglutide", "Tirzepatide", "Other", "Unknown"]


class KnownsResponse(BaseModel):
    t2dm: Literal["Yes", "No", "Unknown"]
    metformin: Literal["Yes", "No", "Unknown"]
    bmi: Union[float, Literal["Unknown"]]
    age: Union[int, Literal["Unknown"]]
    sex: Literal["Male", "Female", "Unknown"]
    start_HbA1c: Union[float, Literal["Unknown"]]  # noqa: N815
    end_HbA1c: Union[float, Literal["Unknown"]]  # noqa: N815
    country: Union[str, Literal["Unknown"]]
    start_weight: Union[float, Literal["Unknown"]]
    end_weight: Union[float, Literal["Unknown"]]
    weight_unit: Literal["kg", "lb", "Unknown"]
    weight_change: Union[float, Literal["Unknown"]]
    percentage_weight_change: Union[float, Literal["Unknown"]]
    duration_days: Union[int, Literal["Unknown"]]
    treatment: Literal[
        "Semaglutide",
        "Tirzepatide",
        "Ozempic",
        "Wegovy",
        "Rybelsus",
        "Mounjaro",
        "Zepbound",
        "Unknown",
    ]
    dosage: Union[float, Literal["Unknown"]]
    target_achieved: Literal["Yes", "No", "Unknown"]


class ImputationsResponse(BaseModel):
    bmi: Union[float, Literal["Unknown"]]
    age: Union[int, Literal["Unknown"]]
    sex: Literal["Male", "Female", "Unknown"]
    start_HbA1c: Union[float, Literal["Unknown"]]  # noqa: N815
    country: Union[str, Literal["Unknown"]]
    start_weight: Union[float, Literal["Unknown"]]
    duration_days: Union[int, Literal["Unknown"]]


def check_nonplacebo(intervention_names: Optional[list[str]]) -> bool:
    nonplacebo_interventions = [
        name for name in (intervention_names or []) if "placebo" not in name.lower()
    ]
    return len(nonplacebo_interventions) > 0


def check_noncontrol(intervention_type: Optional[ArmGroupType]) -> bool:
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
        arm_groups: Optional[list[ArmGroup]] = get_nested_value(
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

                endpoints: Optional[list[Outcome]] = get_nested_value(
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


def get_nested_value(data: Any, path: str) -> Optional[Any]:
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


def enumerate_strings(dct, string=True):
    keys = list(dct.keys())
    keys.reverse()
    num = len(keys)
    all_enumerated = dct[keys[0]]
    all_enumerated = ["A" + str(num) + ": " + e for e in all_enumerated]
    for key in keys[1:]:
        num -= 1
        cur_len = len(all_enumerated)
        all_enumerated *= len(dct[key])
        for j in range(len(dct[key])):
            all_enumerated[j * cur_len : (j + 1) * cur_len] = [
                dct[key][j] + ", " + e
                for e in all_enumerated[j * cur_len : (j + 1) * cur_len]
            ]
        all_enumerated = ["A" + str(num) + ": " + e for e in all_enumerated]
    return all_enumerated


def enum_to_dcts(enumerated, to_enum):
    return_dcts = []
    for elem in enumerated:
        separate = [i.split(":") for i in elem.split(",")]
        dct = {}
        for field in range(len(to_enum)):
            dct[to_enum[field]] = separate[field][1][1:]
        return_dcts.append(dct)
    return return_dcts


def get_sample_text(a_dct, q_dct):
    all_keys = list(a_dct.keys())
    return_text = "\n\nQuestions and their correct answers"
    for key in all_keys:
        return_text += "\nQ: " + q_dct[key] + " A: " + str(a_dct[key]) + "."
    return return_text
