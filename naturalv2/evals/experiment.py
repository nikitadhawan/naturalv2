import os
from ast import literal_eval
from typing import Any, Literal, Optional

import pandas as pd
import yaml
from omegaconf import DictConfig
from pydantic import BaseModel

from naturalv2.evals.clinical_trial import (
    ArmGroup,
    BaselineMeasure,
    ClinicalTrial,
    MeasureGroup,
    Measurement,
    Outcome,
    OutcomeMeasure,
    OutcomeMeasureType,
    Reference,
)
from naturalv2.models.lm import LM
from naturalv2.utils import (
    check_binary_endpoint,
    check_noncontrol,
    check_nonplacebo,
    get_nested_value,
)


class ListResponse(BaseModel):
    output: Optional[list[str]]


class Experiment:
    def __init__(
        self,
        data_path: str,
        nct_id: str,
        split: Literal["train", "val", "test"] = "train",
    ) -> None:
        if split == "test":
            self.trial_path = os.path.join(data_path, f"nct_reports_test/{nct_id}.json")
        else:
            self.trial_path = os.path.join(data_path, f"nct_reports/{nct_id}.json")
        self.split = split

        trial = ClinicalTrial.from_json_file(self.trial_path)

        # extract relevant information from the trial
        self.nct_id = trial.protocolSection.identificationModule.nctId
        self.title = trial.protocolSection.identificationModule.officialTitle
        self.date: Optional[str] = (
            get_nested_value(
                trial, "protocolSection.statusModule.completionDateStruct.date"
            )
            if self.split == "test"
            else get_nested_value(
                trial,
                "protocolSection.statusModule.resultsFirstPostDateStruct.date",
            )
        )

        references: Optional[list[Reference]] = get_nested_value(
            trial, "protocolSection.referencesModule.references"
        )
        self.references: list[str] = (
            [ref.citation for ref in references if ref.citation] if references else []
        )

        baseline_measures: Optional[list[BaselineMeasure]] = get_nested_value(
            trial, "resultsSection.baselineCharacteristicsModule.measures"
        )
        self.covariate_names: list[str] = [
            base.title for base in baseline_measures or []
        ] + ["Duration"]
        self.extended_covariate_names: list[str] = [
            "Dosage"
        ]  # inclusion-related binary variables

        self.inclusion_criteria: Optional[str] = get_nested_value(
            trial, "protocolSection.eligibilityModule.eligibilityCriteria"
        )

        self._set_outcome_treatment_effects(trial)
        self.treatment_common_names: dict[str, list[str]] = {}
        self.outcome_common_names: dict[str, list[str]] = {}

        self.source_paths: dict[str, str] = {}  # paths to curated data, one per source
        self.curated_data_path = ""  # path to curated data -- probably unnecessary
        self.options: dict[str, Any] = {}
        self.question_prompts: dict[str, str] = {}

    def to_yaml(self, filename):
        with open(filename, "w") as file:
            yaml.safe_dump(self.__dict__, file)

    @classmethod
    def from_yaml(cls, filename):
        with open(filename, "r") as file:
            data = yaml.safe_load(file)
        exp = cls.__new__(cls)
        exp.__dict__.update(data)
        return exp

    def set_common_names(
        self,
        attr: Literal["treatment", "outcome"],
        source_name: str,
        lm_cfg: DictConfig,
        prompt_dct: dict,
    ) -> None:
        if attr not in ["treatment", "outcome"]:
            raise ValueError(f"Expected 'treatment' or 'outcome', got {attr}")

        if getattr(self, f"{attr}_common_names"):
            return

        lm = LM(**lm_cfg)
        system_msg = {"role": "system", "content": prompt_dct["system"]}
        common_names = []
        for name in getattr(self, f"{attr}_names"):
            prompt = prompt_dct[attr].format(**{"keyword": name})
            messages = [system_msg, {"role": "user", "content": prompt}]
            lm_response = lm(messages=messages, response_format=ListResponse)
            common_names.extend(self._parse_lm_response(lm_response[0]))

        getattr(self, f"{attr}_common_names").update(
            {source_name: list(set(common_names))}
        )

    def hard_filter_ty(self, extractions):
        for name in self.treatment_names + self.outcome_names:
            extractions = extractions[extractions[name].isin(self.options[name])]
        return extractions

    def hard_filter_inclusion(self, extractions):
        for name in self.extended_covariate_names:
            extractions = extractions[
                extractions[name].lower().isin(["yes", "unknown"])
            ]
        return extractions

    def discretize(self, extractions: pd.DataFrame) -> pd.DataFrame:
        for cov in self.covariate_names:
            all_answers = extractions[cov].unique()
            if len(all_answers) > 10:
                extractions[cov] = pd.to_numeric(extractions[cov], errors="coerce")
                quant_50 = extractions[cov].describe()["50%"]
                extractions.loc[extractions[cov] <= quant_50, cov] = 0
                extractions.loc[extractions[cov] > quant_50, cov] = 1
                self.options.update(
                    {
                        cov: [
                            f"Less than or equal to {quant_50}",
                            f"Greater than {quant_50}",
                        ]
                    }
                )
            else:
                self.options.update({cov: list(all_answers)})
                cov_map = {name: i for (i, name) in enumerate(self.options[cov])}
                extractions[cov] = extractions[cov].replace(cov_map)

        binary_map_num = {"No": 0, "Yes": 1}
        for feat in self.extended_covariate_names + self.outcome_names:
            extractions[feat] = extractions[feat].replace(binary_map_num)
            self.options.update({feat: ["No", "Yes"]})

        treatment_map = {name: i for (i, name) in enumerate(self.treatment_names)}
        extractions["treatment"] = extractions["treatment"].replace(treatment_map)
        self.options.update({"treatment": self.treatment_names})

        self._set_transforms()

        return extractions

    def _set_outcome_treatment_effects(self, trial: ClinicalTrial) -> None:
        self.outcome_treatment: list[tuple[str, tuple[str, str]]] = []

        if self.split == "test":  # use arm information to find outcome-treatment pairs
            primary_outcomes: Optional[list[Outcome]] = get_nested_value(
                trial, "protocolSection.outcomesModule.primaryOutcomes"
            )
            arm_groups: Optional[list[ArmGroup]] = get_nested_value(
                trial, "protocolSection.armsInterventionsModule.armGroups"
            )

            outcomes: list[Outcome] = [
                outcome
                for outcome in primary_outcomes or []
                if check_binary_endpoint(outcome.measure)
            ]
            treatments: list[ArmGroup] = [
                arm
                for arm in arm_groups or []
                if check_noncontrol(arm.type)
                and check_nonplacebo(arm.interventionNames)
            ]

            for outcome in outcomes:
                for i, arm1 in enumerate(treatments):
                    for j, arm2 in enumerate(treatments):
                        if i < j:
                            self.outcome_treatment.append(
                                (outcome.measure, (arm1.label, arm2.label))
                            )
        else:  # use outcome information to find outcome-treatment pairs
            trial_outcome_measures: Optional[list[OutcomeMeasure]] = get_nested_value(
                trial,
                "resultsSection.outcomeMeasuresModule.outcomeMeasures",
            )
            outcomes: list[OutcomeMeasure] = [
                outcome
                for outcome in trial_outcome_measures or []
                if outcome.type == OutcomeMeasureType.PRIMARY
                and check_binary_endpoint(outcome.title)
            ]

            treatments: list[MeasureGroup] = []
            self.effect_sizes: list[float] = []

            for outcome in outcomes:
                measure_groups: list[MeasureGroup] = [
                    cohort
                    for cohort in outcome.groups or []
                    if check_nonplacebo([cohort.title])
                ]
                treatments.extend(measure_groups)

                for i, cohort1 in enumerate(measure_groups):
                    for j, cohort2 in enumerate(measure_groups):
                        if i < j:
                            measure1: Optional[Measurement] = outcome.get_group_stats(
                                cohort1
                            )
                            measure2: Optional[Measurement] = outcome.get_group_stats(
                                cohort2
                            )

                            denom1 = literal_eval(
                                cohort1.extract_denom_value_by_id(outcome.denoms)
                            )
                            denom2 = literal_eval(
                                cohort2.extract_denom_value_by_id(outcome.denoms)
                            )

                            if (
                                (measure1 is not None and measure1.value != "None")
                                and (measure2 is not None and measure2.value != "None")
                                and (denom1 is not None and denom1 > 0)
                                and (denom2 is not None and denom2 > 0)
                            ):
                                effect1: float = literal_eval(measure1.value)
                                effect2: float = literal_eval(measure2.value)
                            else:
                                continue

                            # divide by cohort size or 100 if result is a percentage
                            unit = (
                                outcome.unitOfMeasure.lower()
                                if outcome.unitOfMeasure
                                else ""
                            )
                            effect1 = (
                                effect1 / 100 if "percent" in unit else effect1 / denom1
                            )
                            effect2 = (
                                effect2 / 100 if "percent" in unit else effect2 / denom2
                            )
                            effect_size = effect2 - effect1  # always cohort2 - cohort1

                            self.outcome_treatment.append(
                                (outcome.title, (cohort1.title, cohort2.title))
                            )
                            self.effect_sizes.append(effect_size)

        self.treatment_names: list[str] = [
            treatment.title if isinstance(treatment, MeasureGroup) else treatment.label
            for treatment in treatments
        ]
        self.outcome_names: list[str] = [
            outcome.title if isinstance(outcome, OutcomeMeasure) else outcome.measure
            for outcome in outcomes
        ]
        # TODO: maybe use timeframes in question_prompts, e.g.
        # outcome_q = "What was the patient's reported {out.title}?"
        # if out.timeFrame: outcome_q.replace("?", "after a duration of {out.timeFrame.split(",")[-1]}?")
        self.outcome_timeframes: list[Optional[str]] = [
            outcome.timeFrame for outcome in outcomes
        ]

    def _set_transforms(self):
        binary_map_num = {"No": 0, "Yes": 1}
        binary_map_lang = {0: "No", 1: "Yes"}

        self.numerical_repr = {
            "treatment": {name: i for (i, name) in enumerate(self.treatment_names)}
        }
        self.numerical_repr.update(
            {outcome: binary_map_num for outcome in self.outcome_names}
        )
        self.numerical_repr.update(
            {cov: binary_map_num for cov in self.extended_covariate_names}
        )
        self.numerical_repr.update(
            {
                cov: {name: i for (i, name) in enumerate(self.options[cov])}
                for cov in self.covariate_names
            }
        )

        self.language_repr = dict(enumerate(self.treatment_names))
        self.language_repr.update(
            {outcome: binary_map_lang for outcome in self.outcome_names}
        )
        self.language_repr.update(
            {cov: binary_map_lang for cov in self.extended_covariate_names}
        )
        self.numerical_repr.update(
            {cov: dict(enumerate(self.options[cov])) for cov in self.covariate_names}
        )

    def _parse_lm_response(self, lm_response: str) -> list[str]:
        return (
            ListResponse.model_validate_json(lm_response).output if lm_response else []
        )
