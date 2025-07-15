"""Experiment class for managing a clinical trial and related data."""

import logging
import os
from ast import literal_eval
from pathlib import Path
from typing import Literal

import pandas as pd
import yaml

from naturalv2.evals.clinical_trial import (
    ArmGroup,
    BaselineMeasure,
    ClinicalTrial,
    MeasureGroup,
    Measurement,
    Mesh,
    Outcome,
    OutcomeMeasure,
    OutcomeMeasureType,
    Reference,
)
from naturalv2.pipeline import INCLUSION_COL_NAME, OUTCOME_COL_NAME, TREATMENT_COL_NAME
from naturalv2.utils import (
    check_binary_endpoint,
    check_noncontrol,
    check_nonplacebo,
    get_nested_value,
    load_prompt,
)


logger = logging.getLogger(__name__)


class Experiment:
    def __init__(
        self,
        data_path: str,
        nct_id: str,
        status: Literal["completed", "active"] = "completed",
    ) -> None:
        if status == "active":
            self.trial_path = os.path.join(data_path, f"nct_reports_test/{nct_id}.json")
        else:
            self.trial_path = os.path.join(data_path, f"nct_reports/{nct_id}.json")

        self.status = status
        self.studies: list[list[str]] = []

        trial = ClinicalTrial.from_json_file(self.trial_path)

        # Extract relevant information from the trial
        self._nct_id = trial.protocolSection.identificationModule.nctId
        self._title = trial.protocolSection.identificationModule.officialTitle
        self._date: str | None = (
            get_nested_value(
                trial, "protocolSection.statusModule.completionDateStruct.date"
            )
            if self.status == "active"
            else get_nested_value(
                trial,
                "protocolSection.statusModule.resultsFirstPostDateStruct.date",
            )
        )
        self._trial_keywords: list[str] | None = get_nested_value(
            trial, "protocolSection.conditionsModule.keywords"
        )
        self._conditions: list[str] = get_nested_value(
            trial, "protocolSection.conditionsModule.conditions"
        )
        meshes: list[Mesh] | None = get_nested_value(
            trial, "derivedSection.conditionBrowseModule.meshes"
        )
        self._trial_disease_mesh = [mesh.term for mesh in meshes] if meshes else []

        ancestors: list[Mesh] | None = get_nested_value(
            trial, "derivedSection.conditionBrowseModule.ancestors"
        )
        self._trial_disease_ancestors = (
            [ancestor.term for ancestor in ancestors] if ancestors else []
        )

        references: list[Reference] | None = get_nested_value(
            trial, "protocolSection.referencesModule.references"
        )
        self._references: list[str] = (
            [ref.citation for ref in references if ref.citation] if references else []
        )

        self._inclusion_criteria: str | None = get_nested_value(
            trial, "protocolSection.eligibilityModule.eligibilityCriteria"
        )

        baseline_measures: list[BaselineMeasure] | None = get_nested_value(
            trial, "resultsSection.baselineCharacteristicsModule.measures"
        )
        self._covariate_names: list[str] = [
            base.title for base in baseline_measures or []
        ] + ["Duration"]
        self._covariate_desc: dict[str, str] = {}
        if baseline_measures is not None:
            self._covariate_desc.update(
                {
                    covariate.title: covariate.description
                    for covariate in baseline_measures
                    if covariate.description
                }
            )
        self.covariate_desc["Duration"] = (
            "The number of days that the treatment was taken (rounded to the nearest integer)."
        )
        self.extended_covariate_names: list[str] = [
            INCLUSION_COL_NAME  # , "Dosage"
        ]  # inclusion-related binary variables
        self.covariate_desc[INCLUSION_COL_NAME] = (
            "Whether the report indicates that the individual meets the inclusion criteria."
        )

        # Set treatment and outcome names and common names
        self._set_outcome_treatment_effects(trial)
        self.treatment_common_names: dict[str, list[str]] = {}
        self.outcome_common_names: dict[str, list[str]] = {}

        # Set expected choices for each feature
        # NOTE: options for covariates can only be set when we get some data
        # (see `discretize` method)
        self.options: dict[str, list[str]] = {}
        for feat in self.extended_covariate_names + self.outcome_names:
            self.options.update({feat: ["No", "Yes"]})
        self.options.update({TREATMENT_COL_NAME: self.treatment_names})

        # Store different representations of the features
        self._numerical_repr: dict[str, dict[str, int]] = {}
        self._language_repr: dict[str, dict[int, str]] = {}

        # Set question prompts for each feature
        self.question_prompts: dict[str, str] = {}
        self._set_questions()

        # Store list of paths to curated data for the experiment, one per source
        self.source_paths: dict[str, list] = {}

    @property
    def nct_id(self) -> str:
        """National Clinical Trial (NCT) ID of the clinical trial."""
        return self._nct_id

    @property
    def title(self) -> str | None:
        """Official title of the clinical trial."""
        return self._title

    @property
    def date(self) -> str | None:
        """Completion or results first post date of the clinical trial."""
        return self._date

    @property
    def trial_keywords(self) -> list[str] | None:
        """List of keywords associated with the clinical trial."""
        return self._trial_keywords

    @property
    def conditions(self) -> list[str]:
        """List of disease conditions that the clinical trial is focused on."""
        return self._conditions

    @property
    def trial_disease_mesh(self) -> list[str]:
        """List of MeSH terms associated with the disease in the clinical trial."""
        return self._trial_disease_mesh

    @property
    def trial_disease_ancestors(self) -> list[str]:
        """List of MeSH terms that are ancestors of the trial disease."""
        return self._trial_disease_ancestors

    @property
    def references(self) -> list[str]:
        """List of references associated with the clinical trial for the experiment."""
        return self._references

    @property
    def inclusion_criteria(self) -> str | None:
        """Inclusion/exclusion criteria for the experiment."""
        return self._inclusion_criteria

    @property
    def covariate_names(self) -> list[str]:
        """List of covariate names used in the experiment."""
        return self._covariate_names

    @property
    def covariate_desc(self) -> dict[str, str]:
        """Dictionary mapping covariate names to their descriptions."""
        return self._covariate_desc

    @property
    def treatment_names(self) -> list[str]:
        """List of treatment names."""
        return self._treatment_names

    @property
    def outcome_names(self) -> list[str]:
        """List of outcome names."""
        return self._outcome_names

    @property
    def outcome_timeframes(self) -> list[str | None]:
        """List of timeframes for each outcome."""
        return self._outcome_timeframes

    @property
    def outcome_treatment(self) -> list[list[str, list[str, str]]]:
        """List of tuples containing outcome and treatment pairs."""
        return self._outcome_treatment

    @property
    def treatment_desc(self) -> dict[str, str | None]:
        """Dictionary mapping treatment names to their descriptions."""
        return self._treatment_desc

    @property
    def outcome_desc(self) -> dict[str, str | None]:
        """Dictionary mapping outcome names to their descriptions."""
        return self._outcome_desc

    @property
    def effect_sizes(self) -> list[float]:
        """List of effect sizes for each outcome-treatment pair."""
        return self._effect_sizes

    @classmethod
    def from_yaml(cls, filename: str) -> "Experiment":
        """Load experiment details from a YAML file.

        Parameters
        ----------
        filename : str
            Path to the YAML file containing experiment details.

        Returns
        -------
        Experiment
            An instance of the Experiment class initialized with data from the
            YAML file.

        Raises
        ------
        ValueError
            If the data in the YAML file is not a dictionary or if required fields
            are missing.
        FileNotFoundError
            If the specified YAML file does not exist.
        TypeError
            If the data in the YAML file is not of the expected type.
        """
        with open(filename, "r") as file:
            data = yaml.safe_load(file)

        if not isinstance(data, dict):
            raise ValueError(
                f"Invalid data format in {filename}. Expected a dictionary."
            )

        exp = cls.__new__(cls)
        exp.__dict__.update(data)
        return exp

    def to_yaml(self, filename: str) -> None:
        """Save the experiment details to a YAML file.

        Parameters
        ----------
        filename : str
            Path to the YAML file where the experiment details will be saved.
        """
        with open(filename, "w") as file:
            yaml.safe_dump(self.__dict__, file)

    def hard_filter_ty(self, extractions: pd.DataFrame) -> pd.DataFrame:
        """Filter out rows that don't mention the treatment(s) and outcome(s) of interest.

        Parameters
        ----------
        extractions : pd.DataFrame
            DataFrame containing the extractions from the pipeline.

        Returns
        -------
        pd.DataFrame
            Filtered DataFrame containing only rows that mention the treatment(s)
            and outcome(s) of interest.

        Raises
        ------
        ValueError
            If the `TREATMENT_COL_NAME` or `OUTCOME_COL_NAME` columns are missing
            from the extractions DataFrame.
        """
        if TREATMENT_COL_NAME not in extractions.columns:
            raise ValueError(
                f"`{TREATMENT_COL_NAME}` column is missing from extractions."
            )

        if OUTCOME_COL_NAME not in extractions.columns:
            raise ValueError(
                f"`{OUTCOME_COL_NAME}` column is missing from extractions."
            )

        return extractions[
            extractions[TREATMENT_COL_NAME].isin(self.options[TREATMENT_COL_NAME])
            & (extractions[OUTCOME_COL_NAME] == "Yes")
        ].reset_index(drop=True)

    def hard_filter_inclusion(self, extractions: pd.DataFrame) -> pd.DataFrame:
        """Filter out rows that don't meet the inclusion criteria.

        Parameters
        ----------
        extractions : pd.DataFrame
            DataFrame containing the extractions from the pipeline.

        Returns
        -------
        pd.DataFrame
            Filtered DataFrame containing only rows that meet the inclusion criteria
            or where it's impossible to determine inclusion.

        Raises
        ------
        ValueError
            If the extractions DataFrame does not contain the expected column
            listed in `extended_covariate_names`.
        """
        for name in self.extended_covariate_names:
            if name not in extractions.columns:
                raise ValueError(f"{name} column is missing from extractions.")

            extractions = extractions[
                extractions[name].str.lower().isin(["yes", "unknown"])
            ].reset_index(drop=True)

        return extractions

    def discretize(self, extractions: pd.DataFrame) -> pd.DataFrame:
        """Discretize the features in the extractions DataFrame.

        This method converts continuous covariates into binary or categorical
        representations based on the number of unique values. It also converts
        treatment and outcome columns into numerical encodings. If more than
        half of the values in a covariate are numeric, it will be discretized
        into binary based on the median value. If a covariate has more than 10
        unique values, it will be converted to binary or categorical based on
        the frequency of values. If a covariate has fewer than 10 unique values,
        it will be converted to categorical encoding.
        The method also updates the `self.options` dictionary with the unique
        string values for each covariate and sets the numerical and language
        representations for the covariates, treatments, and outcomes.

        Parameters
        ----------
        extractions : pd.DataFrame
            DataFrame containing the extractions from the pipeline.

        Returns
        -------
        pd.DataFrame
            DataFrame with discretized features, including binary and categorical
            encodings for covariates, treatments, and outcomes.

        Raises
        ------
        ValueError
            If the extractions DataFrame does not contain the expected columns
            listed in `covariate_names`, `extended_covariate_names`, `OUTCOME_COL_NAME`
            or `TREATMENT_COL_NAME`.

        """
        for covariate in self.covariate_names:
            if covariate not in extractions.columns:
                raise ValueError(f"`{covariate}` column is missing from extractions.")

            covariate_data = extractions[covariate]
            all_answers = extractions[covariate].unique()

            if len(all_answers) > 10:  # many unique values, convert to binary
                # Try to convert to numeric first
                numeric_series = pd.to_numeric(covariate_data, errors="coerce")

                if (
                    numeric_series.notna().sum() > len(extractions) * 0.5
                ):  # mostly numeric
                    quant_50 = numeric_series.describe()["50%"]
                    binary_codes = (numeric_series > quant_50).astype(int)
                    extractions[covariate] = binary_codes

                    self.options.update(
                        {
                            covariate: [
                                f"Less than or equal to {quant_50}",
                                f"Greater than {quant_50}",
                            ]
                        }
                    )
                else:  # mostly non-numeric strings, use frequency-based approach
                    # Get top N most frequent values and group rest as "Other"
                    value_counts = covariate_data.value_counts()
                    top_values = value_counts.head(9).index.tolist()

                    # Replace infrequent values with "Other"
                    extractions.loc[~covariate_data.isin(top_values), covariate] = (
                        "Other"
                    )

                    # Convert to categorical
                    updated_answers = covariate_data.unique()
                    cov_map = {name: i for (i, name) in enumerate(updated_answers)}
                    extractions[covariate] = extractions[covariate].replace(cov_map)

                    # Only update the options if the covariate values are strings
                    # and not numeric. When this method is called multiple times,
                    # this prevents overwriting the options with numeric values.
                    if pd.api.types.is_string_dtype(updated_answers):
                        self.options.update({covariate: updated_answers.tolist()})
            else:  # few unique values, convert to categorical
                cov_map = {name: i for (i, name) in enumerate(all_answers)}
                extractions[covariate] = extractions[covariate].replace(cov_map)

                if pd.api.types.is_string_dtype(all_answers):
                    self.options.update({covariate: all_answers.tolist()})

        # Convert binary columns to numerical encoding
        binary_map_num = {"No": 0, "Yes": 1}
        for feat in self.extended_covariate_names + [OUTCOME_COL_NAME]:
            if feat not in extractions.columns:
                raise ValueError(f"`{feat}` column is missing from extractions.")

            extractions[feat] = extractions[feat].replace(binary_map_num)

        # Convert treatment column to categorical encoding
        if TREATMENT_COL_NAME not in extractions.columns:
            raise ValueError(
                f"`{TREATMENT_COL_NAME}` column is missing from extractions."
            )

        treatment_map = {name: i for (i, name) in enumerate(self.treatment_names)}
        extractions[TREATMENT_COL_NAME] = extractions[TREATMENT_COL_NAME].replace(
            treatment_map
        )

        self._set_transforms()

        return extractions

    def apply_transform(
        self,
        input_dict: dict[str, str] | dict[str, int],
        repr_type: Literal["numeric", "language"] = "numeric",
    ) -> dict[str, str] | dict[str, int]:
        """Apply the numerical or language representation to the input dictionary.

        Parameters
        ----------
        input_dict : dict[str, str] | dict[str, int]
            Dictionary containing the input data to be transformed.
        repr_type : Literal["numeric", "language"], default="numeric"
            The type of representation to apply. Can be either "numeric" or
            "language".

        Returns
        -------
        dict[str, str] | dict[str, int]
            Transformed dictionary with the specified representation applied to
            the input data.

        Raises
        ------
        ValueError
            If `repr_type` is not either "numeric" or "language".
        """
        if repr_type not in ["numeric", "language"]:
            raise ValueError("repr_type must be either 'numeric' or 'language'.")

        output_dict: dict[str, str] | dict[str, int] = {}
        for field in input_dict:
            field_map = (
                self._numerical_repr[field]
                if repr_type == "numeric"
                else self._language_repr[field]
            )
            output_dict[field] = field_map[input_dict[field]]
        return output_dict

    def build_prompt_for_report(
        self,
        prompt_type: str,
        outcome: str,
        source_name: str,
        report: str,
        return_format: Literal["prompt", "messages"] = "messages",
    ) -> str | list[dict[str, str]]:
        """Create a prompt for a given report.

        This prompt loads a prompt template specified by `prompt_type` and formats it
        with the relevant information from the experiment, such as conditions,
        treatments, outcomes, covariates, and their descriptions. The formatted prompt
        can be returned either as a single string or as a list of dictionaries with
        'role' and 'content' keys, depending on the `return_format` parameter.

        Parameters
        ----------
        prompt_type : str
            Type of the prompt to be loaded.
        outcome : str
            The outcome for which the prompt is being created.
        source_name : str
            Name of the source from which the data is being processed.
        report : str
            The report for which the prompt is being created.
        return_format : Literal["prompt", "messages"], default="messages"
            The format in which the prompt should be returned. Can be either
            "prompt" for a single string or "messages" for a list of dictionaries
            with 'role' and 'content' keys.

        Returns
        -------
        str | list[dict[str, str]]
            The formatted prompt as a string or a list of dictionaries, depending
            on the `return_format` parameter.

        """
        prompts_dir = str(Path(__file__).resolve().parents[1] / "prompts")

        outcome_desc = {outcome: self.outcome_desc[outcome]}
        format_inputs = {
            "conditions": str(self._conditions),
            "treatments": str(
                self.treatment_names + self.treatment_common_names[source_name]
            ),
            "outcome": outcome,
            "outcome_common_names": self.outcome_common_names.get(source_name, []),
            "covariates": str(self.covariate_names + self.extended_covariate_names),
            "ty_desc": "".join(
                [
                    f"\n{k}: {v}"
                    for k, v in {**self.treatment_desc, **outcome_desc}.items()
                ]
            ),
            "covariate_desc": "".join(
                [f"\n{k}: {v}" for k, v in self.covariate_desc.items()]
            ),
            "inclusion_criteria": self.inclusion_criteria,
            "report": report,
        }

        return load_prompt(
            prompts_dir, prompt_type, return_format=return_format, **format_inputs
        )

    def _set_outcome_treatment_effects(self, trial: ClinicalTrial) -> None:
        """Set variables related to outcomes, treatments and their effect sizes."""
        # NOTE: we use lists instead of tuples in _outcome_treatement because
        # of YAML serialization issues with tuples
        self._outcome_treatment: list[list[str, list[str, str]]] = []
        self._treatment_desc, self._outcome_desc = {}, {}

        if (
            self.status == "active"
        ):  # use arm information to find outcome-treatment pairs
            primary_outcomes: list[Outcome] | None = get_nested_value(
                trial, "protocolSection.outcomesModule.primaryOutcomes"
            )
            arm_groups: list[ArmGroup] | None = get_nested_value(
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
                            self._outcome_treatment.append(
                                [outcome.measure, [arm1.label, arm2.label]]
                            )
        else:  # use outcome information to find outcome-treatment pairs
            trial_outcome_measures: list[OutcomeMeasure] | None = get_nested_value(
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
            self._effect_sizes: list[float] = []

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
                            measure1: Measurement | None = outcome.get_group_stats(
                                cohort1
                            )
                            measure2: Measurement | None = outcome.get_group_stats(
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

                            self._outcome_treatment.append(
                                [outcome.title, [cohort1.title, cohort2.title]]
                            )
                            self._effect_sizes.append(effect_size)

        self._treatment_names: list[str] = [
            treatment.title if isinstance(treatment, MeasureGroup) else treatment.label
            for treatment in treatments
        ]
        self._outcome_names: list[str] = [
            outcome.title if isinstance(outcome, OutcomeMeasure) else outcome.measure
            for outcome in outcomes
        ]
        # TODO: maybe use timeframes in question_prompts, e.g.
        # outcome_q = "What was the patient's reported {out.title}?"
        # if out.timeFrame: outcome_q.replace("?", "after a duration of {out.timeFrame.split(",")[-1]}?")
        self._outcome_timeframes: list[str | None] = [
            outcome.timeFrame for outcome in outcomes
        ]

        self._treatment_desc = {
            treatment.title
            if isinstance(treatment, MeasureGroup)
            else treatment.label: treatment.description
            for treatment in treatments
        }
        self._outcome_desc = {
            outcome.title
            if isinstance(outcome, OutcomeMeasure)
            else outcome.measure: outcome.description
            for outcome in outcomes
        }

    def _set_transforms(self) -> None:
        """Set the numerical and language representations for the features.

        This can be used for converting the features into numerical or language
        representations. The numerical representation maps each feature value to an
        integer index, while the language representation maps each index to its
        corresponding feature value. The mappings are created based on the options
        provided for each feature, including treatments, outcomes, and covariates.
        """
        binary_map_num = {"No": 0, "Yes": 1}
        binary_map_lang = {0: "No", 1: "Yes"}

        self._numerical_repr = {
            TREATMENT_COL_NAME: {
                name: i for (i, name) in enumerate(self.options[TREATMENT_COL_NAME])
            }
        }
        self._numerical_repr.update(dict.fromkeys(self.outcome_names, binary_map_num))
        self._numerical_repr.update(
            dict.fromkeys(self.extended_covariate_names, binary_map_num)
        )
        self._numerical_repr.update(
            {
                cov: {name: i for (i, name) in enumerate(self.options[cov])}
                for cov in self.covariate_names
            }
        )

        self._language_repr = {
            TREATMENT_COL_NAME: dict(enumerate(self.options[TREATMENT_COL_NAME]))
        }
        self._language_repr.update(dict.fromkeys(self.outcome_names, binary_map_lang))
        self._language_repr.update(
            dict.fromkeys(self.extended_covariate_names, binary_map_lang)
        )
        self._language_repr.update(
            {cov: dict(enumerate(self.options[cov])) for cov in self.covariate_names}
        )

        # Persist the transforms for later use
        # NOTE: This helps in the case where the experiment is run multiple times
        # so that transforms are available after the first run.
        exp_dir = Path(self.trial_path).parents[1] / "experiments"
        exp_dir.mkdir(mode=755, parents=True, exist_ok=True)
        self.to_yaml(str(exp_dir / f"{self.nct_id}.yaml"))

    def _set_questions(self) -> None:
        """Set the prompts for each feature in the experiment."""
        prompts_dir = str(Path(__file__).resolve().parents[1] / "prompts")

        self.question_prompts[INCLUSION_COL_NAME] = load_prompt(
            prompts_dir,
            "question_inclusion",
            return_format="prompt",
            inclusion_criteria=self.inclusion_criteria,
        )

        for cov in self.covariate_names:
            self.question_prompts[cov] = load_prompt(
                prompts_dir, "question_covariate", return_format="prompt", covariate=cov
            )

        self.question_prompts[TREATMENT_COL_NAME] = load_prompt(
            prompts_dir, "question_treatment", return_format="prompt"
        )

        for outcome in self.outcome_names:
            self.question_prompts[outcome] = load_prompt(
                prompts_dir, "question_outcome", return_format="prompt", outcome=outcome
            )
