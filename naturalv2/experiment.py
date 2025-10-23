"""Experiment class for managing a clinical trial and related data."""

import importlib.resources
import logging
import os
import re
from ast import literal_eval
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yaml

from naturalv2.clinical_trial import (
    ArmGroup,
    BaselineMeasure,
    ClinicalTrial,
    Location,
    MeasureGroup,
    Measurement,
    Mesh,
    Outcome,
    OutcomeMeasure,
    OutcomeMeasureType,
    Reference,
)
from naturalv2.pipeline.constants import (
    INCLUSION_COL_NAME,
    OUTCOME_COL_NAME,
    TREATMENT_COL_NAME,
)
from naturalv2.prompts import load_prompt
from naturalv2.sources.drugbank_cache import get_drugbank_aliases
from naturalv2.utils import (
    check_arm,
    check_binary_endpoint,
    check_nonplacebo,
    get_nested_value,
    normalize_treatment,
)


logger = logging.getLogger(__name__)

pd.set_option("future.no_silent_downcasting", True)

DRUG_NAME_SANITIZER = re.compile(
    r"\s+\d+([./]\d+)*\s*(mg|g|mcg|ug|ml|iu|units|tablets?|capsules?)?\b.*$",
    flags=re.IGNORECASE,
)


class Experiment:
    """Class representing an experiment based on a clinical trial.

    This class encapsulates the details of a clinical trial, including its
    identification, conditions, treatments, outcomes, and other relevant
    information. It provides methods to load experiment details from a YAML file,
    filter extractions based on treatment and outcome, discretize features,
    apply transformations, and build prompts for reports.

    Parameters
    ----------
    data_path : str
        Path to the directory containing clinical trial JSON files.
    nct_id : str
        National Clinical Trial (NCT) ID of the clinical trial.
    status : Literal["completed", "active"], default="completed"
        Status of the clinical trial, either "completed" or "active". If "active",
        the trial is expected to be in progress, and the relevant JSON file will be
        loaded from the `nct_reports_test` directory. If "completed", it will be
        loaded from the `nct_reports` directory.

    Attributes
    ----------
    data_path : str
        Path to the directory containing clinical trial JSON files.
    trial_path : str
        Path to the JSON file for the clinical trial based on the NCT ID and status.
    status : Literal["completed", "active"]
        Status of the clinical trial, either "completed" or "active".
    studies : list[list[str]]
        List of studies associated with the experiment, along with the split it is in.
    nct_id : str
        National Clinical Trial (NCT) ID of the clinical trial.
    title : str | None
        Official title of the clinical trial, or None if not available.
    date : str | None
        Completion or results first post date of the clinical trial, or None if not
        available.
    trial_keywords : list[str] | None
        List of keywords associated with the clinical trial, or None if not available.
    conditions : list[str]
        List of disease conditions that the clinical trial is focused on.
    trial_disease_mesh : list[str]
        List of MeSH terms associated with the disease in the clinical trial.
    trial_disease_ancestors : list[str]
        List of MeSH terms that are ancestors of the trial disease.
    references : list[str]
        List of references associated with the clinical trial for the experiment.
    inclusion_criteria : str | None
        Inclusion/exclusion criteria for the experiment, or None if not available.
    countries : list[str]
        List of countries covered by the clinical trial.
    covariate_names : list[str]
        List of covariate names used in the experiment.
    covariate_desc : dict[str, str]
        Dictionary mapping covariate names to their descriptions. Note that not
        all covariates have descriptions. "Duration" is always included as a covariate
        and represents the duration of the treatment in terms of number of days.
    treatment_names : list[str]
        List of treatment names used in the experiment.
    outcome_names : list[str]
        List of outcome names used in the experiment.
    outcome_timeframes : list[str | None]
        List of timeframes for each outcome, or None if not available.
    outcome_treatment : list[list[str | list[str]]]
        List of lists containing outcome and treatment pairs, where each pair
        consists of an outcome name and a list of treatment names.
    treatment_desc : dict[str, str | None]
        Dictionary mapping treatment names to their descriptions, or None if not
        available.
    outcome_desc : dict[str, str | None]
        Dictionary mapping outcome names to their descriptions, or None if not
        available.
    treatment_common_names : dict[str, list[str]]
        Dictionary mapping treatment names to their common names, which are
        alternative names or identifiers for the treatments used in the experiment.
    outcome_common_names : dict[str, list[str]]
        Dictionary mapping outcome names to their common names, which are
        alternative names or identifiers for the outcomes used in the experiment.
    effect_sizes : list[float]
        List of effect sizes for each outcome-treatment pair.
    drugbank_names : dict[str, list[str]]
        Dictionary mapping treatment names to their DrugBank aliases, which are
        alternative names or identifiers for the treatments used in the experiment.
    options : dict[str, list[str]]
        Dictionary containing expected choices for each feature, including treatments,
        outcomes, and covariates. The keys are feature names, and the values are lists
        of possible values for those features.
    numerical_repr : dict[str, dict[str, int]]
        Dictionary mapping feature names to their numerical representations,
        where each feature value is mapped to an integer index.
    language_repr : dict[str, dict[int, str]]
        Dictionary mapping feature names to their language representations,
        where each integer index is mapped to its corresponding feature value.
    source_paths : dict[str, str]
        Dictionary storing paths to curated data for the experiment, for each source.

    """

    def __init__(
        self,
        data_path: str,
        nct_id: str,
        status: Literal["completed", "active"] = "completed",
    ) -> None:
        self.data_path = data_path
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
        _enrollment_info = get_nested_value(
            trial, "protocolSection.designModule.enrollmentInfo"
        )
        self._enrollment: int = _enrollment_info.count if _enrollment_info else -1

        self._enrollment_type: str = (
            str(_enrollment_info.type) if _enrollment_info else ""
        )

        self._trial_keywords: list[str] | None = get_nested_value(
            trial, "protocolSection.conditionsModule.keywords"
        )
        self._conditions: list[str] = (
            get_nested_value(trial, "protocolSection.conditionsModule.conditions") or []
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
        locations: list[Location] | None = get_nested_value(
            trial, "protocolSection.contactsLocationsModule.locations"
        )
        self._countries = None
        if locations:
            self._countries: list[str] = list(
                {location.country for location in locations}
            )

        baseline_measures: list[BaselineMeasure] | None = get_nested_value(
            trial, "resultsSection.baselineCharacteristicsModule.measures"
        )
        self._covariate_names: list[str] = [
            base.title for base in baseline_measures or []
        ] + ["Country", "Duration"]
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
            "The duration of the treatment in terms of number of days, represented as an integer."
        )
        self.covariate_desc["Country"] = "The patient's country of residence."
        self.covariate_desc[INCLUSION_COL_NAME] = (
            "Whether the report indicates that the individual meets the inclusion criteria."
        )

        self._effect_sizes: list[float] = []

        # Set treatment and outcome names and common names per data source
        # E.g. {"reddit": {
        #  "Erenumab": [Aimovig],
        #  "Topiramate": [Topamax]
        # }}
        self._set_outcome_treatment_effects(trial)
        self.treatment_common_names: dict[str, dict[str, list[str]]] = {}
        self.outcome_common_names: dict[str, dict[str, list[str]]] = {}

        self._drugbank_names: dict[str, list[str]] = {}

        # Set expected choices for each feature
        # NOTE: options for covariates can only be set when we get some data
        # (see `discretize` method)
        self.options: dict[str, list[str]] = {}
        for feat in [INCLUSION_COL_NAME] + self.outcome_names:
            self.options.update({feat: ["No", "Yes"]})
        self.options.update({TREATMENT_COL_NAME: self.treatment_names})

        # Store different representations of the features
        self._numerical_repr: dict[str, dict[str, int]] = {}
        self._language_repr: dict[str, dict[int, str]] = {}

        # Store list of paths to curated data for the experiment, one per source
        self.source_paths: dict[str, str] = {}

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
    def enrollment(self) -> int | None:
        """Enrollment count of the clinical trial."""
        return self._enrollment

    @property
    def enrollment_type(self) -> str | None:
        """Enrollment type of the clinical trial: ACTUAL or ESTIMATED."""
        return self._enrollment_type

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
    def countries(self) -> str | None:
        """Countries where the experiment has been conducted."""
        return self._countries

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
    def outcome_treatment(self) -> list[list[str | list[str]]]:
        """List of tuples containing outcome and treatment pairs."""
        return self._outcome_treatment

    @property
    def outcome_treatment_stats(self) -> list[list[str | list[str]]]:
        """List of tuples containing outcome dispersion type, cohort dispersion, and cohort sizes."""
        return self._outcome_treatment_stats

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

    @property
    def drugbank_names(self) -> dict[str, list[str]]:
        """Get DrugBank names for the treatments in the experiment."""
        if self._drugbank_names:
            return self._drugbank_names

        for drug_name in self._treatment_names:
            drug_name_stripped = DRUG_NAME_SANITIZER.sub("", drug_name).strip()
            self._drugbank_names[drug_name] = get_drugbank_aliases(
                self.data_path, drug_name_stripped
            )

        return self._drugbank_names

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
        # Make sure self._drugbank_names is populated
        # This is a potentially expensive operation
        _ = self.drugbank_names

        # Make sure parent folders exist first
        Path(os.path.dirname(filename)).mkdir(parents=True, exist_ok=True)

        with open(filename, "w") as file:
            yaml.safe_dump(self.__dict__, file)

    def get_all_treatment_names_for_source(self, source: str) -> list[str]:
        """Get all treatment names for a given data source.

        This method retrieves all treatment names associated with a specified data
        source. It combines treatment names from DrugBank aliases and common names
        specific to the source, ensuring that all names are in lowercase for
        consistency.

        Parameters
        ----------
        source : str
            The name of the data source for which to retrieve treatment names.

        Returns
        -------
        list[str]
            A list of treatment names associated with the specified data source.
        """
        drugbank_names = [
            item for sublist in self.drugbank_names.values() for item in sublist
        ]
        common_names = set()
        for name, aliases in self.treatment_common_names.get(source, {}).items():
            common_names.add(name.lower())
            for alias in aliases:
                common_names.add(alias.lower())

        return list(common_names.union(set(drugbank_names)))

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
        if f"{TREATMENT_COL_NAME}_filter" not in extractions.columns:
            raise ValueError(
                f"`{TREATMENT_COL_NAME}_filter` column is missing from extractions."
            )

        if f"{OUTCOME_COL_NAME}_filter" not in extractions.columns:
            raise ValueError(
                f"`{OUTCOME_COL_NAME}_filter` column is missing from extractions."
            )

        return extractions[
            extractions[f"{TREATMENT_COL_NAME}_filter"].isin(
                self.options[TREATMENT_COL_NAME]
            )
            & (extractions[f"{OUTCOME_COL_NAME}_filter"].isin(["Yes", "No"]))
        ]

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
            called INCLUSION_COL_NAME.
        """
        if INCLUSION_COL_NAME not in extractions.columns:
            raise ValueError(
                f"{INCLUSION_COL_NAME} column is missing from extractions."
            )

        return extractions[
            extractions[INCLUSION_COL_NAME].str.lower().isin(["yes", "unknown"])
        ]

    def discretize(self, extractions: pd.DataFrame) -> pd.DataFrame:
        """Discretize the covariates in the extractions DataFrame.

        This method converts continuous covariates into binary or categorical
        representations based on the number of unique values. It also converts
        treatment columns into numerical encodings.

        Parameters
        ----------
        extractions : pd.DataFrame
            DataFrame containing the extractions from the pipeline.

        Returns
        -------
        pd.DataFrame
            DataFrame with discretized covariates and treatments, including binary and
            categorical encodings for covariates.

        Raises
        ------
        ValueError
            If the extractions DataFrame does not contain the expected columns
            listed in `covariate_names`.
        """
        for covariate in self.covariate_names:
            self._discretize_covariate(extractions, covariate)
        self._set_transforms()
        return extractions

    def discretize_ty(self, extractions: pd.DataFrame) -> pd.DataFrame:
        self._discretize_outcome_column(extractions)
        self._discretize_treatment_column(extractions)
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
        covariate_answers: dict | None = None,
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
        prompts_dir = str(importlib.resources.files("naturalv2.prompts.templates"))

        format_inputs = {
            "conditions": self._conditions,
            "source": source_name,
            "treatments": self.treatment_common_names[source_name],
            "outcome": outcome,
            "covariates": self.covariate_names,
            "treatment_desc": self.treatment_desc,
            "outcome_desc": self.outcome_desc[outcome],
            "covariate_desc": self.covariate_desc,
            "inclusion_criteria": self.inclusion_criteria,
            "covariate_answers": covariate_answers,
            "treatment_options": self.options[TREATMENT_COL_NAME],
            "outcome_options": ["No", "Yes"],
            "report": report,
        }

        return load_prompt(
            prompts_dir, prompt_type, return_format=return_format, **format_inputs
        )

    def get_question_prompts(self) -> dict[str, str]:
        """Get the prompts for each feature in the experiment.

        This method loads the question prompts for inclusion criteria, covariates,
        treatments, and outcomes from the prompt templates directory. The prompts
        are formatted with the relevant information from the experiment, such as
        inclusion criteria, covariate names, treatment names, and outcome names.

        Returns
        -------
        dict[str, str]
            A dictionary where keys are feature names (e.g., inclusion criteria,
            covariates, treatments, outcomes) and values are the corresponding
            formatted question prompts.
        """
        prompts_dir = str(importlib.resources.files("naturalv2.prompts.templates"))
        question_prompts: dict[str, str] = {}

        question_prompts[INCLUSION_COL_NAME] = load_prompt(  # type: ignore[assignment]
            prompts_dir,
            "question_inclusion",
            return_format="prompt",
            inclusion_criteria=self.inclusion_criteria,
        )

        for cov in self.covariate_names:
            question_prompts[cov] = load_prompt(
                prompts_dir, "question_covariate", return_format="prompt", covariate=cov
            )

        question_prompts[TREATMENT_COL_NAME] = load_prompt(
            prompts_dir,
            "question_treatment",
            return_format="prompt",
        )

        for idx, outcome in enumerate(self.outcome_names):
            question_prompts[outcome] = load_prompt(
                prompts_dir,
                "question_outcome",
                return_format="prompt",
                outcome=outcome,
                outcome_timeframe=self.outcome_timeframes[idx],
            )

        return question_prompts

    def _discretize_covariate(
        self, extractions: pd.DataFrame, covariate_name: str
    ) -> None:
        """Discretize a covariate in the extractions DataFrame, after imputations."""
        imputation_col_name = f"{covariate_name}_imputed"
        discrete_covariate_name = f"{covariate_name}_discretized"
        if covariate_name not in extractions.columns:
            raise ValueError(f"`{covariate_name}` column is missing from extractions.")
        if imputation_col_name not in extractions.columns:
            raise ValueError(
                f"`{imputation_col_name}` column is missing from extractions."
            )

        knowns = extractions[covariate_name].copy().astype(str)
        unknowns = knowns.str.lower().isin(["unknown", ""]) | knowns.isna()
        covariate_data = knowns.mask(unknowns, extractions[imputation_col_name])
        all_answers = covariate_data.unique()

        if len(all_answers) > 10:
            self._discretize_many_unique(
                extractions,
                covariate_data,
                discrete_covariate_name,
                covariate_name,
            )
        else:
            self._discretize_few_unique(
                extractions,
                covariate_data,
                all_answers,
                discrete_covariate_name,
                covariate_name,
            )

    def _discretize_many_unique(
        self,
        extractions: pd.DataFrame,
        covariate_data: pd.Series,
        discrete_covariate_name: str,
        covariate_name: str,
    ) -> None:
        """Discretize a covariate with many unique values in the extractions DataFrame.

        This method converts numeric covariates into binary representations based on
        the median value, or categorizes non-numeric covariates into frequent and
        infrequent categories, grouping the infrequent ones into "Other".
        """
        # if covariate_name == "Duration":
        #     numeric_series = pd.to_timedelta(covariate_data, errors="coerce")
        # else:
        numeric_series = pd.to_numeric(covariate_data, errors="coerce")

        # Most extractions are numerical values.
        if numeric_series.notna().sum() > len(extractions) * 0.8:
            quant_50 = numeric_series.describe()["50%"]
            binary_codes = (numeric_series > quant_50).astype(int)
            extractions[discrete_covariate_name] = binary_codes

            self.options.update(
                {
                    covariate_name: [
                        f"Less than or equal to {quant_50}",
                        f"Greater than {quant_50}",
                    ]
                }
            )
        # There are non-numerical extractions, so we categorize them.
        else:
            value_counts = covariate_data.value_counts()
            cumulative_counts = value_counts.cumsum()

            # For each category, calculate the number of samples in all the categories
            # that come after it in the sorted order
            tail_sums = len(covariate_data) - cumulative_counts

            # Create a mask for categories that have more samples than the tail sum
            # This means, if we kept the category and grouped all the categories
            # after it into "Other", the "Other" category would have at most as many samples
            # as the category itself
            tail_mask = tail_sums <= value_counts

            if tail_mask.any():
                # Find the position of the first category that has more samples
                # than the tail sum
                # This is the cutoff position for renaming categories to "Other"
                cutoff_position = value_counts.index.get_loc(tail_mask.idxmax())
                categories = value_counts.index[: cutoff_position + 1].tolist()
            else:
                # If all categories are frequent enough, only rename the last
                # as "Other"
                categories = value_counts.index.tolist()[:-1]

            extractions[discrete_covariate_name] = np.where(
                covariate_data.isin(categories), covariate_data.astype(str), "Other"
            )
            updated_answers = categories + ["Other"]
            cov_map = {name: i for (i, name) in enumerate(updated_answers)}
            extractions[discrete_covariate_name] = (
                extractions[discrete_covariate_name].replace(cov_map).astype(int)
            )
            self.options.update(
                {covariate_name: [str(name) for name in updated_answers]}
            )

    def _discretize_few_unique(
        self,
        extractions: pd.DataFrame,
        covariate_data: pd.Series,
        all_answers: np.ndarray,
        discrete_covariate_name: str,
        covariate_name: str,
    ) -> None:
        """Discretize a covariate with few unique values in the extractions DataFrame."""
        cov_map = {name: i for (i, name) in enumerate(all_answers)}
        extractions[discrete_covariate_name] = covariate_data.replace(cov_map).astype(
            int
        )
        self.options.update({covariate_name: [str(name) for name in all_answers]})

    def _discretize_outcome_column(self, extractions: pd.DataFrame) -> None:
        """ "Discretize binary columns in the extractions DataFrame."""
        if OUTCOME_COL_NAME not in extractions.columns:
            raise ValueError(
                f"`{OUTCOME_COL_NAME}` column is missing from extractions."
            )
        binary_map_num = {"No": 0, "Yes": 1}
        extractions[OUTCOME_COL_NAME + "_discretized"] = extractions[
            OUTCOME_COL_NAME
        ].replace(binary_map_num)

    def _discretize_treatment_column(self, extractions: pd.DataFrame) -> None:
        """ "Discretize the treatment column in the extractions DataFrame."""
        if TREATMENT_COL_NAME not in extractions.columns:
            raise ValueError(
                f"`{TREATMENT_COL_NAME}` column is missing from extractions."
            )
        treatment_map = {name: i for (i, name) in enumerate(self.treatment_names)}
        extractions[TREATMENT_COL_NAME + "_discretized"] = (
            extractions[TREATMENT_COL_NAME].replace(treatment_map).astype(int)
        )

    def _set_outcome_treatment_effects(self, trial: ClinicalTrial) -> None:
        """Set variables related to outcomes, treatments and their effect sizes.

        This method initializes the `_outcome_treatment` list, `_treatment_desc`,
        and `_outcome_desc` dictionaries based on the status of the trial. It builds
        the outcome-treatment pairs and their descriptions for both active and
        completed trials. It also sets the treatment and outcome names, timeframes,
        and their descriptions.
        """
        self._outcome_treatment: list[list[str | list[str]]] = []
        self._outcome_treatment_stats: list[list[str | list[float]]] = []
        self._treatment_desc, self._outcome_desc = {}, {}

        if self.status == "active":
            outcomes, treatments = self._get_active_outcomes_treatments(trial)
            self._build_active_outcome_treatment(outcomes, treatments)
        else:
            outcomes, treatments = self._get_completed_outcomes_treatments(trial)
            self._build_completed_outcome_treatment(outcomes)

        self._treatment_names: list[str] = [
            treatment.title if hasattr(treatment, "title") else treatment.label
            for treatment in treatments
        ]
        self._outcome_names: list[str] = [
            outcome.title if hasattr(outcome, "title") else outcome.measure
            for outcome in outcomes
        ]

        self._outcome_timeframes: list[str | None] = [
            getattr(outcome, "timeFrame", None) for outcome in outcomes
        ]

        self._treatment_desc = {
            (
                treatment.title if hasattr(treatment, "title") else treatment.label
            ): getattr(treatment, "description", None)
            for treatment in treatments
        }
        self._outcome_desc = {
            (outcome.title if hasattr(outcome, "title") else outcome.measure): getattr(
                outcome, "description", None
            )
            for outcome in outcomes
        }

    def _get_active_outcomes_treatments(
        self, trial: ClinicalTrial
    ) -> tuple[list[Outcome], list[ArmGroup]]:
        """Get active outcomes and treatments from the trial."""
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
            if check_arm(arm.type) and check_nonplacebo([arm.label])
        ]
        return outcomes, treatments

    def _build_active_outcome_treatment(
        self, outcomes: list[Outcome], treatments: list[ArmGroup]
    ) -> None:
        """Build the outcome-treatment pairs for active trials."""
        for outcome in outcomes:
            for i, arm1 in enumerate(treatments):
                for j, arm2 in enumerate(treatments):
                    if i < j:
                        normalized = [
                            normalize_treatment(arm1.label),
                            normalize_treatment(arm2.label),
                        ]
                        unique_treatments = set(normalized)
                        if len(unique_treatments) > 1:
                            self._outcome_treatment.append(
                                [outcome.measure, [arm1.label, arm2.label]]
                            )

    def _get_completed_outcomes_treatments(
        self, trial: ClinicalTrial
    ) -> tuple[list[OutcomeMeasure], list[MeasureGroup]]:
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
        for outcome in outcomes:
            measure_groups: list[MeasureGroup] = [
                cohort
                for cohort in outcome.groups or []
                if check_nonplacebo([cohort.title])
            ]
            treatments.extend(measure_groups)
        return outcomes, treatments

    def _build_completed_outcome_treatment(
        self, outcomes: list[OutcomeMeasure]
    ) -> None:
        for outcome in outcomes:
            measure_groups: list[MeasureGroup] = [
                cohort
                for cohort in getattr(outcome, "groups", []) or []
                if check_nonplacebo([cohort.title])
            ]
            for i, cohort1 in enumerate(measure_groups):
                for j, cohort2 in enumerate(measure_groups):
                    if i < j:
                        normalized = [
                            normalize_treatment(cohort1.title),
                            normalize_treatment(cohort2.title),
                        ]
                        unique_treatments = set(normalized)
                        if len(unique_treatments) <= 1:
                            continue
                        measure1: Measurement | None = outcome.get_group_stats(cohort1)
                        measure2: Measurement | None = outcome.get_group_stats(cohort2)
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
                            if outcome.dispersionType is not None:
                                dispersion1 = (
                                    (measure1.lowerLimit, measure1.upperLimit)
                                    if "confidence" in outcome.dispersionType.lower()
                                    else measure1.spread
                                )
                                dispersion2 = (
                                    (measure2.lowerLimit, measure2.upperLimit)
                                    if "confidence" in outcome.dispersionType.lower()
                                    else measure2.spread
                                )
                            else:
                                dispersion1, dispersion2 = None, None
                        else:
                            continue
                        unit = (
                            outcome.unitOfMeasure.lower()
                            if outcome.unitOfMeasure is not None
                            else ""
                        )
                        effect1 = (
                            effect1 / 100 if "percent" in unit else effect1 / denom1
                        )
                        effect2 = (
                            effect2 / 100 if "percent" in unit else effect2 / denom2
                        )
                        effect_size = effect2 - effect1
                        self._outcome_treatment.append(
                            [outcome.title, [cohort1.title, cohort2.title]]
                        )
                        self._outcome_treatment_stats.append(
                            [
                                outcome.dispersionType,
                                [dispersion1, dispersion2],
                                [denom1, denom2],
                            ]
                        )
                        self._effect_sizes.append(effect_size)

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
            {cov: dict(enumerate(self.options[cov])) for cov in self.covariate_names}
        )

        # Persist the transforms for later use
        # NOTE: This helps in the case where the experiment is run multiple times
        # so that transforms are available after the first run.
        exp_dir = Path(self.trial_path).parents[1] / "experiments"
        exp_dir.mkdir(mode=755, parents=True, exist_ok=True)
        self.to_yaml(str(exp_dir / f"{self.nct_id}.yaml"))
