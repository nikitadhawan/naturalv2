import logging
from collections import Counter
from typing import Optional

import pandas as pd
from presidio_analyzer import AnalyzerEngine, BatchAnalyzerEngine
from presidio_anonymizer import AnonymizerEngine, OperatorConfig
from tqdm import tqdm


logger = logging.getLogger(__name__)
logging.getLogger("presidio-analyzer").setLevel(logging.ERROR)


def _get_sample_size(
    pop_size: int, z: float = 1.96, p: float = 0.5, e: float = 0.01
) -> int:
    """
    Calculate the sample size needed for a given population size, confidence level,
    proportion of success, and margin of error.
    """
    return int(
        (pop_size * z**2 * p * (1 - p)) / ((pop_size - 1) * e**2 + z**2 * p * (1 - p))
    )


class Anonymizer:
    """
    A class for anonymizing sensitive information in text data.

    This class uses Presidio's AnalyzerEngine and AnonymizerEngine to detect and
    anonymize sensitive entities such as credit card numbers, email addresses, and
    personal identifiers.

    Parameters
    ----------
    score_threshold : float, optional
        The score threshold for entity detection. Entities with a score below this
        threshold will not be anonymized. Default is 0.85.
    """

    ENTITIES = [
        "CREDIT_CARD",
        "CRYPTO",
        "EMAIL_ADDRESS",
        "IBAN_CODE",
        "IP_ADDRESS",
        "PERSON",
        "PHONE_NUMBER",
        "MEDICAL_LICENSE",
        "US_BANK_NUMBER",
        "US_DRIVER_LICENSE",
        "US_ITIN",
        "US_PASSPORT",
        "US_SSN",
        "UK_NHS",
        "UK_NINO",
        "ES_NIF",
        "ES_NIE",
        "IT_FISCAL_CODE",
        "IT_DRIVER_LICENSE",
        "IT_VAT_CODE",
        "IT_PASSPORT",
        "IT_IDENTITY_CARD",
        "PL_PESEL",
        "SG_NRIC_FIN",
        "SG_UEN",
        "AU_ABN",
        "AU_ACN",
        "AU_TFN",
        "AU_MEDICARE",
        "IN_PAN",
        "IN_AADHAAR",
        "IN_VEHICLE_REGISTRATION",
        "IN_VOTER",
        "IN_PASSPORT",
        "FI_PERSONAL_IDENTITY_CODE",
    ]

    def __init__(self, score_threshold: float = 0.85) -> None:
        """Initialize the Anonymizer with a score threshold."""
        self._score_threshold = score_threshold
        self._analyzer = AnalyzerEngine(
            default_score_threshold=score_threshold, supported_languages=["en"]
        )
        self._anonymizer = AnonymizerEngine()

        self.operators = {}
        for entity in self.ENTITIES:
            self.operators[entity] = OperatorConfig(
                operator_name="replace",
                params={"new_value": f"<{entity}>"},
            )

    def anonymize_text(self, text: str) -> tuple[str, dict[str, int]]:
        """Anonymize sensitive entities in a given text.

        Parameters
        ----------
        text : str
            The input text to be anonymized.

        Returns
        -------
        tuple[str, dict[str, int]]
            A tuple containing the anonymized text and a dictionary with entity
            counts.
        """
        results = self._analyzer.analyze(
            text=text, language="en", entities=self.ENTITIES
        )
        anon_result = self._anonymizer.anonymize(text=text, analyzer_results=results)

        entity_stats = Counter()

        for item in anon_result.items:
            entity = item.entity_type
            entity_stats[entity] += 1

        return anon_result.text, dict(entity_stats)

    def anonymize_dataframe(
        self,
        df: pd.DataFrame,
        cols_to_keep: list[str],
        cols_to_anonymize: list[str] = None,
        data_source_name: Optional[str] = None,
        batch_size: int = 1000,
        num_workers: int = 1,
    ) -> pd.DataFrame:
        """Anonymize specified columns in a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing the data to be anonymized.
        cols_to_keep : list[str]
            List of column names to keep in the DataFrame. All other columns,
            besides those specified in `cols_to_anonymize`, will be dropped from
            the DataFrame.
        cols_to_anonymize : list[str], optional
            List of column names to anonymize. If not provided, no columns will be
            anonymized. If provided, only these columns will be anonymized.
        data_source_name : str, optional
            Name of the data source for logging purposes. If not provided, defaults
            to "DataFrame".
        batch_size : int, default=1000
            The size of each batch of data to process for anonymization.
        num_workers : int, default=1
            The number of parallel workers to use for anonymization. Must be a
            positive integer greater than 0.

        Returns
        -------
        pd.DataFrame
            A new DataFrame with the specified columns anonymized. If no columns are
            specified for anonymization, the original DataFrame is returned,
            possibly with some columns dropped based on `cols_to_keep`.

        Raises
        ------
        ValueError
            If `num_workers` or `batch_size` are not positive integers greater than 0,
            or if `cols_to_keep` or `cols_to_anonymize` are not lists of strings.

        """
        cols_to_keep, cols_to_anonymize = self._validate_anonymize_dataframe_args(
            cols_to_keep, cols_to_anonymize, batch_size, num_workers
        )

        cols_to_keep = [col for col in cols_to_keep if col in df.columns]
        cols_to_anonymize = [col for col in cols_to_anonymize if col in df.columns]
        if not cols_to_keep and not cols_to_anonymize:
            logging.warning(
                "No columns to keep or anonymize. Returning the original DataFrame."
            )
            return df.copy()

        anonymized_df = (
            df.loc[:, cols_to_keep + cols_to_anonymize].copy()
            if cols_to_keep
            else df.copy()
        )

        if not cols_to_anonymize:
            logging.warning(
                "No columns to anonymize. Returning the DataFrame with only kept columns."
            )
            return anonymized_df

        df_entity_stats = Counter()

        batch_analyzer = BatchAnalyzerEngine(self._analyzer)
        for col in cols_to_anonymize:
            if col in anonymized_df.columns and (
                pd.api.types.is_string_dtype(anonymized_df[col])
            ):
                anonymized_col, col_stats = self._anonymize_column(
                    anonymized_df.loc[:, col],
                    batch_analyzer,
                    batch_size=batch_size,
                    num_workers=num_workers,
                )
                anonymized_df.loc[:, col] = anonymized_col
                df_entity_stats.update(col_stats)
            else:
                tqdm.write(
                    f"Column `{col}` does not exist in the DataFrame or is not a "
                    "string type. Skipping anonymization for this column."
                )

        logging.info(
            f"Anonymization stats for {data_source_name if data_source_name else 'DataFrame'}:"
        )
        self._log_anonymization_stats(df_entity_stats)

        return anonymized_df

    def _anonymize_column(
        self,
        col: pd.Series,
        batch_analyzer: BatchAnalyzerEngine,
        batch_size: int = 1,
        num_workers: int = 1,
    ) -> tuple[pd.Series, dict[str, int]]:
        """Anonymize a single column in a DataFrame."""
        col_stats_agg = Counter()

        col_data = col.fillna("").astype(str).to_list()
        if not col_data:
            tqdm.write(
                f"Column {col.name} is empty or contains only NaN values. "
                "Skipping anonymization and returning original column."
            )
            return col, dict(col_stats_agg)

        try:
            analyzer_results = batch_analyzer.analyze_iterator(
                col_data,
                language="en",
                batch_size=batch_size,
                n_process=num_workers,
                entities=self.ENTITIES,
            )
        except Exception as e:
            tqdm.write(
                f"Error analyzing column {col.name}: {e}. "
                "Skipping anonymization and returning original column."
            )
            return col, dict(col_stats_agg)

        processed_text = []
        for original_text, results in zip(col_data, analyzer_results):
            if results:
                try:
                    anonymizer_engine_result = self._anonymizer.anonymize(
                        text=original_text, analyzer_results=results
                    )

                    processed_text.append(anonymizer_engine_result.text)

                    for item in anonymizer_engine_result.items:
                        col_stats_agg[item.entity_type] += 1
                except Exception as e:
                    tqdm.write(
                        f"Error anonymizing text in column {col.name}: {e}. "
                        "Returning original text."
                    )
                    processed_text.append(original_text)
                    continue
            else:
                processed_text.append(original_text)

        return pd.Series(
            processed_text, index=col.index, dtype=col.dtype, name=col.name
        ), dict(col_stats_agg)

    def _validate_anonymize_dataframe_args(
        self,
        cols_to_keep: list[str],
        cols_to_anonymize: Optional[list[str]],
        batch_size: int,
        num_workers: int,
    ):
        """Validate the arguments for the `anonymize_dataframe` method."""
        if num_workers <= 0:
            raise ValueError(
                "Expected ``num_workers`` to be a positive integer greater than 0 "
                f"but got {num_workers}"
            )
        if batch_size <= 0:
            raise ValueError(
                "Expected ``batch_size`` to be a positive integer greater than 0 "
                f"but got {batch_size}"
            )

        if cols_to_anonymize is None:
            cols_to_anonymize = []

        if not isinstance(cols_to_keep, list) or not all(
            isinstance(col, str) for col in cols_to_keep
        ):
            raise ValueError(
                "Expected `cols_to_keep` to be a list of strings but got "
                f"{type(cols_to_keep)}"
            )
        if not isinstance(cols_to_anonymize, list) or not all(
            isinstance(col, str) for col in cols_to_anonymize
        ):
            raise ValueError(
                "Expected `cols_to_anonymize` to be a list of strings but got "
                f"{type(cols_to_anonymize)}"
            )

        return cols_to_keep, cols_to_anonymize

    def _log_anonymization_stats(self, df_entity_stats: Counter) -> None:
        if df_entity_stats:
            stats_table = pd.DataFrame(
                list(df_entity_stats.items()), columns=["Entity", "Count"]
            ).sort_values(by="Count", ascending=False)
        else:
            stats_table = pd.DataFrame(columns=["Entity", "Count"])

        stats_table.loc[len(stats_table)] = ["-" * 5, "-" * 5]
        stats_table.loc[len(stats_table)] = ["Total", sum(df_entity_stats.values())]
        stats_table.loc[len(stats_table)] = ["Score Threshold", self._score_threshold]

        logging.info(stats_table.to_string(index=False))
