import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import pandas as pd
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from tqdm import tqdm


logger = logging.getLogger(__name__)


class Anonymizer:
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

    def __init__(self, score_threshold: float = 0.7) -> None:
        self._analyzer = AnalyzerEngine(default_score_threshold=score_threshold)
        self._anonymizer = AnonymizerEngine()

    def anonymize_text(self, text: str) -> str:
        results = self._analyzer.analyze(
            text=text, language="en", entities=self.ENTITIES
        )
        return self._anonymizer.anonymize(text=text, analyzer_results=results)

    def _anonymize_column(self, col: pd.Series) -> pd.Series:
        return col.apply(
            lambda x: self.anonymize_text(x).text if isinstance(x, str) else x
        )

    def anonymize_dataframe(
        self,
        df: pd.DataFrame,
        cols: list[str] = None,
        num_workers: Optional[int] = None,
    ) -> pd.DataFrame:
        if num_workers is not None and num_workers <= 0:
            raise ValueError(
                "Expected ``num_workers`` to be a positive integer greater than 0 "
                f"but got {num_workers}"
            )

        if cols is None:
            cols = df.columns

        cols_to_anonymize = [
            col
            for col in cols
            if (
                col in df.columns
                and (
                    pd.api.types.is_string_dtype(df[col].dtype)
                    or pd.api.types.is_object_dtype(df[col].dtype)
                )
            )
        ]

        if not cols_to_anonymize:
            logger.warning("\nNo columns to anonymize. Returning original DataFrame.")
            return df
        logger.info(
            f"\nAnonymizing {len(cols_to_anonymize)} columns: {cols_to_anonymize}"
        )

        missing_cols = set(cols) - set(cols_to_anonymize)
        if missing_cols:
            logger.warning(
                f"Columns {missing_cols} not found in DataFrame or not string/object type. "
                "They will not be anonymized."
            )

        progress_bar = tqdm(total=len(cols_to_anonymize), desc="Anonymizing columns")

        with ProcessPoolExecutor(num_workers) as executor:
            futures = {
                executor.submit(self._anonymize_column, df[col]): col
                for col in cols_to_anonymize
            }

            for future in as_completed(futures):
                col = futures[future]
                try:
                    df.loc[:, col] = future.result()
                    progress_bar.update(1)
                except Exception as e:
                    logger.error(f"Error anonymizing column {col}: {e}")
                    continue

        return df
