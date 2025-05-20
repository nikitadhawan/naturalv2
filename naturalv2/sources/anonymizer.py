import multiprocessing as mp
from typing import Optional

import dask.dataframe as dd
import pandas as pd
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from tqdm.dask import TqdmCallback


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

    def __init__(self, score_threshold: float = 0.3) -> None:
        self._analyzer = AnalyzerEngine(default_score_threshold=score_threshold)
        self._anonymizer = AnonymizerEngine()

    def anonymize(self, text: str) -> str:
        results = self._analyzer.analyze(
            text=text, language="en", entities=self.ENTITIES
        )
        return self._anonymizer.anonymize(text=text, analyzer_results=results)

    def anonymize_dataframe(
        self,
        adf: pd.DataFrame,
        cols: Optional[list[str]] = None,
        num_workers: Optional[int] = None,
    ) -> pd.DataFrame:
        if num_workers is not None and num_workers <= 0:
            raise ValueError(
                "Expected ``num_workers`` to be a positive integer greater than 0 "
                f"but got {num_workers}"
            )

        if cols is None:
            cols = adf.columns

        # verify that all 'cols' are in df.columns
        for col in cols:
            if col not in adf.columns:
                raise ValueError(f"Column {col} not found in DataFrame")

        n_cores = num_workers or mp.cpu_count() - 1 or 1
        ddf: dd.DataFrame = dd.from_pandas(adf, npartitions=n_cores)

        for col in cols:
            dtype = ddf[col].dtype

            if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(
                dtype
            ):
                ddf[col] = ddf[col].apply(
                    lambda x: self.anonymize(x).text if isinstance(x, str) else x,
                    meta=pd.Series(dtype=pd.StringDtype("pyarrow"), name=col),
                )

        with TqdmCallback():
            return ddf.compute()
