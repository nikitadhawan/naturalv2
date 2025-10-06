"""Date-related helpers shared across source stages."""

import logging

import pandas as pd


logger = logging.getLogger(__name__)


def filter_by_date(
    adf: pd.DataFrame, cutoff_dt: pd.Timestamp, date_col: str
) -> pd.DataFrame:
    """Filter a DataFrame by a date cutoff.

    Parameters
    ----------
    adf : pd.DataFrame
        The DataFrame to filter.
    cutoff_dt : pd.Timestamp
        The cutoff timestamp. Only rows with dates before this date will be kept.
    date_col : str
        The name of the column in the DataFrame containing date information.

    Returns
    -------
    pd.DataFrame
        A DataFrame filtered to include only rows with dates on or before the cutoff
        date.
    """
    if adf.empty:
        return pd.DataFrame()

    date_series: pd.Series = pd.to_datetime(adf[date_col], errors="coerce")
    num_no_date = date_series.isna().sum()
    if num_no_date > 0:
        logger.debug(
            "Found %d rows with NaN values in '%s' column.", num_no_date, date_col
        )

    mask = (date_series.notna()) & (date_series < cutoff_dt)
    return adf.loc[mask].reset_index(drop=True)
