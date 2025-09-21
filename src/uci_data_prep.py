import pandas as pd
import numpy as np

def load_uci_electricity(path: str, freq: str = "15min") -> pd.DataFrame:
    """
    Load UCI ElectricityLoadDiagrams20112014 dataset.
    - Semicolon delimiter
    - Comma as decimal separator
    - First column is datetime
    - Returns dataframe with datetime index and float32 values
    """
    df = pd.read_csv(
        path,
        sep=";",
        decimal=",",
        low_memory=False
    )

    # first col is datetime
    ts_col = df.columns[0]
    df[ts_col] = pd.to_datetime(df[ts_col])
    df = df.set_index(ts_col)

    # numeric conversion
    df = df.apply(pd.to_numeric, errors="coerce").astype("float32")

    # resample if needed
    if freq:
        df = df.asfreq(freq)

    return df


def load_single_series(df: pd.DataFrame, client_id: str) -> pd.Series:
    """
    Extract a single client's time series.
    """
    if client_id not in df.columns:
        raise ValueError(f"{client_id} not found in dataset")
    return df[client_id].copy()


def preprocess_series(series: pd.Series, method: str = "ffill_then_bfill") -> pd.Series:
    """
    Handle missing values and clip extreme outliers.
    """
    # impute
    if method == "ffill_then_bfill":
        series = series.ffill().bfill()
    elif method == "interpolate":
        series = series.interpolate(limit_direction="both")

    # robust z-score outlier clip
    med = series.median()
    mad = (np.abs(series - med)).median() + 1e-9
    series = series.clip(lower=med - 4 * mad / 0.6745,
                         upper=med + 4 * mad / 0.6745)

    return series
