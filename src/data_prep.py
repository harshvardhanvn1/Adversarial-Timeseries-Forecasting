# src/data_prep.py
import pandas as pd
import numpy as np
from typing import Dict

# ---------- UCI (unchanged semantics; copied from your uci_data_prep.py) ----------

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

    # note: your original used asfreq (no aggregation). keep identical behavior.
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
    s = series.copy()

    # impute
    if method == "ffill_then_bfill":
        s = s.ffill().bfill()
    elif method == "interpolate":
        s = s.interpolate(limit_direction="both")

    # robust z-score outlier clip
    med = s.median()
    mad = (np.abs(s - med)).median() + 1e-9
    s = s.clip(lower=med - 4 * mad / 0.6745,
               upper=med + 4 * mad / 0.6745)

    return s

# ---------- M4 utilities ----------

def load_m4_dataset(path: str) -> Dict[str, pd.Series]:
    """
    Load an M4 train CSV (e.g., Hourly-train.csv, Daily-train.csv, etc.).

    Expected format (typical M4):
      - First column: series id
      - Remaining columns: observations in chronological order (left->right)
      - Rows may have different effective lengths (trailing empties/NaNs)

    Returns:
      dict: {series_id: pd.Series(values, index=RangeIndex)}
    """
    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError("M4 file must have at least 2 columns (id + values).")

    id_col = df.columns[0]
    value_cols = list(df.columns[1:])

    series_dict: Dict[str, pd.Series] = {}
    for _, row in df.iterrows():
        sid = str(row[id_col])
        vals = row[value_cols].astype("float64")
        # drop trailing NaNs (common in M4 CSVs)
        vals = vals[~vals.isna()]
        s = pd.Series(vals.values, dtype="float32")
        s.index = pd.RangeIndex(start=0, stop=len(s))
        series_dict[sid] = s

    return series_dict


def m4_to_wide_dataframe(series_dict: Dict[str, pd.Series]) -> pd.DataFrame:
    """
    Convert dict of {id: Series} (variable lengths) into a right-aligned wide DataFrame:
      - Rows: integer time index (0..max_len-1)
      - Cols: series ids
      - Each series is right-aligned (last observation at the last row)
      - Leading values padded with NaN (safe: your feature pipeline drops NaNs)
    """
    if not series_dict:
        return pd.DataFrame()

    max_len = max(len(s) for s in series_dict.values())
    out = {}
    for sid, s in series_dict.items():
        pad_len = max_len - len(s)
        if pad_len > 0:
            padded = np.concatenate([np.full(pad_len, np.nan, dtype="float32"), s.values.astype("float32")])
        else:
            padded = s.values.astype("float32")
        out[sid] = pd.Series(padded)

    wide = pd.DataFrame(out)
    wide.index = pd.RangeIndex(start=0, stop=max_len)
    return wide


def load_m4_wide(path: str) -> pd.DataFrame:
    """
    Convenience loader: read M4 CSV and return a right-aligned wide DataFrame.
    """
    d = load_m4_dataset(path)
    return m4_to_wide_dataframe(d)
