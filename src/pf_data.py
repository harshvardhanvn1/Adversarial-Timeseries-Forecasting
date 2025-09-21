# src/pf_data.py
import pandas as pd
from typing import Optional
from src.data_prep import load_uci_electricity, load_m4_wide, preprocess_series

def to_long_from_wide(df: pd.DataFrame, max_clients: Optional[int] = None) -> pd.DataFrame:
    """
    Convert wide DF (index: time; columns: series ids) to long format:
      columns = ['unique_id', 'time_idx', 'y'].
    Index can be datetime or integer; we map to 0..T-1 with time_idx.
    """
    if max_clients is not None and max_clients > 0:
        df = df.iloc[:, :max_clients].copy()
    df = df.copy().reset_index(drop=True)  # ensure 0..T-1
    long = df.reset_index().melt(id_vars="index", var_name="unique_id", value_name="y")
    long = long.rename(columns={"index": "time_idx"})
    long = long.dropna(subset=["y"]).reset_index(drop=True)
    long["unique_id"] = long["unique_id"].astype(str)
    long["time_idx"] = long["time_idx"].astype(int)
    long["y"] = long["y"].astype("float32")
    return long

def load_long_dataset(dataset: str, csv_path: str, freq: str = "15min",
                      max_clients: Optional[int] = None) -> pd.DataFrame:
    """
    Load UCI or M4 and return long dataframe [unique_id, time_idx, y].
    Applies your preprocess_series() per column before melting.
    """
    if dataset.lower() == "uci":
        wide = load_uci_electricity(csv_path, freq=freq)
    elif dataset.lower() == "m4":
        wide = load_m4_wide(csv_path)
    else:
        raise ValueError("dataset must be one of {uci, m4}")

    if max_clients is not None and max_clients > 0:
        wide = wide.iloc[:, :max_clients].copy()

    for c in wide.columns:
        wide[c] = preprocess_series(wide[c])

    return to_long_from_wide(wide, max_clients=None)
