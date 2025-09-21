import math
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def make_lag_features(y: pd.Series, lags, roll_windows=None) -> pd.DataFrame:
    """
    Create supervised lagged features and optional rolling statistics.
    """
    df = pd.DataFrame({"y": y})
    for L in lags:
        df[f"lag_{L}"] = y.shift(L)
    if roll_windows:
        for w in roll_windows:
            df[f"rollmean_{w}"] = y.rolling(w).mean().shift(1)
            df[f"rollstd_{w}"] = y.rolling(w).std(ddof=0).shift(1)
    return df.dropna()


def train_test_split_time(df: pd.DataFrame, test_size: float):
    """
    Split time series data into train and test sets by index order.
    """
    n = len(df)
    split = int(n * (1 - test_size))
    return df.iloc[:split], df.iloc[split:]


def rmse(y_true, y_pred) -> float:
    return math.sqrt(mean_squared_error(y_true, y_pred))


def mape(y_true, y_pred) -> float:
    """
    Smoothed MAPE: avoids explosion when y_true is zero or near-zero
    by using an epsilon floor on the denominator.
    """
    y_true = np.asarray(y_true).astype(float)
    y_pred = np.asarray(y_pred).astype(float)
    eps = 1e-3   # floor for small denominators; adjust if needed
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100.0

