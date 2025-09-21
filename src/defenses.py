import pandas as pd

try:
    import pywt
    _HAS_PYWT = True
except ImportError:
    _HAS_PYWT = False


def tsas_smooth_series(s: pd.Series, method: str = "moving_avg", window: int = 5) -> pd.Series:
    """
    Temporal smoothing defense (TSAS).
    """
    if method == "wavelet" and _HAS_PYWT:
        coeffs = pywt.wavedec(s.values, "db2", level=2, mode="symmetric")
        sigma = (abs(coeffs[-1])).mean() + 1e-9
        uthresh = sigma * (2 * len(s))**0.5
        coeffs[1:] = [pywt.threshold(c, value=uthresh, mode="soft") for c in coeffs[1:]]
        rec = pywt.waverec(coeffs, "db2", mode="symmetric")[: len(s)]
        return pd.Series(rec, index=s.index)
    return s.rolling(window, min_periods=1, center=True).mean()


def tsas_smooth_feature_matrix(X: pd.DataFrame,
                               cols_like: str = "lag_",
                               window: int = 5,
                               method: str = "moving_avg") -> pd.DataFrame:
    """
    Apply TSAS smoothing to all lag features in X.
    """
    Xs = X.copy()
    lag_cols = [c for c in X.columns if c.startswith(cols_like)]
    for c in lag_cols:
        Xs[c] = tsas_smooth_series(X[c], method=method, window=window)
    return Xs
