import numpy as np
import pandas as pd


def fgsm_attack_features(X: pd.DataFrame, model, epsilon: float) -> pd.DataFrame:
    """
    FGSM-like attack on linear models in feature space.
    """
    X_adv = X.copy()
    feat = [c for c in X.columns if c != "y"]
    w = model.coef_
    y_hat = model.predict(X[feat])
    resid = (X["y"].values - y_hat)
    grad = (-2.0 * resid.reshape(-1, 1)) * w.reshape(1, -1)
    perturb = epsilon * np.sign(grad)
    X_adv[feat] = X[feat].values + perturb
    return X_adv


def pgd_attack_features(X: pd.DataFrame, model, epsilon: float,
                        step: float = 0.01, iters: int = 10) -> pd.DataFrame:
    """
    PGD attack with iterative updates in feature space.
    """
    X_adv = X.copy()
    feat = [c for c in X.columns if c != "y"]
    Xs0 = X[feat].values.astype(float)
    delta = np.zeros_like(Xs0)
    for _ in range(iters):
        X_tmp = X.copy()
        X_tmp[feat] = Xs0 + delta
        y_hat = model.predict(X_tmp[feat])
        resid = (X["y"].values - y_hat).reshape(-1, 1)
        w = model.coef_.reshape(1, -1)
        grad = (-2.0 * resid) * w
        delta = delta + step * np.sign(grad)
        delta = np.clip(delta, -epsilon, epsilon)
    X_adv[feat] = Xs0 + delta
    return X_adv
