import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

from src.data_prep import load_uci_electricity, load_m4_wide, preprocess_series
from src.utils import make_lag_features, train_test_split_time, rmse, mape
from src.attacks import fgsm_attack_features, pgd_attack_features
from src.defenses import tsas_smooth_feature_matrix


def run_pipeline(dataset: str,
                 csv_path: str,
                 freq: str = "15min",
                 alpha: float = 1.0,
                 attack: str = "fgsm",
                 epsilon: float = 0.05,
                 defense: str = "tsas",
                 smoother: str = "moving_avg",
                 window: int = 5,
                 test_size: float = 0.2,
                 max_clients: int = 5,
                 output_metrics_path: str = "results/metrics.csv",
                 output_plot_dir: str = "results/plots",
                 save_plots: bool = True,
                 write_metrics: bool = True) -> pd.DataFrame:

    if save_plots:
        os.makedirs(output_plot_dir, exist_ok=True)

    # ---------- Load data ----------
    if dataset.lower() == "uci":
        df = load_uci_electricity(csv_path, freq=freq)
    elif dataset.lower() == "m4":
        df = load_m4_wide(csv_path)
    else:
        raise ValueError("dataset must be one of {uci, m4}")

    all_metrics = []
    cols = list(df.columns)
    if max_clients is not None and max_clients > 0:
        cols = cols[:max_clients]

    for client in cols:
        print(f"\n[CLIENT] {client}")
        y = preprocess_series(df[client])
        sup = make_lag_features(y, lags=[1, 24, 168], roll_windows=[3, 24])
        if sup.empty:
            print("  - empty after feature generation; skip")
            continue

        train_df, test_df = train_test_split_time(sup, test_size=test_size)

        model = Ridge(alpha=alpha).fit(train_df.drop(columns="y"), train_df["y"])
        y_hat_clean = model.predict(test_df.drop(columns="y"))

        if attack == "fgsm":
            X_adv = fgsm_attack_features(test_df, model, epsilon)
        elif attack == "pgd":
            X_adv = pgd_attack_features(test_df, model, epsilon)
        else:
            X_adv = test_df.copy()
        y_hat_adv = model.predict(X_adv.drop(columns="y"))

        if defense == "tsas":
            X_def = tsas_smooth_feature_matrix(X_adv, window=window, method=smoother)
            y_hat_def = model.predict(X_def.drop(columns="y"))
        else:
            y_hat_def = y_hat_adv

        y_true = test_df["y"].values
        metrics = {
            "client": client,
            "MAE_CLEAN": mean_absolute_error(y_true, y_hat_clean),
            "RMSE_CLEAN": rmse(y_true, y_hat_clean),
            "MAPE_CLEAN": mape(y_true, y_hat_clean),
            "MAE_ADV": mean_absolute_error(y_true, y_hat_adv),
            "RMSE_ADV": rmse(y_true, y_hat_adv),
            "MAPE_ADV": mape(y_true, y_hat_adv),
            "MAE_DEF": mean_absolute_error(y_true, y_hat_def),
            "RMSE_DEF": rmse(y_true, y_hat_def),
            "MAPE_DEF": mape(y_true, y_hat_def),
        }
        all_metrics.append(metrics)

        if save_plots:
            plt.figure()
            plt.plot(y_true, label="Actual")
            plt.plot(y_hat_clean, label="Clean")
            plt.plot(y_hat_adv, label="Attacked")
            plt.plot(y_hat_def, label="Defended")
            plt.legend()
            plt.title(f"{dataset.upper()} | {client}")
            plt.tight_layout()
            plt.savefig(f"{output_plot_dir}/{dataset}_{client}.png", bbox_inches="tight")
            plt.close()

    df_metrics = pd.DataFrame(all_metrics)
    if write_metrics:
        os.makedirs(os.path.dirname(output_metrics_path), exist_ok=True)
        df_metrics.to_csv(output_metrics_path, index=False)
        print(f"\n[INFO] Metrics saved to {output_metrics_path}")

    return df_metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["uci", "m4"])
    p.add_argument("--csv", required=True)
    p.add_argument("--freq", default="15min")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--attack", default="fgsm", choices=["fgsm", "pgd", "none"])
    p.add_argument("--epsilon", type=float, default=0.05)
    p.add_argument("--defense", default="tsas", choices=["tsas", "none"])
    p.add_argument("--smoother", default="moving_avg", choices=["moving_avg", "wavelet"])
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--max_clients", type=int, default=5)
    p.add_argument("--output_metrics_path", default="results/metrics_all_clients.csv")
    p.add_argument("--output_plot_dir", default="results/plots")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    max_clients = None if (args.max_clients is not None and args.max_clients < 0) else args.max_clients
    # CLI usage keeps default behavior (write CSV + plots)
    _ = run_pipeline(
        dataset=args.dataset,
        csv_path=args.csv,
        freq=args.freq,
        alpha=args.alpha,
        attack=args.attack,
        epsilon=args.epsilon,
        defense=args.defense,
        smoother=args.smoother,
        window=args.window,
        test_size=args.test_size,
        max_clients=max_clients,
        output_metrics_path=args.output_metrics_path,
        output_plot_dir=args.output_plot_dir,
        save_plots=True,
        write_metrics=True,
    )
