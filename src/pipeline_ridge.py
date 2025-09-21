import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

from uci_data_prep import load_uci_electricity, preprocess_series
from utils import make_lag_features, train_test_split_time, rmse, mape
from attacks import fgsm_attack_features, pgd_attack_features
from defenses import tsas_smooth_feature_matrix



def run_pipeline(csv_path: str,
                 freq: str = "15min",
                 attack: str = "fgsm",
                 epsilon: float = 0.05,
                 defense: str = "tsas",
                 smoother: str = "moving_avg",
                 window: int = 5,
                 output_metrics_path: str = "results/metrics.csv",
                 output_plot_dir: str = "results/plots"):

    df = load_uci_electricity(csv_path, freq)
    os.makedirs(output_plot_dir, exist_ok=True)

    all_metrics = []
    for client in df.columns[:5]:   # limit for demo, remove [:5] for all clients
        print(f"\n[CLIENT] {client}")
        y = preprocess_series(df[client])

        sup = make_lag_features(y, lags=[1,24,168], roll_windows=[3,24])
        if sup.empty:
            continue
        train_df, test_df = train_test_split_time(sup, test_size=0.2)

        model = Ridge(alpha=1.0).fit(train_df.drop(columns="y"), train_df["y"])
        y_hat_clean = model.predict(test_df.drop(columns="y"))

        # attack
        if attack == "fgsm":
            X_adv = fgsm_attack_features(test_df, model, epsilon)
        elif attack == "pgd":
            X_adv = pgd_attack_features(test_df, model, epsilon)
        else:
            X_adv = test_df.copy()
        y_hat_adv = model.predict(X_adv.drop(columns="y"))

        # defense
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

        # save plot
        plt.figure()
        plt.plot(y_true, label="Actual")
        plt.plot(y_hat_clean, label="Clean")
        plt.plot(y_hat_adv, label="Attacked")
        plt.plot(y_hat_def, label="Defended")
        plt.legend()
        plt.title(client)
        plt.savefig(f"{output_plot_dir}/{client}.png", bbox_inches="tight")
        plt.close()

    pd.DataFrame(all_metrics).to_csv(output_metrics_path, index=False)
    print(f"\n[INFO] Metrics saved to {output_metrics_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to dataset")
    parser.add_argument("--freq", default="15min", help="Frequency for resampling (default: 15min)")
    parser.add_argument("--attack", default="fgsm", choices=["fgsm", "pgd", "none"])
    parser.add_argument("--epsilon", type=float, default=0.05, help="Attack strength")
    parser.add_argument("--defense", default="tsas", choices=["tsas", "none"])
    parser.add_argument("--smoother", default="moving_avg", choices=["moving_avg", "wavelet"])
    parser.add_argument("--window", type=int, default=5, help="Window size for smoothing")
    parser.add_argument("--test_size", type=float, default=0.2, help="Fraction of data for testing")
    parser.add_argument("--output_metrics_path", default="results/metrics_all_clients.csv")
    parser.add_argument("--output_plot_dir", default="results/plots")

    args = parser.parse_args()

    run_pipeline(
        csv_path=args.csv,
        freq=args.freq,
        attack=args.attack,
        epsilon=args.epsilon,
        defense=args.defense,
        smoother=args.smoother,
        window=args.window,
        output_metrics_path=args.output_metrics_path,
        output_plot_dir=args.output_plot_dir,
    )

