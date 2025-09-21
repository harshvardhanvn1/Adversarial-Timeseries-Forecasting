# src/pipeline_lstm.py
import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from src.data_prep import load_uci_electricity, load_m4_wide, preprocess_series
from src.utils import make_lag_features, train_test_split_time, rmse, mape
from src.attacks import fgsm_attack_torch, pgd_attack_torch
from src.defenses import tsas_smooth_feature_matrix

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out.squeeze()

def to_tensor_from_df(X_df: pd.DataFrame) -> torch.Tensor:
    arr = X_df.values.astype("float32")
    t = torch.tensor(arr, dtype=torch.float32, device=DEVICE)
    return t.unsqueeze(1)  # [n, 1, feat]

def run_pipeline(dataset: str,
                 csv_path: str,
                 freq: str = "15min",
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 attack: str = "none",
                 epsilon: float = 0.05,
                 defense: str = "none",
                 smoother: str = "moving_avg",
                 window: int = 5,
                 test_size: float = 0.2,
                 epochs: int = 5,
                 batch_size: int = 32,
                 max_clients: int = 5,
                 output_metrics_path: str = "results/lstm_metrics.csv",
                 output_plot_dir: str = "results/lstm_plots",
                 save_plots: bool = True,
                 write_metrics: bool = True) -> pd.DataFrame:

    if save_plots:
        os.makedirs(output_plot_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    if dataset.lower() == "uci":
        df = load_uci_electricity(csv_path, freq=freq)
    elif dataset.lower() == "m4":
        df = load_m4_wide(csv_path)
    else:
        raise ValueError("dataset must be one of {uci, m4}")

    clients = list(df.columns)
    if max_clients is not None and max_clients > 0:
        clients = clients[:max_clients]

    all_metrics = []
    for client in clients:
        print(f"\n[CLIENT] {client}")
        y = preprocess_series(df[client])
        sup = make_lag_features(y, lags=[1, 24, 168], roll_windows=[3, 24])
        if sup.empty:
            print("[WARN] empty supervised table, skipping")
            continue

        train_df, test_df = train_test_split_time(sup, test_size=test_size)

        X_train = to_tensor_from_df(train_df.drop(columns="y"))
        y_train = torch.tensor(train_df["y"].values, dtype=torch.float32, device=DEVICE)
        X_test_df = test_df.drop(columns="y").copy()
        y_test = torch.tensor(test_df["y"].values, dtype=torch.float32, device=DEVICE)

        input_size = X_train.shape[-1]
        model = LSTMForecaster(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        train_ds = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        model.train()
        for _ in range(epochs):
            for xb, yb in train_loader:
                optimizer.zero_grad()
                preds = model(xb)
                loss = loss_fn(preds.reshape(yb.shape), yb)
                loss.backward()
                optimizer.step()

        model.eval()
        X_test = to_tensor_from_df(X_test_df)
        with torch.no_grad():
            y_hat_clean = model(X_test).cpu().numpy().reshape(-1)

        # Attack
        if attack.lower() == "none":
            X_test_adv = X_test
        elif attack.lower() == "fgsm":
            X_test_adv = fgsm_attack_torch(model, X_test, y_test, epsilon, loss_fn)
        elif attack.lower() == "pgd":
            X_test_adv = pgd_attack_torch(model, X_test, y_test, epsilon,
                                          step=epsilon/5.0, iters=10, loss_fn=loss_fn)
        else:
            raise ValueError(f"Unknown attack: {attack}")

        with torch.no_grad():
            y_hat_adv = model(X_test_adv).cpu().numpy().reshape(-1)

        # Defense
        if defense.lower() == "tsas":
            X_test_def_df = tsas_smooth_feature_matrix(
                test_df.drop(columns="y"),
                cols_like="lag_", window=window, method=smoother
            )
            X_test_def = to_tensor_from_df(X_test_def_df)
        else:
            X_test_def = X_test_adv

        with torch.no_grad():
            y_hat_def = model(X_test_def).cpu().numpy().reshape(-1)

        y_true = test_df["y"].values
        metrics = {
            "client": client,
            "RMSE_CLEAN": rmse(y_true, y_hat_clean),
            "MAPE_CLEAN": mape(y_true, y_hat_clean),
            "RMSE_ADV": rmse(y_true, y_hat_adv),
            "MAPE_ADV": mape(y_true, y_hat_adv),
            "RMSE_DEF": rmse(y_true, y_hat_def),
            "MAPE_DEF": mape(y_true, y_hat_def),
        }
        print(metrics)
        all_metrics.append(metrics)

        if save_plots:
            plt.figure()
            plt.plot(y_true, label="Actual")
            plt.plot(y_hat_clean, label="Clean")
            if attack.lower() != "none":
                plt.plot(y_hat_adv, label=f"Attacked:{attack}")
            if defense.lower() != "none":
                plt.plot(y_hat_def, label=f"Defended:{defense}")
            plt.legend()
            plt.title(f"{dataset.upper()} | {client}")
            plt.tight_layout()
            plt.savefig(os.path.join(output_plot_dir, f"{dataset}_{client}.png"))
            plt.close()

    df_metrics = pd.DataFrame(all_metrics)
    if write_metrics:
        os.makedirs(os.path.dirname(output_metrics_path), exist_ok=True)
        df_metrics.to_csv(output_metrics_path, index=False)
        print(f"[INFO] Saved metrics to {output_metrics_path}")
    return df_metrics

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["uci", "m4"])
    p.add_argument("--csv", required=True)
    p.add_argument("--freq", default="15min")
    p.add_argument("--hidden_size", type=int, default=64)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--attack", default="none", choices=["fgsm","pgd","none"])
    p.add_argument("--epsilon", type=float, default=0.05)
    p.add_argument("--defense", default="none", choices=["tsas","none"])
    p.add_argument("--smoother", default="moving_avg", choices=["moving_avg","wavelet"])
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_clients", type=int, default=5)
    p.add_argument("--output_metrics_path", default="results/lstm_metrics.csv")
    p.add_argument("--output_plot_dir", default="results/lstm_plots")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    max_clients = None if (args.max_clients is not None and args.max_clients < 0) else args.max_clients
    _ = run_pipeline(
        dataset=args.dataset,
        csv_path=args.csv,
        freq=args.freq,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        attack=args.attack,
        epsilon=args.epsilon,
        defense=args.defense,
        smoother=args.smoother,
        window=args.window,
        test_size=args.test_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_clients=max_clients,
        output_metrics_path=args.output_metrics_path,
        output_plot_dir=args.output_plot_dir,
        save_plots=True,
        write_metrics=True,
    )
