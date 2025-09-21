# src/pipeline_lstm.py
import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from uci_data_prep import load_uci_electricity, preprocess_series
from utils import make_lag_features, train_test_split_time, rmse, mape
from attacks import fgsm_attack_torch, pgd_attack_torch
from defenses import tsas_smooth_feature_matrix

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: [batch, seq_len, feat]
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out.squeeze()

def to_tensor_from_df(X_df: pd.DataFrame) -> torch.Tensor:
    # X_df shape [n_samples, n_features] -> convert to [n, seq_len=1, feat]
    arr = X_df.values.astype("float32")
    t = torch.tensor(arr, dtype=torch.float32, device=DEVICE)
    return t.unsqueeze(1)  # [n, 1, feat]

def run_pipeline(csv_path, freq="15min", attack="none", epsilon=0.05,
                 defense="none", smoother="moving_avg", window=5,
                 test_size=0.2, epochs=5, output_metrics_path="results/lstm_metrics.csv"):
    # load dataset (all clients); we'll loop over a few for demonstration
    df = load_uci_electricity(csv_path, freq=freq)
    os.makedirs("results", exist_ok=True)
    clients = list(df.columns)  # full list; you can limit with clients[:N]

    all_metrics = []
    for client in clients[:5]:  # demo: first 5 clients; remove [:5] to run all
        print(f"\n[CLIENT] {client}")
        y = preprocess_series(df[client])
        sup = make_lag_features(y, lags=[1,24,168], roll_windows=[3,24])
        if sup.empty:
            print("[WARN] empty supervised table, skipping")
            continue
        train_df, test_df = train_test_split_time(sup, test_size=test_size)

        # Prepare tensors
        X_train = to_tensor_from_df(train_df.drop(columns="y"))
        y_train = torch.tensor(train_df["y"].values, dtype=torch.float32, device=DEVICE)
        X_test_df = test_df.drop(columns="y").copy()
        y_test = torch.tensor(test_df["y"].values, dtype=torch.float32, device=DEVICE)

        # model
        input_size = X_train.shape[-1]
        model = LSTMForecaster(input_size=input_size).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        # training dataset (small example)
        train_ds = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

        model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for xb, yb in train_loader:
                optimizer.zero_grad()
                preds = model(xb)  # xb shape [batch, seq_len, feat]
                loss = loss_fn(preds.reshape(yb.shape), yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            # optional: print epoch loss
            # print(f"epoch {epoch+1}/{epochs} loss: {epoch_loss/len(train_ds):.6f}")

        model.eval()
        # Clean predictions
        X_test = to_tensor_from_df(X_test_df)
        with torch.no_grad():
            y_hat_clean = model(X_test).cpu().numpy().reshape(-1)

        # Attack
        if attack.lower() == "none":
            X_test_adv = X_test
        elif attack.lower() == "fgsm":
            X_test_adv = fgsm_attack_torch(model, X_test, y_test, epsilon, loss_fn)
        elif attack.lower() == "pgd":
            X_test_adv = pgd_attack_torch(model, X_test, y_test, epsilon, step=epsilon/5.0, iters=10, loss_fn=loss_fn)
        else:
            raise ValueError(f"Unknown attack: {attack}")

        with torch.no_grad():
            y_hat_adv = model(X_test_adv).cpu().numpy().reshape(-1)

        # Defense: smooth the features (apply to pandas then to tensor)
        if defense.lower() == "tsas":
            X_test_def_df = tsas_smooth_feature_matrix(test_df.drop(columns="y"), cols_like="lag_", window=window, method=smoother)
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

    pd.DataFrame(all_metrics).to_csv(output_metrics_path, index=False)
    print(f"[INFO] Saved metrics to {output_metrics_path}")
    return all_metrics

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--freq", default="15min")
    p.add_argument("--attack", default="none", choices=["fgsm","pgd","none"])
    p.add_argument("--epsilon", type=float, default=0.05)
    p.add_argument("--defense", default="none", choices=["tsas","none"])
    p.add_argument("--smoother", default="moving_avg", choices=["moving_avg","wavelet"])
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--output_metrics_path", default="results/lstm_metrics.csv")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        csv_path=args.csv,
        freq=args.freq,
        attack=args.attack,
        epsilon=args.epsilon,
        defense=args.defense,
        smoother=args.smoother,
        window=args.window,
        test_size=args.test_size,
        epochs=args.epochs,
        output_metrics_path=args.output_metrics_path,
    )
