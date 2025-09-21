from __future__ import annotations

import argparse
import os
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# local utils
from uci_data_prep import load_uci_electricity, preprocess_series

# TFT stack
import pytorch_lightning as pl
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import MAE


def make_calendar_feats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add calendar/time features from datetime index.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index must be datetime")

    df = df.copy()
    d = df.index  # datetime index

    df["hour"] = d.hour
    df["dayofweek"] = d.dayofweek
    df["month"] = d.month
    df["year"] = d.year
    return df



def build_client_frame(df, client_id):
    dfl = pd.DataFrame({"datetime": df.index, "value": df[client_id]})
    dfl = dfl.set_index("datetime")  # ✅ make datetime the index
    dfl["value"] = preprocess_series(dfl["value"])
    dfl = make_calendar_feats(dfl)   # now index is datetime
    return dfl



def train_one_client_tft(
    df_client: pd.DataFrame,
    max_encoder_length: int = 96,    # 1 day history @ 15-min
    max_prediction_length: int = 24, # 6-hour horizon @ 15-min
    batch_size: int = 64,
    max_epochs: int = 3,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Train+evaluate a small TFT for one client.
    Returns a dict of metrics on the validation set (RMSE, MAPE).
    """
    # time-based split
    training_cutoff = df_client["time_idx"].max() - max_prediction_length * 7  # one week of validation
    df_train = df_client[df_client["time_idx"] <= training_cutoff]
    df_val = df_client[df_client["time_idx"] > training_cutoff]

    # dataset spec
    ts_train = TimeSeriesDataSet(
        df_train,
        time_idx="time_idx",
        target="value",
        group_ids=["group_id"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_known_reals=["time_idx", "hour", "dow", "month"],
        time_varying_unknown_reals=["value"],
        target_normalizer=None,  # keep raw scale (easier to compare)
        add_relative_time_idx=True,
        add_target_scales=False,
        add_encoder_length=True,
    )

    ts_val = TimeSeriesDataSet.from_dataset(ts_train, df_val, stop_randomization=True)

    # dataloaders (num_workers=0 to avoid mac fork issues)
    train_loader = ts_train.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_loader = ts_val.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    # small TFT; set hidden sizes modest to keep CPU training light
    model = TemporalFusionTransformer.from_dataset(
        ts_train,
        learning_rate=1e-3,
        hidden_size=16,
        attention_head_size=2,
        dropout=0.1,
        hidden_continuous_size=8,
        loss=MAE(),  # simple, robust metric
        log_interval=50,
        reduce_on_plateau_patience=2,
    )

    # trainer on CPU unless you pass mps/gpu explicitly
    accelerator = "cpu" if device == "cpu" else device
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        deterministic=True,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # predictions on validation set
    preds, idx = model.predict(val_loader, return_index=True, show_progress_bar=False)
    # preds shape: [N, max_prediction_length]; flatten along time dim to line up with targets
    preds = preds.reshape(-1)

    # collect matching targets
    # the dataset stores targets per row as 'value' shifted into decoder; easiest is to rebuild y_true via ts_val
    y_list = []
    for batch in val_loader:
        # batch is dict of tensors; 'decoder_target' has shape [B, max_prediction_length]
        y = batch[ts_val.target_names[0]]  # decoder_target
        y_list.append(y.detach().cpu().numpy())
    y_true = np.concatenate([y.reshape(-1) for y in y_list], axis=0)

    # same length check
    n = min(len(preds), len(y_true))
    y_true = y_true[:n]
    preds = preds[:n]

    # metrics
    rmse = float(np.sqrt(np.mean((y_true - preds) ** 2)))
    eps = 1e-8
    mape = float(np.mean(np.abs((y_true - preds) / np.maximum(np.abs(y_true), eps))) * 100.0)

    return {"RMSE": rmse, "MAPE": mape}


def run_pipeline(
    csv_path: str,
    freq: str = "15min",
    n_clients: int = 5,
    max_epochs: int = 3,
    batch_size: int = 64,
    device: str = "cpu",
):
    # load raw UCI data (datetime as index, float32 values)
    df = load_uci_electricity(csv_path, freq=freq)

    # bring datetime back as a column named "datetime"
    df = df.reset_index().rename(columns={df.columns[0]: "datetime"})

    # pick first N clients
    all_clients = [c for c in df.columns if c != "datetime"]
    clients = all_clients[:n_clients]

    os.makedirs("results", exist_ok=True)
    rows = []
    for client in clients:
        print(f"[CLIENT] {client}")
        df_client = build_client_frame(df, client)

        metrics = train_one_client_tft(
            df_client=df_client,
            max_encoder_length=96,
            max_prediction_length=24,
            batch_size=batch_size,
            max_epochs=max_epochs,
            device=device,
        )
        print({**{"client": client}, **metrics})
        rows.append({"client": client, **metrics})

    pd.DataFrame(rows).to_csv("results/tft_metrics.csv", index=False)
    print("[INFO] Saved metrics to results/tft_metrics.csv")


def parse_args():
    p = argparse.ArgumentParser(description="Temporal Fusion Transformer on UCI Electricity (per-client).")
    p.add_argument("--csv", required=True, help="Path to UCI LD2011_2014 .txt")
    p.add_argument("--freq", default="15min", help="Resample frequency (keep '15min' for this dataset)")
    p.add_argument("--n_clients", type=int, default=5, help="How many client series to train")
    p.add_argument("--max_epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", default="cpu", help="cpu | mps | gpu (if available)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        csv_path=args.csv,
        freq=args.freq,
        n_clients=args.n_clients,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        device=args.device,
    )
