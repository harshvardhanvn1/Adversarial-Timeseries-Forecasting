import argparse, os
import numpy as np
import pandas as pd

from lightning.pytorch import Trainer, seed_everything  # ✅ Lightning 2.x namespace
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.models import DeepAR

from src.pf_data import load_long_dataset

# --- silence noisy warnings, keep progress bar ---
import warnings, logging

# target the sklearn one you saw
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but StandardScaler was fitted with feature names",
    category=UserWarning,
)

# (optional) quiet common libs without touching the progress bar
warnings.filterwarnings("ignore", category=FutureWarning, module="pytorch_forecasting")
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_forecasting")
warnings.filterwarnings("ignore", category=UserWarning, module="lightning")
logging.getLogger("pytorch_forecasting").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


def build_datasets(df_long: pd.DataFrame,
                   max_encoder_length: int = 48,
                   max_prediction_length: int = 1):
    training = TimeSeriesDataSet(
        df_long,
        time_idx="time_idx",
        target="y",
        group_ids=["unique_id"],
        min_encoder_length=max(8, max_encoder_length // 2),
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,
        time_varying_unknown_reals=["y"],
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    validation = TimeSeriesDataSet.from_dataset(training, df_long, predict=True, stop_randomization=True)
    train_dl = training.to_dataloader(train=True, batch_size=128, num_workers=0)
    val_dl = validation.to_dataloader(train=False, batch_size=256, num_workers=0)
    return training, validation, train_dl, val_dl

def evaluate_predictions(preds: np.ndarray, x) -> pd.DataFrame:
    y_true = x["decoder_target"].reshape(-1).cpu().numpy()
    y_pred = preds.reshape(-1)
    eps = 1e-8
    uid = np.array([g[0] for g in x["groups"]]).reshape(-1)
    df = pd.DataFrame({"unique_id": uid, "y_true": y_true, "y_pred": y_pred})
    per_series = df.groupby("unique_id").apply(
        lambda g: pd.Series({
            "RMSE_CLEAN": np.sqrt(((g.y_true - g.y_pred) ** 2).mean()),
            "MAPE_CLEAN": (np.abs((g.y_true - g.y_pred) / (np.abs(g.y_true) + eps))).mean() * 100.0
        })
    ).reset_index()
    overall = pd.DataFrame([{
        "unique_id": "__MEAN__",
        "RMSE_CLEAN": per_series["RMSE_CLEAN"].mean(),
        "MAPE_CLEAN": per_series["MAPE_CLEAN"].mean()
    }])
    return pd.concat([per_series, overall], ignore_index=True)

def run_pipeline(dataset: str,
                 csv_path: str,
                 freq: str = "15min",
                 max_clients: int = 50,
                 max_encoder_length: int = 48,
                 max_prediction_length: int = 1,
                 epochs: int = 10,
                 seed: int = 42,
                 output_metrics_path: str = "results/deepar_metrics.csv") -> pd.DataFrame:

    os.makedirs("results", exist_ok=True)
    seed_everything(seed, workers=True)

    df_long = load_long_dataset(dataset=dataset, csv_path=csv_path, freq=freq, max_clients=max_clients)
    _, _, train_dl, val_dl = build_datasets(df_long, max_encoder_length, max_prediction_length)

    model = DeepAR.from_dataset(
        train_dl.dataset,
        learning_rate=1e-3,
        hidden_size=64,
        dropout=0.1,
    )
    trainer = Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices="auto",
        logger=False,
        enable_checkpointing=False,
    )
    # ✅ Lightning 2.x style, and same namespace as model
    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    preds, x = model.predict(val_dl, return_x=True, trainer=trainer)
    preds = preds.numpy().reshape(-1)
    metrics = evaluate_predictions(preds, x)
    os.makedirs(os.path.dirname(output_metrics_path), exist_ok=True)
    metrics.to_csv(output_metrics_path, index=False)
    print(f"[INFO] Saved metrics to {output_metrics_path}")
    return metrics

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["uci", "m4"])
    ap.add_argument("--csv", required=True)
    ap.add_argument("--freq", default="15min")
    ap.add_argument("--max_clients", type=int, default=50)
    ap.add_argument("--max_encoder_length", type=int, default=48)
    ap.add_argument("--max_prediction_length", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_metrics_path", default="results/deepar_metrics.csv")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    _ = run_pipeline(
        dataset=args.dataset,
        csv_path=args.csv,
        freq=args.freq,
        max_clients=args.max_clients,
        max_encoder_length=args.max_encoder_length,
        max_prediction_length=args.max_prediction_length,
        epochs=args.epochs,
        seed=args.seed,
        output_metrics_path=args.output_metrics_path,
    )
