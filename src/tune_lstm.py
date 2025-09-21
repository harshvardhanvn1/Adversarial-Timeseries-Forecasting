# src/tune_lstm.py
import os
import itertools
import pandas as pd
from datetime import datetime

from src.pipeline_lstm import run_pipeline

def summarize_df(df: pd.DataFrame) -> dict:
    return {
        "mean_RMSE_CLEAN": df["RMSE_CLEAN"].mean(),
        "mean_RMSE_ADV": df["RMSE_ADV"].mean(),
        "mean_RMSE_DEF": df["RMSE_DEF"].mean(),
        "robust_drop": df["RMSE_ADV"].mean() - df["RMSE_CLEAN"].mean(),
        "defense_recovery": df["RMSE_ADV"].mean() - df["RMSE_DEF"].mean(),
    }

def main():
    os.makedirs("results/tuning_lstm", exist_ok=True)

    # === Grid (start moderate; expand if you like) ===
    DATASETS = [("m4", "datasets/M4/Hourly-train.csv", None)]
    HIDDEN = [32, 64, 128]
    LAYERS = [1, 2]
    EPOCHS = [5, 10]
    ATTACKS = ["none", "fgsm", "pgd"]
    EPSILONS = [0.01, 0.05]          # keep small first
    DEFENSES = ["none", "tsas"]
    WINDOWS = [3, 5, 12]

    rows = []
    for (dataset, csv_path, _), hs, nl, ep, attack, eps, defense, window in itertools.product(
        DATASETS, HIDDEN, LAYERS, EPOCHS, ATTACKS, EPSILONS, DEFENSES, WINDOWS
    ):
        tag = f"{dataset}_hs{hs}_nl{nl}_ep{ep}_atk{attack}_e{eps}_def{defense}_w{window}"
        print(f"\n=== RUN {tag} ===")

        df_metrics = run_pipeline(
            dataset=dataset,
            csv_path=csv_path,
            freq="ignore",
            hidden_size=hs,
            num_layers=nl,
            attack=attack,
            epsilon=eps,
            defense=defense,
            smoother="moving_avg",
            window=window,
            test_size=0.2,
            epochs=ep,
            batch_size=32,
            max_clients=20,            # raise to 50/100 when stable
            output_metrics_path="",    # ignored
            output_plot_dir="",        # ignored
            save_plots=False,
            write_metrics=False,
        )

        s = summarize_df(df_metrics)
        s.update({
            "dataset": dataset, "csv": csv_path,
            "hidden_size": hs, "num_layers": nl, "epochs": ep,
            "attack": attack, "epsilon": eps, "defense": defense, "window": window,
            "num_clients": len(df_metrics)
        })
        rows.append(s)

    leaderboard = pd.DataFrame(rows)

    # Overall leaderboard: prioritize low defended error, then small robust drop, then higher recovery
    leaderboard = leaderboard.sort_values(
        ["mean_RMSE_DEF", "robust_drop", "defense_recovery"],
        ascending=[True, True, False]
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = f"results/tuning_lstm/leaderboard_lstm_{ts}.csv"
    leaderboard.to_csv(out_csv, index=False)

    print(f"\n[LSTM TUNING] Leaderboard saved to {out_csv}")
    print("\nTop-5:")
    print(leaderboard.head(5).to_string(index=False))

    # Robust-only leaderboard (attack != none)
    robust = leaderboard[leaderboard["attack"] != "none"].copy()
    if not robust.empty:
        robust = robust.sort_values(
            ["mean_RMSE_DEF", "robust_drop", "defense_recovery"],
            ascending=[True, True, False]
        )
        out_csv_robust = out_csv.replace("leaderboard_lstm_", "leaderboard_lstm_robust_")
        robust.to_csv(out_csv_robust, index=False)
        print(f"\n[LSTM TUNING] Robust-only leaderboard saved to {out_csv_robust}")
        print("\nTop-5 (robust-only):")
        print(robust.head(5).to_string(index=False))
        best_r = robust.iloc[0].to_dict()
        print("\nBest robust configuration (attack ∈ {fgsm,pgd}):")
        for k, v in best_r.items():
            print(f"  {k}: {v}")
    else:
        print("\n[LSTM TUNING] No robust rows found.")

if __name__ == "__main__":
    main()