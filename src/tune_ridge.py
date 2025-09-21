# src/tune_ridge.py
import os
import itertools
import pandas as pd
from datetime import datetime

from src.pipeline_ridge import run_pipeline

def summarize_df(df: pd.DataFrame) -> dict:
    return {
        "mean_RMSE_CLEAN": df["RMSE_CLEAN"].mean(),
        "mean_RMSE_ADV": df["RMSE_ADV"].mean(),
        "mean_RMSE_DEF": df["RMSE_DEF"].mean(),
        "robust_drop": df["RMSE_ADV"].mean() - df["RMSE_CLEAN"].mean(),
        "defense_recovery": df["RMSE_ADV"].mean() - df["RMSE_DEF"].mean(),
    }

def main():
    os.makedirs("results/tuning", exist_ok=True)

    # === Full grid (unchanged) ===
    DATASETS = [("m4", "datasets/M4/Hourly-train.csv", None)]  # (name, path, freq_unused)
    ALPHAS = [0.1, 0.5, 1.0, 2.0, 5.0]
    ATTACKS = ["none", "fgsm", "pgd"]
    EPSILONS = [0.01, 0.05, 0.1]
    DEFENSES = ["none", "tsas"]
    WINDOWS = [3, 5, 12]
    SMOOTHERS = ["moving_avg"]

    rows = []
    for (dataset, csv_path, _), alpha, attack, eps, defense, window, smoother in itertools.product(
        DATASETS, ALPHAS, ATTACKS, EPSILONS, DEFENSES, WINDOWS, SMOOTHERS
    ):
        tag = f"{dataset}_a{alpha}_atk{attack}_e{eps}_def{defense}_w{window}"
        print(f"\n=== RUN {tag} ===")

        # Run without saving per-config CSVs or plots
        df_metrics = run_pipeline(
            dataset=dataset,
            csv_path=csv_path,
            freq="15min" if dataset == "uci" else "ignore",
            alpha=alpha,
            attack=attack,
            epsilon=eps,
            defense=defense,
            smoother=smoother,
            window=window,
            test_size=0.2,
            max_clients=20,             # same as before
            output_metrics_path="",     # ignored when write_metrics=False
            output_plot_dir="",         # ignored when save_plots=False
            save_plots=False,
            write_metrics=False,
        )

        s = summarize_df(df_metrics)
        s.update({
            "dataset": dataset, "csv": csv_path, "alpha": alpha,
            "attack": attack, "epsilon": eps, "defense": defense,
            "window": window, "smoother": smoother,
            "num_clients": len(df_metrics)
        })
        rows.append(s)

    leaderboard = pd.DataFrame(rows)

    # Choose "best" as minimum mean_RMSE_DEF (you can switch to other criteria)
    leaderboard = leaderboard.sort_values(["mean_RMSE_DEF", "robust_drop", "defense_recovery"], ascending=[True, True, False])

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = f"results/tuning/leaderboard_ridge_{ts}.csv"
    leaderboard.to_csv(out_csv, index=False)

    # Print top-5 and best config
    print(f"\n[RIDGE TUNING] Leaderboard saved to {out_csv}")
    print("\nTop-5:")
    print(leaderboard.head(5).to_string(index=False))

    best = leaderboard.iloc[0].to_dict()
    print("\nBest configuration (by mean_RMSE_DEF):")
    for k, v in best.items():
        print(f"  {k}: {v}")

    robust = leaderboard[leaderboard["attack"] != "none"].copy()
    if not robust.empty:
        robust = robust.sort_values(["mean_RMSE_DEF", "robust_drop", "defense_recovery"],
                                    ascending=[True, True, False])
        out_csv_robust = out_csv.replace("leaderboard_ridge_", "leaderboard_ridge_robust_")
        robust.to_csv(out_csv_robust, index=False)
        print(f"\n[RIDGE TUNING] Robust-only leaderboard saved to {out_csv_robust}")
        print("\nTop-5 (robust-only):")
        print(robust.head(5).to_string(index=False))
        best_r = robust.iloc[0].to_dict()
        print("\nBest robust configuration (attack ∈ {fgsm,pgd}):")
        for k, v in best_r.items():
            print(f"  {k}: {v}")
    else:
        print("\n[RIDGE TUNING] No robust rows found (this would happen only if ATTACKS=['none']).")

if __name__ == "__main__":
    main()
