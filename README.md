# Adversarial Time-Series Forecasting (UCI + M4)

Robustness experiments for time-series forecasting under adversarial perturbations.  
Implements **Ridge** and **LSTM** pipelines with **FGSM/PGD** attacks and a **TSAS** (temporal smoothing) defense.  
Supports **UCI Electricity** and **M4** datasets. Includes grid-search tuners that output a single leaderboard CSV (no per-config clutter).

---

## ✨ Highlights
- **Datasets:** UCI Electricity (15-min), M4 (Hourly/Daily/…)
- **Models:** Ridge (lagged features), LSTM (PyTorch)
- **Attacks:** FGSM & PGD (feature-space for Ridge, tensor-space for LSTM)
- **Defense:** TSAS smoothing (`moving_avg`, optional `wavelet` if `pywavelets` installed)
- **Tuning:** One-command grid search → **single leaderboard CSV**
- **Clean outputs:** Metrics CSVs + optional plots per run (plots off for big grids)

---

## 🗂️ Repo Structure
```
src/
  attacks.py
  data_prep.py            # UCI + M4 loaders
  defenses.py
  pipeline_ridge.py
  pipeline_lstm.py
  tune_ridge.py
  tune_lstm.py
  uci_data_prep.py        # (kept for back-compat; data_prep.py is used now)
  utils.py
datasets/
  UCI/LD2011_2014.txt     # (ignored by git)
  M4/Hourly-train.csv     # (ignored by git)
results/                  # metrics, plots, leaderboards (ignored by git)
```
> `datasets/` and `results/` are git-ignored. Keep an empty `results/.gitkeep` if you want the folder tracked.

---

## 🔧 Setup
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**`requirements.txt` (minimum):**
```
pandas
numpy
scikit-learn
matplotlib
torch
pywavelets    # optional, only for wavelet TSAS
```

---

## 📥 Datasets

### UCI Electricity
Place the file at:
```
datasets/UCI/LD2011_2014.txt
```
The loader expects `sep=";"`, `decimal=","`, first column is datetime (15-min sampling).

### M4 (e.g., Hourly)
Put train CSVs like:
```
datasets/M4/Hourly-train.csv
```
The loader reads “wide” format (`id, v1, v2, ...`) and right-aligns variable-length series internally.

---

## ▶️ Quickstart

### Ridge on UCI
```bash
python -m src.pipeline_ridge   --dataset uci   --csv datasets/UCI/LD2011_2014.txt   --freq 15min   --alpha 1.0   --attack fgsm --epsilon 0.05   --defense tsas --smoother moving_avg --window 5   --test_size 0.2 --max_clients 5   --output_metrics_path results/uci_ridge_metrics.csv   --output_plot_dir results/uci_plots
```

### Ridge on M4 (Hourly)
```bash
python -m src.pipeline_ridge   --dataset m4   --csv datasets/M4/Hourly-train.csv   --alpha 0.1   --attack none --defense tsas --window 3   --test_size 0.2 --max_clients 50   --output_metrics_path results/m4_hourly_ridge_metrics.csv   --output_plot_dir results/m4_hourly_plots
```

### LSTM on UCI
```bash
python -m src.pipeline_lstm   --dataset uci   --csv datasets/UCI/LD2011_2014.txt   --freq 15min   --hidden_size 64 --num_layers 2   --attack none --defense none   --epochs 5 --max_clients 5   --output_metrics_path results/uci_lstm_metrics.csv   --output_plot_dir results/uci_lstm_plots
```

### LSTM on M4 (Hourly)
```bash
python -m src.pipeline_lstm   --dataset m4   --csv datasets/M4/Hourly-train.csv   --hidden_size 64 --num_layers 2   --attack fgsm --epsilon 0.05   --defense tsas --window 5   --epochs 5 --max_clients 20   --output_metrics_path results/m4_hourly_lstm_metrics.csv   --output_plot_dir results/m4_hourly_lstm_plots
```
> `--max_clients -1` runs **all** series (can be heavy).

---

## 🛡️ Attacks & Defense
- **Ridge attacks:** feature-space FGSM/PGD → `attacks.fgsm_attack_features`, `attacks.pgd_attack_features`
- **LSTM attacks:** tensor-space FGSM/PGD → `attacks.fgsm_attack_torch`, `attacks.pgd_attack_torch`
- **Defense (TSAS):** `defenses.tsas_smooth_feature_matrix` with `moving_avg` (default) or `wavelet` (requires `pywavelets`)

---

## 🔎 Tuning (Leaderboards Only)

Both tuners run a grid and **do not** save per-config plots/CSVs. They aggregate each run’s metrics in-memory and save a **single leaderboard CSV**.

### Ridge
```bash
python -m src.tune_ridge
```
- Output: `results/tuning/leaderboard_ridge_*.csv`  
- Also writes a **robust-only** leaderboard (filters `attack != none`).

**Sample top results (20 clients):**
```
Best (by mean_RMSE_DEF):
  dataset=m4, alpha=0.1, attack=none, defense=tsas, window=3
  mean_RMSE_CLEAN≈199.52, mean_RMSE_DEF≈164.09, defense_recovery≈35.43

Best robust (attack ∈ {fgsm, pgd}):
  dataset=m4, alpha=0.1, attack=fgsm, epsilon=0.01, defense=tsas, window=3
  mean_RMSE_DEF≈164.10, robust_drop≈0.021, defense_recovery≈35.44
`````

### LSTM
```bash
python -m src.tune_lstm
```
- Output: `results/tuning_lstm/leaderboard_lstm_*.csv`  
- Also writes a **robust-only** leaderboard.

> Tune lists (hidden size, layers, epochs, epsilon, etc.) are defined at the top of each tuner. Increase `max_clients` for more stable averages.

---

## 📈 Reproducing Final Tables

1) **Baseline best** (e.g., Ridge M4 Hourly):
```bash
python -m src.pipeline_ridge   --dataset m4 --csv datasets/M4/Hourly-train.csv   --alpha 0.1 --attack none --defense tsas --window 3   --test_size 0.2 --max_clients -1   --output_metrics_path results/final_baseline_m4_hourly_ridge.csv   --output_plot_dir results/final_baseline_plots
```

2) **Best robust** (from robust leaderboard top row):
```bash
python -m src.pipeline_ridge   --dataset m4 --csv datasets/M4/Hourly-train.csv   --alpha 0.1 --attack fgsm --epsilon 0.01   --defense tsas --window 3   --test_size 0.2 --max_clients -1   --output_metrics_path results/final_robust_m4_hourly_ridge.csv   --output_plot_dir results/final_robust_plots
```

> Suppress plotting in programmatic runs via `run_pipeline(..., save_plots=False)` (tuners already do this).

---

## 🧪 Implementation Notes
- **Feature engineering:** `utils.make_lag_features` builds `lag_*` and optional rolling stats.
- **Temporal split:** `utils.train_test_split_time` is order-preserving.
- **Metrics:** `utils.rmse`, `utils.mape` (+ MAE where used).
- **M4 loading:** `data_prep.load_m4_wide` right-aligns variable-length series into a wide DF; feature generation safely drops NaNs.
- **Reproducibility:** Use fixed seeds where needed; small fluctuations are expected with stochastic training.

---

## 📦 .gitignore (recommended)
```
.venv/
__pycache__/
*.pyc
*.pyo
*.pyd
*.DS_Store
.vscode/
.idea/
datasets/
results/
!results/.gitkeep
*.ipynb_checkpoints
```

---

## 🗺️ Roadmap
- Add **TFT** or other deep forecasters (N-BEATS, PatchTST)
- Add **cross-dataset evaluations** (UCI + multiple M4 frequencies)
- Add **composite robust score** (e.g., `mean_RMSE_DEF + λ·robust_drop`)
- Finalize a **paper-style report** + curated plots for the README

---

## 📝 Citation
If you use this repo, consider citing the M4 competition:
> Spyros Makridakis, Evangelos Spiliotis, and Vassilios Assimakopoulos. *The M4 Competition: 100,000 time series and 61 forecasting methods*. IJF 2018.
