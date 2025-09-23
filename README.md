# Adversarial Time‑Series Forecasting (UCI + M4)

Robustness experiments for time‑series forecasting under adversarial perturbations.

Implements:
- **Ridge** (lag features)
- **LSTM** (PyTorch)
- **DeepAR** (PyTorch Forecasting)
- **TFT** - Temporal Fusion Transformer (PyTorch Forecasting)

Attacks: **FGSM/PGD** (feature‑space for Ridge, tensor‑space for LSTM; DeepAR/TFT baseline now, adversarial hooks next).  
Defense: **TSAS** temporal smoothing (`moving_avg`, optional `wavelet`).  
Datasets: **UCI Electricity** (15‑min), **M4** (Hourly, etc.).  
Includes grid‑search tuners with **single leaderboard CSV** outputs (no per‑config clutter).

---

## 🗂️ Repo Structure
```
src/
  attacks.py
  data_prep.py            # UCI + M4 loaders
  defenses.py
  pipeline_ridge.py
  pipeline_lstm.py
  pipeline_deepar.py
  pipeline_tft.py
  tune_ridge.py
  tune_lstm.py
  pf_data.py              # bridge: wide → long for PF models (DeepAR/TFT)
  uci_data_prep.py        # (legacy; kept for back‑compat)
  utils.py
datasets/
  UCI/LD2011_2014.txt     # (ignored by git)
  M4/Hourly-train.csv     # (ignored by git)
results/                  # metrics, plots, leaderboards (ignored by git)
```
> `datasets/` and `results/` are git‑ignored. Keep an empty `results/.gitkeep` if you want the folder tracked.

---

## 🔧 Setup (with `requirements.txt`)

Create a virtual environment and install all dependencies via the pinned **`requirements.txt`** in this repo:

```bash
# 1) create + activate venv
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2) install all deps
pip install -r requirements.txt

# 3) (optional) verify key versions
python - <<'PY'
import lightning as L, pytorch_forecasting as pf, torchmetrics as tm
print("lightning:", L.__version__, "| PF:", pf.__version__, "| TM:", tm.__version__)
PY
```

> If you are on macOS with Apple Silicon and hit MPS attention errors with TFT, run that pipeline with `--device cpu` or a smaller `--batch_size` (see commands below).

---

## 📥 Datasets
**UCI Electricity** → `datasets/UCI/LD2011_2014.txt`  
- CSV with `sep=";"`, `decimal=","`, first column datetime. We resample to `--freq` (default `15min`).

**M4** (e.g., Hourly) → `datasets/M4/Hourly-train.csv`  
- Wide format (`id, v1, v2, ...`). Loader right‑aligns series internally; `pf_data.py` converts to long for PF models.

---

## ▶️ Quickstart

### Ridge (UCI)
```bash
python -m src.pipeline_ridge   --dataset uci   --csv datasets/UCI/LD2011_2014.txt   --freq 15min   --alpha 1.0   --attack fgsm --epsilon 0.05   --defense tsas --smoother moving_avg --window 5   --test_size 0.2 --max_clients 5   --output_metrics_path results/uci_ridge_metrics.csv   --output_plot_dir results/uci_plots
```

### Ridge (M4 Hourly)
```bash
python -m src.pipeline_ridge   --dataset m4   --csv datasets/M4/Hourly-train.csv   --alpha 0.1   --attack none --defense tsas --window 3   --test_size 0.2 --max_clients 50   --output_metrics_path results/m4_hourly_ridge_metrics.csv   --output_plot_dir results/m4_hourly_ridge_plots
```

### LSTM (UCI)
```bash
python -m src.pipeline_lstm   --dataset uci   --csv datasets/UCI/LD2011_2014.txt   --freq 15min   --hidden_size 64 --num_layers 2   --attack none --defense none   --epochs 5 --max_clients 5   --output_metrics_path results/uci_lstm_metrics.csv   --output_plot_dir results/uci_lstm_plots
```

### LSTM (M4 Hourly)
```bash
python -m src.pipeline_lstm   --dataset m4   --csv datasets/M4/Hourly-train.csv   --hidden_size 64 --num_layers 2   --attack fgsm --epsilon 0.05   --defense tsas --window 5   --epochs 5 --max_clients 20   --output_metrics_path results/m4_hourly_lstm_metrics.csv   --output_plot_dir results/m4_hourly_lstm_plots
```

### DeepAR (UCI)
```bash
python -m src.pipeline_deepar   --dataset uci   --csv datasets/UCI/LD2011_2014.txt   --freq 15min   --max_clients 50   --max_encoder_length 48 --max_prediction_length 1   --epochs 10   --output_metrics_path results/uci_deepar_metrics.csv
```

### DeepAR (M4 Hourly)
```bash
python -m src.pipeline_deepar   --dataset m4   --csv datasets/M4/Hourly-train.csv   --max_clients 50   --max_encoder_length 48 --max_prediction_length 1   --epochs 10   --output_metrics_path results/m4_hourly_deepar_metrics.csv
```

### TFT (UCI) — CPU default (MPS can error on attention)
```bash
python -m src.pipeline_tft   --dataset uci   --csv datasets/UCI/LD2011_2014.txt   --freq 15min   --max_clients 10   --max_encoder_length 48 --max_prediction_length 1   --hidden_size 64 --attention_head_size 4 --dropout 0.1   --epochs 3   --device cpu --batch_size 32   --output_metrics_path results/uci_tft_metrics.csv
```

### TFT (M4 Hourly) — try `--device mps --batch_size 16` if you want GPU on Mac
```bash
python -m src.pipeline_tft   --dataset m4   --csv datasets/M4/Hourly-train.csv   --max_clients 20   --max_encoder_length 48 --max_prediction_length 1   --hidden_size 64 --attention_head_size 4 --dropout 0.1   --epochs 3   --device cpu --batch_size 32   --output_metrics_path results/m4_hourly_tft_metrics.csv
```

> PF models (DeepAR/TFT) use **`pf_data.py`** to convert wide → long (`unique_id`, `time_idx`, `y`). Horizon=1 for parity with Ridge/LSTM metrics.

---

## 🔎 Tuning

### Ridge
```bash
python -m src.tune_ridge
```
- Saves: `results/tuning/leaderboard_ridge_*.csv` (and a **robust‑only** leaderboard).  
- Sample best (20 clients, Hourly): `alpha=0.1, defense=tsas(window=3)`.

### LSTM
```bash
python -m src.tune_lstm
```
- Saves: `results/tuning/leaderboard_lstm_*.csv`.  
- Grid includes `hidden_size`, `attack ∈ {none, fgsm, pgd}`, `epsilon`, `defense`, `window`.  
- Defaults are light (epochs/clients) for speed—bump for final sweeps.

> DeepAR/TFT tuners can be added next in the same leaderboard‑only style once adversarial hooks are wired.

---

## 🧪 Notes & Tips
- **Suppress warnings but keep progress bars** (optional): add in PF pipelines:
  ```python
  import warnings, logging
  warnings.filterwarnings("ignore", message="X does not have valid feature names, but StandardScaler was fitted with feature names")
  warnings.filterwarnings("ignore", category=FutureWarning, module="pytorch_forecasting")
  logging.getLogger("pytorch_forecasting").setLevel(logging.ERROR)
  logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
  ```
- **TFT on macOS/MPS:** known `MPSNDArray` buffer errors with attention. Use `--device cpu` or try smaller `--batch_size` on `--device mps`.
- **Reproducibility:** we set `seed_everything(seed, workers=True)` in PF pipelines. Some variability is expected.

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
- Add adversarial evaluation to DeepAR/TFT (FGSM/PGD on encoder inputs).  
- Extend experiments to multiple M4 frequencies.  
- Composite robust score for ranking configs.  
- Final report + curated plots for README.

---

## 📝 References
- Makridakis, Spiliotis, Assimakopoulos. *The M4 Competition: 100,000 time series and 61 forecasting methods*. IJF, 2018.
