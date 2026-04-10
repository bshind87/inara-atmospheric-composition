# INARA — Exoplanet Atmospheric Retrieval

**CS 6140 · Machine Learning · Northeastern University**

A machine learning pipeline that predicts molecular surface abundances in exoplanet atmospheres from CLIMA atmospheric profiles. Two models are compared: a **capped Random Forest baseline** and a **simple 1D CNN deep model**.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset](#2-dataset)
3. [Project Structure](#3-project-structure)
4. [Environment Setup](#4-environment-setup)
5. [Pipeline Stages](#5-pipeline-stages)
6. [Running Locally (Mac M5)](#6-running-locally-mac-m5)
7. [Running on HPC (Northeastern Explorer)](#7-running-on-hpc-northeastern-explorer)
8. [Running Individual Steps](#8-running-individual-steps)
9. [Configuration Reference](#9-configuration-reference)
10. [Dashboard — Viewing Results](#10-dashboard--viewing-results)
11. [EDA Notebook](#11-eda-notebook)
12. [Results & Output Files](#12-results--output-files)

---

## 1. Project Overview

**Task:** Given a simulated exoplanet's atmospheric CLIMA profile (12 variables × 101 altitude levels), predict the log₁₀ surface volume mixing ratios of 12 molecular species.

**Models:**

| Model | Description | Training samples |
|---|---|---|
| Random Forest (baseline) | Per-molecule RF on PCA-reduced features | ≤ 10,000 (always capped) |
| 1D CNN (deep) | Simple 1D CNN + per-molecule output heads | Full training split |

**Molecules predicted:** H₂O · CO₂ · O₂ · O₃ · CH₄ · N₂ · N₂O · CO · H₂ · H₂S · SO₂ · NH₃

---

## 2. Dataset

**Source:** INARA ATMOS — synthetic exoplanet atmospheric simulations  
**Full size:** 3,112,620 samples · this run uses 25,000 samples

| Array | Shape | Description |
|---|---|---|
| `spectra.npy` | (N, 12, 101) | CLIMA atmospheric profiles — input features |
| `molecules.npy` | (N, 12) | log₁₀ surface molecular mixing ratios — targets |
| `aux_params.npy` | (N, 11) | Input fluxes & conditions metadata |
| `wavelengths.npy` | (101,) | Altitude axis (km, 0 = surface) |

Raw data: 10 tar.gz archives (`dir_0.tar.gz` … `dir_9.tar.gz` + `Dir_alpha.tar.gz`) + `pyatmos_summary.csv`.

---

## 3. Project Structure

```
inara_vscode/
│
├── src/                           # Core ML modules (do not modify)
│   ├── data_utils.py              # Loading, splitting, normalisation, PCA, metrics
│   ├── baseline_model.py          # Per-molecule Random Forest model
│   └── deep_model.py              # CNN1D + Trainer + SpectralDataset
│
├── pipeline/
│   ├── config.yaml                # All configuration (paths, hyperparameters, toggles)
│   └── steps/                     # Numbered pipeline steps — run in order
│       ├── config_loader.py       # Shared config reader (used by all steps)
│       ├── 01_extract.py          # Step 1: raw archives → processed numpy arrays
│       ├── 02_feature_engineer.py # Step 2: split + normalise + PCA → artifacts
│       ├── 03_train_baseline.py   # Step 3: Random Forest (≤10k samples)
│       ├── 04_train_deep.py       # Step 4: 1D CNN training (with target normalisation)
│       └── 05_evaluate.py         # Step 5: unified test eval + comparison report
│
├── jobs/
│   ├── slurm/                     # Northeastern Explorer HPC job scripts
│   │   ├── 01_extract.sh
│   │   ├── 02_feature_engineer.sh
│   │   ├── 03_train_baseline.sh
│   │   ├── 04_train_deep.sh
│   │   ├── 05_evaluate.sh
│   │   └── submit_pipeline.sh     # Submit all jobs with dependency chain
│   └── local/
│       └── run_pipeline.sh        # Mac M5 — runs all steps sequentially
│
├── notebooks/
│   └── eda.ipynb                  # Exploratory Data Analysis (standalone)
│
├── inara_data/
│   ├── processed/                 # Output of Step 1 (spectra.npy etc.)
│   └── engineered/                # Output of Step 2 (normalised arrays + PCA)
│
├── models/                        # Saved model weights
│   ├── baseline_rf.joblib
│   └── cnn1d.pt
│
├── results/                       # Metrics, predictions, training history
│
├── dashboard.py                   # Streamlit interactive results dashboard
├── process_inara.py               # Standalone data extraction script
├── run_baseline.py                # Standalone baseline training script
└── run_deep_model.py              # Standalone deep model training script
```

---

## 4. Environment Setup

### Local (Mac)

```bash
# Create and activate conda environment
conda create -n inara_env python=3.11 -y
conda activate inara_env

# Install dependencies
pip install numpy pandas scikit-learn torch joblib tqdm
pip install streamlit plotly pyyaml jupyter

# Verify PyTorch sees MPS (Apple Silicon)
python -c "import torch; print(torch.backends.mps.is_available())"
# Expected: True
```

### HPC (Northeastern Explorer)

```bash
# SSH into Explorer
ssh shinde.b@login.explorer.northeastern.edu

# Load base modules (check available with: module avail)
module load anaconda3

# Create environment (run once, from login node)
conda create -n inara_env python=3.11 -y
conda activate inara_env

# Install packages — include CUDA-enabled PyTorch
#pip install numpy pandas scikit-learn joblib tqdm pyyaml
#pip install torch --index-url https://download.pytorch.org/whl/cu118
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install scikit-learn numpy pandas matplotlib

# Verify GPU packages
#python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

> **Important:** Run `conda activate inara_env` before any pipeline commands.  
> Update `slurm.conda_env` in `pipeline/config.yaml` if you use a different environment name.

### Upload code and data to HPC

```bash
# Upload code to /home (never purged)
scp -r /path/to/inara_vscode  shinde.b@login.explorer.neu.edu:/home/shinde.b/

# Upload raw data to /scratch (fast Lustre filesystem)
scp /path/to/full_inara/*.tar.gz  shinde.b@login.explorer.neu.edu:/scratch/shinde.b/inara/raw/
scp /path/to/full_inara/pyatmos_summary.csv  shinde.b@login.explorer.neu.edu:/scratch/shinde.b/inara/raw/

# Or rsync (resumes interrupted transfers)
rsync -avzP /path/to/full_inara/  shinde.b@login.explorer.neu.edu:/scratch/shinde.b/inara/raw/
```

---

## 5. Pipeline Stages

```
[Step 1] Extract          Raw tar.gz archives → processed numpy arrays
              ↓
[Step 2] Feature Engineer  Train/val/test split → Z-normalise spectra → PCA
              ↓                        ↓
[Step 3] Train Baseline   ──────────── │ ──── Random Forest (≤10k samples)
[Step 4] Train Deep       ─────────────┘ ──── 1D CNN (full training set)
              ↓                ↓
[Step 5] Evaluate         Unified test metrics + baseline vs deep comparison
```

Steps 3 and 4 both depend on Step 2 but are independent of each other — on HPC they run **in parallel**.

### Step toggles

Any step can be disabled in `pipeline/config.yaml`:

```yaml
steps:
  extract:          true   # set false if processed/ already exists
  feature_engineer: true   # set false if engineered/ already exists
  train_baseline:   true
  train_deep:       true
  evaluate:         true
```

---

## 6. Running Locally (Mac M5)

### Full pipeline

```bash
cd /path/to/inara_vscode
bash jobs/local/run_pipeline.sh
```

### Skip extraction (data already processed)

```bash
bash jobs/local/run_pipeline.sh --skip-extract
```

### Skip extraction + feature engineering (engineered/ exists)

```bash
bash jobs/local/run_pipeline.sh --skip-extract --skip-feature-eng
```

**Expected runtimes (Mac M5 Pro, 25k dataset):**

| Step | Time |
|---|---|
| Step 1 — Extraction | ~10–15 min |
| Step 2 — Feature Engineering | ~3–5 min |
| Step 3 — Baseline RF (10k) | < 1 min |
| Step 4 — Deep Model (MPS) | ~10–20 min |
| Step 5 — Evaluation | < 1 min |

---

## 7. Running on HPC (Northeastern Explorer)

### First-time setup checklist

1. Code uploaded to `/home/shinde.b/inara_vscode/`
2. Raw data uploaded to `/scratch/shinde.b/inara/raw/`
3. `inara_env` conda environment created and tested
4. `pipeline/config.yaml` updated:
   - `slurm.conda_env` → your environment name
   - `slurm.email` → your email for job notifications
   - Check partition names match what your account can access (`sinfo -s`)

### Submit the full pipeline

```bash
cd /home/shinde.b/inara_vscode
bash jobs/slurm/submit_pipeline.sh
```

This automatically submits all 5 jobs with the correct dependency chain.

### Submit with flags

```bash
# Skip extraction (processed/ already on scratch)
bash jobs/slurm/submit_pipeline.sh --skip-extract

# Skip extraction + feature engineering
bash jobs/slurm/submit_pipeline.sh --skip-extract --skip-feature-eng
```

### Monitor jobs

```bash
# View your queue
squeue -u shinde.b

# Watch live (refreshes every 5s)
watch -n 5 squeue -u shinde.b

# View job output in real-time
tail -f /scratch/shinde.b/inara/logs/04_train_deep_<JOBID>.out

# Cancel a specific job
scancel <JOBID>

# Cancel all your jobs
scancel -u shinde.b
```

### Copy results home after completion

Step 5 automatically copies CSV result files to `/home/shinde.b/inara_results/` at the end.  
To copy manually:

```bash
cp /scratch/shinde.b/inara/results/*.csv  /home/shinde.b/inara_results/
cp /scratch/shinde.b/inara/models/*.pt    /home/shinde.b/inara_results/
cp /scratch/shinde.b/inara/models/*.joblib /home/shinde.b/inara_results/
```

> **Warning:** `/scratch` is purged each semester. Always copy important results to `/home` or your course directory before the semester ends.

### SLURM resource allocation per step

| Step | Partition | CPUs | Memory | GPU | Walltime |
|---|---|---|---|---|---|
| 1 — Extract | `short` | 8 | 32 GB | — | 4h |
| 2 — Feature Engineering | `short` | 4 | 16 GB | — | 1h |
| 3 — Baseline | `short` | 8 | 16 GB | — | 1h |
| 4 — Deep Model | `gpu` | 4 | 32 GB | 1 × GPU | 8h |
| 5 — Evaluate | `short` | 2 | 8 GB | — | 30 min |

> Partition names (`short`, `gpu`) are common defaults — verify with `sinfo -s` on Explorer and update in the SLURM scripts if different.

---

## 8. Running Individual Steps

Each step is a standalone Python script. Run any step directly:

```bash
# Step 1 — Extract data
python pipeline/steps/01_extract.py --profile local

# Step 1 — Resume a partial extraction
python pipeline/steps/01_extract.py --profile local --resume

# Step 1 — Override n_samples from config
python pipeline/steps/01_extract.py --profile local --n-samples 10000

# Step 2 — Feature engineering
python pipeline/steps/02_feature_engineer.py --profile local

# Step 3 — Train baseline (saves model)
python pipeline/steps/03_train_baseline.py --profile local --save

# Step 3 — Override sample cap
python pipeline/steps/03_train_baseline.py --profile local --max-train-samples 5000

# Step 4 — Train deep model (saves checkpoint)
python pipeline/steps/04_train_deep.py --profile local --save

# Step 4 — Resume from checkpoint
python pipeline/steps/04_train_deep.py --profile local --resume models/cnn1d.pt

# Step 4 — Override hyperparameters
python pipeline/steps/04_train_deep.py --profile local --epochs 50 --patience 10

# Step 5 — Evaluate and compare
python pipeline/steps/05_evaluate.py --profile local
```

For HPC, add `--profile hpc` to any command above.

All steps accept `--config` to point to an alternative config file:
```bash
python pipeline/steps/03_train_baseline.py --config pipeline/config.yaml --profile hpc
```

---

## 9. Configuration Reference

`pipeline/config.yaml` controls all pipeline behaviour.

### Step toggles
```yaml
steps:
  extract:          true   # Step 1
  feature_engineer: true   # Step 2
  train_baseline:   true   # Step 3
  train_deep:       true   # Step 4
  evaluate:         true   # Step 5
```

### Data & extraction
```yaml
data:
  val_frac:  0.15    # 15% validation
  test_frac: 0.15    # 15% test
  seed:      42      # reproducibility seed for splits

extraction:
  n_samples: 124000  # samples to extract (reduce for dev/test)
  n_workers: 4       # parallel archive workers
  seed:      42
```

### Model & training
```yaml
model:
  in_channels:    12    # 12 CLIMA channels
  pca_components: 300   # PCA features for RF baseline

baseline:
  max_train_samples: 10000   # RF cap — always enforced

training:
  epochs:       150
  batch_size:   32
  lr:           0.001
  weight_decay: 0.0001
  patience:     30
```

### Profiles
```yaml
profiles:
  local:
    raw_dir:        /Users/bhalchandra/Downloads/full_inara
    processed_dir:  inara_data/processed       # relative paths OK for local
    engineered_dir: inara_data/engineered
    results_dir:    results/processed
    models_dir:     models
    device:         auto                        # auto → MPS / CPU
    n_workers:      4

  hpc:
    raw_dir:        /scratch/shinde.b/inara/raw
    processed_dir:  /scratch/shinde.b/inara/processed
    engineered_dir: /scratch/shinde.b/inara/engineered
    results_dir:    /scratch/shinde.b/inara/results/processed
    models_dir:     /scratch/shinde.b/inara/models
    device:         auto                        # auto → CUDA / CPU
    n_workers:      8
```

### SLURM
```yaml
slurm:
  account:   ""                          # leave blank if not required
  email:     shinde.b@northeastern.edu
  conda_env: inara_env
  code_dir:  /home/shinde.b/inara_vscode
```

---

## 10. Dashboard — Viewing Results

The dashboard is a **Streamlit** app that reads the results files generated by Steps 3–5.  
It auto-detects which results are available and updates when new files are written.

### Start the dashboard

```bash
# From the project root
conda activate inara_env
streamlit run dashboard.py
```

Then open: **http://localhost:8501**

### Dashboard sections

| Section | What it shows |
|---|---|
| **Dataset Overview** | Molecule distribution histograms, CLIMA spectral channels, dataset statistics |
| **Model Metrics** | Val/test R², RMSE, MAE — Baseline vs Deep Model, per molecule and mean |
| **Prediction Detail** | Scatter plots (predicted vs true), residual distributions per molecule |
| **Training History** | Deep model train/val loss curves, early stopping point |

### Updating the dashboard after a training run

The dashboard reads files from `results/processed/` at page load — no restart needed.  
**Just reload the browser tab** after a step completes and new CSV/npy files are written.

Files the dashboard reads (all in `results/processed/`):

```
baseline_val_metrics.csv      ← written by Step 3
baseline_test_metrics.csv     ← written by Step 3 or 5
deep_val_metrics.csv          ← written by Step 4
deep_test_metrics.csv         ← written by Step 4 or 5
model_comparison.csv          ← written by Step 5
deep_training_history.csv     ← written by Step 4
baseline_test_pred.npy        ← written by Step 3
deep_test_pred.npy            ← written by Step 4
test_targets.npy              ← written by Step 3 or 4 (whichever runs first)
```

> The dashboard gracefully degrades — if only the baseline has run, only baseline sections appear.

### Viewing dashboard for HPC results

After copying results from scratch to local:

```bash
# 1. Copy results back from HPC
scp shinde.b@login.explorer.neu.edu:/home/shinde.b/inara_results/*.csv  results/
scp shinde.b@login.explorer.neu.edu:/home/shinde.b/inara_results/*.npy  results/

# 2. Launch dashboard locally
streamlit run dashboard.py
```

### Dashboard tips

- Use **Ctrl+C** in the terminal to stop the dashboard
- If port 8501 is in use: `streamlit run dashboard.py --server.port 8502`
- To share on a network: `streamlit run dashboard.py --server.address 0.0.0.0`

---

## 11. EDA Notebook

The EDA notebook (`notebooks/eda.ipynb`) is **standalone** — it is not part of the pipeline and can be run at any time after Step 1 completes. It loads ~500 sample records for fast exploration.

### Start the notebook

```bash
conda activate inara_env
cd /path/to/inara_vscode
jupyter notebook notebooks/eda.ipynb
# or with JupyterLab:
jupyter lab notebooks/eda.ipynb
```

### Notebook contents

| Cell | What it explores |
|---|---|
| Load dataset | Shapes, dtype, sample counts |
| CLIMA profile structure | Channel names, value ranges per channel |
| Mean altitude profiles | Per-channel mean ± 1σ across all samples |
| Example individual profiles | 5 random planet profiles overlaid |
| Target distributions | Histograms of all 12 molecules (log₁₀ space) |
| Summary statistics table | Mean, std, % at log-floor, learnable flag |
| Molecule correlation matrix | Pearson correlations between molecule pairs |
| Input–output scatter | CLIMA surface value vs top-4 most variable molecules |
| Spectral variability heatmap | Coefficient of variation per channel × altitude |
| Split preview | Verifies train/val/test distributions match |

> No GPU or large memory required — runs on any machine with 4 GB RAM.

---

## 12. Results & Output Files

### Local paths (after full pipeline run)

```
results/processed/
├── baseline_val_metrics.csv     # Per-molecule R², RMSE, MAE on validation set
├── baseline_test_metrics.csv    # Per-molecule R², RMSE, MAE on test set
├── baseline_test_pred.npy       # RF predictions on test set (N_test, 12)
├── deep_val_metrics.csv
├── deep_test_metrics.csv
├── deep_test_pred.npy           # CNN predictions on test set (N_test, 12)
├── deep_training_history.csv    # train_loss, val_loss per epoch
├── model_comparison.csv         # Filtered Baseline_R2, DeepModel_R2, Delta_R2 (H2O/NH3 excluded)
└── test_targets.npy             # Ground truth for test set (N_test, 12)

models/
├── baseline_rf.joblib           # Trained Random Forest (12 per-molecule RFs)
└── cnn1d.pt                     # Trained 1D CNN state dict

inara_data/
├── processed/                   # Step 1 output
│   ├── spectra.npy
│   ├── molecules.npy
│   ├── aux_params.npy
│   ├── system_ids.npy
│   ├── wavelengths.npy
│   └── dataset_info.json
└── engineered/                  # Step 2 output
    ├── train_indices.npy
    ├── val_indices.npy
    ├── test_indices.npy
    ├── spectra_{train,val,test}.npy    # Z-normalised spectra
    ├── molecules_{train,val,test}.npy  # log10 targets (unchanged)
    ├── feat_{train,val,test}.npy       # PCA features for RF
    ├── scaler.joblib                   # Fitted SpectraScaler
    ├── pca.joblib                      # Fitted PCA
    └── feature_info.json               # Split sizes, explained variance, config
```

### Reading results in Python

```python
import numpy as np
import pandas as pd

# Model comparison
compare = pd.read_csv('results/processed/model_comparison.csv')
print(compare)

# Predictions vs ground truth
test_targets = np.load('results/processed/test_targets.npy')        # (N, 12)
bl_pred      = np.load('results/processed/baseline_test_pred.npy')  # (N, 12)
deep_pred    = np.load('results/processed/deep_test_pred.npy')      # (N, 12)

# Training curve
history = pd.read_csv('results/processed/deep_training_history.csv')
history.plot(title='Training History')
```

---

## Notes

- **NH₃** values are all at the log-floor (−40) in this dataset — the model learns to predict the floor but the molecule is not physically recoverable from CLIMA profiles alone.
- **CO, H₂, N₂O** have low R² in both models due to low variance and weak spectral signatures — this is a known dataset characteristic, not a bug.
- The **10k RF cap** is intentional and fixed. Do not increase it — it represents the "baseline" condition for the academic comparison.
- Re-running Step 2 with the same seed always produces identical splits. Steps 3 and 4 will be compared on the same test set.
- The `requirements.txt` lists core ML packages only. Add `streamlit plotly pyyaml jupyter` for dashboard and notebook use.
