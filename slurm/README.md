
# 🚀 INARA ML Pipeline (SLURM Execution Guide)

This document explains how to run the full **end-to-end ML pipeline** on HPC using SLURM.

---

# 📌 Pipeline Overview

The pipeline consists of 3 main stages:


Feature Engineering → Model Training → Prediction + Report


Each stage is executed as a **separate SLURM job**, chained using dependencies.

---

# 📁 Directory Structure


project/
├── src/
│ ├── xgboost_feature_engineering.py
│ ├── xgboost_train.py
│ ├── predict.py
│ ├── report_generator.py
│
├── slurm/
│ ├── feature_job.sh
│ ├── train_job.sh
│ ├── predict_job.sh
│ ├── run_pipeline.sh
│ ├── README.md
│
├── logs/
├── models/
├── reports/


---

# ⚙️ Prerequisites

## 1. Python Environment

Make sure your environment is available on HPC:

```bash
module load python
source ~/envs/ml/bin/activate

Install dependencies:

pip install numpy pandas scikit-learn xgboost matplotlib joblib scipy
2. Required Data

Ensure processed data exists:

../inara_data/processed/
  ├── spectra.npy
  ├── molecules.npy
  ├── aux_params.npy
🚀 Running the Pipeline
Step 1 — Navigate to SLURM folder
cd slurm
Step 2 — Submit Pipeline
bash run_pipeline.sh
🔁 What Happens Internally
feature_job.sh   → builds features
        ↓
train_job.sh     → trains XGBoost models
        ↓
predict_job.sh   → runs inference + generates report

SLURM automatically manages dependencies.

📊 Monitor Jobs
Check running jobs
squeue -u $USER
Check job history
sacct -j <job_id>
View logs
cat logs/feature_<jobid>.out
cat logs/train_<jobid>.out
cat logs/predict_<jobid>.out
📁 Output Files
Models
models/
  ├── xgb_model_mol_0.pkl
  ├── ...
  ├── scaler.pkl
  ├── variance_selector.pkl
Reports
reports/
  ├── prediction_YYYYMMDD_HHMMSS.csv
  ├── prediction_report.html
  ├── figures/
🧪 Running Individual Steps (Optional)
Feature Engineering
sbatch feature_job.sh
Training
sbatch train_job.sh
Prediction
sbatch predict_job.sh
⚠️ Common Issues & Fixes
❌ Feature mismatch error
X has N features, scaler expects M

✔ Fix:

Ensure variance_selector.pkl is saved and loaded
Ensure same feature pipeline used
❌ Unexpected "Processing 10000 samples" during prediction

✔ Fix:

Ensure xgboost_feature_engineering.py has:
if __name__ == "__main__":
    build_feature_matrix()
❌ Missing reports

✔ Fix:

Check reports/ directory
Ensure predict.py creates directory:
os.makedirs("../reports", exist_ok=True)
🔥 Advanced Usage
Parallel Hyperparameter Tuning

Use SLURM array jobs:

#SBATCH --array=0-5
GPU Support (if available)
#SBATCH --gres=gpu:1
Email Notifications
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your_email@domain.com
🧠 Design Summary
Feature engineering optimized for XGBoost
Per-molecule models for better accuracy
Batch inference supported
HTML + CSV reports generated
Fully scalable via SLURM
🎯 Final Command (TL;DR)
cd slurm
bash run_pipeline.sh