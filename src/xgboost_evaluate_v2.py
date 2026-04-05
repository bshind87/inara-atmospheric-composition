# xgboost_evaluate_v2.py

import os
import numpy as np
import joblib
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

DATA_DIR = "../inara_data/features/"
MODEL_DIR = "../models/"
N_MOLECULES = 12

MOLECULE_NAMES = [
    "H2O","CO2","O2","O3","CH4","N2",
    "N2O","CO","H2","H2S","SO2","NH3"
]

# ─────────────────────────────────────────────
# LOAD TEST DATA
# ─────────────────────────────────────────────
X = np.load(os.path.join(MODEL_DIR, "X_test.npy"))
y = np.load(os.path.join(MODEL_DIR, "y_test.npy"))

print("Loaded test:", X.shape, y.shape)

# ─────────────────────────────────────────────
# LOAD FEATURE SCALER
# ─────────────────────────────────────────────
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
X = scaler.transform(X)

# ─────────────────────────────────────────────
# LOAD TARGET SCALER (CRITICAL FIX)
# ─────────────────────────────────────────────
y_mean, y_std = joblib.load(os.path.join(MODEL_DIR, "target_scaler.pkl"))

# ─────────────────────────────────────────────
# PREDICTIONS
# ─────────────────────────────────────────────
preds = []

for i in range(N_MOLECULES):
    model_path = os.path.join(MODEL_DIR, f"xgb_model_mol_{i}.pkl")
    model = joblib.load(model_path)

    pred = model.predict(X)

    # 🔥 DENORMALIZE (CRITICAL FIX)
    pred = pred * y_std[i] + y_mean[i]

    preds.append(pred)

y_pred = np.column_stack(preds)

# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────

# Per-molecule R² (correct way)
r2_per_mol = [
    r2_score(y[:, i], y_pred[:, i])
    for i in range(N_MOLECULES)
]

global_r2 = np.mean(r2_per_mol)

rmse = np.sqrt(mean_squared_error(y, y_pred))
mae  = mean_absolute_error(y, y_pred)

print("\n=== TEST METRICS (OVERALL) ===")
print("R2   :", global_r2)
print("RMSE :", rmse)
print("MAE  :", mae)

# ─────────────────────────────────────────────
# PER-MOLECULE METRICS
# ─────────────────────────────────────────────
print("\n=== PER MOLECULE METRICS ===")

for i in range(N_MOLECULES):
    r2_i = r2_per_mol[i]
    rmse_i = np.sqrt(mean_squared_error(y[:, i], y_pred[:, i]))
    mae_i = mean_absolute_error(y[:, i], y_pred[:, i])

    print(f"{MOLECULE_NAMES[i]} → R2={r2_i:.3f}, RMSE={rmse_i:.3f}, MAE={mae_i:.3f}")

print("\nEvaluation complete 🚀")