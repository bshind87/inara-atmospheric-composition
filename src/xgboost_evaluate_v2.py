# evaluate.py

import os
import numpy as np
import joblib
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

DATA_DIR = "../inara_data/features/"
MODEL_DIR = "../models/"
N_MOLECULES = 12

# Load data
#X = np.load(os.path.join(DATA_DIR, "X.npy"))
#y = np.load(os.path.join(DATA_DIR, "targets.npy"))

X = np.load(os.path.join(MODEL_DIR, "X_test.npy"))
y = np.load(os.path.join(MODEL_DIR, "y_test.npy"))

# Load scaler
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
X = scaler.transform(X)

# Load models and predict
preds = []

for i in range(N_MOLECULES):
    model_path = os.path.join(MODEL_DIR, f"xgb_model_mol_{i}.pkl")
    model = joblib.load(model_path)

    pred = model.predict(X)
    preds.append(pred)

y_pred = np.column_stack(preds)

# ================================
# METRICS
# ================================
print("\nOverall Metrics (TEST):")
print("R2   :", r2_score(y, y_pred))
print("RMSE :", np.sqrt(mean_squared_error(y, y_pred)))
print("MAE  :", mean_absolute_error(y, y_pred))

print("\nPer Molecule Metrics:")
for i in range(N_MOLECULES):
    r2_i = r2_score(y[:, i], y_pred[:, i])
    rmse_i = np.sqrt(mean_squared_error(y[:, i], y_pred[:, i]))
    mae_i = mean_absolute_error(y[:, i], y_pred[:, i])

    print(f"Molecule {i:02d} → R2={r2_i:.3f}, RMSE={rmse_i:.3f}, MAE={mae_i:.3f}")