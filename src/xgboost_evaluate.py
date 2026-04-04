# evaluate.py

import os
import numpy as np
import joblib
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

MODEL_DIR = "../models/"

MOLECULES = [
    "H2O","CO2","O2","O3","CH4",
    "N2","N2O","CO","H2","H2S","SO2","NH3"
]

# ================================
# LOAD TEST DATA (SQRT SPACE)
# ================================
X = np.load(os.path.join(MODEL_DIR, "X_test.npy"))
y = np.load(os.path.join(MODEL_DIR, "y_test.npy"))

# ================================
# LOAD MODELS
# ================================
strong_models = joblib.load(os.path.join(MODEL_DIR, "strong_models.pkl"))
weak_models = joblib.load(os.path.join(MODEL_DIR, "weak_models.pkl"))

# ================================
# PREDICT (SQRT SPACE)
# ================================
y_pred = np.zeros_like(y)

for i, model in enumerate(strong_models):
    y_pred[:, i] = model.predict(X)

for i, model in enumerate(weak_models):
    y_pred[:, i + 5] = model.predict(X)

y_pred = y_pred
y_true = y

# ================================
# METRICS
# ================================
print("\n=== TEST METRICS ===")
print("R2   :", r2_score(y_true, y_pred))
print("RMSE :", np.sqrt(mean_squared_error(y_true, y_pred)))
print("MAE  :", mean_absolute_error(y_true, y_pred))

print("\n=== PER MOLECULE ===")

for i in range(12):
    r2_i = r2_score(y_true[:, i], y_pred[:, i])
    rmse_i = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
    mae_i = mean_absolute_error(y_true[:, i], y_pred[:, i])

    print(f"{MOLECULES[i]} → R2={r2_i:.3f}, RMSE={rmse_i:.3f}, MAE={mae_i:.3f}")