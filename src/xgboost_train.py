# xgboost_train.py

import os
import numpy as np
import joblib
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ================================
# CONFIG
# ================================
DATA_DIR = "../inara_data/features/"
MODEL_DIR = "../models/"
os.makedirs(MODEL_DIR, exist_ok=True)

STRONG_IDX = list(range(5))
WEAK_IDX = list(range(5, 12))

MOLECULES = [
    "H2O","CO2","O2","O3","CH4",
    "N2","N2O","CO","H2","H2S","SO2","NH3"
]


# ================================
# LOAD DATA
# ================================
X = np.load(os.path.join(DATA_DIR, "X.npy"))
y = np.load(os.path.join(DATA_DIR, "targets.npy"))

print("Loaded:", X.shape)

# ================================
# SAFE SQRT TRANSFORM
# ================================
#y = np.sqrt(np.clip(y, a_min=0, a_max=None))
# small stabilization only
y = y + 1e-8
# ================================
# SPLIT
# ================================
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

print("X shape:", X.shape)
print("y shape:", y.shape)

# ================================
# SCALE FEATURES
# ================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

# ================================
# TRAIN STRONG MODELS
# ================================
print("\n=== Training STRONG molecules ===")

strong_models = []
strong_preds = []

for i in STRONG_IDX:
    model = xgb.XGBRegressor(
        max_depth=8,
        learning_rate=0.05,
        n_estimators=800,
        subsample=0.8,
        colsample_bytree=0.7,
        tree_method="hist",
        n_jobs=-1,
        random_state=42
    )

    model.fit(X_train, y_train[:, i])
    strong_models.append(model)
    strong_preds.append(model.predict(X_val))

# ================================
# TRAIN WEAK MODELS
# ================================
print("\n=== Training WEAK molecules ===")

weak_models = []
weak_preds = []

for i in WEAK_IDX:
    weights = 1 + np.abs(y_train[:, i]) * 3

    model = xgb.XGBRegressor(
        max_depth=5,
        learning_rate=0.03,
        n_estimators=1000,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=1.0,
        reg_lambda=2.0,
        tree_method="hist",
        n_jobs=-1,
        random_state=42
    )

    model.fit(X_train, y_train[:, i], sample_weight=weights)
    weak_models.append(model)
    weak_preds.append(model.predict(X_val))


# ================================
# COMBINE PREDICTIONS
# ================================
y_pred = np.zeros_like(y_val)

for idx, p in zip(STRONG_IDX, strong_preds):
    y_pred[:, idx] = p

for idx, p in zip(WEAK_IDX, weak_preds):
    y_pred[:, idx] = p

y_val_real = y_val  # no transform

# ================================
# METRICS
# ================================
print("\n=== VALIDATION METRICS ===")
print("R2   :", r2_score(y_val_real, y_pred))
print("RMSE :", np.sqrt(mean_squared_error(y_val_real, y_pred)))
print("MAE  :", mean_absolute_error(y_val_real, y_pred))

print("\n=== PER MOLECULE ===")

for i in range(12):
    r2_i = r2_score(y_val_real[:, i], y_pred[:, i])
    rmse_i = np.sqrt(mean_squared_error(y_val_real[:, i], y_pred[:, i]))
    mae_i = mean_absolute_error(y_val_real[:, i], y_pred[:, i])

    print(f"{MOLECULES[i]} → R2={r2_i:.3f}, RMSE={rmse_i:.3f}, MAE={mae_i:.3f}")

# ================================
# SAVE
# ================================
joblib.dump(strong_models, os.path.join(MODEL_DIR, "strong_models.pkl"))
joblib.dump(weak_models, os.path.join(MODEL_DIR, "weak_models.pkl"))

np.save(os.path.join(MODEL_DIR, "X_test.npy"), X_test)
np.save(os.path.join(MODEL_DIR, "y_test.npy"), y_test)

print("\nTraining complete 🚀")