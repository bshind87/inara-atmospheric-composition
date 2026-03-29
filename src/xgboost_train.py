# xgboost_train.py

import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import joblib
from xgboost.callback import EarlyStopping

# CONFIG
DATA_DIR = "../inara_data/features/"
MODEL_DIR = "../models/"
os.makedirs(MODEL_DIR, exist_ok=True)

N_MOLECULES = 12

# LOAD
X = np.load(os.path.join(DATA_DIR, "X.npy"))
y = np.load(os.path.join(DATA_DIR, "targets.npy"))

print("Loaded:", X.shape, y.shape)

# SPLIT
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# SCALE
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

# TRAIN
models = []
val_preds = []

print("\nTraining per-molecule models...\n")

for i in range(N_MOLECULES):
    print(f"Training molecule {i}...")

    model = xgb.XGBRegressor(
        n_estimators=1200,
        max_depth=10,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        n_jobs=-1,
        random_state=42
    )

    model.fit(
        X_train,
        y_train[:, i]
    )

    joblib.dump(model, os.path.join(MODEL_DIR, f"xgb_model_mol_{i}.pkl"))

    pred = model.predict(X_val)
    val_preds.append(pred)
    models.append(model)

# STACK
y_pred = np.column_stack(val_preds)

# METRICS
print("\n=== VALIDATION RESULTS ===")
print("R2   :", r2_score(y_val, y_pred))
print("RMSE :", np.sqrt(mean_squared_error(y_val, y_pred)))
print("MAE  :", mean_absolute_error(y_val, y_pred))

print("\nPer Molecule:")
for i in range(N_MOLECULES):
    print(
        f"Molecule {i:02d} → "
        f"R2={r2_score(y_val[:, i], y_pred[:, i]):.3f}, "
        f"RMSE={np.sqrt(mean_squared_error(y_val[:, i], y_pred[:, i])):.3f}"
    )

print("\nModels saved 🚀")