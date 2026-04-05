# xgboost_train.py

import os
import numpy as np
import joblib
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

DATA_DIR = "../inara_data/features/"
MODEL_DIR = "../models/"
os.makedirs(MODEL_DIR, exist_ok=True)

X = np.load(os.path.join(DATA_DIR, "X.npy"))
y = np.load(os.path.join(DATA_DIR, "targets.npy"))

print("Loaded:", X.shape)

# Load selector and move to model dir
selector = joblib.load(os.path.join(DATA_DIR, "variance_selector.pkl"))
joblib.dump(selector, os.path.join(MODEL_DIR, "variance_selector.pkl"))

# Split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# SAVE TEST SPLIT
np.save(os.path.join(MODEL_DIR, "X_test.npy"), X_test)
np.save(os.path.join(MODEL_DIR, "y_test.npy"), y_test)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

# Train per molecule
preds = []

for i in range(y.shape[1]):
    print(f"Training molecule {i}")

    model = xgb.XGBRegressor(
        n_estimators=1200,
        max_depth=10,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        n_jobs=-1
    )

    model.fit(X_train, y_train[:, i])

    joblib.dump(model, os.path.join(MODEL_DIR, f"xgb_model_mol_{i}.pkl"))

    preds.append(model.predict(X_val))

#y_pred = np.column_stack(preds)

#print("\nR2:", r2_score(y_val, y_pred))
#print("RMSE:", np.sqrt(mean_squared_error(y_val, y_pred)))

#print("Training complete 🚀")

from sklearn.metrics import mean_absolute_error

y_pred = np.column_stack(preds)

# ================================
# OVERALL METRICS
# ================================
print("\n=== VALIDATION METRICS (OVERALL) ===")
print("R2   :", r2_score(y_val, y_pred))
print("RMSE :", np.sqrt(mean_squared_error(y_val, y_pred)))
print("MAE  :", mean_absolute_error(y_val, y_pred))


# ================================
# PER-MOLECULE METRICS
# ================================
MOLECULE_NAMES = [
        "H2O","CO2","O2","O3","CH4","N2",
        "N2O","CO","H2","H2S","SO2","NH3"
    ]

print("\n=== PER MOLECULE METRICS ===")

for i in range(y.shape[1]):
    r2_i = r2_score(y_val[:, i], y_pred[:, i])
    rmse_i = np.sqrt(mean_squared_error(y_val[:, i], y_pred[:, i]))
    mae_i = mean_absolute_error(y_val[:, i], y_pred[:, i])

    #print(f"Molecule {i:02d} → R2={r2_i:.3f}, RMSE={rmse_i:.3f}, MAE={mae_i:.3f}")
    print(f"{MOLECULE_NAMES[i]} → R2={r2_i:.3f}, RMSE={rmse_i:.3f}, MAE={mae_i:.3f}")

print("\nTraining complete 🚀")