import os
import numpy as np
import joblib
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

DATA_DIR = "../inara_data/features/"
MODEL_DIR = "../models/"
os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
X = np.load(os.path.join(DATA_DIR, "X.npy"))
y = np.load(os.path.join(DATA_DIR, "targets.npy"))

print("Loaded:", X.shape, y.shape)

# ─────────────────────────────────────────────
# FEATURE SELECTOR
# ─────────────────────────────────────────────
selector = joblib.load(os.path.join(DATA_DIR, "variance_selector.pkl"))
joblib.dump(selector, os.path.join(MODEL_DIR, "variance_selector.pkl"))

# ─────────────────────────────────────────────
# SPLIT
# ─────────────────────────────────────────────
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# Save test split
np.save(os.path.join(MODEL_DIR, "X_test.npy"), X_test)
np.save(os.path.join(MODEL_DIR, "y_test.npy"), y_test)

# ─────────────────────────────────────────────
# TARGET NORMALIZATION (TRAIN ONLY)
# ─────────────────────────────────────────────
y_mean = y_train.mean(axis=0)
y_std  = y_train.std(axis=0) + 1e-6

y_train = (y_train - y_mean) / y_std
y_val   = (y_val - y_mean) / y_std
y_test  = (y_test - y_mean) / y_std

joblib.dump((y_mean, y_std), os.path.join(MODEL_DIR, "target_scaler.pkl"))

# ─────────────────────────────────────────────
# FEATURE SCALING
# ─────────────────────────────────────────────
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

# ─────────────────────────────────────────────
# TRAIN PER MOLECULE
# ─────────────────────────────────────────────
preds_val = []
preds_test = []

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

    preds_val.append(model.predict(X_val))
    preds_test.append(model.predict(X_test))

# Stack predictions
y_pred_val = np.column_stack(preds_val)
y_pred_test = np.column_stack(preds_test)

# ─────────────────────────────────────────────
# DENORMALIZE
# ─────────────────────────────────────────────
y_val_real = y_val * y_std + y_mean
y_test_real = y_test * y_std + y_mean

y_pred_val = y_pred_val * y_std + y_mean
y_pred_test = y_pred_test * y_std + y_mean

# ─────────────────────────────────────────────
# METRICS FUNCTION
# ─────────────────────────────────────────────
def evaluate(y_true, y_pred, split="VAL"):
    print(f"\n=== {split} METRICS (OVERALL) ===")

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)

    # Correct global R²
    r2_per_mol = [
        r2_score(y_true[:, i], y_pred[:, i])
        for i in range(y_true.shape[1])
    ]
    global_r2 = np.mean(r2_per_mol)

    print("R2   :", global_r2)
    print("RMSE :", rmse)
    print("MAE  :", mae)

    print("\n=== PER MOLECULE METRICS ===")

    MOLECULE_NAMES = [
        "H2O","CO2","O2","O3","CH4","N2",
        "N2O","CO","H2","H2S","SO2","NH3"
    ]

    for i in range(y_true.shape[1]):
        r2_i = r2_score(y_true[:, i], y_pred[:, i])
        rmse_i = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        mae_i = mean_absolute_error(y_true[:, i], y_pred[:, i])

        print(f"{MOLECULE_NAMES[i]} → R2={r2_i:.3f}, RMSE={rmse_i:.3f}, MAE={mae_i:.3f}")

# ─────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────
evaluate(y_val_real, y_pred_val, "VALIDATION")
evaluate(y_test_real, y_pred_test, "TEST")

print("\nTraining complete 🚀")