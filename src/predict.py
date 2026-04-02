# predict.py

import os
import numpy as np
import joblib
import argparse
import pandas as pd
from datetime import datetime

from download_inara import parse_csv, build_tensor
from xgboost_feature_engineering import extract_features
from report_generator import generate_prediction_report

REPORT_DIR = "../reports/"
os.makedirs(REPORT_DIR, exist_ok=True)

MODEL_DIR = "../models/"

MOLECULE_NAMES = [
    "H2O", "CO2", "O2", "O3",
    "CH4", "N2", "N2O", "CO",
    "H2", "H2S", "SO2", "NH3"
]


def load_models():
    return [
        joblib.load(os.path.join(MODEL_DIR, f"xgb_model_mol_{i}.pkl"))
        for i in range(12)
    ]

def load_scaler():
    return joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

def load_selector():
    return joblib.load(os.path.join(MODEL_DIR, "variance_selector.pkl"))

def get_aux():
    return np.zeros(8, dtype=np.float32)


def process_file(path, models, scaler, selector):
    parsed = parse_csv(path)
    tensor = build_tensor(parsed['snr_raw'], parsed['depth_logppm'])

    features = extract_features(tensor, get_aux()).reshape(1, -1)

    # CRITICAL FIX
    features = selector.transform(features)
    features = scaler.transform(features)

    preds = [m.predict(features)[0] for m in models]
    return preds


def main(input_path):
    models = load_models()
    scaler = load_scaler()
    selector = load_selector()

    files = (
        [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith(".csv")]
        if os.path.isdir(input_path)
        else [input_path]
    )

    results = []
    names = []

    for f in files:
        print(f"\nProcessing: {os.path.basename(f)}")

        preds = process_file(f, models, scaler, selector)

        for i, val in enumerate(preds):
            print(f"{MOLECULE_NAMES[i]:>4} : {val:.4f}")

        results.append(preds)
        names.append(os.path.basename(f))

    df = pd.DataFrame(results, columns=MOLECULE_NAMES)
    df.insert(0, "file", names)
    #df.to_csv("prediction_report.csv", index=False)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(REPORT_DIR, f"prediction_{timestamp}.csv")

    # Save CSV
    df.to_csv(csv_path, index=False)
    # Generate HTML report
    generate_prediction_report(df)
    print(f"\nCSV saved → {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    args = parser.parse_args()

    main(args.input)