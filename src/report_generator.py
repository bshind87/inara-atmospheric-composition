# report_generator.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG
# -----------------------------
REPORT_DIR = "../reports"
FIG_DIR = os.path.join(REPORT_DIR, "figures")

os.makedirs(FIG_DIR, exist_ok=True)

MOLECULES = [
    "H2O","CO2","O2","O3",
    "CH4","N2","N2O","CO",
    "H2","H2S","SO2","NH3"
]

# -----------------------------
# PLOT FUNCTIONS
# -----------------------------
def plot_abundance(y_pred_log, sample_id):
    y_linear = 10 ** y_pred_log
    idx = np.argsort(-y_linear)

    plt.figure(figsize=(8,4))
    plt.bar(np.array(MOLECULES)[idx], y_linear[idx])
    plt.yscale("log")
    plt.xticks(rotation=45)
    plt.title(f"Abundance - {sample_id}")
    plt.tight_layout()

    path = os.path.join(FIG_DIR, f"abundance_{sample_id}.png")
    plt.savefig(path)
    plt.close()

    return path


# -----------------------------
# REPORT BUILDER (PREDICTION MODE)
# -----------------------------
def generate_prediction_report(df):

    html = "<h1> INARA Atmospheric Prediction Report</h1>"

    # -----------------------------
    # Table
    # -----------------------------
    html += "<h2>Predictions Table</h2>"
    html += df.to_html(index=False)

    # -----------------------------
    # Plots per sample
    # -----------------------------
    html += "<h2>Sample Visualizations</h2>"

    for _, row in df.iterrows():
        sample_id = row["file"]
        y_log = row[MOLECULES].values.astype(float)

        fig_path = plot_abundance(y_log, sample_id)

        html += f"<h3>{sample_id}</h3>"
        html += f"<img src='figures/{os.path.basename(fig_path)}' width='500'>"

    # -----------------------------
    # Save
    # -----------------------------
    report_path = os.path.join(REPORT_DIR, "prediction_report.html")

    with open(report_path, "w") as f:
        f.write(html)

    print(f"✅ HTML report generated → {report_path}")