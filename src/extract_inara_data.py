# extract__inara_data.py
# ─────────────────────────────────────────────────────────────────────────────
# Updated for PSG / INARA dataset using psg_models.csv (PlanetIndex mapping)
# Supports partial dataset processing via --n
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import logging

# ── CONFIG (adjust paths if needed) ───────────────────────────────────────────
DATA_DIR = "../inara_data"
LABELS_FILE = os.path.join(DATA_DIR, "psg_models.csv")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

SPECTRAL_LENGTH = 4378
INPUT_CHANNELS = 3


MOLECULES = [
    'H2O','CO2','O2','O3','CH4','N2',
    'N2O','CO','SO2','NH3','C2H6','NO2'
]

LOG_DEPTH_MIN = -6
LOG_DEPTH_MAX = 6

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# LABEL LOADER (PSG format)
# ─────────────────────────────────────────────────────────────────────────────

def load_labels(labels_path):
    df = pd.read_csv(labels_path)
    df.columns = [c.lower() for c in df.columns]

    df = df.rename(columns={
        'planetindex': 'system_id',
        'h2o': 'H2O',
        'co2': 'CO2',
        'o2': 'O2',
        'o3': 'O3',
        'ch4': 'CH4',
        'n2': 'N2',
        'n2o': 'N2O',
        'co': 'CO',
        'h2': 'H2',
        'h2s': 'H2S',
        'so2': 'SO2',
        'nh3': 'NH3',
        'c2h6': 'C2H6',
        'no2': 'NO2',
    })

    df['system_id'] = df['system_id'].astype(int)
    return df.set_index('system_id')

# ─────────────────────────────────────────────────────────────────────────────
# PARSE CSV
# ─────────────────────────────────────────────────────────────────────────────

def parse_csv(filepath):
    df = pd.read_csv(filepath)

    # Column assumptions (PSG format)
    wl = df.iloc[:, 0].values
    sp = df.iloc[:, 1].values
    noise = df.iloc[:, 2].values
    star = df.iloc[:, 3].values
    planet = df.iloc[:, 4].values

    usable = noise > 0
    wl = wl[usable]
    sp = sp[usable]
    noise = noise[usable]
    star = star[usable]
    planet = planet[usable]

    snr_raw = (sp / (noise + 1e-10)).astype(np.float32)
    depth = np.clip(planet / (star + 1e-10) * 1e6, 1e-3, 1e6)
    depth_log = np.log10(depth).astype(np.float32)

    sid = int(os.path.splitext(os.path.basename(filepath))[0])

    return {
        "system_id": sid,
        "snr_raw": snr_raw,
        "depth_log": depth_log,
        "wavelengths": wl
    }

# ─────────────────────────────────────────────────────────────────────────────
# BUILD TENSOR
# ─────────────────────────────────────────────────────────────────────────────

def build_tensor(snr_raw, depth_log):
    lo, hi = snr_raw.min(), snr_raw.max()

    ch0 = (snr_raw - lo) / (hi - lo + 1e-10)
    ch1 = ch0 - ch0.mean()
    ch2 = (np.clip(depth_log, LOG_DEPTH_MIN, LOG_DEPTH_MAX)
           - LOG_DEPTH_MIN) / (LOG_DEPTH_MAX - LOG_DEPTH_MIN)

    return np.stack([ch0, ch1, ch2], axis=0).astype(np.float32)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN DATA BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(n_samples=None):

    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    csv_files = [f for f in csv_files if os.path.basename(f)[:-4].isdigit()]

    if not csv_files:
        raise FileNotFoundError("No spectrum CSV files found")

    if n_samples:
        csv_files = csv_files[:n_samples]

    log.info(f"Using {len(csv_files)} CSV files")

    labels = load_labels(LABELS_FILE)
    log.info(f"Loaded labels: {labels.shape}")

    spectra, mols, ids = [], [], []
    failed = 0

    for i, fpath in enumerate(csv_files):

        if i % 500 == 0 and i > 0:
            log.info(f"{i}/{len(csv_files)} processed")

        try:
            parsed = parse_csv(fpath)
            sid = parsed["system_id"]

            if sid not in labels.index:
                failed += 1
                if failed <= 5:
                    log.warning(f"Missing label for {sid}")
                continue

            tensor = build_tensor(parsed["snr_raw"], parsed["depth_log"])

            row = labels.loc[sid]
            #mol_row = np.array([float(row[m]) for m in MOLECULES],
            #                   dtype=np.float32)
            mol_row = np.array([float(row[m]) for m in MOLECULES], dtype=np.float32)

            # convert to log10(VMR)
            mol_row = np.log10(np.clip(mol_row, 1e-12, 1.0))

            spectra.append(tensor)
            mols.append(mol_row)
            ids.append(sid)

        except Exception as e:
            failed += 1
            if failed <= 5:
                log.warning(f"{fpath}: {e}")

    log.info(f"Parsed {len(spectra)} ok | {failed} failed")

    if not spectra:
        raise RuntimeError(
            f"No samples were successfully parsed ({failed} failed). "
            "Check warnings above for the root cause."
        )

    spectra = np.stack(spectra)
    mols = np.stack(mols)
    ids = np.array(ids)

    save_data(spectra, mols, ids)

# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────────

def save_data(spectra, mols, ids):
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    np.save(os.path.join(PROCESSED_DIR, "spectra.npy"), spectra)
    np.save(os.path.join(PROCESSED_DIR, "molecules.npy"), mols)
    np.save(os.path.join(PROCESSED_DIR, "system_ids.npy"), ids)

    log.info("Saved processed data")
    log.info(f"spectra: {spectra.shape}")
    log.info(f"molecules: {mols.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None,
                    help="Number of samples to process")

    args = ap.parse_args()

    build_dataset(n_samples=args.n)
