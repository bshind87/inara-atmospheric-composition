# xgboost_feature_engineering.py

import numpy as np
import os
import joblib
from scipy.signal import find_peaks
from sklearn.feature_selection import VarianceThreshold

DATA_DIR = "../inara_data/processed/"
OUTPUT_DIR = "../inara_data/features/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# BASIC FEATURES
# ----------------------------
def extract_stat_features(s):
    return [
        np.mean(s), np.std(s), np.min(s), np.max(s),
        np.percentile(s, 25), np.percentile(s, 50), np.percentile(s, 75)
    ]

def gradient_features(s):
    grad = np.gradient(s)
    return [np.mean(grad), np.std(grad)]

def fft_features(s):
    fft = np.abs(np.fft.rfft(s))
    return [np.mean(fft[:50]), np.std(fft[:50]), np.max(fft[:50])]

def peak_features(s):
    prom = 0.01 * np.std(s)
    peaks, _ = find_peaks(-s, prominence=prom)
    return [
        len(peaks),
        np.mean(s[peaks]) if len(peaks) > 0 else 0,
        np.std(s[peaks]) if len(peaks) > 0 else 0
    ]

def band_contrast_features(s):
    bands = np.array_split(s, 20)
    return [np.max(b) - np.min(b) for b in bands]

def normalize_signal(s):
    return (s - np.mean(s)) / (np.std(s) + 1e-6)

# ----------------------------
# FEATURE EXTRACTION
# ----------------------------
def extract_features(spec, aux):
    ch0, ch1, ch2 = spec
    f = []

    # Core stats
    for ch in [ch0, ch1, ch2]:
        f += extract_stat_features(ch)
        f += gradient_features(ch)

    # Spectral features
    f += fft_features(ch2)
    f += peak_features(ch2)
    f += band_contrast_features(ch2)

    # 🔥 HIGH-RES SIGNAL (critical improvement)
    f += list(ch2[::2])   # ~2000 features (main)
    f += list(ch1[::4])   # support
    f += list(ch0[::6])   # NEW (SNR info)

    # 🔥 Absorption edge strength
    grad = np.gradient(ch2)
    f.append(np.mean(np.abs(grad)))
    f.append(np.max(np.abs(grad)))

    # 🔥 Band ratios (powerful)
    bands = np.array_split(ch2, 20)
    for b in bands:
        f.append(np.mean(b) / (np.std(b) + 1e-6))

    # Normalized signal
    ch2n = normalize_signal(ch2)
    f += extract_stat_features(ch2n)

    # Interaction
    f.append(np.mean(ch2 * ch1))

    # AUX (important)
    f += list(aux)

    return np.array(f, dtype=np.float32)

# ----------------------------
# BUILD FEATURE MATRIX
# ----------------------------
def build_feature_matrix():
    print("Loading data...")

    spectra = np.load(os.path.join(DATA_DIR, "spectra.npy"), mmap_mode="r")
    targets = np.load(os.path.join(DATA_DIR, "molecules.npy"), mmap_mode="r")
    aux = np.load(os.path.join(DATA_DIR, "aux_params.npy"), mmap_mode="r")

    X = []
    for i in range(len(spectra)):
        if i % 2000 == 0:
            print(f"Processing {i}/{len(spectra)}")
        X.append(extract_features(spectra[i], aux[i]))

    X = np.array(X)
    X = np.nan_to_num(X)

    print("Raw shape:", X.shape)

    # Variance filtering
    selector = VarianceThreshold()
    X = selector.fit_transform(X)

    print("Final shape:", X.shape)

    np.save(os.path.join(OUTPUT_DIR, "X.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "targets.npy"), targets)
    joblib.dump(selector, os.path.join(OUTPUT_DIR, "selector.pkl"))

    print("Features saved 🚀")

if __name__ == "__main__":
    build_feature_matrix()

