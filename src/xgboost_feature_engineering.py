# xgboost_feature_engineering.py

import numpy as np
import os
import joblib
from scipy.signal import find_peaks
from sklearn.feature_selection import VarianceThreshold

# ================================
# CONFIG
# ================================
DATA_DIR = "../inara_data/processed/"
OUTPUT_DIR = "../inara_data/features/"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================================
# FEATURE FUNCTIONS
# ================================
def extract_stat_features(s):
    return [
        np.mean(s), np.std(s), np.min(s), np.max(s),
        np.percentile(s, 25), np.percentile(s, 50), np.percentile(s, 75)
    ]


def band_features(s, n_bands=12):
    return [np.mean(b) for b in np.array_split(s, n_bands)]


def gradient_features(s):
    grad = np.gradient(s)
    return [np.mean(grad), np.std(grad)]


def fft_features(s, k=40):
    return np.abs(np.fft.fft(s)[:k])


def peak_features(s):
    peaks, _ = find_peaks(-s, prominence=0.01)
    return [
        len(peaks),
        np.mean(s[peaks]) if len(peaks) > 0 else 0,
        np.std(s[peaks]) if len(peaks) > 0 else 0
    ]


def snr_feature(signal, noise):
    return np.mean(signal) / (np.std(noise) + 1e-6)


def downsample_signal(s, n_points):
    idx = np.linspace(0, len(s) - 1, n_points).astype(int)
    return s[idx]


def extract_spectral_windows(s, n_windows=20, window_size=50):
    features = []
    step = len(s) // n_windows

    for i in range(0, len(s) - window_size, step):
        window = s[i:i + window_size]
        features.append(np.mean(window))
        features.append(np.std(window))

    return features


# ================================
# MAIN FEATURE EXTRACTION
# ================================
def extract_features(spec, aux_features):
    """
    This function is SAFE to import (used in predict.py)
    """
    ch0, ch1, ch2 = spec
    features = []

    # Channel-wise features
    for ch in [ch0, ch1, ch2]:
        features += extract_stat_features(ch)
        features += band_features(ch)
        features += gradient_features(ch)

    # Spectral features
    features += list(fft_features(ch2))
    features += peak_features(ch2)
    features.append(snr_feature(ch2, ch1))

    # Local windows (important for weak molecules)
    features += extract_spectral_windows(ch2)

    # Raw spectral shape (critical)
    features += list(downsample_signal(ch2, 400))
    features += list(downsample_signal(ch1, 200))

    # Auxiliary parameters
    features += list(aux_features)

    return np.array(features, dtype=np.float32)


# ================================
# BUILD FEATURE MATRIX (TRAINING ONLY)
# ================================
def build_feature_matrix():
    print("Loading data...")

    spectra = np.load(os.path.join(DATA_DIR, "spectra.npy"), mmap_mode='r')
    targets = np.load(os.path.join(DATA_DIR, "molecules.npy"), mmap_mode='r')
    aux = np.load(os.path.join(DATA_DIR, "aux_params.npy"), mmap_mode='r')

    print("Loaded data:")
    print("Spectra:", spectra.shape)
    print("Targets:", targets.shape)

    X = []

    for i in range(len(spectra)):
        if i % 5000 == 0:
            print(f"Processing {i}/{len(spectra)}")

        X.append(extract_features(spectra[i], aux[i]))

    X = np.array(X)

    print("Raw feature matrix shape:", X.shape)

    # Clean NaNs/Infs
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Feature selection
    selector = VarianceThreshold()
    X = selector.fit_transform(X)

    print("Final feature matrix shape:", X.shape)

    # Save
    np.save(os.path.join(OUTPUT_DIR, "X.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "targets.npy"), targets)

    joblib.dump(selector, os.path.join(OUTPUT_DIR, "variance_selector.pkl"))

    print("Features + selector saved successfully 🚀")


# ================================
# MAIN ENTRY (CRITICAL FIX)
# ================================
if __name__ == "__main__":
    build_feature_matrix()