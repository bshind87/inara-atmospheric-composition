# xgboost_feature_eng_v2.py

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
# BASIC FEATURES
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


def second_derivative_features(s):
    d2 = np.gradient(np.gradient(s))
    return [np.mean(d2), np.std(d2)]


# ================================
# SPECTRAL FEATURES
# ================================
def fft_features(s, k=50):
    return np.abs(np.fft.fft(s)[:k])


def peak_features(s):
    peaks, _ = find_peaks(-s, prominence=0.001)
    return [
        len(peaks),
        np.mean(s[peaks]) if len(peaks) > 0 else 0,
        np.std(s[peaks]) if len(peaks) > 0 else 0
    ]


def snr_feature(signal, noise):
    return np.mean(signal) / (np.std(noise) + 1e-6)


# ================================
# LOCAL WINDOWS
# ================================
def extract_spectral_windows(s, n_windows=25, window_size=60):
    features = []
    step = len(s) // n_windows

    for i in range(0, len(s) - window_size, step):
        window = s[i:i + window_size]
        features += [np.mean(window), np.std(window), np.min(window), np.max(window)]

    return features


# ================================
# HIGH-RES
# ================================
def downsample_signal(s, n_points):
    idx = np.linspace(0, len(s) - 1, n_points).astype(int)
    return s[idx]


# ================================
# NORMALIZATION
# ================================
def normalize_signal(s):
    return (s - np.mean(s)) / (np.std(s) + 1e-6)


def log_transform(s):
    return np.log(np.abs(s) + 1e-6)


# ================================
# MAIN FEATURE FUNCTION
# ================================
def extract_features(spec, aux_features):
    ch0, ch1, ch2 = spec
    features = []

    # ORIGINAL FEATURES
    for ch in [ch0, ch1, ch2]:
        features += extract_stat_features(ch)
        features += band_features(ch)
        features += gradient_features(ch)

    features += second_derivative_features(ch2)
    features += list(fft_features(ch2))
    features += peak_features(ch2)
    features.append(snr_feature(ch2, ch1))
    features += extract_spectral_windows(ch2)

    # HIGH RESOLUTION
    features += list(downsample_signal(ch2, 500))
    features += list(downsample_signal(ch1, 150))

    # NEW FEATURES
    ch2_norm = normalize_signal(ch2)
    features += extract_stat_features(ch2_norm)
    features += gradient_features(ch2_norm)

    ch2_log = log_transform(ch2)
    features += extract_stat_features(ch2_log)

    # INTERACTIONS
    features.append(np.mean(ch2 * ch1))
    features.append(np.std(ch2 * ch1))

    # AUX
    features += list(aux_features)

    return np.array(features, dtype=np.float32)


# ================================
# BUILD MATRIX
# ================================
def build_feature_matrix():
    print("Loading data...")

    spectra = np.load(os.path.join(DATA_DIR, "spectra.npy"), mmap_mode='r')

    # 🔒 SAFE TARGET LOAD
    targets_original = np.load(os.path.join(DATA_DIR, "molecules.npy"), mmap_mode='r')

    aux = np.load(os.path.join(DATA_DIR, "aux_params.npy"), mmap_mode='r')

    print("Loaded data:")
    print("Spectra:", spectra.shape)
    print("Targets:", targets_original.shape)

    X = []

    for i in range(len(spectra)):
        if i % 5000 == 0:
            print(f"Processing {i}/{len(spectra)}")

        features = extract_features(spectra[i], aux[i])
        X.append(features)

    X = np.array(X)

    print("Raw feature matrix shape:", X.shape)

    # CLEAN
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # FEATURE SELECTION
    selector = VarianceThreshold()
    X = selector.fit_transform(X)

    print("Final feature matrix shape:", X.shape)

    # 🔍 FINAL CHECK (VERY IMPORTANT)
    print("\nFINAL CHECK:")
    print("X shape:", X.shape)
    print("y shape:", targets_original.shape)

    assert targets_original.shape[1] == 12, "❌ Targets corrupted!"

    # SAVE
    np.save(os.path.join(OUTPUT_DIR, "X.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "targets.npy"), targets_original)

    joblib.dump(selector, os.path.join(OUTPUT_DIR, "variance_selector.pkl"))

    print("\nFeatures + targets saved successfully 🚀")


# ================================
# MAIN
# ================================
if __name__ == "__main__":
    build_feature_matrix()