import numpy as np
import os
from scipy.signal import find_peaks
from sklearn.feature_selection import VarianceThreshold

# ================================
# CONFIG
# ================================
DATA_DIR = "../inara_data/processed/"
OUTPUT_DIR = "../inara_data/features/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================
# LOAD DATA
# ================================
spectra = np.load(os.path.join(DATA_DIR, "spectra.npy"), mmap_mode='r')
targets = np.load(os.path.join(DATA_DIR, "molecules.npy"), mmap_mode='r')
aux = np.load(os.path.join(DATA_DIR, "aux_params.npy"), mmap_mode='r')

print("Loaded data:")
print("Spectra:", spectra.shape)
print("Targets:", targets.shape)

# Optional subset (for dev)
# spectra = spectra[:10000]
# targets = targets[:10000]
# aux = aux[:10000]

# ================================
# FEATURE FUNCTIONS
# ================================
def extract_stat_features(s):
    return [
        np.mean(s),
        np.std(s),
        np.min(s),
        np.max(s),
        np.percentile(s, 25),
        np.percentile(s, 50),
        np.percentile(s, 75)
    ]

def band_features(s, n_bands=10):
    bands = np.array_split(s, n_bands)
    return [np.mean(b) for b in bands]

def gradient_features(s):
    grad = np.gradient(s)
    return [np.mean(grad), np.std(grad)]

def fft_features(s, k=50):
    fft = np.fft.fft(s)
    return np.abs(fft[:k])

def peak_features(s):
    peaks, _ = find_peaks(-s)
    return [len(peaks)]

def snr_feature(signal, noise):
    return np.mean(signal) / (np.std(noise) + 1e-6)

# ================================
# FEATURE EXTRACTION
# ================================
def extract_features(spec, aux_features):
    ch0, ch1, ch2 = spec
    s = ch2

    features = []
    features += extract_stat_features(s)
    features += band_features(s)
    features += gradient_features(s)
    features += list(fft_features(s))
    features += peak_features(s)
    features.append(snr_feature(ch2, ch1))
    features += list(aux_features)

    return np.array(features, dtype=np.float64)

# ================================
# BUILD FEATURE MATRIX
# ================================
X = []
for i in range(len(spectra)):
    X.append(extract_features(spectra[i], aux[i]))

X = np.array(X)

print("Raw feature matrix shape:", X.shape)

# ================================
# CLEAN DATA
# ================================
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

# Remove zero-variance features
selector = VarianceThreshold()
X = selector.fit_transform(X)

print("Final feature matrix shape:", X.shape)

# ================================
# SAVE FEATURES
# ================================
np.save(os.path.join(OUTPUT_DIR, "X.npy"), X)
np.save(os.path.join(OUTPUT_DIR, "targets.npy"), targets)

print("Features saved successfully 🚀")