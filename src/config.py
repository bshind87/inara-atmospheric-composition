# src/config.py
# ─────────────────────────────────────────────────────────────────────────────
# All project constants. Import from any file in src/ or notebooks/ like:
#
#   import sys, os
#   sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
#   from config import *
#
# Project layout assumed:
#   inara/
#   ├── src/            ← this file lives here
#   ├── notebooks/
#   └── inara_data/     ← numbered CSVs + parameters.csv
# ─────────────────────────────────────────────────────────────────────────────

import os

# ── Root is one level above src/ ──────────────────────────────────────────────
ROOT_DIR        = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR         = os.path.join(ROOT_DIR, 'src')
NOTEBOOKS_DIR   = os.path.join(ROOT_DIR, 'notebooks')
DATA_DIR        = os.path.join(ROOT_DIR, 'inara_data')      # numbered CSVs live here
RESULTS_DIR     = os.path.join(ROOT_DIR, 'results')
CHECKPOINT_DIR  = os.path.join(ROOT_DIR, 'checkpoints')
PLOTS_DIR       = os.path.join(RESULTS_DIR, 'plots')

for _d in [RESULTS_DIR, CHECKPOINT_DIR, PLOTS_DIR]:
    os.makedirs(_d, exist_ok=True)

# ── Labels file ───────────────────────────────────────────────────────────────
# The INARA download bundle includes a master parameters CSV alongside the
# numbered spectrum CSVs. Expected at inara_data/parameters.csv.
# Columns: system_id, H2O, CO2, O2, O3, CH4, N2, N2O, CO, H2, H2S, SO2, NH3,
#          stellar_type, stellar_teff, stellar_radius, planet_radius,
#          planet_mass, surface_gravity, orbital_distance, surface_pressure
LABELS_FILE     = os.path.join(DATA_DIR, 'parameters.csv')

# ── Real INARA CSV format (confirmed from sample files) ──────────────────────
#   Total rows per CSV : 15,346  (full PSG output 0.2–639.9 µm)
#   Usable rows        :  4,378  (where noise > 0, i.e. 0.2–2.0 µm)
#   Instrument         : HWO/LUVOIR  UV–optical–NIR
#   File naming        : {system_id:07d}.csv   e.g. 0001100.csv
CSV_TOTAL_ROWS  = 15_346
SPECTRAL_LENGTH = 4_378          # usable wavelength points
WL_MIN_UM       = 0.2
WL_MAX_UM       = 2.0

# CSV column names
COL_WAVELENGTH  = 'wavelength_(um)'
COL_SP_SIGNAL   = 'star_planet_signal_(erg/s/cm2)'
COL_NOISE       = 'noise_(erg/s/cm2)'
COL_STAR        = 'stellar_signal_(erg/s/cm2)'
COL_PLANET      = 'planet_signal_(erg/s/cm2)'

# ── 3-channel ML input tensor  shape: (3, 4378) ───────────────────────────────
#   Ch 0 — SNR = star_planet / noise,  normalised [0, 1] per spectrum
#   Ch 1 — Mean-subtracted SNR          isolates absorption, removes continuum
#   Ch 2 — Log₁₀ transit depth (ppm)   scaled [0, 1]
INPUT_CHANNELS  = 3

# Transit depth scaling bounds (log₁₀ ppm)
LOG_DEPTH_MIN   = -3.0    # 0.001 ppm
LOG_DEPTH_MAX   =  6.0    # 10⁶ ppm

# ── Target molecules ──────────────────────────────────────────────────────────
MOLECULES = [
    'H2O',   # habitability; 0.7–2.0 µm bands
    'CO2',   # greenhouse gas
    'O2',    # biosignature — 0.76 µm A-band
    'O3',    # biosignature — UV Hartley band
    'CH4',   # biosignature if co-present with O2
    'N2',    # bulk filler, weak spectral features
    'N2O',   # microbial biosignature
    'CO',    # volcanic / photochemical
    'H2',    # primordial outgassing
    'H2S',   # volcanic outgassing
    'SO2',   # volcanic activity
    'NH3',   # nitrogen cycle
]
N_MOLECULES     = len(MOLECULES)   # 12

BIOSIGNATURES   = ['O2', 'O3', 'CH4', 'N2O']
STRONG_FEATURES = ['H2O', 'O2', 'O3', 'CO2']   # strong features in 0.2–2 µm
WEAK_FEATURES   = ['N2', 'H2']                  # weak / no features

LOG_VMR_MIN     = -8.0
LOG_VMR_MAX     =  0.0

# ── Auxiliary features (stellar + planetary params) ───────────────────────────
AUX_FEATURES = [
    'stellar_type',       # encoded: F=0, G=1, K=2, M=3
    'stellar_teff',
    'stellar_radius',
    'planet_radius',
    'planet_mass',
    'surface_gravity',
    'orbital_distance',
    'surface_pressure',
]
N_AUX           = len(AUX_FEATURES)   # 8
STELLAR_ENC     = {'F': 0, 'G': 1, 'K': 2, 'M': 3}

# ── Model ─────────────────────────────────────────────────────────────────────
CONV_LAYERS = [
    (64,  7, 'tanh'),
    (64,  5, 'relu'),
    (128, 3, 'relu'),
    (256, 3, 'relu'),
]
POOL_SIZE       = 2
FC_LAYERS       = [256, 128]
DROPOUT_RATE    = 0.2

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE      = 256
LR              = 1e-3
WEIGHT_DECAY    = 1e-4
LR_PATIENCE     = 5
LR_FACTOR       = 0.5
EARLY_STOP_PAT  = 15
MAX_EPOCHS      = 150
RANDOM_SEED     = 42

TRAIN_FRAC      = 0.7
VAL_FRAC        = 0.15
TEST_FRAC       = 0.15

# ── MC Dropout inference ──────────────────────────────────────────────────────
MC_SAMPLES      = 200       # forward passes at inference
CONFIDENCE      = 0.95
