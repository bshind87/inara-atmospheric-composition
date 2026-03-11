# src/download_inara.py
# ─────────────────────────────────────────────────────────────────────────────
# Prepares INARA data for training by parsing the numbered CSV spectrum files
# and joining them with the labels from parameters.csv.
#
# EXPECTED inara_data/ LAYOUT:
#   inara_data/
#   ├── 0001100.csv        ← spectrum files (one per planetary system)
#   ├── 0001108.csv
#   ├── ...
#   └── parameters.csv     ← labels: system_id + 12 molecule VMRs + aux params
#
# CONFIRMED CSV FORMAT (from real sample files):
#   Columns  : wavelength_(um) | star_planet_signal | noise | stellar_signal | planet_signal
#   Total rows: 15,346  (full PSG output, 0.2–639.9 µm)
#   Usable    :  4,378  (rows where noise > 0, covering 0.2–2.0 µm)
#
# OUTPUT (saved to inara_data/processed/):
#   spectra.npy     (N, 3, 4378)  — 3-channel input tensors
#   molecules.npy   (N, 12)       — log₁₀ VMR targets
#   aux_params.npy  (N, 8)        — stellar + planetary auxiliary features
#   system_ids.npy  (N,)          — INARA system IDs (for traceability)
#   wavelengths.npy (4378,)       — wavelength grid in µm
#
# USAGE:
#   # Parse real data:
#   python src/download_inara.py
#
#   # Generate synthetic demo data (no files needed):
#   python src/download_inara.py --demo
#
#   # Validate a single CSV is parsed correctly:
#   python src/download_inara.py --validate inara_data/0001100.csv
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import logging

# ── Make src/ importable from anywhere ────────────────────────────────────────
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    DATA_DIR, LABELS_FILE, SPECTRAL_LENGTH, MOLECULES, N_MOLECULES, N_AUX,
    COL_WAVELENGTH, COL_SP_SIGNAL, COL_NOISE, COL_STAR, COL_PLANET,
    LOG_DEPTH_MIN, LOG_DEPTH_MAX, LOG_VMR_MIN, LOG_VMR_MAX,
    WL_MIN_UM, WL_MAX_UM, STELLAR_ENC, INPUT_CHANNELS,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')


# ─────────────────────────────────────────────────────────────────────────────
# Core: parse one CSV → 3-channel tensor
# ─────────────────────────────────────────────────────────────────────────────

def parse_csv(filepath: str) -> dict:
    """
    Parse one INARA spectrum CSV into raw signal arrays.

    Filters to rows where noise > 0 (the 4,378 usable wavelength points
    covering 0.2–2.0 µm). Rows beyond 2 µm have noise=0 and are discarded.

    Returns dict with keys:
        system_id   : int
        snr_raw     : (4378,) float32  — star_planet / noise
        depth_logppm: (4378,) float32  — log10(planet/stellar × 1e6)
        wavelengths : (4378,) float64  — wavelength grid µm
    """
    df      = pd.read_csv(filepath, dtype=np.float64,
                          usecols=[COL_WAVELENGTH, COL_SP_SIGNAL,
                                   COL_NOISE, COL_STAR, COL_PLANET])
    usable  = df[df[COL_NOISE] > 0].reset_index(drop=True)

    if len(usable) == 0:
        raise ValueError(f'No usable rows (noise>0) in {filepath}')

    wl      = usable[COL_WAVELENGTH].values
    sp      = usable[COL_SP_SIGNAL].values
    noise   = usable[COL_NOISE].values
    star    = usable[COL_STAR].values
    planet  = usable[COL_PLANET].values

    snr_raw      = (sp / (noise + 1e-40)).astype(np.float32)
    depth_ppm    = np.clip(planet / (star + 1e-40) * 1e6, 1e-3, 1e6)
    depth_logppm = np.log10(depth_ppm).astype(np.float32)

    sid = int(os.path.splitext(os.path.basename(filepath))[0])
    return dict(system_id=sid, snr_raw=snr_raw,
                depth_logppm=depth_logppm, wavelengths=wl)


def build_tensor(snr_raw: np.ndarray, depth_logppm: np.ndarray) -> np.ndarray:
    """
    Build the (3, SPECTRAL_LENGTH) input tensor from a parsed CSV.

    Ch 0: min-max normalised SNR                 → [0, 1]
    Ch 1: mean-subtracted Ch 0                   → removes stellar continuum
    Ch 2: log depth scaled to [LOG_DEPTH_MIN/MAX] → [0, 1]
    """
    lo, hi = snr_raw.min(), snr_raw.max()
    ch0 = (snr_raw - lo) / (hi - lo + 1e-10)
    ch1 = ch0 - ch0.mean()
    ch2 = (np.clip(depth_logppm, LOG_DEPTH_MIN, LOG_DEPTH_MAX)
           - LOG_DEPTH_MIN) / (LOG_DEPTH_MAX - LOG_DEPTH_MIN)
    return np.stack([ch0, ch1, ch2], axis=0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Build processed dataset from inara_data/
# ─────────────────────────────────────────────────────────────────────────────

def build_processed_dataset(csv_dir: str = DATA_DIR,
                             labels_path: str = LABELS_FILE,
                             output_dir: str = PROCESSED_DIR,
                             n_samples: int = None):
    """
    Parse all spectrum CSVs, join with labels, save processed numpy arrays.

    Args:
        csv_dir     : directory containing numbered CSV files
        labels_path : path to parameters.csv with system_id + molecule VMRs
        output_dir  : where to write processed .npy files
        n_samples   : cap on number of files to process (None = all)
    """
    csv_files = sorted(glob.glob(os.path.join(csv_dir, '[0-9]*.csv')))
    if not csv_files:
        raise FileNotFoundError(
            f'No numbered CSV files found in {csv_dir}\n'
            f'Download INARA from: https://exoplanetarchive.ipac.caltech.edu'
            f'/cgi-bin/FDL/nph-fdl'
        )
    if n_samples:
        csv_files = csv_files[:n_samples]
    log.info(f'Found {len(csv_files):,} CSV files in {csv_dir}')

    if not os.path.exists(labels_path):
        raise FileNotFoundError(
            f'Labels file not found: {labels_path}\n'
            f'The INARA download bundle includes a parameters.csv alongside '
            f'the numbered CSVs. Download it from the same FDL page.'
        )
    labels = pd.read_csv(labels_path).set_index('system_id')
    log.info(f'Labels loaded: {labels.shape}  columns: {list(labels.columns)}')

    os.makedirs(output_dir, exist_ok=True)

    spectra_list, mol_list, aux_list, sid_list = [], [], [], []
    failed = 0

    for i, fpath in enumerate(csv_files):
        if i % 5000 == 0 and i > 0:
            log.info(f'  {i:,}/{len(csv_files):,} parsed ...')
        try:
            parsed  = parse_csv(fpath)
            sid     = parsed['system_id']
            if sid not in labels.index:
                failed += 1
                continue
            row     = labels.loc[sid]
            tensor  = build_tensor(parsed['snr_raw'], parsed['depth_logppm'])

            mol_row = np.array([float(row[m]) for m in MOLECULES], dtype=np.float32)
            aux_row = _build_aux(row)

            spectra_list.append(tensor)
            mol_list.append(mol_row)
            aux_list.append(aux_row)
            sid_list.append(sid)
        except Exception as e:
            failed += 1
            if failed <= 5:
                log.warning(f'  {os.path.basename(fpath)}: {e}')

    log.info(f'Parsed {len(spectra_list):,} ok | {failed} failed/missing labels')

    spectra_arr = np.stack(spectra_list, axis=0)   # (N, 3, 4378)
    mol_arr     = np.stack(mol_list,    axis=0)    # (N, 12)
    aux_arr     = np.stack(aux_list,    axis=0)    # (N, 8)
    sid_arr     = np.array(sid_list,    dtype=np.int32)

    _save(spectra_arr, mol_arr, aux_arr, sid_arr, output_dir,
          wavelengths=parsed['wavelengths'])

    return spectra_arr, mol_arr, aux_arr


def _build_aux(row) -> np.ndarray:
    return np.array([
        float(STELLAR_ENC.get(str(row.get('stellar_type', 'G')), 1)),
        float(row.get('stellar_teff',     5778.0)),
        float(row.get('stellar_radius',   1.0)),
        float(row.get('planet_radius',    1.0)),
        float(row.get('planet_mass',      1.0)),
        float(row.get('surface_gravity',  9.81)),
        float(row.get('orbital_distance', 1.0)),
        float(row.get('surface_pressure', 1.0)),
    ], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Demo mode: generate synthetic data in exact same format
# ─────────────────────────────────────────────────────────────────────────────

def generate_demo(output_dir: str = PROCESSED_DIR,
                  n_samples: int = 20_000,
                  seed: int = 42):
    """
    Generate synthetic data in exact INARA processed format.
    Produces the same files as build_processed_dataset() — drop-in replacement.
    Use while the real download is pending or for unit tests.
    """
    rng = np.random.default_rng(seed)
    log.info(f'Generating {n_samples:,} synthetic samples (demo mode)...')
    os.makedirs(output_dir, exist_ok=True)

    WL = np.logspace(np.log10(WL_MIN_UM), np.log10(WL_MAX_UM), SPECTRAL_LENGTH)

    # Molecular absorption bands present in 0.2–2.0 µm
    bands = [
        (0.72, 0.030, 0.30, 'H2O'), (0.82, 0.030, 0.35, 'H2O'),
        (0.94, 0.040, 0.40, 'H2O'), (1.14, 0.050, 0.45, 'H2O'),
        (1.38, 0.080, 0.60, 'H2O'), (1.87, 0.100, 0.55, 'H2O'),
        (0.76, 0.008, 0.25, 'O2'),  (1.27, 0.020, 0.15, 'O2'),
        (0.30, 0.020, 0.20, 'O3'),  (0.60, 0.040, 0.10, 'O3'),
        (1.60, 0.060, 0.08, 'CO2'), (2.00, 0.100, 0.10, 'CO2'),
        (1.66, 0.040, 0.12, 'CH4'), (1.73, 0.030, 0.08, 'CH4'),
        (0.97, 0.030, 0.05, 'N2O'), (1.57, 0.020, 0.04, 'CO'),
    ]
    mol_dist = {
        'H2O':(-3.0,1.5),'CO2':(-3.5,1.5),'O2': (-4.0,1.5),
        'O3': (-6.5,1.0),'CH4':(-5.0,1.5),'N2': (-1.0,0.8),
        'N2O':(-6.5,1.0),'CO': (-5.5,1.5),'H2': (-4.5,1.5),
        'H2S':(-6.0,1.2),'SO2':(-6.5,1.2),'NH3':(-6.0,1.5),
    }
    mol_arr  = np.zeros((n_samples, N_MOLECULES), dtype=np.float32)
    mol_idx  = {m: i for i, m in enumerate(MOLECULES)}
    for i, m in enumerate(MOLECULES):
        mu, sig = mol_dist[m]
        mol_arr[:, i] = np.clip(rng.normal(mu, sig, n_samples), -8., 0.)

    stypes   = rng.choice(['F','G','K','M'], n_samples, p=[0.08,0.50,0.27,0.15])
    teff_map = {'F':(6500,400),'G':(5778,300),'K':(4500,400),'M':(3200,500)}

    snr_base = np.ones((n_samples, SPECTRAL_LENGTH), dtype=np.float32)
    for st in ['F','G','K','M']:
        mask = stypes == st
        cont = np.exp(-WL / (teff_map[st][0] / 3000.)) + 0.5
        cont /= cont.max()
        lvl  = rng.uniform(0.5, 3.5, mask.sum()).astype(np.float32)
        snr_base[mask] = (cont * lvl[:, None]).astype(np.float32)

    for wl_c, bw, mx, mol in bands:
        if mol not in mol_idx:
            continue
        feat   = np.exp(-((WL - wl_c)**2) / (2 * bw**2))
        norm   = (mol_arr[:, mol_idx[mol]] + 8.) / 8.
        snr_base += (mx * norm[:, None] * feat[None, :]).astype(np.float32)

    noise_std = rng.uniform(0.02, 0.08, (n_samples, 1))
    snr_obs   = np.clip(snr_base + (rng.normal(0,1,snr_base.shape)*noise_std).astype(np.float32), 0, None)

    planet_r  = rng.lognormal(0.1, 0.35, n_samples).astype(np.float32)
    depth_ppm = np.clip(1e4 * planet_r[:,None]**2 * (1 + 0.1*snr_base), 1e-3, 1e6)
    depth_log = np.log10(depth_ppm).astype(np.float32)

    spectra = np.zeros((n_samples, INPUT_CHANNELS, SPECTRAL_LENGTH), dtype=np.float32)
    for i in range(n_samples):
        spectra[i] = build_tensor(snr_obs[i], depth_log[i])

    teff_vals = np.array([float(rng.normal(*teff_map[s])) for s in stypes], dtype=np.float32)
    aux_arr   = np.column_stack([
        np.array([float(STELLAR_ENC[s]) for s in stypes], dtype=np.float32),
        teff_vals,
        rng.lognormal(0.0,  0.25, n_samples).astype(np.float32),
        planet_r,
        rng.lognormal(0.15, 0.60, n_samples).astype(np.float32),
        rng.lognormal(2.28, 0.35, n_samples).astype(np.float32),
        rng.lognormal(-0.35,0.55, n_samples).astype(np.float32),
        rng.lognormal(0.0,  0.40, n_samples).astype(np.float32),
    ]).astype(np.float32)

    sid_arr = np.arange(n_samples, dtype=np.int32)
    _save(spectra, mol_arr, aux_arr, sid_arr, output_dir, wavelengths=WL, stypes=stypes)
    return spectra, mol_arr, aux_arr


# ─────────────────────────────────────────────────────────────────────────────
# Validate a single CSV file
# ─────────────────────────────────────────────────────────────────────────────

def validate_csv(filepath: str):
    """Quick sanity-check on a single CSV. Prints stats, saves a plot."""
    import matplotlib.pyplot as plt

    parsed = parse_csv(filepath)
    tensor = build_tensor(parsed['snr_raw'], parsed['depth_logppm'])
    wl     = parsed['wavelengths']

    print(f'\nFile : {os.path.basename(filepath)}')
    print(f'System ID    : {parsed["system_id"]}')
    print(f'Usable pts   : {len(wl)}  ({wl.min():.4f}–{wl.max():.4f} µm)')
    print(f'Tensor shape : {tensor.shape}  dtype={tensor.dtype}')
    for ch, label in enumerate(['SNR (norm)', 'Mean-sub SNR', 'Log depth (scaled)']):
        t = tensor[ch]
        print(f'  Ch{ch} {label:<20}: min={t.min():.4f}  max={t.max():.4f}  mean={t.mean():.6f}')

    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    fig.suptitle(f'INARA CSV validation — {os.path.basename(filepath)}', fontweight='bold')
    labels = ['Ch0: SNR (normalised)', 'Ch1: Mean-subtracted SNR', 'Ch2: Log transit depth (scaled)']
    colors = ['#2196F3', '#E91E63', '#4CAF50']
    for i in range(3):
        axes[i].plot(wl, tensor[i], color=colors[i], lw=0.8)
        axes[i].set_ylabel(labels[i], fontsize=9)
    axes[-1].set_xlabel('Wavelength (µm)')
    plt.tight_layout()
    out = os.path.join(os.path.dirname(filepath),
                       os.path.splitext(os.path.basename(filepath))[0] + '_validation.png')
    plt.savefig(out, dpi=120)
    print(f'\nPlot saved: {out}')


# ─────────────────────────────────────────────────────────────────────────────
# Save helper
# ─────────────────────────────────────────────────────────────────────────────

def _save(spectra, mol_arr, aux_arr, sid_arr, output_dir,
          wavelengths=None, stypes=None):
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'spectra.npy'),    spectra)
    np.save(os.path.join(output_dir, 'molecules.npy'),  mol_arr)
    np.save(os.path.join(output_dir, 'aux_params.npy'), aux_arr)
    np.save(os.path.join(output_dir, 'system_ids.npy'), sid_arr)
    if wavelengths is not None:
        np.save(os.path.join(output_dir, 'wavelengths.npy'), wavelengths)

    mol_df = pd.DataFrame(mol_arr, columns=MOLECULES)
    aux_cols = ['stellar_type_enc','stellar_teff','stellar_radius','planet_radius',
                'planet_mass','surface_gravity','orbital_distance','surface_pressure']
    aux_df = pd.DataFrame(aux_arr, columns=aux_cols)
    if stypes is not None:
        aux_df['stellar_type'] = stypes
    pd.concat([pd.DataFrame({'system_id': sid_arr}), mol_df, aux_df], axis=1)\
      .to_csv(os.path.join(output_dir, 'parameters_processed.csv'), index=False)

    log.info(f'Saved to {output_dir}:')
    log.info(f'  spectra.npy     {spectra.shape}')
    log.info(f'  molecules.npy   {mol_arr.shape}')
    log.info(f'  aux_params.npy  {aux_arr.shape}')
    print(f'\n  {"Molecule":<8} {"Mean":>8} {"Std":>7} {"Min":>7} {"Max":>7}')
    print(f'  {"-"*44}')
    for i, m in enumerate(MOLECULES):
        v = mol_arr[:, i]
        print(f'  {m:<8} {v.mean():>8.2f} {v.std():>7.2f} {v.min():>7.2f} {v.max():>7.2f}')


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Prepare INARA dataset')
    ap.add_argument('--demo',     action='store_true',
                    help='Generate synthetic demo data (no CSVs needed)')
    ap.add_argument('--validate', metavar='CSV',
                    help='Validate a single CSV file and exit')
    ap.add_argument('--n',        type=int, default=None,
                    help='Max number of CSV files to process')
    ap.add_argument('--out',      default=PROCESSED_DIR,
                    help=f'Output directory (default: {PROCESSED_DIR})')
    args = ap.parse_args()

    if args.validate:
        validate_csv(args.validate)
    elif args.demo:
        generate_demo(args.out, n_samples=args.n or 20_000)
    else:
        build_processed_dataset(DATA_DIR, LABELS_FILE, args.out, args.n)
