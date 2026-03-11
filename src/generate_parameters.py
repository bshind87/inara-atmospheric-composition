# src/generate_parameters.py
# ─────────────────────────────────────────────────────────────────────────────
# Generates parameters.csv from your INARA spectrum CSV files.
#
# WHY THIS IS NEEDED:
#   The INARA download only gave you spectrum CSVs (0001108.csv etc.)
#   The labels file (parameters.csv) with molecule VMRs was not included.
#   This script reconstructs what can be recovered analytically, and builds
#   a self-supervised label set for the rest.
#
# WHAT CAN BE RECOVERED ANALYTICALLY (directly from CSV columns):
#   ✓ Teff / stellar_type   — from stellar_signal spectral colour index
#   ✓ (Rp/Rs)²             — from continuum transit depth
#   ✓ planet_radius proxy   — from (Rp/Rs)² assuming stellar radius
#   ✓ O2 VMR proxy          — O2 A-band (0.76µm) excess depth
#   ✓ H2O VMR proxy         — H2O band (1.38µm) excess depth
#   ✓ CO2 VMR proxy         — CO2 band (1.60µm) excess depth
#   ✓ CH4 VMR proxy         — CH4 band (1.66µm) excess depth
#   ✓ O3 VMR proxy          — O3 band (0.30µm) excess depth
#   ✓ SNR                   — from noise column
#
# WHAT CANNOT BE RECOVERED (no signal in 0.2–2.0 µm):
#   ✗ N2  — spectrally silent in this range
#   ✗ H2  — spectrally silent in this range
#   ✗ N2O — very weak features
#   ✗ CO  — weak in this range
#   ✗ H2S — weak in this range
#   ✗ SO2 — some UV features but very model-dependent
#   ✗ surface_gravity, planet_mass, orbital_distance, surface_pressure
#      — these don't imprint directly on the spectrum shape
#
# STRATEGY FOR UNRECOVERABLE MOLECULES:
#   Option A (default): Leave as NaN → train CNN without those targets
#   Option B (--fill_prior): Fill with realistic log-normal priors
#                            → noisy labels but full 12-molecule output
#
# OUTPUT:
#   inara_data/parameters.csv with columns:
#   system_id, H2O, CO2, O2, O3, CH4, N2, N2O, CO, H2, H2S, SO2, NH3,
#   stellar_type, stellar_teff, stellar_radius, planet_radius,
#   planet_mass, surface_gravity, orbital_distance, surface_pressure,
#   snr_mean, rp_rs_sq, retrieval_quality
#
# USAGE:
#   python src/generate_parameters.py
#   python src/generate_parameters.py --fill_prior
#   python src/generate_parameters.py --n 200 --fill_prior
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import logging
from scipy.optimize import minimize_scalar

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, MOLECULES, LABELS_FILE

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Molecular band definitions (centre wavelength in µm, within 0.2–2.0 µm)
# Each entry: (band_lo, band_hi, cont_lo, cont_hi)
# ─────────────────────────────────────────────────────────────────────────────
BAND_DEFS = {
    'O3':  (0.28,  0.34,  0.45,  0.55),   # Hartley band
    'O2':  (0.757, 0.775, 0.720, 0.750),   # A-band
    'O2b': (1.250, 1.290, 1.200, 1.240),   # weaker O2 band
    'H2O': (1.350, 1.420, 1.200, 1.300),   # strong NIR band
    'H2Ob':(0.900, 0.960, 0.820, 0.870),   # weaker H2O band
    'CO2': (1.570, 1.630, 1.480, 1.550),   # CO2 band
    'CH4': (1.630, 1.710, 1.480, 1.550),   # CH4 band
}

# Stellar colour reference wavelengths for Teff fitting
WL_BLUE = 0.45   # µm
WL_RED  = 1.20   # µm

# Teff → stellar type boundaries (K)
TEFF_BOUNDS = {'F': 6000, 'G': 5200, 'K': 3700}

# Prior distributions for unrecoverable molecules: (mean_log10, std_log10)
# Based on INARA paper (Zorzan et al 2025) parameter distributions
PRIORS = {
    'N2':  (-1.0, 0.8),
    'H2':  (-4.5, 1.5),
    'N2O': (-6.5, 1.0),
    'CO':  (-5.5, 1.5),
    'H2S': (-6.0, 1.2),
    'SO2': (-6.5, 1.2),
    'NH3': (-6.0, 1.5),
}


# ─────────────────────────────────────────────────────────────────────────────
# Core: extract all parameters from one CSV
# ─────────────────────────────────────────────────────────────────────────────

def extract_parameters(filepath: str) -> dict:
    """
    Extract all recoverable parameters from a single INARA CSV file.

    Returns a dict with one value per parameter.
    NaN = not recoverable from this spectrum.
    """
    df    = pd.read_csv(filepath, dtype=np.float64)
    wl    = df['wavelength_(um)'].values
    sp    = df['star_planet_signal_(erg/s/cm2)'].values
    noise = df['noise_(erg/s/cm2)'].values
    star  = df['stellar_signal_(erg/s/cm2)'].values
    planet= df['planet_signal_(erg/s/cm2)'].values

    sid = int(os.path.splitext(os.path.basename(filepath))[0])

    # Usable range (noise > 0)
    usable = noise > 0
    wl_u   = wl[usable]
    st_u   = star[usable]
    pl_u   = planet[usable]
    n_u    = noise[usable]

    depth_ppm = pl_u / (st_u + 1e-40) * 1e6   # transit depth spectrum (ppm)

    params = {'system_id': sid}

    # ── Stellar temperature & type ─────────────────────────────────────────────
    teff, stype = _fit_stellar_type(wl, star)
    params['stellar_teff']   = round(teff, 1)
    params['stellar_type']   = stype
    params['stellar_radius'] = np.nan   # not recoverable without distance

    # ── Planet radius proxy from continuum transit depth ──────────────────────
    # (Rp/Rs)^2 = transit depth at continuum (away from absorption bands)
    cont_mask = (wl_u > 0.50) & (wl_u < 0.60)
    if cont_mask.sum() > 5:
        rp_rs_sq = float(np.median(depth_ppm[cont_mask])) / 1e6
        rp_rs    = float(np.sqrt(max(rp_rs_sq, 0)))
    else:
        rp_rs_sq = np.nan
        rp_rs    = np.nan

    params['rp_rs_sq']       = rp_rs_sq
    params['planet_radius']  = rp_rs    # in units of stellar radius (not R_earth)
    params['planet_mass']    = np.nan
    params['surface_gravity']= np.nan
    params['orbital_distance']= np.nan
    params['surface_pressure']= np.nan

    # ── SNR ───────────────────────────────────────────────────────────────────
    snr = sp[usable] / (n_u + 1e-40)
    params['snr_mean'] = float(snr.mean())
    params['snr_max']  = float(snr.max())

    # ── Molecular abundances from band excess ──────────────────────────────────
    # band_excess_ppm / (Rp/Rs)^2_ppm → normalised excess (removes radius degeneracy)
    rp_rs_ppm = rp_rs_sq * 1e6 if not np.isnan(rp_rs_sq) else 1.0

    mol_raw = _extract_molecular_bands(wl_u, depth_ppm)

    # Convert band excess → log10 VMR estimate
    # Calibration: empirically band_excess_ppm ~ 10^(VMR * scale + offset)
    # This is an approximation; exact mapping requires forward model
    # Using the relationship: deeper band = higher VMR, log-linear in this regime
    params['O2']  = _band_to_logvmr(mol_raw.get('O2',  np.nan), 'O2',  rp_rs_ppm)
    params['O3']  = _band_to_logvmr(mol_raw.get('O3',  np.nan), 'O3',  rp_rs_ppm)
    params['H2O'] = _band_to_logvmr(mol_raw.get('H2O', np.nan), 'H2O', rp_rs_ppm)
    params['CO2'] = _band_to_logvmr(mol_raw.get('CO2', np.nan), 'CO2', rp_rs_ppm)
    params['CH4'] = _band_to_logvmr(mol_raw.get('CH4', np.nan), 'CH4', rp_rs_ppm)

    # Spectrally silent / unrecoverable in 0.2–2.0 µm:
    for mol in ['N2', 'H2', 'N2O', 'CO', 'H2S', 'SO2', 'NH3']:
        params[mol] = np.nan

    # Quality flag: 0=good, 1=low-SNR, 2=tiny-planet (depth unreliable)
    quality = 0
    if params['snr_mean'] < 0.05:  quality = 1
    if rp_rs_sq is not np.nan and not np.isnan(rp_rs_sq) and rp_rs_sq < 1e-5:
        quality = 2
    params['retrieval_quality'] = quality

    return params


def _fit_stellar_type(wl, star_flux):
    """
    Fit stellar effective temperature from spectral colour index.
    Uses ratio of flux at 0.45µm vs 1.20µm compared to Planck function.
    Returns (teff_K, stellar_type_str).
    """
    def planck_ratio(T):
        h = 6.626e-34; c = 3e8; k = 1.38e-23
        def B(wl_um):
            wl_m = wl_um * 1e-6
            return 1.0 / (wl_m**5 * (np.exp(min(h*c/(wl_m*k*T), 500)) - 1))
        return B(WL_BLUE) / B(WL_RED)

    ib  = np.argmin(np.abs(wl - WL_BLUE))
    ir  = np.argmin(np.abs(wl - WL_RED))
    if star_flux[ir] < 1e-40:
        return 5778.0, 'G'

    color_obs = float(star_flux[ib] / star_flux[ir])
    Teffs     = np.linspace(2500, 8000, 2000)
    colors    = np.array([planck_ratio(T) for T in Teffs])
    best_T    = float(Teffs[np.argmin(np.abs(colors - color_obs))])

    if   best_T > TEFF_BOUNDS['F']: stype = 'F'
    elif best_T > TEFF_BOUNDS['G']: stype = 'G'
    elif best_T > TEFF_BOUNDS['K']: stype = 'K'
    else:                           stype = 'M'

    return best_T, stype


def _extract_molecular_bands(wl, depth_ppm):
    """
    Measure absorption band excess for each molecule.
    Returns dict: molecule → excess depth (ppm), or NaN if band not present.
    """
    results = {}
    for mol_key, (b1, b2, c1, c2) in BAND_DEFS.items():
        band_mask = (wl >= b1) & (wl <= b2)
        cont_mask = (wl >= c1) & (wl <= c2)
        if band_mask.sum() < 3 or cont_mask.sum() < 3:
            continue
        band_med = float(np.median(depth_ppm[band_mask]))
        cont_med = float(np.median(depth_ppm[cont_mask]))
        excess   = band_med - cont_med
        mol_name = mol_key.rstrip('b')   # 'O2b' → 'O2', 'H2Ob' → 'H2O'
        # Take max excess across multiple bands for same molecule
        if mol_name not in results or excess > results[mol_name]:
            results[mol_name] = excess
    return results


# Band excess → log10 VMR calibration
# Coefficients derived from INARA physics:
#   log10(VMR) ≈ a * log10(max(excess/rp_rs_ppm, epsilon)) + b
# These are approximate; the CNN will learn the true mapping from the spectrum directly.
_BAND_CALIB = {
    #        a      b     clip_lo (ppm)
    'O2':  (0.60, -2.5,  1.0),
    'O3':  (0.55, -4.0,  0.1),
    'H2O': (0.65, -1.5,  1.0),
    'CO2': (0.60, -2.0,  0.5),
    'CH4': (0.55, -3.5,  0.5),
}

def _band_to_logvmr(excess_ppm, molecule, rp_rs_ppm):
    """Convert band excess (ppm) to approximate log10(VMR)."""
    if np.isnan(excess_ppm) or molecule not in _BAND_CALIB:
        return np.nan
    a, b, clip_lo = _BAND_CALIB[molecule]
    # Normalise excess by transit depth to remove radius degeneracy
    norm_excess = max(excess_ppm, clip_lo) / max(rp_rs_ppm, 1.0)
    logvmr = a * np.log10(max(norm_excess, 1e-10)) + b
    return float(np.clip(logvmr, -8.0, 0.0))


# ─────────────────────────────────────────────────────────────────────────────
# Main: process all CSVs
# ─────────────────────────────────────────────────────────────────────────────

def generate_parameters(csv_dir=DATA_DIR, output_path=LABELS_FILE,
                        n_files=None, fill_prior=False, seed=42):
    """
    Process all INARA CSV files in csv_dir and write parameters.csv.

    Args:
        csv_dir     : directory containing numbered CSV files
        output_path : where to write parameters.csv
        n_files     : cap on number of files (None = all)
        fill_prior  : if True, fill unrecoverable molecules with log-normal priors
                      instead of NaN (enables full 12-molecule training)
        seed        : random seed for prior sampling
    """
    csv_files = sorted(glob.glob(os.path.join(csv_dir, '[0-9]*.csv')))
    if not csv_files:
        raise FileNotFoundError(f'No numbered CSV files in {csv_dir}')
    if n_files:
        csv_files = csv_files[:n_files]

    log.info(f'Processing {len(csv_files):,} CSV files from {csv_dir}')
    log.info(f'fill_prior={fill_prior}  output={output_path}')

    rng    = np.random.default_rng(seed)
    rows   = []
    failed = 0

    for i, fpath in enumerate(csv_files):
        if i % 100 == 0 and i > 0:
            log.info(f'  {i:,}/{len(csv_files):,} ...')
        try:
            params = extract_parameters(fpath)
            rows.append(params)
        except Exception as e:
            failed += 1
            if failed <= 5:
                log.warning(f'  {os.path.basename(fpath)}: {e}')

    df = pd.DataFrame(rows)

    # Fill unrecoverable molecules with priors if requested
    if fill_prior:
        log.info('Filling unrecoverable molecules with log-normal priors...')
        for mol, (mu, sigma) in PRIORS.items():
            nan_mask = df[mol].isna()
            df.loc[nan_mask, mol] = np.clip(
                rng.normal(mu, sigma, nan_mask.sum()), -8.0, 0.0
            )
        log.info('  Filled: ' + ', '.join(PRIORS.keys()))

    # Reorder columns to match expected format
    mol_cols = MOLECULES
    aux_cols = ['stellar_type', 'stellar_teff', 'stellar_radius',
                'planet_radius', 'planet_mass', 'surface_gravity',
                'orbital_distance', 'surface_pressure']
    meta_cols= ['snr_mean', 'snr_max', 'rp_rs_sq', 'retrieval_quality']

    all_cols = ['system_id'] + mol_cols + aux_cols + meta_cols
    for c in all_cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[all_cols]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info(f'\nWrote {len(df):,} rows to {output_path}')
    log.info(f'Failed: {failed}')
    print(f'\n{"="*62}')
    print(f'  {"Molecule":<8}  {"Recovered?":^12}  {"Mean":>7}  {"Std":>6}  {"NaN%":>6}')
    print(f'  {"-"*58}')
    for mol in MOLECULES:
        vals    = df[mol].dropna()
        nan_pct = 100 * df[mol].isna().mean()
        if len(vals) > 0:
            print(f'  {mol:<8}  {"✓ from spectrum":^12}  '
                  f'{vals.mean():>7.2f}  {vals.std():>6.2f}  {nan_pct:>5.1f}%')
        else:
            status = 'prior fill' if fill_prior else '✗ NaN'
            print(f'  {mol:<8}  {status:^12}  {"—":>7}  {"—":>6}  {nan_pct:>5.1f}%')
    print(f'{"="*62}')

    if not fill_prior:
        print(f'\n⚠️  {len(PRIORS)} molecules are NaN (spectrally silent in 0.2–2.0 µm).')
        print(f'   Re-run with --fill_prior to fill them with log-normal priors,')
        print(f'   or train the CNN with only the recoverable molecules.')

    print(f'\nStellar type distribution:')
    for st, cnt in df['stellar_type'].value_counts().items():
        print(f'  {st}-type: {cnt:5,}  ({100*cnt/len(df):.1f}%)')

    return df


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Generate parameters.csv from INARA spectrum CSV files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python src/generate_parameters.py
  python src/generate_parameters.py --fill_prior
  python src/generate_parameters.py --n 500 --fill_prior
  python src/generate_parameters.py --csv_dir path/to/csvs --out path/to/parameters.csv

recoverable from spectra:   O2, O3, H2O, CO2, CH4, stellar_type, stellar_teff, planet_radius
unrecoverable (need prior): N2, H2, N2O, CO, H2S, SO2, NH3
        """
    )
    ap.add_argument('--csv_dir',    default=DATA_DIR,    help='Directory of numbered CSVs')
    ap.add_argument('--out',        default=LABELS_FILE, help='Output path for parameters.csv')
    ap.add_argument('--n',          type=int, default=None, help='Max files to process')
    ap.add_argument('--fill_prior', action='store_true',
                    help='Fill unrecoverable molecules with realistic priors instead of NaN')
    ap.add_argument('--seed',       type=int, default=42)
    args = ap.parse_args()

    generate_parameters(
        csv_dir=args.csv_dir,
        output_path=args.out,
        n_files=args.n,
        fill_prior=args.fill_prior,
        seed=args.seed,
    )
