# src/predict.py
# ─────────────────────────────────────────────────────────────────────────────
# Run atmospheric retrieval on a single INARA CSV spectrum.
# Outputs log₁₀ VMR predictions with MC Dropout uncertainty.
#
# Usage:
#   python src/predict.py inara_data/0001100.csv
#   python src/predict.py inara_data/0001100.csv --ckpt checkpoints/best_model.pt
#   python src/predict.py inara_data/0001100.csv --aux "G,5778,1.0,1.0,1.0,9.81,1.0,1.0"
#                                                        stellar_type, teff, srad, prad,
#                                                        pmass, grav, dist, pressure
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    CHECKPOINT_DIR, PLOTS_DIR, MOLECULES, N_MOLECULES, N_AUX,
    MC_SAMPLES, CONFIDENCE, STELLAR_ENC, BIOSIGNATURES,
    LOG_VMR_MIN, LOG_VMR_MAX,
)
from download_inara import parse_csv, build_tensor
from cnn1d           import ExoplanetCNN, mc_predict

BEST_CKPT = os.path.join(CHECKPOINT_DIR, 'best_model.pt')


def predict_spectrum(csv_path: str,
                     ckpt_path: str = BEST_CKPT,
                     aux_str: str   = None,
                     n_mc: int      = MC_SAMPLES,
                     save_plot: bool= True) -> dict:
    """
    Retrieve atmospheric composition from a single INARA CSV spectrum.

    Args:
        csv_path  : path to an INARA CSV file (e.g. inara_data/0001100.csv)
        ckpt_path : model checkpoint
        aux_str   : comma-separated auxiliary features:
                    stellar_type,teff,srad,prad,pmass,grav,dist,pressure
                    e.g. "G,5778,1.0,1.0,1.0,9.81,1.0,1.0"
                    (defaults to solar/Earth values if not provided)
        n_mc      : MC Dropout forward passes
        save_plot : save bar chart to results/plots/

    Returns:
        dict with 'molecules', 'mean', 'std', 'lo', 'hi' arrays (length 12)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Load model ────────────────────────────────────────────────────────────
    ckpt  = torch.load(ckpt_path, map_location=device)
    model = ExoplanetCNN().to(device)
    model.load_state_dict(ckpt['model'])
    scaler = ckpt.get('scaler')

    # ── Parse spectrum ────────────────────────────────────────────────────────
    parsed = parse_csv(csv_path)
    tensor = build_tensor(parsed['snr_raw'], parsed['depth_logppm'])
    spec   = torch.tensor(tensor[None], dtype=torch.float32, device=device)  # (1,3,4378)

    # ── Auxiliary features ────────────────────────────────────────────────────
    aux_vals = _parse_aux(aux_str)
    if scaler is not None:
        aux_vals = scaler.transform(aux_vals.reshape(1, -1)).ravel()
    aux = torch.tensor(aux_vals[None], dtype=torch.float32, device=device)  # (1, 8)

    # ── MC Dropout inference ──────────────────────────────────────────────────
    result = mc_predict(model, spec, aux, n_samples=n_mc, confidence=CONFIDENCE)
    mean   = result['mean'].cpu().numpy().ravel()   # (12,)
    std    = result['std'].cpu().numpy().ravel()
    lo     = result['lo'].cpu().numpy().ravel()
    hi     = result['hi'].cpu().numpy().ravel()

    # ── Print results ─────────────────────────────────────────────────────────
    sid = parsed['system_id']
    print(f'\n{"═"*58}')
    print(f'  Atmospheric Retrieval: {os.path.basename(csv_path)}  (System {sid})')
    print(f'  MC Dropout passes: {n_mc}  |  {int(CONFIDENCE*100)}% credible intervals')
    print(f'{"═"*58}')
    print(f'  {"Molecule":<8} {"log VMR":>8} {"±σ":>7} {"95% CI":>18}  {"Note":<12}')
    print(f'  {"─"*56}')
    for i, mol in enumerate(MOLECULES):
        bio  = ' ← BIOSIG' if mol in BIOSIGNATURES else ''
        print(f'  {mol:<8} {mean[i]:>8.3f} {std[i]:>7.3f}  '
              f'[{lo[i]:>6.3f}, {hi[i]:>6.3f}]{bio}')
    print(f'{"═"*58}\n')

    if save_plot:
        _plot_retrieval(mean, std, lo, hi, sid, csv_path)

    return dict(molecules=MOLECULES, mean=mean, std=std, lo=lo, hi=hi,
                system_id=sid, wavelengths=parsed['wavelengths'])


def _parse_aux(aux_str: str) -> np.ndarray:
    """Parse auxiliary feature string or return Earth/Sun defaults."""
    if aux_str is None:
        return np.array([
            float(STELLAR_ENC.get('G', 1)),  # stellar_type
            5778.0,                           # teff
            1.0,                              # stellar_radius
            1.0,                              # planet_radius
            1.0,                              # planet_mass
            9.81,                             # surface_gravity
            1.0,                              # orbital_distance
            1.0,                              # surface_pressure
        ], dtype=np.float32)
    parts = [p.strip() for p in aux_str.split(',')]
    return np.array([
        float(STELLAR_ENC.get(parts[0], 1)),  # stellar_type → encoded
        *[float(p) for p in parts[1:]],
    ], dtype=np.float32)


def _plot_retrieval(mean, std, lo, hi, system_id, csv_path):
    """Horizontal bar chart with ±σ error bars and 95% CI whiskers."""
    palette = ['#E91E63' if m in BIOSIGNATURES else '#2196F3' for m in MOLECULES]

    fig, ax = plt.subplots(figsize=(10, 7))
    y  = np.arange(N_MOLECULES)

    ax.barh(y, mean - LOG_VMR_MIN,
            left=LOG_VMR_MIN, height=0.55,
            color=palette, alpha=0.75, edgecolor='white')
    ax.errorbar(mean, y, xerr=std,
                fmt='none', color='black', capsize=4, lw=1.5, zorder=3)
    # 95% CI whiskers
    for i in range(N_MOLECULES):
        ax.plot([lo[i], hi[i]], [y[i], y[i]], color='navy', lw=2.5,
                solid_capstyle='round', zorder=4, alpha=0.6)

    ax.set_yticks(y)
    ax.set_yticklabels([f'{"⭐" if m in BIOSIGNATURES else "  "} {m}'
                        for m in MOLECULES], fontsize=11)
    ax.set_xlabel('log₁₀(Volume Mixing Ratio)', fontsize=11)
    ax.set_xlim(LOG_VMR_MIN - 0.3, LOG_VMR_MAX + 0.3)
    ax.set_title(f'Atmospheric Retrieval — {os.path.basename(csv_path)}\n'
                 f'System {system_id}   (MC Dropout ±σ  |  navy bar = 95% CI)',
                 fontweight='bold', fontsize=12)
    ax.axvline(0, color='grey', lw=0.8, linestyle='--', alpha=0.5)
    patches = [mpatches.Patch(color='#E91E63', label='Biosignature molecule'),
               mpatches.Patch(color='#2196F3', label='Other molecule')]
    ax.legend(handles=patches, fontsize=9, loc='lower right')
    plt.tight_layout()

    fname = f'retrieval_{system_id}.png'
    path  = os.path.join(PLOTS_DIR, fname)
    plt.savefig(path, dpi=150); plt.close()
    print(f'Plot saved: {path}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Atmospheric retrieval for one spectrum')
    ap.add_argument('csv',          help='Path to INARA CSV file')
    ap.add_argument('--ckpt',       default=BEST_CKPT)
    ap.add_argument('--aux',        default=None,
                    help='Aux features: "stellar_type,teff,srad,prad,pmass,grav,dist,pressure"')
    ap.add_argument('--mc',         type=int, default=MC_SAMPLES)
    ap.add_argument('--no-plot',    action='store_true')
    args = ap.parse_args()
    predict_spectrum(args.csv, args.ckpt, args.aux, args.mc,
                     save_plot=not args.no_plot)
