# src/evaluate.py
# ─────────────────────────────────────────────────────────────────────────────
# Evaluate a trained ExoplanetCNN on the test set.
# Produces: per-molecule RMSE/R², scatter plots, calibration curve,
#           and uncertainty vs error analysis.
#
# Usage:
#   python src/evaluate.py                          # uses best_model.pt
#   python src/evaluate.py --ckpt checkpoints/best_model.pt
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import r2_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    CHECKPOINT_DIR, PLOTS_DIR, MOLECULES, N_MOLECULES, MC_SAMPLES, CONFIDENCE,
    LOG_VMR_MIN, LOG_VMR_MAX,
)
from dataset import get_dataloaders, PROCESSED_DIR
from cnn1d   import ExoplanetCNN, mc_predict

BEST_CKPT = os.path.join(CHECKPOINT_DIR, 'best_model.pt')
plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': '#F8F9FA',
                     'axes.grid': True, 'grid.alpha': 0.3,
                     'axes.spines.top': False, 'axes.spines.right': False})


def evaluate(ckpt_path=BEST_CKPT, n_mc=MC_SAMPLES):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Load checkpoint ───────────────────────────────────────────────────────
    ckpt   = torch.load(ckpt_path, map_location=device)
    model  = ExoplanetCNN().to(device)
    model.load_state_dict(ckpt['model'])
    scaler = ckpt.get('scaler')
    print(f'Loaded checkpoint: {ckpt_path}  (epoch {ckpt["epoch"]})')

    # ── Test loader ───────────────────────────────────────────────────────────
    _, _, test_loader, _, wavelengths = get_dataloaders()

    # ── MC Dropout inference ──────────────────────────────────────────────────
    means, stds, lows, highs, truths = [], [], [], [], []
    print(f'Running MC Dropout inference ({n_mc} passes per batch)...')

    for spec, aux, targets in test_loader:
        spec, aux = spec.to(device), aux.to(device)
        result = mc_predict(model, spec, aux, n_samples=n_mc, confidence=CONFIDENCE)
        means.append(result['mean'].cpu().numpy())
        stds.append(result['std'].cpu().numpy())
        lows.append(result['lo'].cpu().numpy())
        highs.append(result['hi'].cpu().numpy())
        truths.append(targets.numpy())

    means  = np.vstack(means)    # (N, 12)
    stds   = np.vstack(stds)
    lows   = np.vstack(lows)
    highs  = np.vstack(highs)
    truths = np.vstack(truths)

    # ── Metrics ───────────────────────────────────────────────────────────────
    rmse    = np.sqrt(((means - truths)**2).mean(axis=0))        # (12,)
    mae     = np.abs(means - truths).mean(axis=0)
    r2      = np.array([r2_score(truths[:, i], means[:, i]) for i in range(N_MOLECULES)])
    coverage= ((truths >= lows) & (truths <= highs)).mean(axis=0) # (12,)
    sharpness= (highs - lows).mean(axis=0)                        # (12,)

    print(f'\n{"Molecule":<8} {"RMSE":>7} {"MAE":>7} {"R²":>7} {"Cov95":>7} {"Width":>7}')
    print('─' * 48)
    for i, mol in enumerate(MOLECULES):
        print(f'{mol:<8} {rmse[i]:>7.3f} {mae[i]:>7.3f} {r2[i]:>7.3f} '
              f'{coverage[i]:>7.3f} {sharpness[i]:>7.3f}')
    print('─' * 48)
    print(f'{"Mean":<8} {rmse.mean():>7.3f} {mae.mean():>7.3f} {r2.mean():>7.3f} '
          f'{coverage.mean():>7.3f} {sharpness.mean():>7.3f}')

    # ── Plots ─────────────────────────────────────────────────────────────────
    _plot_scatter_grid(truths, means, stds)
    _plot_calibration(truths, lows, highs)
    _plot_uncertainty_vs_error(truths, means, stds)
    _plot_training_curves()

    return dict(rmse=rmse, mae=mae, r2=r2, coverage=coverage, sharpness=sharpness)


def _plot_scatter_grid(truths, means, stds):
    """4×3 grid: predicted vs true for all 12 molecules with ±1σ shading."""
    fig, axes = plt.subplots(3, 4, figsize=(18, 13))
    fig.suptitle('Predicted vs True log₁₀(VMR) — Test Set\n(error bars = MC Dropout ±1σ)',
                 fontsize=13, fontweight='bold')
    palette = plt.cm.tab20(np.linspace(0, 1, 12))

    for i, (mol, ax) in enumerate(zip(MOLECULES, axes.ravel())):
        t, m, s = truths[:, i], means[:, i], stds[:, i]
        rmse_i  = np.sqrt(((m - t)**2).mean())
        r2_i    = r2_score(t, m)

        ax.errorbar(t, m, yerr=s, fmt='o', ms=2, alpha=0.25,
                    color=palette[i], elinewidth=0.5, capsize=0)
        lim = [LOG_VMR_MIN, LOG_VMR_MAX]
        ax.plot(lim, lim, 'k--', lw=1.2, alpha=0.8, label='y=x')
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel('True log₁₀(VMR)', fontsize=8)
        ax.set_ylabel('Predicted',       fontsize=8)
        ax.set_title(f'{mol}   RMSE={rmse_i:.3f}   R²={r2_i:.3f}',
                     fontsize=9, fontweight='bold')
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, 'scatter_grid.png')
    plt.savefig(path, dpi=150); plt.close()
    print(f'Saved: {path}')


def _plot_calibration(truths, lows, highs):
    """Coverage vs confidence level — perfect calibration = diagonal."""
    levels   = np.linspace(0.05, 0.99, 40)
    coverage = []
    for alpha in 1 - levels:
        # Symmetrically narrow the CI
        half  = (highs - lows) * (1 - alpha) / 2
        mid   = (highs + lows) / 2
        lo_a  = mid - half
        hi_a  = mid + half
        cov   = ((truths >= lo_a) & (truths <= hi_a)).mean()
        coverage.append(cov)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(levels, coverage,  color='#2196F3', lw=2.5, label='Model')
    ax.plot([0, 1], [0, 1],   'k--',           lw=1.5, label='Perfect calibration')
    ax.fill_between(levels, levels, coverage,   alpha=0.15, color='#2196F3')
    ax.set_xlabel('Nominal confidence level', fontsize=11)
    ax.set_ylabel('Empirical coverage',       fontsize=11)
    ax.set_title('Calibration Curve — MC Dropout', fontweight='bold', fontsize=12)
    ax.legend(fontsize=10)
    cal_err = np.abs(np.array(coverage) - levels).mean()
    ax.text(0.05, 0.92, f'Mean calibration error: {cal_err:.3f}',
            transform=ax.transAxes, fontsize=10, color='navy')
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, 'calibration.png')
    plt.savefig(path, dpi=150); plt.close()
    print(f'Saved: {path}')


def _plot_uncertainty_vs_error(truths, means, stds):
    """Scatter of uncertainty (std) vs absolute error — should correlate."""
    errors = np.abs(means - truths).mean(axis=1)   # (N,)
    uncert = stds.mean(axis=1)                      # (N,)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.hexbin(uncert, errors, gridsize=50, cmap='Blues', mincnt=1)
    ax.set_xlabel('Mean MC Dropout σ (uncertainty)', fontsize=11)
    ax.set_ylabel('Mean |error| (log₁₀ VMR)',        fontsize=11)
    ax.set_title('Uncertainty vs Error\n(good model: positive correlation)',
                 fontweight='bold', fontsize=12)
    # Correlation coefficient
    r = np.corrcoef(uncert, errors)[0, 1]
    ax.text(0.05, 0.92, f'Pearson r = {r:.3f}', transform=ax.transAxes,
            fontsize=11, color='navy')
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, 'uncertainty_vs_error.png')
    plt.savefig(path, dpi=150); plt.close()
    print(f'Saved: {path}')


def _plot_training_curves():
    import json
    hist_path = os.path.join(os.path.dirname(PLOTS_DIR), 'training_history.json')
    if not os.path.exists(hist_path):
        return
    with open(hist_path) as f:
        h = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Training History', fontweight='bold', fontsize=13)
    axes[0].plot(h['train_loss'], label='Train', color='#2196F3')
    axes[0].plot(h['val_loss'],   label='Val',   color='#E91E63')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('MSE Loss')
    axes[0].set_title('Loss Curves'); axes[0].legend()

    rmse_hist = np.array(h['val_rmse_per_mol'])  # (epochs, 12)
    palette   = plt.cm.tab20(np.linspace(0, 1, 12))
    for i, mol in enumerate(MOLECULES):
        axes[1].plot(rmse_hist[:, i], color=palette[i], lw=1.5, label=mol, alpha=0.8)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('RMSE (log₁₀ VMR)')
    axes[1].set_title('Per-Molecule Val RMSE')
    axes[1].legend(fontsize=7, ncol=3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, 'training_curves.png')
    plt.savefig(path, dpi=150); plt.close()
    print(f'Saved: {path}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=BEST_CKPT, help='Path to model checkpoint')
    ap.add_argument('--mc',   type=int, default=MC_SAMPLES, help='MC Dropout samples')
    args = ap.parse_args()
    evaluate(ckpt_path=args.ckpt, n_mc=args.mc)
