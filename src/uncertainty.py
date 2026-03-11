# model/uncertainty.py
# ─────────────────────────────────────────────────────────────────────────────
# MC Dropout inference: run N stochastic forward passes to estimate
# predictive mean and uncertainty for each of the 12 molecular targets.
#
# Key concept:
#   Standard dropout: active during training, OFF at test time.
#   MC Dropout:       active during training AND at test time (model.train()).
#   Each forward pass randomly drops neurons → different prediction.
#   N passes → distribution over predictions → mean + std.
#
# Reference: Gal & Ghahramani (2016); INARA paper (Zorzan et al. 2025)
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MOLECULES, N_MOLECULES, MC_SAMPLES, CONFIDENCE_LEVEL


class MCDropoutPredictor:
    """
    Wraps a trained ExoplanetCNN1D and provides MC Dropout inference.

    Usage:
        predictor = MCDropoutPredictor(model, n_samples=200, device=device)
        mean, std, lower, upper = predictor.predict(spectrum_tensor, aux_tensor)
    """

    def __init__(
        self,
        model:     nn.Module,
        n_samples: int   = MC_SAMPLES,
        device:    torch.device = None,
        conf:      float = CONFIDENCE_LEVEL,
    ):
        self.model     = model
        self.n_samples = n_samples
        self.conf      = conf
        self.device    = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

    def predict(
        self,
        spectrum: torch.Tensor,
        aux:      torch.Tensor = None,
    ) -> dict:
        """
        Run MC Dropout inference on a single spectrum or batch.

        Args:
            spectrum : Tensor (batch, 2, 4379)
            aux      : Tensor (batch, 8) or None

        Returns:
            dict with keys:
              "mean"       : np.ndarray (batch, 12) — predicted log10 VMR
              "std"        : np.ndarray (batch, 12) — predictive std (uncertainty)
              "lower"      : np.ndarray (batch, 12) — lower CI bound
              "upper"      : np.ndarray (batch, 12) — upper CI bound
              "all_preds"  : np.ndarray (n_samples, batch, 12) — raw MC samples
        """
        # ── CRITICAL: keep model in TRAIN mode so dropout stays active ────────
        self.model.train()

        spectrum = spectrum.to(self.device)
        if aux is not None:
            aux = aux.to(self.device)

        # ── N stochastic forward passes ───────────────────────────────────────
        all_preds = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                if aux is not None:
                    pred = self.model(spectrum, aux)
                else:
                    pred = self.model(spectrum)
                all_preds.append(pred.cpu().numpy())

        # all_preds: (n_samples, batch, 12)
        all_preds = np.stack(all_preds, axis=0)

        # ── Compute statistics ────────────────────────────────────────────────
        mean = all_preds.mean(axis=0)       # (batch, 12)
        std  = all_preds.std(axis=0)        # (batch, 12)

        # Percentile-based confidence interval
        alpha = (1 - self.conf) / 2
        lower = np.percentile(all_preds, alpha * 100,       axis=0)  # (batch, 12)
        upper = np.percentile(all_preds, (1 - alpha) * 100, axis=0)  # (batch, 12)

        return {
            "mean":      mean,
            "std":       std,
            "lower":     lower,
            "upper":     upper,
            "all_preds": all_preds,
        }

    def predict_loader(
        self,
        loader,
        desc: str = "MC Dropout inference",
    ) -> dict:
        """
        Run MC Dropout inference over an entire DataLoader.

        Returns:
            dict with same keys as predict(), concatenated across all batches.
            Also includes "targets" if the loader returns labels.
        """
        all_means   = []
        all_stds    = []
        all_lowers  = []
        all_uppers  = []
        all_targets = []

        for batch in tqdm(loader, desc=desc):
            # Handle both (spec, aux, target) and (spec, target) cases
            if len(batch) == 3:
                spec, aux, targets = batch
            else:
                spec, targets = batch
                aux = None

            result = self.predict(spec, aux)
            all_means.append(result["mean"])
            all_stds.append(result["std"])
            all_lowers.append(result["lower"])
            all_uppers.append(result["upper"])
            all_targets.append(targets.numpy())

        return {
            "mean":    np.concatenate(all_means,   axis=0),   # (N, 12)
            "std":     np.concatenate(all_stds,    axis=0),   # (N, 12)
            "lower":   np.concatenate(all_lowers,  axis=0),   # (N, 12)
            "upper":   np.concatenate(all_uppers,  axis=0),   # (N, 12)
            "targets": np.concatenate(all_targets, axis=0),   # (N, 12)
        }


# ── Calibration Metrics ───────────────────────────────────────────────────────

def compute_calibration(
    targets:    np.ndarray,
    lower:      np.ndarray,
    upper:      np.ndarray,
    conf_level: float = CONFIDENCE_LEVEL,
) -> dict:
    """
    Measure how well the MC Dropout uncertainty is calibrated.

    A well-calibrated model should have:
      coverage ≈ conf_level (e.g. 95% of true values within 95% CI)

    Args:
        targets    : (N, 12) — true log10 VMR values
        lower      : (N, 12) — lower CI bound
        upper      : (N, 12) — upper CI bound
        conf_level : float   — nominal confidence level (0.95)

    Returns:
        dict with per-molecule and overall calibration metrics
    """
    # Coverage: fraction of true values inside the predicted interval
    inside  = (targets >= lower) & (targets <= upper)
    coverage_per_mol = inside.mean(axis=0)          # (12,)
    coverage_overall = inside.mean()                 # scalar

    # Sharpness: mean width of confidence intervals (smaller = more certain)
    width             = upper - lower
    sharpness_per_mol = width.mean(axis=0)
    sharpness_overall = width.mean()

    # Calibration error: |empirical coverage - nominal coverage|
    calib_error = abs(coverage_overall - conf_level)

    results = {
        "coverage_overall":  float(coverage_overall),
        "coverage_per_mol":  dict(zip(MOLECULES, coverage_per_mol.tolist())),
        "sharpness_overall": float(sharpness_overall),
        "sharpness_per_mol": dict(zip(MOLECULES, sharpness_per_mol.tolist())),
        "calibration_error": float(calib_error),
        "nominal_conf":      conf_level,
    }

    print(f"\n{'='*55}")
    print(f"  MC Dropout Calibration Report ({conf_level*100:.0f}% CI)")
    print(f"{'='*55}")
    print(f"  Overall Coverage:  {coverage_overall*100:.1f}%  (target: {conf_level*100:.0f}%)")
    print(f"  Calibration Error: {calib_error*100:.2f}%")
    print(f"  Mean CI Width:     {sharpness_overall:.4f} log10 VMR")
    print(f"\n  Per-molecule coverage:")
    for mol, cov in zip(MOLECULES, coverage_per_mol):
        status = "✓" if abs(cov - conf_level) < 0.05 else "✗"
        print(f"    {mol:6s}  {cov*100:5.1f}%  {status}")
    print(f"{'='*55}\n")

    return results


# ── Predictive Entropy ────────────────────────────────────────────────────────

def predictive_entropy(all_preds: np.ndarray) -> np.ndarray:
    """
    Compute predictive entropy as a summary uncertainty measure.
    Higher entropy = model is more uncertain about this spectrum.

    Uses differential entropy approximation for Gaussian:
        H ≈ 0.5 * log(2πe * σ²)

    Args:
        all_preds : (n_samples, batch, 12)

    Returns:
        entropy : (batch, 12)
    """
    std   = all_preds.std(axis=0)
    var   = std ** 2
    # Avoid log(0)
    var   = np.maximum(var, 1e-10)
    H     = 0.5 * np.log(2 * np.pi * np.e * var)
    return H


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from model.cnn1d import ExoplanetCNN1D
    from config import SPECTRAL_LENGTH, INPUT_CHANNELS, N_AUX

    device = torch.device("cpu")
    model  = ExoplanetCNN1D()

    predictor = MCDropoutPredictor(model, n_samples=50, device=device)

    # Fake batch of 4 spectra
    dummy_spec = torch.randn(4, INPUT_CHANNELS, SPECTRAL_LENGTH)
    dummy_aux  = torch.randn(4, N_AUX)

    result = predictor.predict(dummy_spec, dummy_aux)

    print(f"Mean shape:     {result['mean'].shape}")      # (4, 12)
    print(f"Std shape:      {result['std'].shape}")       # (4, 12)
    print(f"All preds shape:{result['all_preds'].shape}") # (50, 4, 12)

    # Calibration test with fake targets
    fake_targets = np.random.uniform(-7, 0, (4, N_MOLECULES))
    cal = compute_calibration(fake_targets, result["lower"], result["upper"])
    print("Uncertainty module smoke test passed ✓")
