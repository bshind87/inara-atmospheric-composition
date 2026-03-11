# src/cnn1d.py
# ─────────────────────────────────────────────────────────────────────────────
# 1D CNN with MC Dropout for exoplanet atmospheric retrieval.
#
# Architecture:
#   Input : (batch, 3, 4378)  — 3-channel spectrum tensor
#   Aux   : (batch, 8)        — standardised stellar/planetary features
#   Output: (batch, 12)       — log₁₀ VMR for 12 molecules (no activation)
#
# MC Dropout:
#   Keep model.train() at inference time, run N forward passes.
#   Mean = point estimate.  Std = predictive uncertainty.
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import math
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    INPUT_CHANNELS, SPECTRAL_LENGTH, N_MOLECULES, N_AUX,
    CONV_LAYERS, POOL_SIZE, FC_LAYERS, DROPOUT_RATE, MC_SAMPLES, CONFIDENCE,
)


class ConvBlock(nn.Module):
    """Conv1D → BatchNorm → Activation → MaxPool → Dropout"""

    def __init__(self, in_ch, out_ch, kernel, activation='relu',
                 pool=POOL_SIZE, dropout=DROPOUT_RATE):
        super().__init__()
        act_fn = nn.Tanh() if activation == 'tanh' else nn.ReLU()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=kernel // 2),
            nn.BatchNorm1d(out_ch),
            act_fn,
            nn.MaxPool1d(pool),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class ExoplanetCNN(nn.Module):
    """
    1D CNN + auxiliary MLP branch for exoplanet atmospheric retrieval.

    Usage:
        model = ExoplanetCNN()
        out   = model(spectrum, aux)   # (batch, 12)

    MC Dropout inference:
        model.train()   # keep dropout active
        preds = torch.stack([model(spec, aux) for _ in range(200)])  # (200, B, 12)
        mean  = preds.mean(0)
        std   = preds.std(0)
    """

    def __init__(self,
                 in_channels=INPUT_CHANNELS,
                 spectral_len=SPECTRAL_LENGTH,
                 n_aux=N_AUX,
                 n_out=N_MOLECULES,
                 conv_layers=CONV_LAYERS,
                 fc_layers=FC_LAYERS,
                 pool_size=POOL_SIZE,
                 dropout=DROPOUT_RATE):
        super().__init__()

        # ── Convolutional backbone ────────────────────────────────────────────
        conv_blocks = []
        ch_in = in_channels
        for (ch_out, kernel, act) in conv_layers:
            conv_blocks.append(ConvBlock(ch_in, ch_out, kernel, act, pool_size, dropout))
            ch_in = ch_out
        self.conv = nn.Sequential(*conv_blocks)

        # Compute flattened size after all conv+pool layers
        L = spectral_len
        for _ in conv_layers:
            L = L // pool_size
        flat_dim = ch_in * L

        # ── Fully connected head (conv flat + aux) ────────────────────────────
        fc_blocks = []
        in_dim = flat_dim + n_aux
        for out_dim in fc_layers:
            fc_blocks += [
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = out_dim
        self.fc  = nn.Sequential(*fc_blocks)
        self.out = nn.Linear(in_dim, n_out)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, spectrum, aux):
        """
        Args:
            spectrum : (batch, 3, 4378)
            aux      : (batch, 8)
        Returns:
            (batch, 12)  — log₁₀ VMR predictions
        """
        x = self.conv(spectrum)         # (batch, 256, L')
        x = x.flatten(1)               # (batch, flat_dim)
        x = torch.cat([x, aux], dim=1) # (batch, flat_dim + 8)
        x = self.fc(x)
        return self.out(x)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# MC Dropout predictor
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def mc_predict(model: ExoplanetCNN,
               spectrum: torch.Tensor,
               aux: torch.Tensor,
               n_samples: int = MC_SAMPLES,
               confidence: float = CONFIDENCE) -> dict:
    """
    Run MC Dropout inference.

    Args:
        model    : ExoplanetCNN instance
        spectrum : (B, 3, 4378) — batch of spectra
        aux      : (B, 8)       — batch of auxiliary features
        n_samples: number of stochastic forward passes
        confidence: confidence level for credible interval

    Returns dict with keys (all arrays shape (B, 12)):
        mean   : predictive mean
        std    : predictive std  (uncertainty)
        lo     : lower credible interval
        hi     : upper credible interval
        entropy: predictive entropy (scalar per sample)
    """
    model.train()   # keep dropout active
    preds = torch.stack([model(spectrum, aux) for _ in range(n_samples)], dim=0)
    # preds: (n_samples, B, 12)

    alpha  = (1 - confidence) / 2
    lo_q   = alpha
    hi_q   = 1 - alpha

    mean   = preds.mean(dim=0)
    std    = preds.std(dim=0)
    lo     = torch.quantile(preds, lo_q, dim=0)
    hi     = torch.quantile(preds, hi_q, dim=0)

    # Predictive entropy across molecules (mean over batch dim)
    entropy = std.mean(dim=-1)   # (B,)

    return dict(mean=mean, std=std, lo=lo, hi=hi, entropy=entropy)


# ─────────────────────────────────────────────────────────────────────────────
# Quick architecture summary
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    model = ExoplanetCNN()
    print(model)
    print(f'\nTrainable parameters: {model.count_params():,}')

    # Shape check
    B = 4
    spec = torch.randn(B, INPUT_CHANNELS, SPECTRAL_LENGTH)
    aux  = torch.randn(B, N_AUX)
    out  = model(spec, aux)
    print(f'\nInput  spectrum : {tuple(spec.shape)}')
    print(f'Input  aux      : {tuple(aux.shape)}')
    print(f'Output          : {tuple(out.shape)}   (expected: ({B}, {N_MOLECULES}))')

    # MC Dropout check
    result = mc_predict(model, spec, aux, n_samples=10)
    print(f'\nMC Dropout (10 passes):')
    print(f'  mean   : {tuple(result["mean"].shape)}')
    print(f'  std    : {tuple(result["std"].shape)}')
    print(f'  entropy: {tuple(result["entropy"].shape)}')
    print('Architecture check passed ✓')
