# src/dataset.py
# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset + DataLoader factory for INARA.
#
# Reads processed numpy arrays produced by download_inara.py:
#   inara_data/processed/
#   ├── spectra.npy     (N, 3, 4378)
#   ├── molecules.npy   (N, 12)
#   └── aux_params.npy  (N, 8)
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    DATA_DIR, MOLECULES, N_MOLECULES, N_AUX, INPUT_CHANNELS, SPECTRAL_LENGTH,
    LOG_VMR_MIN, LOG_VMR_MAX, BATCH_SIZE, RANDOM_SEED,
    TRAIN_FRAC, VAL_FRAC, TEST_FRAC,
)

PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')


class INARADataset(Dataset):
    """
    PyTorch Dataset for INARA spectra.

    Each item returns (spectrum, aux, targets):
        spectrum : FloatTensor (3, 4378)
        aux      : FloatTensor (8,)
        targets  : FloatTensor (12,)  — log₁₀ VMR, clipped to [−8, 0]

    Args:
        spectra    : (N, 3, 4378) float32
        molecules  : (N, 12)      float32 — log₁₀ VMR labels
        aux_params : (N, 8)       float32 — stellar/planetary features
        aux_scaler : fitted StandardScaler  (None → fit on this split)
        augment    : bool — add Gaussian noise to Ch0/Ch1 during training
        noise_std  : augmentation noise level
    """

    def __init__(self, spectra, molecules, aux_params,
                 aux_scaler=None, augment=False, noise_std=0.005):
        assert spectra.shape[1] == INPUT_CHANNELS, \
            f'Expected {INPUT_CHANNELS} channels, got {spectra.shape[1]}'
        assert spectra.shape[2] == SPECTRAL_LENGTH, \
            f'Expected {SPECTRAL_LENGTH} wavelength points, got {spectra.shape[2]}'
        assert molecules.shape[1] == N_MOLECULES

        self.spectra  = spectra.astype(np.float32)
        self.targets  = np.clip(molecules.astype(np.float32), LOG_VMR_MIN, LOG_VMR_MAX)
        self.augment  = augment
        self.noise_std= noise_std

        # Fit or apply StandardScaler on auxiliary features
        aux = aux_params.astype(np.float32)
        if aux_scaler is None:
            self.scaler = StandardScaler()
            self.aux    = self.scaler.fit_transform(aux)
        else:
            self.scaler = aux_scaler
            self.aux    = self.scaler.transform(aux)

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, idx):
        spec = self.spectra[idx].copy()   # (3, 4378)

        # Augment Ch0 and Ch1 only — Ch2 (transit depth) stays clean
        if self.augment and self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, spec.shape).astype(np.float32)
            noise[2] = 0.0
            spec += noise

        return (
            torch.tensor(spec,           dtype=torch.float32),
            torch.tensor(self.aux[idx],  dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32),
        )


def get_dataloaders(processed_dir=PROCESSED_DIR,
                    n_total=None,
                    seed=RANDOM_SEED):
    """
    Load processed arrays, split into train/val/test, return DataLoaders.

    If processed files don't exist, auto-generates synthetic demo data.

    Returns:
        train_loader, val_loader, test_loader, aux_scaler, wavelengths
    """
    spec_path = os.path.join(processed_dir, 'spectra.npy')

    if not os.path.exists(spec_path):
        print(f'Processed data not found at {processed_dir}')
        print('Generating demo data — run:  python src/download_inara.py  for real data')
        from download_inara import generate_demo
        generate_demo(processed_dir, n_samples=max(n_total or 0, 25_000))

    spectra   = np.load(os.path.join(processed_dir, 'spectra.npy'),    mmap_mode='r')
    molecules = np.load(os.path.join(processed_dir, 'molecules.npy'),  mmap_mode='r')
    aux       = np.load(os.path.join(processed_dir, 'aux_params.npy'), mmap_mode='r')

    wl_path = os.path.join(processed_dir, 'wavelengths.npy')
    wavelengths = np.load(wl_path) if os.path.exists(wl_path) else None

    N = len(spectra)
    if n_total:
        N = min(N, n_total)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(spectra))[:N]

    n_train = int(N * TRAIN_FRAC)
    n_val   = int(N * VAL_FRAC)
    tr_idx, vl_idx, ts_idx = idx[:n_train], idx[n_train:n_train+n_val], idx[n_train+n_val:]

    # Load subsets into memory (mmap slices)
    def load(i): return np.array(spectra[i]), np.array(molecules[i]), np.array(aux[i])

    tr_spec, tr_mol, tr_aux = load(tr_idx)
    vl_spec, vl_mol, vl_aux = load(vl_idx)
    ts_spec, ts_mol, ts_aux = load(ts_idx)

    train_ds = INARADataset(tr_spec, tr_mol, tr_aux, augment=True)
    scaler   = train_ds.scaler
    val_ds   = INARADataset(vl_spec, vl_mol, vl_aux, aux_scaler=scaler)
    test_ds  = INARADataset(ts_spec, ts_mol, ts_aux, aux_scaler=scaler)

    print(f'Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}')

    kw = dict(batch_size=BATCH_SIZE, num_workers=4, pin_memory=True, persistent_workers=True)
    return (
        DataLoader(train_ds, shuffle=True,  **kw),
        DataLoader(val_ds,   shuffle=False, **kw),
        DataLoader(test_ds,  shuffle=False, **kw),
        scaler,
        wavelengths,
    )
