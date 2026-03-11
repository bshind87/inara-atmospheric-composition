# src/train.py
# ─────────────────────────────────────────────────────────────────────────────
# Training loop for ExoplanetCNN.
#
# Usage:
#   python src/train.py                        # train on demo data
#   python src/train.py --n 500000             # train on 500K real samples
#   python src/train.py --resume checkpoints/best.pt
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    CHECKPOINT_DIR, RESULTS_DIR, MOLECULES, N_MOLECULES,
    LR, WEIGHT_DECAY, LR_PATIENCE, LR_FACTOR, EARLY_STOP_PAT,
    MAX_EPOCHS, BATCH_SIZE, RANDOM_SEED,
)
from dataset  import get_dataloaders, PROCESSED_DIR
from cnn1d    import ExoplanetCNN

BEST_CKPT = os.path.join(CHECKPOINT_DIR, 'best_model.pt')
LAST_CKPT = os.path.join(CHECKPOINT_DIR, 'last_model.pt')


def train(n_total=None, resume=None):
    torch.manual_seed(RANDOM_SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── Data ─────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, scaler, wavelengths = \
        get_dataloaders(n_total=n_total)

    n_train = len(train_loader.dataset)
    if n_train < 500:
        print(f'\n⚠️  Only {n_train} training samples.')
        print('   First generate labels: python src/generate_parameters.py --fill_prior')
        print('   Then process data:     python src/download_inara.py')
        print('   Then re-run training.\n')

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = ExoplanetCNN().to(device)
    print(f'Parameters: {model.count_params():,}')

    optimiser = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # verbose= was removed in PyTorch 2.2 — log LR changes manually instead
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, factor=LR_FACTOR, patience=LR_PATIENCE)

    # NaN-safe MSE: skips targets that are NaN
    # (molecules not recoverable from spectrum have NaN labels unless --fill_prior used)
    def criterion(pred, target):
        mask = ~torch.isnan(target)
        if mask.sum() == 0:
            return pred.sum() * 0.0
        return ((pred[mask] - target[mask]) ** 2).mean()

    start_epoch  = 1
    best_val_loss= float('inf')
    patience_ctr = 0
    prev_lr      = LR
    history      = {'train_loss': [], 'val_loss': [], 'val_rmse_per_mol': []}

    # ── Resume ────────────────────────────────────────────────────────────────
    if resume and os.path.exists(resume):
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimiser.load_state_dict(ckpt['optimiser'])
        start_epoch   = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('best_val_loss', best_val_loss)
        history       = ckpt.get('history', history)
        print(f'Resumed from {resume} at epoch {start_epoch - 1}')

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        t0 = time.time()

        # Train
        model.train()
        train_loss = 0.0
        for spec, aux, targets in train_loader:
            spec, aux, targets = spec.to(device), aux.to(device), targets.to(device)
            optimiser.zero_grad()
            pred = model(spec, aux)
            loss = criterion(pred, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimiser.step()
            train_loss += loss.item() * len(spec)
        train_loss /= len(train_loader.dataset)

        # Validate
        model.eval()
        val_loss = 0.0
        all_preds, all_tgts = [], []
        with torch.no_grad():
            for spec, aux, targets in val_loader:
                spec, aux, targets = spec.to(device), aux.to(device), targets.to(device)
                pred = model(spec, aux)
                val_loss += criterion(pred, targets).item() * len(spec)
                all_preds.append(pred.cpu())
                all_tgts.append(targets.cpu())
        val_loss /= len(val_loader.dataset)

        # Per-molecule RMSE (NaN-safe — ignore molecules with NaN targets)
        all_preds = torch.cat(all_preds).numpy()
        all_tgts  = torch.cat(all_tgts).numpy()
        with np.errstate(invalid='ignore'):
            rmse_per_mol = np.sqrt(np.nanmean((all_preds - all_tgts)**2, axis=0))  # (12,)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_rmse_per_mol'].append(rmse_per_mol.tolist())

        scheduler.step(val_loss)
        elapsed = time.time() - t0

        # Log LR changes manually (verbose= removed in PyTorch 2.2)
        cur_lr = optimiser.param_groups[0]['lr']
        lr_tag = f'  lr→{cur_lr:.2e}' if cur_lr != prev_lr else ''
        prev_lr = cur_lr

        print(f'Ep {epoch:03d}/{MAX_EPOCHS}  '
              f'train={train_loss:.4f}  val={val_loss:.4f}  '
              f'best_mol={MOLECULES[rmse_per_mol.argmin()]}({rmse_per_mol.min():.3f})  '
              f'worst_mol={MOLECULES[rmse_per_mol.argmax()]}({rmse_per_mol.max():.3f})  '
              f't={elapsed:.1f}s{lr_tag}')

        # Checkpoint
        ckpt = dict(epoch=epoch, model=model.state_dict(),
                    optimiser=optimiser.state_dict(),
                    best_val_loss=best_val_loss, history=history,
                    scaler=scaler, wavelengths=wavelengths)
        torch.save(ckpt, LAST_CKPT)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, BEST_CKPT)
            patience_ctr = 0
            print(f'  → New best val loss: {best_val_loss:.4f}  (saved)')
        else:
            patience_ctr += 1
            if patience_ctr >= EARLY_STOP_PAT:
                print(f'Early stopping at epoch {epoch} (patience={EARLY_STOP_PAT})')
                break

    # ── Final per-molecule RMSE ───────────────────────────────────────────────
    print('\nPer-molecule validation RMSE (best epoch):')
    best_rmse = np.array(history['val_rmse_per_mol']).min(axis=0)
    for mol, rmse in zip(MOLECULES, best_rmse):
        bar = '█' * int(rmse * 20)
        print(f'  {mol:<6} {rmse:.3f}  {bar}')

    # Save training history
    import json
    hist_path = os.path.join(RESULTS_DIR, 'training_history.json')
    with open(hist_path, 'w') as f:
        json.dump({k: (v if isinstance(v[0], float) else
                       [x if isinstance(x, float) else x for x in v])
                   for k, v in history.items()}, f, indent=2)
    print(f'\nHistory saved to {hist_path}')
    print(f'Best checkpoint: {BEST_CKPT}')
    return model, history


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n',      type=int, default=None,
                    help='Total samples to use (default: all in processed/)')
    ap.add_argument('--resume', default=None, metavar='CKPT',
                    help='Resume from checkpoint path')
    args = ap.parse_args()
    train(n_total=args.n, resume=args.resume)
