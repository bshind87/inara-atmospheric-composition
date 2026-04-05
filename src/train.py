# src/train.py (UPDATED)

import os
import sys
import time
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import (
    CHECKPOINT_DIR, RESULTS_DIR, MOLECULES, N_MOLECULES,
    LR, WEIGHT_DECAY, LR_PATIENCE, LR_FACTOR, EARLY_STOP_PAT,
    MAX_EPOCHS, BATCH_SIZE, RANDOM_SEED,
)
from dataset import get_dataloaders
from cnn1d import ExoplanetCNN

BEST_CKPT = os.path.join(CHECKPOINT_DIR, 'best_model.pt')
LAST_CKPT = os.path.join(CHECKPOINT_DIR, 'last_model.pt')


# ─────────────────────────────────────────────────────────────
# NaN-safe loss
# ─────────────────────────────────────────────────────────────
def criterion(pred, target):
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return pred.sum() * 0.0
    return ((pred[mask] - target[mask]) ** 2).mean()


# ─────────────────────────────────────────────────────────────
def train(n_total=None, resume=None):
    torch.manual_seed(RANDOM_SEED)

    device = torch.device(
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    )
    print(f'Device: {device}')

    # ── Data ─────────────────────────────────────────────────
    train_loader, val_loader, test_loader, scaler, wavelengths = \
        get_dataloaders(n_total=n_total)

    print(f'Train: {len(train_loader.dataset):,}  '
          f'Val: {len(val_loader.dataset):,}  '
          f'Test: {len(test_loader.dataset):,}')

    # ── Model ────────────────────────────────────────────────
    model = ExoplanetCNN().to(device)
    print(f'Parameters: {model.count_params():,}')

    optimiser = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, factor=LR_FACTOR, patience=LR_PATIENCE
    )

    # ── History ──────────────────────────────────────────────
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_rmse_per_mol': [],
        'val_mae_per_mol': [],
        'val_r2_per_mol': [],
        'global_rmse': [],
        'global_mae': [],
        'global_r2': []
    }

    start_epoch = 1
    best_val_loss = float('inf')
    patience_ctr = 0
    prev_lr = LR

    # ── Resume ───────────────────────────────────────────────
    if resume and os.path.exists(resume):
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimiser.load_state_dict(ckpt['optimiser'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('best_val_loss', best_val_loss)
        history = ckpt.get('history', history)
        print(f'Resumed from {resume}')

    # ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        t0 = time.time()

        # ── TRAIN ─────────────────────────────────────────────
        model.train()
        train_loss = 0.0

        for spec, aux, targets in train_loader:
            spec, aux, targets = spec.to(device), aux.to(device), targets.to(device)

            optimiser.zero_grad()
            pred = model(spec, aux)
            loss = criterion(pred, targets)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()

            train_loss += loss.item() * len(spec)

        train_loss /= len(train_loader.dataset)

        # ── VALIDATION ────────────────────────────────────────
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

        # Convert to numpy
        all_preds = torch.cat(all_preds).numpy()
        all_tgts = torch.cat(all_tgts).numpy()

        # ── DEBUG: target distribution per molecule ───────────────────
        if epoch == 1:  # only print once to avoid spam
            print("\n=== TARGET DISTRIBUTION (VALIDATION) ===")
            for i, mol in enumerate(MOLECULES):
                y_true = all_tgts[:, i]
                mask = ~np.isnan(y_true)
                y_true = y_true[mask]

                print(f"{mol}: "
                      f"mean={np.mean(y_true):.3f}, "
                      f"std={np.std(y_true):.6f}, "
                      f"min={np.min(y_true):.3f}, "
                      f"max={np.max(y_true):.3f}")


        # ── Metrics ──────────────────────────────────────────
        rmse_per_mol, mae_per_mol, r2_per_mol = [], [], []

        for i in range(N_MOLECULES):
            mask = ~np.isnan(all_tgts[:, i])
            if mask.sum() == 0:
                rmse_per_mol.append(np.nan)
                mae_per_mol.append(np.nan)
                r2_per_mol.append(np.nan)
                continue

            y_true = all_tgts[mask, i]
            y_pred = all_preds[mask, i]

            rmse_per_mol.append(np.sqrt(np.mean((y_pred - y_true) ** 2)))
            mae_per_mol.append(np.mean(np.abs(y_pred - y_true)))

            if len(y_true) > 1:
                r2_per_mol.append(r2_score(y_true, y_pred))
            else:
                r2_per_mol.append(np.nan)

        rmse_per_mol = np.array(rmse_per_mol)
        mae_per_mol  = np.array(mae_per_mol)
        r2_per_mol   = np.array(r2_per_mol)

        # ── Global metrics ───────────────────────────────────
        mask = ~np.isnan(all_tgts)

        global_rmse = np.sqrt(np.mean((all_preds[mask] - all_tgts[mask])**2))
        global_mae  = np.mean(np.abs(all_preds[mask] - all_tgts[mask]))

        # Compute per-molecule R² properly
        valid_r2 = [r for r in r2_per_mol if not np.isnan(r)]

        if len(valid_r2) > 0:
            global_r2 = np.mean(valid_r2)
        else:
            global_r2 = np.nan

        #try:
        #    global_r2 = r2_score(all_tgts.reshape(-1), all_preds.reshape(-1))
        #except:
        #    global_r2 = np.nan

        # ── Save history ─────────────────────────────────────
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_rmse_per_mol'].append(rmse_per_mol.tolist())
        history['val_mae_per_mol'].append(mae_per_mol.tolist())
        history['val_r2_per_mol'].append(r2_per_mol.tolist())
        history['global_rmse'].append(float(global_rmse))
        history['global_mae'].append(float(global_mae))
        history['global_r2'].append(float(global_r2))

        scheduler.step(val_loss)

        # LR logging
        cur_lr = optimiser.param_groups[0]['lr']
        lr_tag = f'  lr→{cur_lr:.2e}' if cur_lr != prev_lr else ''
        prev_lr = cur_lr

        elapsed = time.time() - t0

        # ── Print ────────────────────────────────────────────
        print(
            f'Ep {epoch:03d}/{MAX_EPOCHS}  '
            f'train={train_loss:.4f}  val={val_loss:.4f}  '
            f'RMSE={global_rmse:.4f}  MAE={global_mae:.4f}  R2={global_r2:.3f}  '
            f'best_mol={MOLECULES[np.nanargmin(rmse_per_mol)]}({np.nanmin(rmse_per_mol):.3f})  '
            f'worst_mol={MOLECULES[np.nanargmax(rmse_per_mol)]}({np.nanmax(rmse_per_mol):.3f})  '
            f't={elapsed:.1f}s{lr_tag}'
        )

        # ── Checkpoint ───────────────────────────────────────
        ckpt = dict(
            epoch=epoch,
            model=model.state_dict(),
            optimiser=optimiser.state_dict(),
            best_val_loss=best_val_loss,
            history=history,
            scaler=scaler,
            wavelengths=wavelengths
        )

        torch.save(ckpt, LAST_CKPT)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, BEST_CKPT)
            patience_ctr = 0
            print(f'  → New best val loss: {best_val_loss:.4f} (saved)')
        else:
            patience_ctr += 1
            if patience_ctr >= EARLY_STOP_PAT:
                print(f'Early stopping at epoch {epoch}')
                break

    # ── Save history ─────────────────────────────────────────
    hist_path = os.path.join(RESULTS_DIR, 'training_history.json')
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f'\nHistory saved to {hist_path}')
    print(f'Best checkpoint: {BEST_CKPT}')

    # ── BEST PER-MOLECULE METRICS (VALIDATION) ─────────────────────
    print("\n=== BEST PER-MOLECULE (VALIDATION) ===")

    rmse_hist = np.array(history['val_rmse_per_mol'])  # (epochs, 12)
    mae_hist = np.array(history['val_mae_per_mol'])
    r2_hist = np.array(history['val_r2_per_mol'])

    best_rmse = np.nanmin(rmse_hist, axis=0)
    best_mae = np.nanmin(mae_hist, axis=0)
    best_r2 = np.nanmax(r2_hist, axis=0)

    for i, mol in enumerate(MOLECULES):
        # get all true values for this molecule
        y_true = all_tgts[:, i]

        # remove NaNs
        mask = ~np.isnan(y_true)
        y_true = y_true[mask]

        var = np.var(y_true)

        print(f'{mol:<6} → RMSE={best_rmse[i]:.3f}  '
              f'MAE={best_mae[i]:.3f}  R2={best_r2[i]:.3f}  '
              f'Var={var:.6f}')


    # ── BEST GLOBAL METRICS ───────────────────────────────────────
    print("\n=== BEST GLOBAL (VALIDATION) ===")

    best_global_rmse = np.nanmin(history['global_rmse'])
    best_global_mae = np.nanmin(history['global_mae'])
    best_global_r2 = np.nanmax(history['global_r2'])

    print(f'RMSE: {best_global_rmse:.4f}')
    print(f'MAE : {best_global_mae:.4f}')
    print(f'R2  : {best_global_r2:.4f}')

    return model, history


# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=None)
    ap.add_argument('--resume', default=None)
    args = ap.parse_args()

    train(n_total=args.n, resume=args.resume)