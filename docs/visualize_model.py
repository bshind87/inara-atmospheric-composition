#!/usr/bin/env python
"""
Generate a publication-quality architecture diagram for 1D CNN.
Output: docs/resnet_architecture.png  (and .pdf)

Run:  python docs/visualize_model.py

Architecture being visualised (matches deep_model.py exactly):
  Input  : (B, 12, 101)  — 12 CLIMA channels × 101 altitude levels
  Block 1: Conv1d(12→32,  k=9, s=2, p=4) + BN + ReLU + MaxPool1d(2) → (B, 32, 25)
  Block 2: Conv1d(32→64,  k=7, s=2, p=3) + BN + ReLU + MaxPool1d(2) → (B, 64,  6)
  Block 3: Conv1d(64→128, k=5, s=2, p=2) + BN + ReLU + MaxPool1d(2) → (B, 128, 1)
  Block 4: Conv1d(128→256,k=3, s=1, p=1) + BN + ReLU                → (B, 256, 1)
  Pool   : AdaptiveAvgPool1d(1) + Flatten                            → (B, 256)
  Shared : Dropout(0.25) + FC(256→128) + LayerNorm + ReLU           → (B, 128)
  Heads  : 12 × molecule-specific MLP → scalar log10 abundance
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

# ── Colour palette ─────────────────────────────────────────────────────────────
C_INPUT   = '#2D4A2D'   # dark green  — input
C_B1      = '#1A4A7A'   # dark navy   — Block 1
C_B2      = '#2166AC'   # steel blue  — Block 2
C_B3      = '#4393C3'   # mid blue    — Block 3
C_B4      = '#74ADD1'   # light blue  — Block 4
C_POOL    = '#5B8C5A'   # forest green — pooling
C_SHARED  = '#8B4A8B'   # purple      — shared FC
C_HEAD    = '#B2182B'   # crimson     — molecule heads
C_OUTPUT  = '#7A3A00'   # brown       — output
C_BG      = '#F7F9FC'
C_TEXT_LIGHT = '#FFFFFF'
C_TEXT_DARK  = '#1A1A2E'
C_ARROW   = '#555555'
C_BORDER  = '#CCCCCC'

FONT = 'DejaVu Sans'

fig = plt.figure(figsize=(22, 14), facecolor=C_BG)
ax  = fig.add_axes([0, 0, 1, 1], facecolor=C_BG)
ax.set_xlim(0, 22)
ax.set_ylim(0, 14)
ax.axis('off')

# ── Helpers ──────────────────────────────────────────────────────────────────
def rbox(ax, x, y, w, h, color, text_lines, text_color=C_TEXT_LIGHT,
         fontsize=9, border=None):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0",
                         facecolor=color, edgecolor=border or color,
                         linewidth=1.2, zorder=3)
    ax.add_patch(box)
    n = len(text_lines)
    for i, (line, fs, bold) in enumerate(text_lines):
        ty = y + h/2 + (n/2 - i - 0.5) * (fs * 0.014 + 0.05)
        ax.text(x + w/2, ty, line,
                ha='center', va='center',
                fontsize=fs, fontweight='bold' if bold else 'normal',
                color=text_color, fontfamily=FONT, zorder=4)

def arrow(ax, x1, y1, x2, y2, color=C_ARROW, lw=1.8):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw, connectionstyle='arc3,rad=0.0'),
                zorder=5)

def label(ax, x, y, text, color=C_TEXT_DARK, fs=8.5, ha='center', bold=False):
    ax.text(x, y, text, ha=ha, va='center', fontsize=fs,
            color=color, fontfamily=FONT,
            fontweight='bold' if bold else 'normal', zorder=6)

def dim_pill(ax, x, y, text, color='#EEEEEE', tc='#333333'):
    box = FancyBboxPatch((x - 0.55, y - 0.17), 1.1, 0.34,
                         boxstyle="round,pad=0.05",
                         facecolor=color, edgecolor=tc, linewidth=0.8, zorder=5)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center',
            fontsize=7.5, color=tc, fontfamily=FONT, zorder=6)

# ════════════════════════════════════════════════════════════════════════════════
# TITLE
# ════════════════════════════════════════════════════════════════════════════════
ax.text(11, 13.5, '1D CNN — 1D Convolutional Network for Atmospheric Retrieval',
        ha='center', va='center', fontsize=16, fontweight='bold',
        color=C_TEXT_DARK, fontfamily=FONT)
ax.text(11, 13.05,
        'Input: CLIMA profile  (Batch × 12 channels × 101 altitude levels)'
        '   →   Output: 12 log₁₀ molecular abundances',
        ha='center', va='center', fontsize=10, color='#555555', fontfamily=FONT)

ax.plot([0.5, 21.5], [12.72, 12.72], color=C_BORDER, lw=1.2)

# ════════════════════════════════════════════════════════════════════════════════
# MAIN VERTICAL FLOW  (left column)
# ════════════════════════════════════════════════════════════════════════════════
BX, BW = 0.55, 4.2   # block x, width
SX     = 5.2          # shape pill x
DX     = 6.8          # description x

# y-centres from top to bottom
YS = [12.15, 10.9, 9.55, 8.2, 6.85, 5.5, 4.3, 3.1]
BH = 0.85

blocks = [
    (C_INPUT,  'INPUT',          'CLIMA Profile'),
    (C_B1,     'CONV BLOCK 1',   'Conv1d(12→32, k=9, s=2) + BN + ReLU + MaxPool1d(2)'),
    (C_B2,     'CONV BLOCK 2',   'Conv1d(32→64, k=7, s=2) + BN + ReLU + MaxPool1d(2)'),
    (C_B3,     'CONV BLOCK 3',   'Conv1d(64→128, k=5, s=2) + BN + ReLU + MaxPool1d(2)'),
    (C_B4,     'CONV BLOCK 4',   'Conv1d(128→256, k=3, s=1) + BN + ReLU'),
    (C_POOL,   'GLOBAL AVG POOL','AdaptiveAvgPool1d(1) → Flatten'),
    (C_SHARED, 'SHARED FC',      'Dropout(0.25) → FC(256→128) → LayerNorm → ReLU'),
    (C_HEAD,   'HEADS ×12',      '12 × molecule MLP → log₁₀ abundance'),
]

shapes = [
    ('B, 12, 101', '#2D4A2D', '#AADDAA'),
    ('B, 32, 25',  C_B1,      '#BDD5EE'),
    ('B, 64, 6',   C_B2,      '#BDD5EE'),
    ('B, 128, 1',  C_B3,      '#BDD5EE'),
    ('B, 256, 1',  C_B4,      '#BDD5EE'),
    ('B, 256',     C_POOL,    '#C8E6C9'),
    ('B, 128',     C_SHARED,  '#E1BEE7'),
    ('B, 12',      C_HEAD,    '#FFCDD2'),
]

descs = [
    '12 CLIMA channels   ×   101 altitude levels   (Z-normalised per channel)',
    'kernel=9 captures broad altitude patterns   |   stride=2 + MaxPool halves resolution → 25 pts',
    'kernel=7   |   resolution reduced to 6 pts   |   channel depth doubled to 64',
    'kernel=5   |   resolution collapsed to 1 pt  |   channel depth 128',
    'kernel=3, stride=1   |   no resolution change   |   expands to 256 channels',
    'Squeeze altitude dimension  →  single 256-d vector per sample',
    'Shared representation for all 12 molecule heads   (128-d)',
    'Each head: FC(128→hidden→1)   |   output: log₁₀ surface volume mixing ratio',
]

for i, ((col, title, sub), (shape, sc, pill_c), desc) in enumerate(
        zip(blocks, shapes, descs)):
    y_top = YS[i] - BH/2
    rbox(ax, BX, y_top, BW, BH, col,
         [(title, 10, True), (sub, 8.0, False)])
    dim_pill(ax, SX + 0.6, YS[i], shape, pill_c, sc)
    label(ax, DX, YS[i] + 0.06, desc, color=C_TEXT_DARK, fs=8.5, ha='left')
    if i < len(YS) - 1:
        arrow(ax, BX + BW/2, y_top, BX + BW/2, YS[i+1] + BH/2 + 0.02,
              color=C_ARROW, lw=2.0)

# Column headers
label(ax, BX + BW/2, 12.52, 'LAYER',        color='#888888', fs=8, bold=True)
label(ax, SX + 0.6,  12.52, 'OUTPUT SHAPE', color='#888888', fs=8, bold=True)
label(ax, DX + 3.0,  12.52, 'DESCRIPTION',  color='#888888', fs=8, bold=True)
ax.plot([0.5, 14.2], [12.38, 12.38], color=C_BORDER, lw=0.8, ls='--')

# ════════════════════════════════════════════════════════════════════════════════
# CONV BLOCK DETAIL  (right panel, upper)
# ════════════════════════════════════════════════════════════════════════════════
RX = 14.6

panel = FancyBboxPatch((RX - 0.2, 4.5), 7.3, 7.9,
                       boxstyle="round,pad=0.15",
                       facecolor='#FFFFFF', edgecolor='#CCCCCC',
                       linewidth=1.5, zorder=2)
ax.add_patch(panel)
label(ax, RX + 3.3, 12.18, 'Conv Block — Detail', color=C_B1, fs=12, bold=True)
ax.plot([RX - 0.05, RX + 6.9], [11.9, 11.9], color=C_BORDER, lw=1.0)

# Conv block internal flow (sequential — NO skip connection)
RBX = RX + 0.6
RBW = 3.8
cb_blocks = [
    (RBX, 11.35, RBW, 0.58, C_B1,   'CONV 1D',    'kernel=k, stride=s, padding=p'),
    (RBX, 10.45, RBW, 0.58, C_B1,   'BATCH NORM', 'BatchNorm1d(out_channels)'),
    (RBX,  9.55, RBW, 0.58, '#4393C3', 'ReLU',     'in-place activation'),
    (RBX,  8.65, RBW, 0.58, C_POOL, 'MAX POOL',   'MaxPool1d(kernel=2, stride=2)'),
]
# Block 4 has no MaxPool — add a note
note_b4 = '(Block 4 omits MaxPool — stride=1 preserves 1-pt sequence)'

for (rx, ry, rw, rh, rc, rt, rs) in cb_blocks:
    rbox(ax, rx, ry, rw, rh, rc,
         [(rt, 9.5, True), (rs, 8, False)])

# Arrows inside conv block
for i in range(len(cb_blocks) - 1):
    _, y_cur, _, h_cur, _, _, _ = cb_blocks[i]
    _, y_nxt, _, _, _, _, _     = cb_blocks[i+1]
    ax.annotate('', xy=(RBX + RBW/2, y_nxt + 0.58),
                xytext=(RBX + RBW/2, y_cur),
                arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=1.6), zorder=5)

# Input / output labels
label(ax, RBX + RBW/2, cb_blocks[0][1] + 0.58 + 0.2,
      '↓  x  (input tensor)', color='#555555', fs=8.5)
label(ax, RBX + RBW/2, cb_blocks[-1][1] - 0.22,
      '↓  output', color='#555555', fs=8.5)

# Block 4 note box
rbox(ax, RBX - 0.1, 7.6, RBW + 0.2, 0.6,
     '#FFF3E0',
     [('Block 4 only — no MaxPool', 8, True),
      ('stride=1, sequence stays at 1 pt', 7.5, False)],
     text_color='#7A4A00', border='#E0A040')

# Sequential flow equation
label(ax, RX + 3.3, 7.1, 'output = MaxPool( ReLU( BN( Conv1d(x) ) ) )',
      color=C_B1, fs=10, bold=True, ha='center')
label(ax, RX + 3.3, 6.75, 'No residual / skip connections  —  pure sequential',
      color='#555555', fs=9, ha='center')

# ════════════════════════════════════════════════════════════════════════════════
# PER-MOLECULE HEADS  (bottom right panel)
# ════════════════════════════════════════════════════════════════════════════════
HP = FancyBboxPatch((RX - 0.2, 0.35), 7.3, 3.85,
                    boxstyle="round,pad=0.15",
                    facecolor='#FFF5F5', edgecolor='#FFAAAA',
                    linewidth=1.5, zorder=2)
ax.add_patch(HP)
label(ax, RX + 3.3, 3.98, 'Per-Molecule Output Heads (×12)', color=C_HEAD, fs=11, bold=True)
ax.plot([RX - 0.05, RX + 6.9], [3.72, 3.72], color='#FFAAAA', lw=1.0)

rbox(ax, RX + 0.5, 3.25, 5.6, 0.44,
     '#E8D0E8',
     [('Shared 128-d Representation  →  broadcasts to all 12 heads', 9, True)],
     text_color='#5A0A5A', border='#AA66AA')

# 12 molecule heads
mols = ['H₂O', 'CO₂', 'O₂', 'O₃', 'CH₄', 'N₂', 'N₂O', 'CO', 'H₂', 'H₂S', 'SO₂', 'NH₃']
head_hidden = {
    'H₂O': '[128]', 'CO₂': '[64]',  'O₂': '[64]',  'O₃': '[128]',
    'CH₄': '[128]', 'N₂':  '[64]',  'N₂O': '[128]', 'CO': '[128]',
    'H₂':  '[128]', 'H₂S': '[128]', 'SO₂': '[128]', 'NH₃': '[128]',
}
head_drop = {
    'H₂O': 0.20, 'CO₂': 0.15, 'O₂': 0.15, 'O₃': 0.20,
    'CH₄': 0.20, 'N₂':  0.15, 'N₂O': 0.20, 'CO': 0.20,
    'H₂':  0.20, 'H₂S': 0.25, 'SO₂': 0.25, 'NH₃': 0.25,
}
N_HEADS = 12
head_w = (7.3 - 0.5) / N_HEADS
for i, mol in enumerate(mols):
    hx = RX - 0.2 + 0.25 + i * head_w
    hy = 1.65
    hw = head_w - 0.08
    drop = head_drop[mol]
    hidden = head_hidden[mol]
    col_h = '#D45A6A' if drop >= 0.25 else '#E8909A'
    rbox(ax, hx, hy, hw, 1.38, col_h,
         [(mol, 8.5, True),
          (hidden, 7, False),
          (f'p={drop}', 6.5, False)],
         text_color='white', border=C_HEAD)
    ax.annotate('', xy=(hx + hw/2, hy + 1.38),
                xytext=(RX + 0.5 + 5.6 * (i + 0.5)/N_HEADS, 3.25),
                arrowprops=dict(arrowstyle='->', color='#AA66AA', lw=0.9), zorder=5)
    label(ax, hx + hw/2, hy - 0.22, 'scalar', color='#888888', fs=7)

label(ax, RX + 3.3, 0.75, 'Output: (Batch, 12)   —   log₁₀ surface volume mixing ratios',
      color=C_HEAD, fs=9.5, bold=True, ha='center')

# Legend
rbox(ax, RX + 0.3, 0.42, 1.5, 0.28, '#E8909A',
     [('hidden=[128], p≤0.20', 7.5, False)], text_color='white')
rbox(ax, RX + 2.0, 0.42, 1.5, 0.28, '#D45A6A',
     [('hidden=[128], p=0.25', 7.5, False)], text_color='white')
label(ax, RX + 4.2, 0.56, '← Trace molecules (H₂S, SO₂, NH₃) get higher dropout',
      color='#888888', fs=7.5, ha='left')

# ════════════════════════════════════════════════════════════════════════════════
# BOTTOM SECTION — main flow arrow to heads
# ════════════════════════════════════════════════════════════════════════════════
shared_bottom = YS[-1] - BH/2   # bottom of HEADS ×12 block in main column
arrow(ax, BX + BW/2, shared_bottom, BX + BW/2, 2.05, color=C_HEAD, lw=2.0)
rbox(ax, BX, 1.55, BW, 0.6,
     C_HEAD,
     [('OUTPUT  (B, 12)', 10, True),
      ('log₁₀ molecular abundances', 8.5, False)])

# ════════════════════════════════════════════════════════════════════════════════
# SIDE ANNOTATIONS — loss weights & augmentation
# ════════════════════════════════════════════════════════════════════════════════
lw_x = 0.55
rbox(ax, lw_x, 2.0, 4.2, 0.9,
     '#FFF8E1',
     [('Loss: Weighted MSE', 9, True),
      ('w ∈ {1.0 (N₂) … 2.0 (SO₂, NH₃)}', 8.5, False),
      ('Upweights trace / hard molecules', 8, False)],
     text_color='#5A3E00', border='#E0B040')

rbox(ax, lw_x, 1.0, 4.2, 0.85,
     '#E8F5E9',
     [('Training augmentation', 9, True),
      ('x  ←  x + N(0, 0.01)  (train only)', 8.5, False),
      ('Forces robust feature learning', 8, False)],
     text_color='#1A4A1A', border='#4CAF50')

# ════════════════════════════════════════════════════════════════════════════════
# OUTER BORDER & FOOTER
# ════════════════════════════════════════════════════════════════════════════════
outer = FancyBboxPatch((0.1, 0.1), 21.8, 13.78,
                       boxstyle="round,pad=0.1",
                       facecolor='none', edgecolor='#CCCCCC',
                       linewidth=2.0, zorder=1)
ax.add_patch(outer)

ax.text(11, 0.35,
        'CS 6140 · ML · Northeastern University · Spring 2026   |   '
        'Shantanu Wankhare  ·  Bhalchandra Shinde  ·  Asad Mulani',
        ha='center', va='center', fontsize=8.5, color='#888888', fontfamily=FONT)

# ── Save ─────────────────────────────────────────────────────────────────────
out_dir = Path(__file__).parent
fig.savefig(out_dir / 'cnn1d_architecture.png', dpi=180,
            bbox_inches='tight', facecolor=C_BG)
fig.savefig(out_dir / 'cnn1d_architecture.pdf',
            bbox_inches='tight', facecolor=C_BG)
print(f'Saved: {out_dir}/cnn1d_architecture.png')
print(f'Saved: {out_dir}/cnn1d_architecture.pdf')
plt.close()
