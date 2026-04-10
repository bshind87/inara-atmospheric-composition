#!/usr/bin/env python
"""
Generate feature_engineering_notes.pdf — technical reference for Step 2.
Run:  python docs/build_feature_engineering_notes.py
Output: feature_engineering_notes.pdf (project root)
"""

from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether,
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER

# ── Colours ──────────────────────────────────────────────────────────────────
NAVY   = colors.HexColor('#002B5C')
RED    = colors.HexColor('#CC0000')
BLUE   = colors.HexColor('#2166AC')
GREEN  = colors.HexColor('#1A7A3C')
LGREY  = colors.HexColor('#F0F4F9')
MGREY  = colors.HexColor('#CCCCCC')
DGREY  = colors.HexColor('#555555')
YELLOW = colors.HexColor('#FFF8E1')
LGREEN = colors.HexColor('#E8F5E9')
LRED   = colors.HexColor('#FFEEEE')

OUT_PATH = Path(__file__).resolve().parents[1] / 'feature_engineering_notes.pdf'

doc = SimpleDocTemplate(
    str(OUT_PATH),
    pagesize=letter,
    leftMargin=0.85*inch, rightMargin=0.85*inch,
    topMargin=0.9*inch,   bottomMargin=0.9*inch,
)

styles = getSampleStyleSheet()

def S(name, **kw):
    base = styles[name]
    return ParagraphStyle(name + '_custom', parent=base, **kw)

H1  = S('Heading1',  fontSize=20, textColor=NAVY,  spaceAfter=6,  spaceBefore=14)
H2  = S('Heading2',  fontSize=14, textColor=NAVY,  spaceAfter=4,  spaceBefore=10)
H3  = S('Heading3',  fontSize=11, textColor=BLUE,  spaceAfter=3,  spaceBefore=8)
BOD = S('Normal',    fontSize=10, leading=15,       spaceAfter=5)
COD = S('Code',      fontSize=9,  fontName='Courier', leading=13,
         backColor=LGREY, leftIndent=12, spaceAfter=6, spaceBefore=4)
CAP = S('Normal',    fontSize=8.5, textColor=DGREY, leading=12)
BUL = S('Normal',    fontSize=10, leading=15, leftIndent=16, spaceAfter=3)

def h1(t):   return Paragraph(t, H1)
def h2(t):   return Paragraph(t, H2)
def h3(t):   return Paragraph(t, H3)
def p(t):    return Paragraph(t, BOD)
def code(t): return Paragraph(t.replace('\n', '<br/>').replace(' ', '&nbsp;'), COD)
def cap(t):  return Paragraph(t, CAP)
def bul(t):  return Paragraph('• &nbsp;' + t, BUL)
def hr():    return HRFlowable(width='100%', thickness=0.8, color=MGREY, spaceAfter=6)
def sp(n=6): return Spacer(1, n)


def table(headers, rows, col_widths=None, hdr_bg=NAVY, alt_bg=LGREY):
    data = [headers] + rows
    cw   = col_widths or [doc.width / len(headers)] * len(headers)
    t    = Table(data, colWidths=cw, repeatRows=1)
    style = [
        ('BACKGROUND', (0,0), (-1,0), hdr_bg),
        ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
        ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0,0), (-1,-1), 9),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, alt_bg]),
        ('GRID',       (0,0), (-1,-1), 0.4, MGREY),
        ('VALIGN',     (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING',  (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
        ('TOPPADDING',   (0,0), (-1,-1), 4),
        ('BOTTOMPADDING',(0,0), (-1,-1), 4),
    ]
    t.setStyle(TableStyle(style))
    return t


def note_box(text, bg=YELLOW, border=colors.HexColor('#E0B040')):
    t = Table([[Paragraph(text, S('Normal', fontSize=9.5, leading=14))]])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), bg),
        ('BOX',        (0,0), (-1,-1), 1.2, border),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING',(0,0), (-1,-1), 8),
        ('TOPPADDING',  (0,0), (-1,-1), 6),
        ('BOTTOMPADDING',(0,0),(-1,-1), 6),
    ]))
    return t


# ═════════════════════════════════════════════════════════════════════════════
story = []

# ── Title block ───────────────────────────────────────────────────────────────
story.append(h1('INARA Feature Engineering — Technical Notes'))
story.append(p('<b>Pipeline Step 2</b> &nbsp;|&nbsp; CS 6140 · Machine Learning · '
               'Northeastern University · Spring 2026'))
story.append(p('<font color="#555555">Shantanu Wankhare &nbsp;·&nbsp; '
               'Bhalchandra Shinde &nbsp;·&nbsp; Asad Mulani</font>'))
story.append(hr())
story.append(p(
    'This document describes every decision made in <b>pipeline/steps/02_feature_engineer.py</b>: '
    'the train/val/test split, spectral Z-score normalisation, PCA feature reduction for the '
    'Random Forest baseline, and data quality checks. The deep model (1D CNN) uses normalised '
    'spectra directly and does not use PCA.'
))
story.append(sp(8))

# ── 1. Overview ───────────────────────────────────────────────────────────────
story.append(h2('1.  Overview'))
story.append(p(
    'Step 2 runs <b>once</b> before Steps 3–5, producing all engineered artifacts in '
    '<code>inara_data/engineered/</code>. Because both the RF baseline and the 1D CNN '
    'load from these same files, they are guaranteed to train and evaluate on identical '
    'data splits — a prerequisite for a fair academic comparison.'
))

story.append(table(
    ['Input', 'Output'],
    [
        ['spectra.npy  (N, 12, 101) — raw CLIMA profiles', 'spectra_{train,val,test}.npy — Z-normalised'],
        ['molecules.npy  (N, 12) — log₁₀ targets',        'molecules_{train,val,test}.npy — raw log₁₀'],
        ['',                                                'feat_{train,val,test}.npy — PCA (RF only)'],
        ['',                                                'scaler.joblib, pca.joblib, feature_info.json'],
        ['',                                                '{train,val,test}_indices.npy — split indices'],
    ],
    col_widths=[3.5*inch, 3.4*inch],
))
story.append(sp(6))

# ── 2. Dataset (current run) ──────────────────────────────────────────────────
story.append(h2('2.  Current Dataset'))
story.append(table(
    ['Property', 'Value'],
    [
        ['Total samples (this run)', '25,000'],
        ['Train split (70%)',        '17,500 samples'],
        ['Validation split (15%)',   '3,750 samples'],
        ['Test split (15%)',         '3,750 samples'],
        ['Input shape',              '(N, 12, 101) — 12 CLIMA channels × 101 altitude levels'],
        ['Target shape',             '(N, 12) — log₁₀ surface volume mixing ratios'],
        ['Split seed',               '42 (reproducible)'],
        ['Full dataset size',        '3,112,620 samples (Zorzan et al. 2025)'],
    ],
    col_widths=[2.8*inch, 4.1*inch],
))
story.append(sp(4))
story.append(note_box(
    '⚠️  The 25,000-sample run is a local development run. The full 3.1M-sample dataset '
    'is available on Northeastern Explorer HPC. All pipeline steps are designed to scale — '
    'only the extraction count and RF cap need adjusting in pipeline/config.yaml.',
    bg=YELLOW, border=colors.HexColor('#E0B040'),
))
story.append(sp(8))

# ── 3. Train / Val / Test Split ───────────────────────────────────────────────
story.append(h2('3.  Train / Val / Test Split  (70 / 15 / 15)'))
story.append(code(
'train_idx, val_idx, test_idx = split_indices(\n'
'    N, val_frac=0.15, test_frac=0.15, seed=42\n'
')'
))
story.append(p('<b>Why this ratio?</b>'))
for item in [
    '70 % training gives the CNN sufficient samples (17,500 in this run; ~87,000 on 124k).',
    '15 % test set (3,750 samples) is large enough for reliable per-molecule R² estimates.',
    'Identical test set for RF and CNN — directly comparable metrics, no cherry-picking.',
    'Seed 42 is fixed in config.yaml — reproducible across machines and reruns.',
    'Indices saved to disk so Steps 3–5 never re-split; data cannot leak between runs.',
]:
    story.append(bul(item))
story.append(sp(8))

# ── 4. Spectral Z-score Normalisation ────────────────────────────────────────
story.append(h2('4.  Spectral Z-score Normalisation'))
story.append(p(
    'Each of the 12 CLIMA channels is normalised <b>independently</b> across the altitude axis:'
))
story.append(code(
'x_norm[n, c, l] = (x[n, c, l] − μ[c, l]) / σ[c, l]\n'
'\n'
'# μ, σ computed on TRAINING SET ONLY, then applied to val and test\n'
'scaler = SpectraScaler()\n'
'spec_train = scaler.fit_transform(spectra[train_idx])\n'
'spec_val   = scaler.transform(spectra[val_idx])\n'
'spec_test  = scaler.transform(spectra[test_idx])'
))

story.append(table(
    ['Design choice', 'Rationale'],
    [
        ['Per-channel, per-altitude-level',
         'CLIMA channels span vastly different scales (temperature in K vs mixing ratios '
         'in [0,1]). Per-element normalisation puts all inputs on the same gradient scale.'],
        ['Fit on train only',
         'Fitting on val/test would leak their statistics into training — a form of data '
         'leakage. The scaler sees only training data before being applied to held-out sets.'],
        ['std floor = 1e-8',
         'Channels with near-zero variance (constant altitude levels) would cause ÷0. '
         'Any std < 1e-8 is replaced with 1.0, leaving the value unnormalised rather than '
         'producing NaN/Inf.'],
        ['Saved as scaler.joblib',
         'Steps 3–5 and the dashboard load the same scaler to ensure consistent '
         'normalisation at inference time.'],
    ],
    col_widths=[2.0*inch, 4.9*inch],
))
story.append(sp(8))

# ── 5. PCA Feature Reduction ──────────────────────────────────────────────────
story.append(h2('5.  PCA Feature Reduction  (Random Forest only)'))
story.append(p(
    'The RF baseline cannot process the raw <code>(N, 12, 101)</code> tensor. '
    'Feature engineering flattens and reduces it:'
))
story.append(code(
'# Flatten: (N, 12, 101) → (N, 1212)\n'
'flat = spectra.reshape(N, -1).astype(np.float64)  # float64 avoids matmul overflow\n'
'\n'
'pca = PCA(n_components=300, svd_solver="full", whiten=False)\n'
'feat_train = pca.fit_transform(flat_train)   # (N_tr, 300)\n'
'feat_val   = pca.transform(flat_val)         # (N_v,  300)\n'
'feat_test  = pca.transform(flat_test)        # (N_te, 300)'
))

story.append(table(
    ['Parameter', 'Value', 'Rationale'],
    [
        ['n_components', '300',
         'Covers 100 % variance on the 25k run (data is rank-limited). '
         'On larger datasets captures ~95 % variance.'],
        ['svd_solver',   '"full"',
         'Deterministic, numerically stable. Avoids randomised approximation '
         'artefacts in transform().'],
        ['whiten',       'False',
         'Whitening divides by singular values; near-zero values cause ÷0. '
         'Not needed for RF — tree splits are scale-invariant.'],
        ['dtype',        'float64',
         'float32 matmul (17500 × 1212) @ (1212 × 300) can overflow on macOS '
         'Accelerate BLAS. float64 eliminates all RuntimeWarnings.'],
    ],
    col_widths=[1.3*inch, 0.8*inch, 4.8*inch],
))
story.append(sp(4))
story.append(note_box(
    '<b>Why does the 1D CNN NOT use PCA?</b><br/>'
    'The CNN processes the raw 2D tensor (12 × 101) directly. Convolutional layers learn '
    'their own spatial feature extraction — they capture non-linear altitude patterns that '
    'a linear PCA projection cannot represent. PCA is only needed to make the RF tractable.',
    bg=LGREEN, border=GREEN,
))
story.append(sp(8))

# ── 6. Target Normalisation (CNN only) ────────────────────────────────────────
story.append(h2('6.  Target Normalisation  (1D CNN only)'))
story.append(p(
    'The 12 molecular targets span very different log₁₀ ranges — NH₃ is fixed at −40, '
    'while N₂ ranges from −1.1 to −0.01. Without normalisation, the weighted MSE loss '
    'is dominated by molecules with large absolute values, destabilising early training.'
))
story.append(code(
'# In pipeline/steps/04_train_deep.py  (NOT in feature engineering)\n'
'mol_scaler = MoleculeScaler()                   # per-molecule Z-score\n'
'mol_train_scaled = mol_scaler.fit_transform(mol_train)\n'
'mol_val_scaled   = mol_scaler.transform(mol_val)\n'
'mol_test_scaled  = mol_scaler.transform(mol_test)\n'
'\n'
'# After training, inverse-transform predictions before computing metrics:\n'
'test_pred_log10 = mol_scaler.inverse_transform(trainer.predict(test_loader))\n'
'test_df = compute_metrics(mol_test, test_pred_log10)  # metrics in log₁₀ space'
))
story.append(p(
    '<b>Note:</b> MoleculeScaler is fit on <i>training targets only</i> to avoid '
    'leakage. R²/RMSE/MAE are always reported in the original log₁₀ space for '
    'interpretability and comparability with the RF baseline.'
))
story.append(sp(8))

# ── 7. Data Quality Report ────────────────────────────────────────────────────
story.append(h2('7.  Data Quality Report  (Step 2 output)'))
story.append(p(
    'After saving all engineered arrays, Step 2 runs a quality check and prints a '
    'summary table. This is the actual output from the 25k run:'
))
story.append(table(
    ['Array', 'Shape', 'NaN', 'Inf', 'Min', 'Max', 'Status'],
    [
        ['spectra_train',   '(17500, 12, 101)', '0', '0', '−68.12', '132.28', 'OK'],
        ['spectra_val',     '(3750,  12, 101)', '0', '0', '−68.12', ' 76.38', 'OK'],
        ['spectra_test',    '(3750,  12, 101)', '0', '0', '−31.51', '139.48', 'OK'],
        ['molecules_train', '(17500, 12)',       '0', '0', '−40.00', ' −0.01', 'OK'],
        ['molecules_val',   '(3750,  12)',       '0', '0', '−40.00', ' −0.01', 'OK'],
        ['molecules_test',  '(3750,  12)',       '0', '0', '−40.00', ' −0.01', 'OK'],
        ['feat_train',      '(17500, 300)',      '0', '0', '−289.71', '167.16', 'OK'],
        ['feat_val',        '(3750,  300)',      '0', '0', '−289.71', ' 88.16', 'OK'],
        ['feat_test',       '(3750,  300)',      '0', '0', '−134.37', '255.91', 'OK'],
    ],
    col_widths=[1.3*inch, 1.2*inch, 0.45*inch, 0.45*inch, 0.7*inch, 0.7*inch, 0.5*inch],
))
story.append(sp(6))

story.append(h3('Target standard deviations (per molecule, training set)'))
story.append(table(
    ['Molecule', 'Std (log₁₀)', 'Learnable?', 'Notes'],
    [
        ['H₂O',  '0.0002', 'LOW',  'Near-constant — R² unreliable; excluded from mean R²'],
        ['CO₂',  '0.8442', 'YES',  'High variance — strong learnable signal'],
        ['O₂',   '0.1709', 'YES',  ''],
        ['O₃',   '0.1234', 'YES',  'Strong spectral signature — best-predicted molecule'],
        ['CH₄',  '1.8232', 'YES',  'High variance — CNN shows strong improvement'],
        ['N₂',   '0.0446', 'YES',  'Low variance — RF and CNN both struggle'],
        ['N₂O',  '0.1527', 'YES',  ''],
        ['CO',   '1.3330', 'YES',  'Both models struggle — photochemical degeneracy'],
        ['H₂',   '2.5344', 'YES',  'High variance — challenging for both models'],
        ['H₂S',  '0.3954', 'YES',  ''],
        ['SO₂',  '0.1125', 'YES',  ''],
        ['NH₃',  '0.0000', 'NO',   'CONSTANT — all samples at log-floor (−40). '
                                    'RF: R²=1.0 (memorises floor). CNN: R²=0.0. Excluded from mean R².'],
    ],
    col_widths=[0.6*inch, 0.9*inch, 0.8*inch, 4.6*inch],
))
story.append(sp(8))

# ── 8. Artifacts saved ────────────────────────────────────────────────────────
story.append(h2('8.  Artifacts Saved to inara_data/engineered/'))
story.append(table(
    ['File', 'Shape / Type', 'Used by'],
    [
        ['spectra_train.npy',   '(17500, 12, 101) float32', 'Step 4 (CNN training)'],
        ['spectra_val.npy',     '(3750,  12, 101) float32', 'Step 4 (CNN validation)'],
        ['spectra_test.npy',    '(3750,  12, 101) float32', 'Step 4 (CNN test eval)'],
        ['molecules_train.npy', '(17500, 12) float32',      'Steps 3, 4'],
        ['molecules_val.npy',   '(3750,  12) float32',      'Steps 3, 4'],
        ['molecules_test.npy',  '(3750,  12) float32',      'Steps 3, 4, 5'],
        ['feat_train.npy',      '(17500, 300) float64',     'Step 3 (RF training)'],
        ['feat_val.npy',        '(3750,  300) float64',     'Step 3 (RF validation)'],
        ['feat_test.npy',       '(3750,  300) float64',     'Step 3 (RF test eval)'],
        ['scaler.joblib',       'SpectraScaler object',      'Dashboard, inference'],
        ['pca.joblib',          'sklearn PCA object',        'Step 3, dashboard'],
        ['train_indices.npy',   '(17500,) int64',            'Provenance'],
        ['val_indices.npy',     '(3750,)  int64',            'Provenance'],
        ['test_indices.npy',    '(3750,)  int64',            'Provenance'],
        ['feature_info.json',   'JSON metadata',             'Dashboard, verification'],
    ],
    col_widths=[2.1*inch, 1.9*inch, 2.9*inch],
))
story.append(sp(8))

# ── 9. Configuration ──────────────────────────────────────────────────────────
story.append(h2('9.  Relevant Configuration  (pipeline/config.yaml)'))
story.append(code(
'data:\n'
'  val_frac:  0.15\n'
'  test_frac: 0.15\n'
'  seed:      42\n'
'\n'
'model:\n'
'  pca_components: 300   # components for RF PCA features\n'
'\n'
'profiles:\n'
'  local:\n'
'    processed_dir:  inara_data/processed\n'
'    engineered_dir: inara_data/engineered\n'
'    results_dir:    results/processed\n'
'  hpc:\n'
'    processed_dir:  /scratch/shinde.b/inara/processed\n'
'    engineered_dir: /scratch/shinde.b/inara/engineered\n'
'    results_dir:    /scratch/shinde.b/inara/results/processed'
))
story.append(sp(10))

# ── Footer line ───────────────────────────────────────────────────────────────
story.append(hr())
story.append(cap(
    'INARA: Investigating Non-trivial Atmospheres for Retrieval Algorithms  |  '
    'NASA FDL / Zorzan et al. 2025  |  CS 6140 Northeastern University Spring 2026'
))

doc.build(story)
print(f'Saved: {OUT_PATH}')
