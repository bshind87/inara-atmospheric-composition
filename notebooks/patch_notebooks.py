#!/usr/bin/env python3
"""
Patch study_notes.ipynb and visualize.ipynb to reflect the current INARA pipeline state.
Run once from the notebooks/ directory:
    cd /Users/bhalchandra/myPythonCode/inara_changes/notebooks
    python3 patch_notebooks.py

This script is idempotent — safe to run multiple times.
"""

import json
import re

# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def src(cell):
    """Return joined source of a cell."""
    return "".join(cell["source"])


def set_src(cell, text):
    """Replace source of a cell (split into lines preserving newlines)."""
    lines = text.splitlines(keepends=True)
    cell["source"] = lines


def clear_outputs(cell):
    cell["outputs"] = []
    cell.pop("execution_count", None)


# ─────────────────────────────────────────────────────────────────────────────
# study_notes.ipynb
# ─────────────────────────────────────────────────────────────────────────────

print("Patching study_notes.ipynb ...")

with open("study_notes.ipynb") as f:
    sn = json.load(f)

cells = sn["cells"]

for idx, cell in enumerate(cells):
    s = src(cell)

    # ── Cell 0: TOC — "1D ResNet" → "1D CNN" ───────────────────────────────
    if "5. [Deep Model" in s and "1D ResNet" in s:
        new_s = s.replace(
            "5. [Deep Model — 1D ResNet](#5-deep-model)",
            "5. [Deep Model — 1D CNN](#5-deep-model)"
        )
        set_src(cell, new_s)
        print(f"  [{idx}] Fixed TOC entry (1D ResNet → 1D CNN)")

    # ── Cell 10 / RF params code: fix Rationale logic + clear stale output ──
    if "from src.baseline_model import MOLECULE_RF_PARAMS" in s:
        new_s = re.sub(
            r"'Rationale':\s*\(\s*'High variance.*?else '.*?'\s*\)",
            "'Rationale': 'Uniform params (n_estimators=100, max_depth=8, min_samples_leaf=2, max_features=sqrt)'",
            s,
            flags=re.DOTALL,
        )
        set_src(cell, new_s)
        clear_outputs(cell)
        print(f"  [{idx}] Fixed RF params cell source + cleared outputs")

    # ── Cell 11 / RF params rationale markdown table ───────────────────────
    if "| `n_estimators` | More trees" in s and "400 for trace species" in s:
        new_s = (s
            .replace(
                "| `n_estimators` | More trees → less variance, more compute | 400 for trace species (H₂S, SO₂, NH₃); 150 for near-constant N₂ |",
                "| `n_estimators` | More trees → less variance, more compute | 100 (uniform across all molecules) |",
            )
            .replace(
                "| `max_depth` | Controls tree complexity | Deeper for wide-range molecules (H₂O, CH₄); shallower for stable ones |",
                "| `max_depth` | Controls tree complexity | 8 (uniform across all molecules) |",
            )
            .replace(
                "| `min_samples_leaf` | Minimum samples per leaf | Larger (2–3) for stable molecules to prevent overfitting to noise |",
                "| `min_samples_leaf` | Minimum samples per leaf | 2 (uniform across all molecules) |",
            )
            .replace(
                "| `max_features` | Features per split | `'sqrt'` for high-variance targets; fractional (0.4–0.5) for stable ones |",
                "| `max_features` | Features per split | `'sqrt'` (uniform across all molecules) |",
            )
        )
        set_src(cell, new_s)
        print(f"  [{idx}] Fixed RF params rationale table")

    # ── Cell 12 / deep model section: fix 5.2 header + body ────────────────
    if "### 5.2 Why ResNet (Residual Network)?" in s:
        new_s = s.replace(
            "### 5.2 Why ResNet (Residual Network)?",
            "### 5.2 Architecture Overview",
        )
        # Replace the residual-network explanation with CNN1D block description
        resnet_body = (
            "A standard deep network suffers from **vanishing gradients** — gradients shrink "
            "exponentially as they backpropagate through many layers.\n"
            "\n"
            "**Residual connection** (He et al., 2016):\n"
            "$$\\text{output} = F(x) + x$$\n"
            "\n"
            "The shortcut path $x$ gives gradients a direct route back — the network learns the "
            "**residual** $F(x)$ rather than the full transformation. This enables training of "
            "much deeper networks.\n"
            "\n"
            "```\n"
            "Input x ──────────────────────────────┐\n"
            "  │                                   │  skip connection\n"
            "  ↓                                   │\n"
            "Conv1d → BN → ReLU → Conv1d → BN ─→ + → ReLU → output\n"
            "```"
        )
        cnn1d_body = (
            "CNN1D uses 4 sequential convolutional blocks (no residual connections), "
            "each with Conv1d + BatchNorm + ReLU + MaxPool1d (the last block omits MaxPool).\n"
            "\n"
            "```\n"
            "Input (B, 12, 101)\n"
            "  → Block 1: Conv1d(12→32,   k=9, s=2) + BN + ReLU + MaxPool1d(2) → (B,  32, ~25)\n"
            "  → Block 2: Conv1d(32→64,   k=7, s=2) + BN + ReLU + MaxPool1d(2) → (B,  64,  ~6)\n"
            "  → Block 3: Conv1d(64→128,  k=5, s=2) + BN + ReLU + MaxPool1d(2) → (B, 128,   1)\n"
            "  → Block 4: Conv1d(128→256, k=3, s=1) + BN + ReLU               → (B, 256,   1)\n"
            "  → AdaptiveAvgPool1d(1) → Flatten → (B, 256)\n"
            "  → Shared FC: Dropout(0.25) + Linear(256→128) + LayerNorm + ReLU → (B, 128)\n"
            "  → 12 per-molecule heads\n"
            "```"
        )
        if resnet_body in new_s:
            new_s = new_s.replace(resnet_body, cnn1d_body)
        set_src(cell, new_s)
        print(f"  [{idx}] Updated section 5.2 (residual-network → CNN1D architecture overview)")

    # ── Cell 13 / architecture trace code: update model attribute access ────
    if ("model.stage1" in s or "model.stem" in s) and "CNN1D" in s:
        old_trace = (
            "x_stem = model.stem(x)\n"
            "print(f'  After Stem       : {tuple(x_stem.shape)}   Conv1d(12→64, k=11, s=1) + BN + ReLU')\n"
            "x_s1 = model.stage1(x_stem)\n"
            "print(f'  After Stage 1    : {tuple(x_s1.shape)}   2×ResBlock(64→64, stride=1)')\n"
            "x_s2 = model.stage2(x_s1)\n"
            "print(f'  After Stage 2    : {tuple(x_s2.shape)}   2×ResBlock(64→128, stride=2)')\n"
            "x_s3 = model.stage3(x_s2)\n"
            "print(f'  After Stage 3    : {tuple(x_s3.shape)}   2×ResBlock(128→256, stride=2)')\n"
            "x_s4 = model.stage4(x_s3)\n"
            "print(f'  After Stage 4    : {tuple(x_s4.shape)}   2×ResBlock(256→512, stride=2)')\n"
            "x_pool = model.pool(x_s4).squeeze(-1)\n"
            "print(f'  After GlobalPool : {tuple(x_pool.shape)}   AdaptiveAvgPool1d(1)')\n"
            "x_shared = model.shared(x_pool)\n"
            "print(f'  After Shared MLP : {tuple(x_shared.shape)}   Dropout+FC(512→256)+LN+ReLU')"
        )
        new_trace = (
            "x_bb = model.backbone(x)\n"
            "print(f'  After Backbone   : {tuple(x_bb.shape)}   4×(Conv1d+BN+ReLU[+MaxPool])')\n"
            "x_pool = model.pool(x_bb).squeeze(-1)\n"
            "print(f'  After GlobalPool : {tuple(x_pool.shape)}   AdaptiveAvgPool1d(1)')\n"
            "x_shared = model.shared(x_pool)\n"
            "print(f'  After Shared MLP : {tuple(x_shared.shape)}   Dropout(0.25)+FC(256→128)+LN+ReLU')"
        )
        new_s = s.replace(old_trace, new_trace) if old_trace in s else s
        set_src(cell, new_s)
        clear_outputs(cell)
        print(f"  [{idx}] Updated architecture trace code + cleared stale outputs")

    # ── Cell 16 / BatchNorm+stride section: remove ResBlock/ResNet refs ─────
    if "### 5.7 Stride-2 downsampling across stages" in s:
        new_s = (s
            .replace(
                "At each stage transition, the first ResBlock uses `stride=2`, halving the "
                "sequence length while doubling channels:",
                "Each conv block uses stride-2 (and MaxPool1d) to progressively halve the "
                "sequence length while expanding channels:",
            )
            .replace(
                "This is the standard ResNet pattern: the network builds increasingly abstract, "
                "spatially compressed representations.",
                "The network builds increasingly abstract, spatially compressed representations.",
            )
        )
        resnet_skip = (
            "The skip connection in a stride-2 block uses a **1×1 Conv** to project channels before adding:\n"
            "```python\n"
            "if stride != 1 or in_ch != out_ch:\n"
            "    self.shortcut = nn.Sequential(\n"
            "        nn.Conv1d(in_ch, out_ch, 1, stride, bias=False),\n"
            "        nn.BatchNorm1d(out_ch)\n"
            "    )\n"
            "```"
        )
        if resnet_skip in new_s:
            new_s = new_s.replace(
                resnet_skip,
                "CNN1D uses no skip connections — each block is a plain convolutional stack (Conv1d → BN → ReLU → MaxPool1d)."
            )
        set_src(cell, new_s)
        print(f"  [{idx}] Updated stride section (removed ResBlock/ResNet references)")

    # ── Cell 22 / results commentary: ResNet → CNN ─────────────────────────
    if "ResNet's convolutional layers" in s or "ResNet may be" in s:
        new_s = (s
            .replace(
                "The ResNet's convolutional layers learn **spatial patterns** across altitude "
                "levels, which a flat PCA vector loses.",
                "The CNN's convolutional layers learn **spatial patterns** across altitude "
                "levels, which a flat PCA vector loses.",
            )
            .replace(
                "The ResNet may be overfitting on the training set noise for these molecules.",
                "The CNN may be overfitting on the training set noise for these molecules.",
            )
        )
        set_src(cell, new_s)
        print(f"  [{idx}] Updated results commentary (ResNet → CNN)")

    # ── Cell 22 continued: stale test-set size and data ratio ──────────────
    if "Same 19k samples" in s or "~8.7× more training data" in s:
        new_s = (s
            .replace("Same 19k samples", "Same 3,750 test samples")
            .replace(
                "~8.7× more training data and a more expressive architecture",
                "~1.75× more training data and a more expressive architecture",
            )
        )
        set_src(cell, new_s)
        print(f"  [{idx}] Fixed test-set size and training-data ratio")

    # ── Cell 25 / step I/O code: n_samples=124000 → n_samples=25000 ─────────
    if "n_samples=124000" in s and "step_table" in s:
        new_s = s.replace("n_samples=124000", "n_samples=25000")
        set_src(cell, new_s)
        clear_outputs(cell)
        print(f"  [{idx}] Fixed step I/O table n_samples + cleared stale outputs")

    # ── Cell 26 / architecture summary: comment + clear stale outputs ────────
    if "# ResNet architecture summary" in s and "arch_table" in s:
        new_s = s.replace("# ResNet architecture summary", "# CNN1D architecture summary")
        set_src(cell, new_s)
        clear_outputs(cell)
        print(f"  [{idx}] Fixed arch summary comment + cleared stale outputs")


with open("study_notes.ipynb", "w") as f:
    json.dump(sn, f, indent=1, ensure_ascii=False)

print("study_notes.ipynb patched and saved.")
print()

# ─────────────────────────────────────────────────────────────────────────────
# visualize.ipynb
# ─────────────────────────────────────────────────────────────────────────────

print("Patching visualize.ipynb ...")

with open("visualize.ipynb") as f:
    viz = json.load(f)

cells = viz["cells"]

for idx, cell in enumerate(cells):
    s = src(cell)

    # ── Fix double-processed RESULTS_DIR (can happen from partial prior edit) ─
    if "results' / 'processed' / 'processed'" in s:
        new_s = s.replace(
            "ROOT / 'results' / 'processed' / 'processed'",
            "ROOT / 'results' / 'processed'",
        )
        set_src(cell, new_s)
        s = src(cell)
        print(f"  [{idx}] Fixed double-processed RESULTS_DIR")

    # ── Cell 0 (markdown): paths + SpectralResNet/ResNet ────────────────────
    if cell["cell_type"] == "markdown" and idx == 0:
        new_s = (s
            .replace("models/processed/baseline_rf.joblib", "models/baseline_rf.joblib")
            .replace("models/processed/spectral_resnet.pt",  "models/cnn1d.pt")
            .replace("SpectralResNet", "1D CNN")
        )
        new_s = re.sub(r"\bResNet\b", "1D CNN", new_s)
        if new_s != s:
            set_src(cell, new_s)
            print(f"  [{idx}] Updated cell-0 markdown (paths + model name)")

    # ── Cell 1 (imports / path setup) ───────────────────────────────────────
    if "from src.deep_model import SpectralResNet" in s:
        new_s = s.replace(
            "from src.deep_model import SpectralResNet, MOLECULE_HEAD_CONFIGS",
            "from src.deep_model import CNN1D, MOLECULE_HEAD_CONFIGS",
        ).replace(
            "from src.deep_model import SpectralResNet",
            "from src.deep_model import CNN1D",
        )
        set_src(cell, new_s)
        s = src(cell)
        print(f"  [{idx}] Fixed import (SpectralResNet → CNN1D)")

    if "MODELS_DIR" in s and "'models'" in s and "'processed'" in s:
        new_s = (s
            .replace("MODELS_DIR  = ROOT / 'models'  / 'processed'", "MODELS_DIR  = ROOT / 'models'")
            .replace("MODELS_DIR = ROOT / 'models' / 'processed'",   "MODELS_DIR = ROOT / 'models'")
        )
        if new_s != s:
            set_src(cell, new_s)
            s = src(cell)
            print(f"  [{idx}] Fixed MODELS_DIR (removed /processed)")

    # Fix plain `results` RESULTS_DIR (add /processed) — only if not already there
    if cell["cell_type"] == "code" and "RESULTS_DIR = ROOT / 'results'" in s:
        # Check the value after RESULTS_DIR on same line
        m = re.search(r"RESULTS_DIR = ROOT / 'results'([^\n]*)", s)
        if m and "processed" not in m.group(0):
            new_s = s.replace(
                "RESULTS_DIR = ROOT / 'results'\n",
                "RESULTS_DIR = ROOT / 'results' / 'processed'\n",
            )
            if new_s != s:
                set_src(cell, new_s)
                s = src(cell)
                print(f"  [{idx}] Fixed RESULTS_DIR (plain results → results/processed)")

    # ── Cell 5 / load models ─────────────────────────────────────────────────
    if "spectral_resnet.pt" in s or (
        "state['stem.0.weight']" in s
    ):
        new_s = (s
            .replace("spectral_resnet.pt", "cnn1d.pt")
            .replace("state['stem.0.weight'].shape[1]", "state['backbone.0.weight'].shape[1]")
            .replace(
                "resnet = SpectralResNet(head_configs=MOLECULE_HEAD_CONFIGS, in_channels=_in_ch)",
                "cnn_model = CNN1D(head_configs=MOLECULE_HEAD_CONFIGS, in_channels=_in_ch)",
            )
        )
        new_s = re.sub(r"\bresnet\b", "cnn_model", new_s)
        new_s = new_s.replace("SpectralResNet", "CNN1D")
        set_src(cell, new_s)
        s = src(cell)
        print(f"  [{idx}] Fixed load-models cell")

    # ── All remaining "ResNet loaded" or "resnet." in code cells ─────────────
    if cell["cell_type"] == "code" and (
        "ResNet loaded" in s or re.search(r"\bresnet\b", s) or "SpectralResNet" in s
    ):
        new_s = re.sub(r"\bresnet\b", "cnn_model", s)
        new_s = new_s.replace("SpectralResNet", "CNN1D")
        new_s = new_s.replace("ResNet loaded", "CNN1D loaded")
        if new_s != s:
            set_src(cell, new_s)
            s = src(cell)
            print(f"  [{idx}] Replaced resnet/ResNet loaded/SpectralResNet in code cell")

    # ── Cell 6 markdown: "SpectralResNet Loss Curve" ─────────────────────────
    if "SpectralResNet Loss Curve" in s:
        set_src(cell, s.replace("SpectralResNet Loss Curve", "1D CNN Loss Curve"))
        s = src(cell)
        print(f"  [{idx}] Fixed loss curve section header")

    # ── All markdown cells: SpectralResNet / ResNet ──────────────────────────
    if cell["cell_type"] == "markdown" and (
        "SpectralResNet" in s or re.search(r"\bResNet\b", s)
    ):
        new_s = (s
            .replace("SpectralResNet", "1D CNN")
            .replace("ResNet Architecture — Parameter Counts per Stage",
                     "1D CNN Architecture — Parameter Counts per Layer")
            .replace("ResNet Architecture Diagram", "1D CNN Architecture Diagram")
        )
        new_s = re.sub(r"\bResNet\b", "1D CNN", new_s)
        if new_s != s:
            set_src(cell, new_s)
            print(f"  [{idx}] Updated markdown (SpectralResNet/ResNet → 1D CNN)")

    # ── All code cells: remaining SpectralResNet / ResNet / label strings ────
    if cell["cell_type"] == "code" and (
        "SpectralResNet" in s
        or re.search(r"label=['\"]ResNet['\"]", s)
        or "'ResNet'" in s
        or '"ResNet"' in s
        or "ResNet mean R²" in s
        or "ΔR² (ResNet" in s
        or "vs ResNet" in s
    ):
        new_s = s.replace("SpectralResNet", "CNN1D")
        # label= string replacements (both single and double quotes)
        new_s = new_s.replace("label='ResNet'", "label='1D CNN'")
        new_s = new_s.replace('label="ResNet"', 'label="1D CNN"')
        new_s = new_s.replace("'ResNet mean R²", "'1D CNN mean R²")
        new_s = new_s.replace("ΔR² (ResNet − RF)", "ΔR² (1D CNN − RF)")
        new_s = new_s.replace("vs ResNet", "vs 1D CNN")
        new_s = new_s.replace("vs ResNet (red)", "vs 1D CNN (red)")
        if new_s != s:
            set_src(cell, new_s)
            print(f"  [{idx}] Fixed remaining ResNet labels in code cell")

    # ── Cell 17 / parameter count: stages list ───────────────────────────────
    if "('Stem (Conv1d→64)'," in s or "resnet.stem" in s or "('Stage 1" in s:
        new_s = src(cell)
        new_s = re.sub(r"\bresnet\b", "cnn_model", new_s)
        new_s = (new_s
            .replace("('Stem (Conv1d→64)', cnn_model.stem)",  "('Backbone (Conv blocks 1-4)', cnn_model.backbone)")
            .replace("('Stage 1 (64→64)',  cnn_model.stage1),\n", "")
            .replace("('Stage 2 (64→128)', cnn_model.stage2),\n", "")
            .replace("('Stage 3 (128→256)', cnn_model.stage3),\n", "")
            .replace("('Stage 4 (256→512)', cnn_model.stage4),\n", "")
            .replace("('Shared FC (512→256)', cnn_model.shared)", "('Shared FC (256→128)', cnn_model.shared)")
        )
        if new_s != src(cell):
            set_src(cell, new_s)
            print(f"  [{idx}] Updated parameter-count stages list")

    # ── Cell 18/19 / architecture diagram: stage/ResBlock labels ─────────────
    if "ResBlock×2" in s or "Stage 1\\nResBlock" in s:
        new_s = (src(cell)
            .replace("Stage 1\\nResBlock×2", "Block 1\\nConv+BN+ReLU")
            .replace("Stage 2\\nResBlock×2", "Block 2\\nConv+BN+ReLU")
            .replace("Stage 3\\nResBlock×2", "Block 3\\nConv+BN+ReLU")
            .replace("Stage 4\\nResBlock×2", "Block 4\\nConv+BN+ReLU")
        )
        set_src(cell, new_s)
        print(f"  [{idx}] Fixed architecture diagram stage/ResBlock labels")

    # ── Cell 19 / architecture diagram code: resnet.stem/stages → backbone ──
    if "resnet.stem" in src(cell) or "resnet.stage" in src(cell):
        new_s = re.sub(r"\bresnet\b", "cnn_model", src(cell))
        new_s = (new_s
            .replace("cnn_model.stem",   "cnn_model.backbone")
            .replace("cnn_model.stage1", "cnn_model.backbone[0:4]")
            .replace("cnn_model.stage2", "cnn_model.backbone[4:8]")
            .replace("cnn_model.stage3", "cnn_model.backbone[8:12]")
            .replace("cnn_model.stage4", "cnn_model.backbone[12:16]")
        )
        set_src(cell, new_s)
        print(f"  [{idx}] Fixed architecture diagram cell (stem/stages → backbone)")


with open("visualize.ipynb", "w") as f:
    json.dump(viz, f, indent=1, ensure_ascii=False)

print("visualize.ipynb patched and saved.")
print()
print("All done.")
