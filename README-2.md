# CS6140 Machine Learning – Spring 2026 - Northeastern University San Jose
## Project title: Exploring ML Models for detection of Atmospheric composition in Exoplanets

Team Members 
•	Shantanu Wankhare	wankhare.s@northeastern.edu
•	Bhalchandra Shinde 	shinde.b@northeastern.edu
•	Asad Mulani 		mulani.a@northeastern.edu

Using ML models to perform fast atmospheric retrieval on exoplanet spectra from the INARA dataset. We will try to use models which can replaces traditional Bayesian methods (hours per planet) with a models like Random Forest or 1D convolutional neural network that infers 12 molecular abundances in milliseconds.

---

## Project Structure

```
inara/
├── src/
│   ├── config.py               ← all constants, paths, hyperparameters
│   ├── download_inara.py       ← parse CSV files → processed numpy arrays
│   ├── generate_parameters.py  ← extract labels from spectra
│   ├── train.py                ← training loops 
│   ├── evaluate.py             ← test-set metrics, scatter plots, calibration curves
│   └── predict.py              ← single-spectrum atmospheric retrieval
├── notebooks/
│   └── 01_EDA.ipynb            ← exploratory data analysis
├── inara_data/
│   ├── 0001100.csv             ← INARA spectrum files (one per planetary system)
│   ├── 0001108.csv
│   ├── ...
│   └── processed/              ← 
│       ├── spectra.npy             (N, 3, 4378) 
│       ├── molecules.npy           (N, 12)      
│       ├── aux_params.npy          (N, 8) 
│       └── wavelengths.npy         (4378,) 
├── checkpoints/
|
└── results/
    ├── reports
    └── plots/
```

---

## Setup

### Requirements

- Python 3.10+
- PyTorch 2.2+
- `conda` or `pip` with the ML environment active

### Install dependencies

```bash
pip install torch numpy pandas scikit-learn matplotlib scipy
pip install umap-learn   # optional — for UMAP embedding in EDA notebook
```

### Clone and navigate

```bash
git clone <repo-url>
cd inara
```

---


## Dataset: INARA

The [INARA dataset](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/FDL/nph-fdl) (Zorzan et al. 2025) contains 3.1 million synthetic rocky planet spectra generated with NASA's Planetary Spectrum Generator, designed specifically for machine learning retrieval.

**CSV format** (confirmed from sample files):

| Column | Description |
|---|---|
| `wavelength_(um)` | Wavelength in µm |
| `star_planet_signal_(erg/s/cm2)` | Combined star + planet observation |
| `noise_(erg/s/cm2)` | Instrument noise floor |
| `stellar_signal_(erg/s/cm2)` | Stellar flux only |
| `planet_signal_(erg/s/cm2)` | Atmospheric signal only |

- **Total rows per file**: 15,346 (full PSG output, 0.2–639.9 µm)
- **Usable rows**: 4,378 (where `noise > 0`, covering 0.2–2.0 µm — HWO/LUVOIR instrument range)

