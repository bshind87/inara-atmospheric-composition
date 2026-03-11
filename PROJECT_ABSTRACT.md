# PROJECT_ABSTRACT.md

# CS6140 Machine Learning – Spring 2026 - Northeastern University San Jose
## Project title: Exploring ML Models for detection of Atmospheric composition in Exoplanets

Team Members 
•	Shantanu Wankhare	wankhare.s@northeastern.edu
•	Bhalchandra Shinde 	shinde.b@northeastern.edu
•	Asad Mulani 		mulani.a@northeastern.edu

**Dataset**: INARA (NASA FDL / Zorzan et al. 2025)

---

## Overview

This project applies machine learning to the problem of exoplanet atmospheric retrieval — the task of inferring the chemical composition of a planet's atmosphere from its observed transmission spectrum. Traditional retrieval methods based on Bayesian sampling are computationally expensive, often requiring hours or days per planet. As space missions such as the James Webb Space Telescope (JWST) and the planned Habitable Worlds Observatory (HWO) generate spectral data at unprecedented scale, automated and efficient retrieval methods are becoming a scientific necessity.

Using ML models to perform fast atmospheric retrieval on exoplanet spectra from the INARA dataset. We will try to use models which can replaces traditional Bayesian methods (hours per planet) with a models like Random Forest or 1D convolutional neural network that infers 12 molecular abundances in milliseconds.

The INARA dataset of 3.1 million synthetic rocky planet spectra would be used, would try to use sample of it or the full data based on the compute capacity. Project would predict the volume mixing ratios (VMRs) of 12 atmospheric molecules, including the key biosignatures O₂, O₃, CH₄, and N₂O. 

---

## Motivation

Understanding the atmospheric composition of rocky exoplanets is one of the most important goals in modern astronomy. The presence and relative abundance of certain molecules — particularly the co-detection of O₂ and CH₄ — could indicate biological activity, making atmospheric characterisation directly relevant to the search for habitable worlds.

Traditional atmospheric retrieval uses Bayesian algorithms such as nested sampling and MCMC to explore a high-dimensional parameter space of atmospheric models. While rigorous, these methods scale poorly: runtimes grow with the number of free parameters, and applying them to thousands of planets simultaneously is infeasible. Machine learning offers retrieval times of milliseconds per planet once trained, enabling population-level studies that are otherwise impractical.

This project is motivated by the intersection of scientific urgency and methodological opportunity: the volume of high-quality spectral data is growing rapidly, and ML-based retrieval is an established and active research direction (Márquez-Neila et al. 2018; Vasist et al. 2023; Gebhard et al. 2024).

---

## Dataset

The **INARA** (Intelligent Exoplanet Atmospheric Retrieval Algorithm) dataset was created by the NASA Frontier Development Lab (FDL) Astrobiology team and is publicly available through the NASA Exoplanet Archive. It contains 3,112,620 synthetic planetary systems generated with NASA's Planetary Spectrum Generator (PSG), designed specifically as a machine learning training resource.

Each spectrum is modelled after HWO/LUVOIR design criteria: UV to near-infrared coverage from 0.2 to 2.0 µm at 4,378 wavelength points. Each file contains five columns — combined star-planet signal, instrument noise, stellar flux, and planet atmospheric signal — from which a 3-channel input tensor can be constructed, this would be explored more during the Data preprocessing and feature engineering to finalize the number of channels if benefits:

- **Channel 0**: Observed SNR spectrum (star+planet / noise), normalised per spectrum
- **Channel 1**: Mean-subtracted SNR, removes the stellar continuum and isolates molecular absorption features
- **Channel 2**: Log transit depth 

The 12 target molecules span habitability indicators (H₂O, CO₂), biosignatures (O₂, O₃, CH₄, N₂O), and geochemical tracers (CO, H₂S, SO₂, NH₃, N₂, H₂). Four stellar types (F, G, K, M) are represented, with eight auxiliary features per system including stellar temperature, planetary radius, surface gravity, and orbital distance.

---

## Methodology

### Model 
The preferred models would be Random Forest or 1D CNN. We would still want to do source data analysis and EDA to finalize the models being used. We can try to use some baseline model and further some other model as well. This can be confirmed once the project progresses and completes the EDA and feature engineering steps. 


### Source data - INARA Dataset

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

We would read all the csv files (like 0001108.csv) and get the data preprocessed and ready into the below files as processed input data:
1. spectra.npy
2. molecules.npy
3. aux_params.npy
4. wavelengths.npy

### Expected Targets/Outcomes — 

We would target most of the molecules from below:

| Molecule | Scientific role |
|---|---|
| H₂O | Habitability indicator — strong NIR bands |
| CO₂ | Greenhouse gas |
| **O₂** | **Biosignature** — 0.76 µm A-band |
| **O₃** | **Biosignature** — UV Hartley band |
| **CH₄** | **Biosignature** when co-present with O₂ |
| N₂ | Bulk atmospheric filler |
| **N₂O** | **Biosignature** — microbial source |
| CO | Volcanic / photochemical indicator |
| H₂ | Primordial outgassing |
| H₂S | Volcanic outgassing |
| SO₂ | Volcanic activity |
| NH₃ | Nitrogen cycle |

The output can be a vector of 12 values.

---

## Evaluation Plan

The model is evaluated on a held-out test set (15% of data) using:

- **RMSE and MAE** per molecule 
- **MAE** per Molecule
- **R²** per molecule — measures fraction of variance explained
- **Calibration error** — mean absolute deviation between empirical and nominal coverage across confidence levels
correlation


## Deliverables:
1.	Notebook 1: EDA Jupyter notebook with statistical analysis and visualisations of INARA dataset
2.	Codebase: Model training and evaluation codebase 
3.	Results: Experimental results: per-molecule RMSE/R² tables, calibration curves, scatter plots
4.	Validation: Domain transfer analysis: predictions on real JWST spectra vs. published retrievals
5.	Report: Final project report covering background, methodology, results, and discussion


---

## References

1. Zorzan et al. (2025) — *ApJS* 277:38 — INARA dataset and baseline 1D CNN retrieval model
2. Márquez-Neila, P. et al. (2018) — *Nature Astronomy* 2 — random forest atmospheric retrieval (seminal ML retrieval work)
3. Vasist, M. et al. (2023) — *A&A* — neural posterior estimation with normalising flows
4. Gebhard, T. et al. (2024) — *A&A* — flow matching for full posterior atmospheric retrieval (current state of the art)
5. Gal, Y. & Ghahramani, Z. (2016) — *ICML* — MC Dropout as approximate Bayesian inference
6. JWST Transiting Exoplanet Community ERS Team (2023) — *Nature* 614:649 — WASP-39 b benchmark
