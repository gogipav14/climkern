# ClimKern-Retune

Two-Step Radiative Kernel Harmonization and Data-Driven Estimation via Constrained NIPALS-PLS

## Overview

ClimKern-Retune extends [ClimKern v1.2](https://github.com/tyfolino/climkern) (Janoski et al., 2025) with a two-step approach that bridges the gap between discrete pre-computed kernels and fully data-driven kernel estimation.

**Step 1 (Kernel Harmonization):** The 11 pre-computed kernel sets in ClimKern v1.2 disagree by up to 50% in polar regions. Step 1 treats them as an ensemble: a global constrained NIPALS-PLS baseline combines the 11 kernel predictions, then SIMCA discovers 11 data-driven regimes (one per kernel) for regime-weighted retuning. Achieves Q² = 0.554 on real CERES+NCEP holdout data with 76% interkernel RMSE reduction.

**Step 2 (Data-Driven Kernels):** The same PLS/SIMCA/constraint framework learns kernel sensitivities directly from 27 atmospheric state features (temperature profiles, humidity profiles, surface temperature, cloud fraction). Achieves Q² = 0.704 on real holdout data with physically correct feedback signs.

### Key Features

- **Two-stage kernel harmonization** of 11 Janoski et al. kernels via PLS + SIMCA regime routing
- **JAX-accelerated NIPALS-PLS regression** with missing data handling (via [open_nipals](https://github.com/gogipav14/open_nipals))
- **Multi-level radiative constraints** (surface Stefan-Boltzmann, TOA energy balance)
- **Data-driven SIMCA regime classification** (11 kernel regimes, outperforming 16 prescribed climate-state regimes)
- **Q² dashboard** with thermodynamic compliance for principled method selection
- **VIP analysis and PLS diagnostics** (Hotelling's T², DModX, loading plots)
- **Observational training** on CERES EBAF TOA Ed4.2 + NCEP/NCAR Reanalysis 1
- **Validated on real data**: Q² = 0.554 (Step 1), Q² = 0.704 (Step 2)

### Mathematical Framework

**Step 1 — Kernel Harmonization:**

```
Input:    X = [ΔR_kernel1, ΔR_kernel2, ..., ΔR_kernel11]  (n × 11)
Target:   Y = [ΔOLR, ΔOSR]                                 (n × 2)

Global PLS:  Y = X · B + ε   where  B = W(P'W)⁻¹Q'
Regimes:     11 SIMCA regimes (one per kernel, assigned by residual proximity)
Retune:      4 competing approaches compared by Q² + thermodynamic compliance
```

**Step 2 — Data-Driven Kernels:**

```
Input:    X = [ΔT(p₁..p₁₇), Δq(p₁..p₈), ΔT_sfc, Δcloud]  (n × 27)
Target:   Y = [ΔOLR, ΔOSR]                                    (n × 2)

PLS: Y = X · B + ε  with optional physical constraints:
  Minimize: ||Y - XB||² + λ₁||C_SB||² + λ₂||C_cons||²
```

## Installation

```bash
# Clone the repository
git clone https://github.com/gogipav14/climkern.git
cd climkern

# Install with dependencies
pip install -e .

# Or install with development tools
pip install -e ".[dev]"
```

**Note:** Requires the `open_nipals` package:
```bash
pip install git+https://github.com/gogipav14/open_nipals.git
```

## Quick Start

### Step 1: Kernel Harmonization

```python
from kernel_harmonizer import KernelRegimeHarmonizer

# X_kernels: (n_samples, 11) — predictions from 11 kernel sets
# Y_obs: (n_samples, 1) — CERES observed ΔOLR
harmonizer = KernelRegimeHarmonizer(n_components=3)
harmonizer.fit_all(X_kernels_train, Y_obs_train, kernel_names)

# Q² dashboard comparing all methods
dashboard = harmonizer.q2_dashboard(X_kernels_test, Y_obs_test)
print(f"Best method: {dashboard.best_method} (Q² = {dashboard.best_q2:.3f})")

# Predict with best retune approach
Y_pred = harmonizer.predict(X_kernels_new, method="best")
```

### Step 2: Data-Driven Kernel

```python
from tunable_kernel import TunableKernel, KernelConfig

# Configure kernel (27 atmospheric features → TOA flux)
config = KernelConfig(n_components=10)
kernel = TunableKernel(config=config)

# X: (n_samples, 27) — ΔT at 17 levels, Δq at 8 levels, ΔT_sfc, Δcloud
# Y: (n_samples, 2) — [ΔOLR, ΔOSR]
kernel.fit(X_train, Y_train, feature_names=feature_names)

# Compute radiative response
output = kernel.compute(X_test)
print(f"LW response: {output.delta_r_lw.mean():.2f} W/m²")

# Evaluate with Q²
q2 = kernel.evaluate(X_test, Y_test)
print(f"Q² = {q2:.4f}")
```

### Using the Core PLS API

```python
from nipals_pls import ConstrainedNipalsPLS

# Center the data
X_c = X - X.mean(axis=0)
Y_c = Y - Y.mean(axis=0)

# Fit constrained PLS
model = ConstrainedNipalsPLS(n_components=5)
model.fit(X_c, Y_c)
Y_pred = model.predict(X_c)

# Access PLS internals
results = model.results_
print(f"X-variance explained: {results.x_variance_explained}")
print(f"Weights shape: {results.x_weights.shape}")
```

## Project Structure

```
climkern/
├── kernel_harmonizer.py       # Step 1: Two-stage kernel harmonization
├── nipals_pls.py              # Constrained NIPALS-PLS implementation
├── state_classifier.py        # SIMCA + climate regime classification
├── tunable_kernel.py          # Step 2: Data-driven tunable kernel
├── validate_real_data.py      # Real data loading and validation
├── backend.py                 # JAX/NumPy backend selection
├── loaders.py                 # CERES, NCEP data readers
├── preprocessors.py           # Anomaly computation utilities
├── cross_validation.py        # K-fold, time-series CV
├── kernel_compare.py          # Traditional kernel comparison
├── constants.py               # Physical constants (σ, etc.)
├── __init__.py                # Package exports
├── climkern/                  # Original ClimKern v1.2 frontend
│   ├── frontend.py            # calc_alb_feedback, calc_cloud_LW, etc.
│   └── data/                  # Kernel data files (11 kernel sets)
├── paper/
│   ├── climkern_retune.tex    # Paper manuscript
│   ├── generate_figures.py    # Figure generation (11 figures)
│   └── compute_real_q2.py     # Real-data Q² computation
├── tests/
│   └── test_jax_nipals_validation.py  # 44-test validation suite
├── data/
│   └── merged_CERES_NCEP_2003-2020.nc  # Merged observational dataset
└── results/                   # Generated plots and analysis
```

## Data Sources

### CERES EBAF TOA Ed4.2

- **TOA fluxes**: Outgoing longwave radiation (OLR), outgoing shortwave radiation (OSR)
- **Cloud fraction**: Monthly mean cloud area fraction
- **Resolution**: 1° × 1° monthly means, 2003-2020

### NCEP/NCAR Reanalysis 1

- **Temperature profiles**: 17 pressure levels (1000-10 hPa)
- **Humidity profiles**: Specific humidity at 8 levels (1000-300 hPa)
- **Surface temperature**: Skin temperature
- **Resolution**: 2.5° × 2.5° monthly means

### Predictor Variables (X)

```
Step 1: X = 11 kernel ΔR predictions (n × 11)
Step 2: X = atmospheric state anomalies (n × 27):
  - Temperature anomalies: ΔT(p) at 17 pressure levels
  - Humidity anomalies: Δq(p) at 8 pressure levels
  - Surface temperature: ΔT_sfc
  - Cloud fraction: Δcf
```

### Response Variables (Y)

```
- ΔOLR: TOA outgoing longwave radiation anomaly (W/m²)
- ΔOSR: TOA outgoing shortwave radiation anomaly (W/m²)
```

### Training/Test Split

- **Training**: 2003-2017 (80%, ~13,200 samples)
- **Testing**: 2018-2020 (20%, ~3,300 samples)

## State Classification

### Data-Driven Kernel Regimes (Step 1)

11 SIMCA regimes, one per kernel set, assigned by residual proximity to CERES observations. Each regime captures the conditions under which a particular kernel best predicts the observed TOA flux. Q² = 0.544 with regime-weighted blending.

### Prescribed Climate-State Regimes (Step 2)

16 prescribed regimes from physical classification:

| Latitude Band | Cloud State | Stability |
|---|---|---|
| Tropical (\|lat\| < 15°) | Clear (CF < 0.5) | Convective (LTS < 18K) |
| Subtropical (15-35°) | Cloudy (CF >= 0.5) | Stable (LTS >= 18K) |
| Midlatitude (35-60°) | | |
| Polar (\|lat\| >= 60°) | | |

Note: On real CERES+NCEP data, the 11 data-driven kernel regimes (Q² = 0.544) outperform the 16 prescribed regimes (Q² = 0.452).

## Physical Constraints

1. **Stefan-Boltzmann (Surface)**: Enforces linearized blackbody emission
   ```
   ΔF_sfc ≈ 4εσT³ΔT
   ```

2. **Energy Conservation (Global)**: Ensures area-weighted flux balance
   ```
   ∫ ΔR dA = ΔN (global imbalance change)
   ```

3. **TOA Effective Emissivity**: Constrains effective temperature relationship
   ```
   OLR ≈ ε_eff σ T_eff⁴
   ```

## Validation

### Test Results (100% Pass Rate)

| Category | Tests | Passed | Status |
|---|---|---|---|
| Core NIPALS-PLS | 8 | 8 | Pass |
| Convergence Properties | 4 | 4 | Pass |
| NaN Handling | 4 | 4 | Pass |
| NIPALS-PCA | 4 | 4 | Pass |
| Constrained PLS | 4 | 4 | Pass |
| Physical Constraints | 4 | 4 | Pass |
| Numerical Stability | 4 | 4 | Pass |
| Distance Metrics | 4 | 4 | Pass |
| Component Addition | 4 | 4 | Pass |
| Integration | 4 | 4 | Pass |
| **Total** | **44** | **44** | **100%** |

### Real-Data Validation

Validated on CERES EBAF TOA Ed4.2 + NCEP/NCAR Reanalysis 1 (2003-2020):

| Step | Method | Real Q² | Notes |
|---|---|---|---|
| 1 | Best individual kernel | < 0 | Individual kernels fail on real data |
| 1 | Simple mean of 11 | 0.290 | Naive averaging |
| 1 | Global PLS (Stage 1) | 0.544 | Constrained NIPALS-PLS |
| 1 | Best retune (Stage 2) | 0.554 | SIMCA-augmented features |
| 2 | Data-driven (27 features) | 0.704 | 10-component PLS |

Physical consistency verified:
- Planck response: +1.56 W/m²/K (warming increases OLR)
- Water vapor: -1.82 W/m²/K (greenhouse trapping reduces OLR)
- Cloud fraction: -0.12 W/m²/% (cloud greenhouse effect)

### Validation Checklist

| Validation Step | Status | Notes |
|---|---|---|
| Unit tests pass | Pass (44/44) | All tests pass |
| Constraint propagation works | Pass | Score-based prediction implemented |
| Real CERES+NCEP data tested | Pass | Q² = 0.704 on holdout (2018-2020) |
| Physical sign/magnitude correct | Pass | Planck, WV, cloud signs verified |
| Out-of-sample prediction | Pass | Temporal holdout 2018-2020 |
| Compared to Soden kernels | Pending | |

See `VALIDATION_REPORT.md` for details.

### Running Tests

```bash
# Run validation test suite
pytest tests/test_jax_nipals_validation.py -v

# Generate paper figures (requires merged dataset)
python3 paper/generate_figures.py

# Compute real-data Q² values
python3 paper/compute_real_q2.py
```

## References

- Janoski, T., et al. (2025). ClimKern v1.2: A new Python package for calculating radiative feedbacks. *Geoscientific Model Development*, 18, 3065-3083.
- Wold, S., et al. (2001). PLS-regression: a basic tool of chemometrics. *Chemometrics and Intelligent Laboratory Systems*.
- Soden, B.J., et al. (2008). Quantifying climate feedbacks using radiative kernels. *Journal of Climate*.
- Forster, P., et al. (2021). The Earth's Energy Budget, Climate Feedbacks, and Climate Sensitivity. IPCC AR6 WGI Chapter 7.

## License

MIT License
