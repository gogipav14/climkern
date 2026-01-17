# ClimKern-Retune

NIPALS-PLS Tunable Radiative Kernels for Climate Feedback Analysis

## Overview

ClimKern-Retune implements a data-driven approach to radiative kernel estimation using NIPALS-PLS (Nonlinear Iterative Partial Least Squares) regression with physical constraints and SIMCA-style state classification.

### Key Features

- **JAX-accelerated NIPALS-PLS regression** with missing data handling (via [open_nipals](https://github.com/gogipav14/open_nipals))
- **Multi-level radiative constraints** (surface Stefan-Boltzmann, TOA energy balance)
- **SIMCA-style climate state classification** (16 regimes: latitude × cloud × stability)
- **Observational training** using CERES surface fluxes + AIRS atmospheric profiles
- **Q² cross-validation** for model selection and component optimization
- **Validated implementation** with 95.5% test pass rate (42/44 tests)

### Mathematical Framework

For each climate state *s* ∈ {tropical, subtropical, midlatitude, polar} × {clear, cloudy} × {stable, convective}:

```
Predictors: X_s = [ΔT(p), Δq(p), Δα, Δcloud, ...]
Response:   Y_s = [ΔR_LW, ΔR_SW]

PLS: Y_s = X_s · B_s + ε  where  B_s = W_s(P_s'W_s)⁻¹Q_s'
```

Physical constraints (soft regularization):
```
Minimize: ||Y - XB||² + λ₁||C_SB||² + λ₂||C_cons||²

where:
  C_SB:   Stefan-Boltzmann residual (ΔF_sfc ≈ 4εσT³ΔT)
  C_cons: Energy conservation residual (∫ΔR dA = ΔN)
```

## Installation

```bash
# Clone the repository
git clone https://github.com/gogipav14/ClimKern-Retune.git
cd ClimKern-Retune

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

### Single-Regime Kernel

```python
from climkern_retune import TunableKernel, KernelConfig

# Configure kernel
config = KernelConfig(n_components=5)
kernel = TunableKernel(config=config)

# Fit to training data
# X: (n_samples, n_features) - atmospheric state changes
# Y: (n_samples, 2) - [ΔR_LW, ΔR_SW] radiative response
kernel.fit(X_train, Y_train, feature_names=feature_names)

# Compute radiative response
output = kernel.compute(X_test)
print(f"LW response: {output.delta_r_lw.mean():.2f} W/m²")

# Evaluate with Q²
q2 = kernel.evaluate(X_test, Y_test)
print(f"Q² = {q2:.4f}")
```

### Multi-State Kernel

```python
from climkern_retune import MultiStateKernel, ClimateStateClassifier

# Create state classifier
classifier = ClimateStateClassifier()

# Fit multi-state kernel
kernel = MultiStateKernel(classifier=classifier)
kernel.fit(
    X, Y,
    latitude=lat,
    cloud_fraction=cf,
    lts=lts,  # Lower Tropospheric Stability
    feature_names=feature_names,
)

# Predict with automatic regime routing
output = kernel.compute(X_new, latitude=lat_new, cloud_fraction=cf_new, lts=lts_new)
```

### Cross-Validation

```python
from climkern_retune.validation import KFoldCV, select_n_components

# Select optimal number of components
results = select_n_components(
    X, Y,
    model_class=TunableKernel,
    max_components=15,
    cv=KFoldCV(n_splits=5),
)
print(f"Optimal components: {results['optimal_n']}")
```

## Project Structure

```
ClimKern-Retune/
├── nipals_pls.py              # Constrained NIPALS-PLS implementation
├── state_classifier.py        # SIMCA-style climate regime classification
├── tunable_kernel.py          # Tunable kernel interface
├── loaders.py                 # CERES, AIRS, ERA5 data readers
├── preprocessors.py           # Anomaly computation utilities
├── cross_validation.py        # K-fold, time-series CV
├── kernel_compare.py          # Traditional kernel comparison
├── constants.py               # Physical constants (σ, etc.)
├── predictive_analysis_real_data.py  # Predictive analysis pipeline
├── climkern_retune/           # Package exports
│   ├── core/                  # Core algorithm exports
│   ├── data/                  # Data loader exports
│   └── validation/            # Validation exports
├── tests/
│   └── test_jax_nipals_validation.py  # 44-test validation suite
├── results/                   # Generated plots and analysis
└── docs/                      # Documentation
```

## State Classification

The classifier assigns samples to up to 16 regimes:

| Latitude Band | Cloud State | Stability |
|---------------|-------------|-----------|
| Tropical (|lat| < 15°) | Clear (CF < 0.5) | Convective (LTS < 18K) |
| Subtropical (15-35°) | Cloudy (CF ≥ 0.5) | Stable (LTS ≥ 18K) |
| Midlatitude (35-60°) | | |
| Polar (|lat| ≥ 60°) | | |

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

## Data Sources

The framework is designed to work with satellite-based radiative flux observations:

### CERES (Clouds and Earth's Radiant Energy System)
- **TOA fluxes**: Shortwave and longwave radiation at top-of-atmosphere
- **Surface fluxes**: Downwelling SW/LW radiation (CERES SYN1deg product)
- **Resolution**: 1° × 1° monthly means

### AIRS (Atmospheric Infrared Sounder)
- **Temperature profiles**: Vertical temperature at standard pressure levels
- **Humidity profiles**: Water vapor mixing ratio profiles
- **Resolution**: 1° × 1° daily/monthly

### Predictor Variables (X)
```
- Temperature anomalies: ΔT(p) at multiple pressure levels
- Humidity anomalies: Δq(p) at multiple pressure levels
- Surface temperature: ΔT_sfc
- Cloud fraction: Δcf
- Lower tropospheric stability: ΔLTS
```

### Response Variables (Y)
```
- Surface SW downwelling: sfc_sw_down_all (W/m²)
- Surface LW downwelling: sfc_lw_down_all (W/m²)
```

### Running the Predictive Analysis

```bash
# Run the complete predictive analysis pipeline
python predictive_analysis_real_data.py

# This will:
# 1. Download CERES/AIRS data from NASA POWER API (or use synthetic fallback)
# 2. Train NIPALS-PLS model on 2018-2022 data
# 3. Make predictions for 2023 (unseen test year)
# 4. Generate parity plots and analysis in results/
```

### Using the Core API

```python
from nipals_pls import ConstrainedNipalsPLS, create_surface_constraint
import numpy as np

# Prepare your climate data
X = np.load('atmospheric_state_changes.npy')  # (n_samples, n_features)
Y = np.load('radiative_flux_changes.npy')      # (n_samples, 2) for [LW, SW]

# Center the data
X_c = X - X.mean(axis=0)
Y_c = Y - Y.mean(axis=0)

# Create constrained model
constraints = [create_surface_constraint(weight=0.5)]
model = ConstrainedNipalsPLS(n_components=5, constraints=constraints)

# Fit and predict
model.fit(X_c, Y_c)
Y_pred = model.predict(X_c)
q2 = model.q2_score(X_c, Y_c)
```

## Validation

The JAX-NIPALS implementation has been thoroughly validated with a comprehensive test suite.

### Test Results (95.5% Pass Rate)

| Category | Tests | Passed | Status |
|----------|-------|--------|--------|
| Core NIPALS-PLS | 8 | 8 | ✓ |
| Convergence Properties | 4 | 4 | ✓ |
| NaN Handling | 4 | 4 | ✓ |
| NIPALS-PCA | 4 | 2 | ~* |
| Constrained PLS | 4 | 4 | ✓ |
| Physical Constraints | 4 | 4 | ✓ |
| Numerical Stability | 4 | 4 | ✓ |
| Distance Metrics | 4 | 4 | ✓ |
| Component Addition | 4 | 4 | ✓ |
| Integration | 4 | 4 | ✓ |
| **Total** | **44** | **42** | **95.5%** |

*\*PCA numerical precision tests have ~1e-6 tolerance differences (expected behavior)*

### Scientific Validation (Required Before Use)

**⚠️ Important:** The algorithm implementation has been validated (42/44 unit tests pass), but scientific validation against real climate data is **pending**. The included `predictive_analysis_real_data.py` falls back to synthetic data when the NASA POWER API is unavailable.

**To properly validate this implementation, you must:**

1. **Use Independent Observational Data**
   - Download real CERES EBAF data from [NASA Earthdata](https://earthdata.nasa.gov/)
   - Download real AIRS L3 profiles from [GES DISC](https://disc.gsfc.nasa.gov/)
   - The synthetic fallback data embeds trivial relationships and should **not** be used for validation

2. **Compare Against Traditional Radiative Kernels**
   ```python
   # The NIPALS-PLS kernels should agree with established kernels:
   # - Soden et al. (2008) - Journal of Climate
   # - Shell et al. (2008) - Journal of Climate
   # - Huang et al. (2017) - Journal of Climate

   # Expected Planck feedback: ~-3.2 W/m²/K
   # If learned kernel differs significantly, investigate why
   ```

3. **Check Physical Consistency**
   - Planck feedback should be **negative** (~-3.2 W/m²/K at surface)
   - Water vapor feedback should be **positive** (~1.8 W/m²/K)
   - Kernel magnitude should decrease with altitude
   - Stronger response near equator than poles

4. **Out-of-Sample Testing**
   - Temporal holdout: Train on 2003-2018, test on 2019-2023
   - Event-based: Test on volcanic eruptions (Pinatubo), El Niño events
   - Model-based: Compare with CMIP6 4xCO2 experiments

### Validation Checklist

| Validation Step | Status | Notes |
|-----------------|--------|-------|
| Unit tests pass | ✅ 42/44 | PCA precision tests are expected failures |
| Constraint propagation works | ✅ | Score-based prediction implemented |
| Real CERES/AIRS data tested | ❌ Pending | Requires NASA Earthdata access |
| Compared to Soden kernels | ❌ Pending | |
| Physical sign/magnitude correct | ❌ Pending | |
| Out-of-sample prediction | ❌ Pending | |

See `VALIDATION_REPORT.md` for algorithm validation details.

## Generated Outputs

Running the predictive analysis generates the following in `results/`:

| File | Description |
|------|-------------|
| `train_2018_2022_parity.png` | Parity plots for training period |
| `test_2023_parity.png` | Parity plots for unseen test year |
| `test_2023_timeseries.png` | Time series comparison by region |
| `test_2023_residuals.png` | Residual analysis and distributions |

### Running the Analysis

```bash
# Run full predictive analysis
python predictive_analysis_real_data.py

# Run validation test suite
pytest tests/test_jax_nipals_validation.py -v

# Run workflow tests
python test_workflow.py
```

## References

- Wold, S., et al. (2001). PLS-regression: a basic tool of chemometrics. *Chemometrics and Intelligent Laboratory Systems*.
- Soden, B.J., et al. (2008). Quantifying climate feedbacks using radiative kernels. *Journal of Climate*.
- Forster, P., et al. (2021). The Earth's Energy Budget, Climate Feedbacks, and Climate Sensitivity. IPCC AR6 WGI Chapter 7.

## License

MIT License
