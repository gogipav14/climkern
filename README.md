# ClimKern-Retune

NIPALS-PLS Tunable Radiative Kernels for Climate Feedback Analysis

## Overview

ClimKern-Retune implements a data-driven approach to radiative kernel estimation using NIPALS-PLS (Nonlinear Iterative Partial Least Squares) regression with physical constraints and SIMCA-style state classification.

### Key Features

- **NIPALS-PLS regression** with missing data handling (via [open_nipals](https://github.com/gogipav14/open_nipals))
- **Multi-level radiative constraints** (surface Stefan-Boltzmann, TOA energy balance)
- **SIMCA-style climate state classification** (latitude × cloud × stability regimes)
- **Observational training** using CERES TOA fluxes + AIRS atmospheric profiles
- **Q² cross-validation** for model selection and vertical resolution comparison
- **Traditional kernel comparison** for validation

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
├── src/climkern_retune/
│   ├── core/
│   │   ├── nipals_pls.py       # Constrained NIPALS-PLS
│   │   ├── state_classifier.py # Climate regime classification
│   │   └── tunable_kernel.py   # Kernel interface
│   ├── data/
│   │   ├── loaders.py          # CERES, AIRS, ERA5 readers
│   │   └── preprocessors.py    # Anomaly computation
│   ├── validation/
│   │   ├── cross_validation.py # K-fold, time-series CV
│   │   └── kernel_compare.py   # Traditional kernel comparison
│   └── constraints/            # Physical constraint functions
├── tests/                      # Unit tests
├── examples/                   # Usage examples
└── configs/                    # Default configurations
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

## Validation

- Compare Q² between standard 17-level and adaptive vertical resolution
- Validate against IPCC AR6 assessed feedback ranges
- Test extrapolation to 4×CO2 scenarios

## References

- Wold, S., et al. (2001). PLS-regression: a basic tool of chemometrics. *Chemometrics and Intelligent Laboratory Systems*.
- Soden, B.J., et al. (2008). Quantifying climate feedbacks using radiative kernels. *Journal of Climate*.
- Forster, P., et al. (2021). The Earth's Energy Budget, Climate Feedbacks, and Climate Sensitivity. IPCC AR6 WGI Chapter 7.

## License

MIT License
