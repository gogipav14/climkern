"""
ClimKern-Retune: NIPALS-PLS Tunable Radiative Kernels for Climate Feedback Analysis

This package implements a data-driven approach to radiative kernel estimation
using NIPALS-PLS regression with physical constraints and SIMCA-style state
classification for regime-dependent kernels.

Key Features
------------
- NIPALS-PLS regression with missing data handling (via open_nipals)
- Multi-level radiative constraints (surface Stefan-Boltzmann, TOA energy balance)
- SIMCA-style climate state classification (latitude × cloud × stability)
- Support for observational (CERES, AIRS) and model-based training
- Q² cross-validation for model selection
- Validation against traditional radiative kernels

Quick Start
-----------
>>> from climkern_retune import TunableKernel, KernelConfig
>>> from climkern_retune.data import create_feature_matrix
>>>
>>> # Create and fit a tunable kernel
>>> config = KernelConfig(n_components=5)
>>> kernel = TunableKernel(config=config)
>>> kernel.fit(X_train, Y_train, feature_names=feature_names)
>>>
>>> # Compute radiative response
>>> output = kernel.compute(X_test)
>>> print(f"LW response: {output.delta_r_lw.mean():.2f} W/m²")

For state-dependent kernels:
>>> from climkern_retune import MultiStateKernel, ClimateStateClassifier
>>>
>>> kernel = MultiStateKernel()
>>> kernel.fit(X, Y, latitude=lat, cloud_fraction=cf, lts=lts)
>>> output = kernel.compute(X_new, latitude=lat_new)

References
----------
NIPALS algorithm: Wold, S., et al. (2001). PLS-regression: a basic tool of chemometrics.
Radiative kernels: Soden, B.J., et al. (2008). Quantifying climate feedbacks.
"""

from backend import HAS_JAX

from climkern_retune.core import (
    # NIPALS-PLS
    ConstrainedNipalsPLS,
    PhysicalConstraint,
    create_multilevel_constraints,
    # State classification
    ClimateStateClassifier,
    SIMCAClassifier,
    ClimateState,
    compute_lts,
    compute_eis,
    # Tunable kernels
    TunableKernel,
    MultiStateKernel,
    KernelConfig,
    KernelOutput,
)

__version__ = "0.1.0"
__author__ = "ClimKern-Retune Contributors"

__all__ = [
    # Core PLS
    "ConstrainedNipalsPLS",
    "PhysicalConstraint",
    "create_multilevel_constraints",
    # State classification
    "ClimateStateClassifier",
    "SIMCAClassifier",
    "ClimateState",
    "compute_lts",
    "compute_eis",
    # Tunable kernels
    "TunableKernel",
    "MultiStateKernel",
    "KernelConfig",
    "KernelOutput",
]
