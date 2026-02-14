"""
ClimKern-Retune: From Kernel Harmonization to Data-Driven Tunable Radiative Kernels

A two-step extension of ClimKern v1.2 (Janoski et al. 2025):

Step 1 — Kernel Harmonization:
    Optimally combine the 11 pre-computed ClimKern kernel sets using
    constrained NIPALS-PLS with SIMCA regime routing. Reduces interkernel
    spread while preserving interpretability.

Step 2 — Data-Driven Extension:
    Learn tunable kernel sensitivities directly from atmospheric state
    observations using the same PLS/SIMCA/constraint framework.

Key Features
------------
- NIPALS-PLS regression with physical constraints (Stefan-Boltzmann, energy conservation)
- SIMCA-style climate state classification (16 regimes)
- Kernel harmonization (Step 1): optimal weighting of 11 ClimKern kernel sets
- Data-driven kernels (Step 2): learned from CERES + NCEP observations
- Dual JAX/NumPy backend with GPU acceleration and autodiff
- Q² cross-validation for model selection

Quick Start — Step 1 (Kernel Harmonization)
--------------------------------------------
>>> from kernel_harmonizer import KernelHarmonizer
>>> harmonizer = KernelHarmonizer(n_components=3)
>>> harmonizer.fit(X_kernels, Y_ceres, kernel_names=names)
>>> q2 = harmonizer.evaluate(X_test, Y_test)

Quick Start — Step 2 (Data-Driven Kernels)
-------------------------------------------
>>> from tunable_kernel import TunableKernel, KernelConfig
>>> kernel = TunableKernel(config=KernelConfig(n_components=8))
>>> kernel.fit(X_atm_state, Y_ceres, feature_names=features)
>>> output = kernel.compute(X_new)

References
----------
Janoski, T.P. et al. (2025). ClimKern v1.2. Geosci. Model Dev., 18, 3065-3079.
Wold, S. et al. (2001). PLS-regression: a basic tool of chemometrics.
Soden, B.J. et al. (2008). Quantifying climate feedbacks using radiative kernels.
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
    # Tunable kernels (Step 2)
    TunableKernel,
    MultiStateKernel,
    KernelConfig,
    KernelOutput,
)

# Step 1: Kernel Harmonization
from kernel_harmonizer import KernelHarmonizer, HarmonizationResult, MultiKernelLoader

# Step 1 wrapper with TunableKernel-compatible interface
from tunable_kernel import HarmonizedKernel

__version__ = "0.2.0"
__author__ = "Gorgi Pavlov"

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
    # Step 1: Kernel Harmonization
    "KernelHarmonizer",
    "HarmonizationResult",
    "MultiKernelLoader",
    "HarmonizedKernel",
    # Step 2: Data-Driven Kernels
    "TunableKernel",
    "MultiStateKernel",
    "KernelConfig",
    "KernelOutput",
]
