"""
Core module for ClimKern-Retune.

Contains the main components for tunable radiative kernel estimation:
- Constrained NIPALS-PLS regression
- Climate state classification
- Tunable kernel interface
"""

from climkern_retune.core.nipals_pls import (
    ConstrainedNipalsPLS,
    PhysicalConstraint,
    create_multilevel_constraints,
)
from climkern_retune.core.state_classifier import (
    ClimateStateClassifier,
    SIMCAClassifier,
    ClimateState,
    compute_lts,
    compute_eis,
)
from climkern_retune.core.tunable_kernel import (
    TunableKernel,
    MultiStateKernel,
    KernelConfig,
    KernelOutput,
)

__all__ = [
    # NIPALS-PLS
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
