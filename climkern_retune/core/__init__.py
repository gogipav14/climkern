"""
Core module for ClimKern-Retune.
"""

import sys
import os

# Add parent directory to path to import from root-level modules
_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

from nipals_pls import (
    ConstrainedNipalsPLS,
    ConstrainedPLSResults,
    PhysicalConstraint,
    stefan_boltzmann_constraint,
    energy_conservation_constraint,
    toa_emissivity_constraint,
    create_surface_constraint,
    create_conservation_constraint,
    create_toa_constraint,
    create_multilevel_constraints,
    STEFAN_BOLTZMANN,
)

from state_classifier import (
    ClimateStateClassifier,
    ClassifierConfig,
    ClimateState,
    LatitudeBand,
    CloudState,
    StabilityState,
    SIMCAClassifier,
    DataDrivenClassifier,
    compute_lts,
    compute_eis,
)

from tunable_kernel import (
    TunableKernel,
    MultiStateKernel,
    KernelConfig,
    KernelOutput,
    VerticalKernelProfile,
)

__all__ = [
    # NIPALS-PLS
    "ConstrainedNipalsPLS",
    "ConstrainedPLSResults",
    "PhysicalConstraint",
    "stefan_boltzmann_constraint",
    "energy_conservation_constraint",
    "toa_emissivity_constraint",
    "create_surface_constraint",
    "create_conservation_constraint",
    "create_toa_constraint",
    "create_multilevel_constraints",
    "STEFAN_BOLTZMANN",
    # State Classification
    "ClimateStateClassifier",
    "ClassifierConfig",
    "ClimateState",
    "LatitudeBand",
    "CloudState",
    "StabilityState",
    "SIMCAClassifier",
    "DataDrivenClassifier",
    "compute_lts",
    "compute_eis",
    # Tunable Kernels
    "TunableKernel",
    "MultiStateKernel",
    "KernelConfig",
    "KernelOutput",
    "VerticalKernelProfile",
]
