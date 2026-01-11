"""
Data loading and preprocessing utilities for ClimKern-Retune.

Supports:
- CERES-EBAF TOA and surface radiative fluxes
- AIRS L3 atmospheric temperature and humidity profiles
- ERA5 reanalysis for atmospheric state
- Traditional radiative kernel formats
"""

from climkern_retune.data.loaders import (
    ObservationalData,
    load_ceres_ebaf,
    load_airs_l3,
    load_traditional_kernel,
    STANDARD_PRESSURE_LEVELS,
)
from climkern_retune.data.preprocessors import (
    FeatureMatrix,
    create_feature_matrix,
)

__all__ = [
    # Data loaders
    "ObservationalData",
    "load_ceres_ebaf",
    "load_airs_l3",
    "load_traditional_kernel",
    "STANDARD_PRESSURE_LEVELS",
    # Preprocessors
    "FeatureMatrix",
    "create_feature_matrix",
]
