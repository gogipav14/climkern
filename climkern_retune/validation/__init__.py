"""
Validation utilities for ClimKern-Retune.

Provides:
- Q²-based cross-validation for model selection
- Comparison utilities for tunable vs traditional kernels
"""

from climkern_retune.validation.cross_validation import (
    CVResult,
    CrossValidator,
    KFoldCV,
    TimeSeriesCV,
    compute_q2_score,
)
from climkern_retune.validation.kernel_compare import (
    KernelComparison,
    compare_kernels,
    validate_against_ipcc,
    IPCC_AR6_FEEDBACKS,
)

__all__ = [
    # Cross-validation
    "CVResult",
    "CrossValidator",
    "KFoldCV",
    "TimeSeriesCV",
    "compute_q2_score",
    # Kernel comparison
    "KernelComparison",
    "compare_kernels",
    "validate_against_ipcc",
    "IPCC_AR6_FEEDBACKS",
]
