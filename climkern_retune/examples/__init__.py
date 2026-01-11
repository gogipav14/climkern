"""
Example workflows for ClimKern-Retune.

Contains demonstration scripts showing how to:
- Train tunable radiative kernels
- Perform cross-validation for model selection
- Compare with traditional kernels
"""

from climkern_retune.examples.basic_workflow import (
    generate_synthetic_climate_data,
    demo_single_regime_kernel,
    demo_multistate_kernel,
)

__all__ = [
    "generate_synthetic_climate_data",
    "demo_single_regime_kernel",
    "demo_multistate_kernel",
]
