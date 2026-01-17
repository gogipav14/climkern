"""
Data loading and preprocessing module.
"""

import sys
import os

# Add parent directory to path
_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

from preprocessors import (
    compute_anomalies,
    create_feature_matrix,
)

__all__ = [
    "compute_anomalies",
    "create_feature_matrix",
]
