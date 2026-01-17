"""
Validation and cross-validation module.
"""

import sys
import os

# Add parent directory to path
_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

from cross_validation import (
    KFoldCV,
    TimeSeriesCV,
    compute_q2_score,
)

__all__ = [
    "KFoldCV",
    "TimeSeriesCV",
    "compute_q2_score",
]
