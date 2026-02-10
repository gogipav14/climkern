"""
Dual-backend abstraction for JAX/NumPy compatibility.

Auto-detects JAX availability and provides a consistent array API.
When JAX is available, uses JAX arrays for GPU acceleration.
When JAX is not available, falls back to NumPy.

Usage:
    from backend import get_array_module, to_array, to_numpy, HAS_JAX
    xp = get_array_module()
    x = to_array(data)
    result_np = to_numpy(result)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

try:
    import jax

    # Enable float64 for scientific computing (JAX defaults to float32)
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    HAS_JAX = True
except ImportError:
    HAS_JAX = False


def get_array_module():
    """Return jnp if JAX available, else np."""
    return jnp if HAS_JAX else np


def to_array(x, dtype=None):
    """Convert to appropriate backend array type.

    Parameters
    ----------
    x : array-like
        Input data.
    dtype : dtype, optional
        Desired data type.

    Returns
    -------
    array
        JAX array if JAX available, else NumPy array.
    """
    xp = get_array_module()
    if dtype is not None:
        return xp.asarray(x, dtype=dtype)
    return xp.asarray(x)


def to_numpy(x) -> NDArray:
    """Ensure result is a NumPy array.

    Needed for sklearn interop, xarray, and output to users.
    """
    if HAS_JAX and isinstance(x, jax.Array):
        return np.asarray(x)
    return np.asarray(x)


def get_nipals_pls_class():
    """Return the appropriate NipalsPLS class for the available backend."""
    if HAS_JAX:
        try:
            from open_nipals.jax import NipalsPLS

            return NipalsPLS
        except ImportError:
            pass
    try:
        from open_nipals.nipalsPLS import NipalsPLS

        return NipalsPLS
    except ImportError:
        return None


def get_nipals_pca_class():
    """Return the appropriate NipalsPCA class for the available backend."""
    if HAS_JAX:
        try:
            from open_nipals.jax import NipalsPCA

            return NipalsPCA
        except ImportError:
            pass
    try:
        from open_nipals.nipalsPCA import NipalsPCA

        return NipalsPCA
    except ImportError:
        return None
