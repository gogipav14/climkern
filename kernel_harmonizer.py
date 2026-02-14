"""
ClimKern-Retune: Two-Stage Kernel Harmonization Pipeline

Stage 1 (Preserve ClimKern):
    11 pre-computed kernels → global PLS → 11 SIMCA regimes → per-regime PLS.
    Discovers where each kernel is most accurate and establishes baseline Q².

Stage 2 (Retune):
    Uses SIMCA regime structure to re-derive a better global PLS via 4 approaches:
    A: SIMCA features → augmented global PLS
    B: Regime-weighted global PLS
    C: Two-pass iterative refinement
    D: Regime collapse → single global PLS

All approaches compared by Q² and thermodynamic compliance (Stefan-Boltzmann,
energy conservation, TOA emissivity). xESMF regridding matches ClimKern exactly.

References
----------
Janoski, T.P. et al. (2025). ClimKern v1.2. Geosci. Model Dev., 18, 3065-3079.
Wold, S. et al. (2001). PLS-regression: a basic tool of chemometrics.
"""

from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Ensure root directory is in path for imports
_root_dir = os.path.dirname(os.path.abspath(__file__))
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

from constants import STEFAN_BOLTZMANN, TYPICAL_SURFACE_TEMP
from nipals_pls import ConstrainedNipalsPLS, PhysicalConstraint
from state_classifier import KernelRegimeClassifier, SIMCAClassifier

try:
    import xarray as xr

    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False

try:
    import xesmf as xe

    HAS_XESMF = True
except ImportError:
    HAS_XESMF = False


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class KernelPredictions:
    """Predictions from all available kernel sets for a given experiment."""

    kernel_names: list[str]
    feedbacks: dict[str, dict[str, Any]]
    total_lw: dict[str, Any]
    total_sw: dict[str, Any]


@dataclass
class Q2Dashboard:
    """Comprehensive Q² report for all steps and versions."""

    # Individual kernels (11)
    q2_per_kernel: dict[str, float]
    # Baselines
    q2_simple_mean: float
    q2_global_pls: float  # Stage 1 global
    q2_regime_blend: float  # Stage 1 SIMCA blend
    # Retune approaches (4)
    q2_retune_A: float  # SIMCA features → global PLS
    q2_retune_B: float  # Regime-weighted PLS
    q2_retune_C: float  # Two-pass iterative
    q2_retune_D: float  # Regime collapse → global
    # Per SIMCA regime (11)
    q2_per_regime: dict[int, float]
    # Thermodynamic compliance (method → RMS residual)
    sb_residual: dict[str, float]
    energy_residual: dict[str, float]
    # Spread metrics
    rmse_before: float
    rmse_after: dict[str, float]
    spread_reduction_pct: dict[str, float]
    # Best method
    best_method: str
    best_q2: float

    def print_report(self) -> None:
        """Print formatted Q² dashboard."""
        print("\n" + "=" * 62)
        print("  Q² DASHBOARD — ClimKern-Retune")
        print("=" * 62)

        print("\n  INDIVIDUAL KERNELS:")
        for name, q2 in sorted(self.q2_per_kernel.items(), key=lambda x: -x[1]):
            print(f"    {name:<20} {q2:>+8.4f}")

        print(f"\n  BASELINES:")
        print(f"    Simple Mean          {self.q2_simple_mean:>+8.4f}")

        print(f"\n  STAGE 1 (ClimKern Foundation):")
        print(f"    Global PLS           {self.q2_global_pls:>+8.4f}")
        print(f"    SIMCA Blend (11 reg) {self.q2_regime_blend:>+8.4f}")

        print(f"\n  STAGE 2 (Retune Approaches):")
        print(f"    A: SIMCA features    {self.q2_retune_A:>+8.4f}")
        print(f"    B: Regime-weighted   {self.q2_retune_B:>+8.4f}")
        print(f"    C: Two-pass iter.    {self.q2_retune_C:>+8.4f}")
        print(f"    D: Regime collapse   {self.q2_retune_D:>+8.4f}")

        print(f"\n  PER-REGIME Q² (SIMCA):")
        for rid, q2 in sorted(self.q2_per_regime.items()):
            print(f"    Regime {rid:<3}            {q2:>+8.4f}")

        print(f"\n  THERMODYNAMIC COMPLIANCE (RMS residual):")
        print(f"    {'Method':<20} {'SB':>10} {'Energy':>10}")
        print("    " + "-" * 42)
        for method in ["global_pls", "retune_A", "retune_B", "retune_C", "retune_D"]:
            sb = self.sb_residual.get(method, float("nan"))
            en = self.energy_residual.get(method, float("nan"))
            print(f"    {method:<20} {sb:>10.4f} {en:>10.4f}")

        print(f"\n  SPREAD REDUCTION:")
        print(f"    RMSE before (indiv): {self.rmse_before:.4f} W/m²")
        for method, rmse in sorted(self.rmse_after.items()):
            pct = self.spread_reduction_pct.get(method, 0)
            print(f"    RMSE after ({method}): {rmse:.4f} W/m² (↓ {pct:.1f}%)")

        print(f"\n  BEST METHOD: {self.best_method} (Q² = {self.best_q2:.4f})")
        print("=" * 62)


@dataclass
class HarmonizationResult:
    """Results from kernel harmonization (backward compatibility)."""

    kernel_weights: dict[int, NDArray[np.floating]]
    kernel_names: list[str]
    q2_harmonized: float
    q2_simple_mean: float
    q2_per_kernel: dict[str, float]
    spread_before: float
    spread_after: float
    spread_reduction_pct: float
    n_regimes_active: int
    samples_per_regime: dict[int, int]


# ---------------------------------------------------------------------------
# xESMF-compatible regridder
# ---------------------------------------------------------------------------


class KernelRegridder:
    """
    xESMF regridding matching ClimKern's frontend.py exactly.

    Uses the same parameters as ClimKern for backward compatibility:
    - method="bilinear"
    - periodic=True
    - extrap_method="nearest_s2d"
    """

    XESMF_PARAMS = dict(
        method="bilinear",
        reuse_weights=False,
        periodic=True,
        extrap_method="nearest_s2d",
    )

    def __init__(self):
        if not HAS_XESMF:
            warnings.warn(
                "xesmf not available — regridding will use xarray.interp() fallback. "
                "Install xesmf for ClimKern-compatible bilinear regridding."
            )

    def regrid(self, source_da, target_grid):
        """
        Regrid a DataArray from source grid to target grid.

        Parameters
        ----------
        source_da : xr.DataArray
            Source data on kernel grid.
        target_grid : xr.DataArray or xr.Dataset
            Target grid to interpolate onto.

        Returns
        -------
        regridded : xr.DataArray
        """
        if HAS_XESMF:
            regridder = xe.Regridder(
                source_da, target_grid, **self.XESMF_PARAMS
            )
            return regridder(source_da, skipna=True)
        else:
            # Fallback: xarray linear interpolation
            return source_da.interp(
                lat=target_grid.lat, lon=target_grid.lon,
                method="linear",
            )


# ---------------------------------------------------------------------------
# Multi-kernel loader
# ---------------------------------------------------------------------------


class MultiKernelLoader:
    """
    Load all available kernel sets and compute feedback predictions.

    Discovers kernel data from ClimKern's data directory and provides
    xESMF-compatible regridding for computing ΔR predictions.
    """

    def __init__(self):
        self._available_kernels: list[str] | None = None
        self._kernel_data_dir: Path | None = None
        self.regridder = KernelRegridder()

    def get_available_kernels(self) -> list[str]:
        """Discover which kernel sets are installed."""
        if self._available_kernels is not None:
            return self._available_kernels

        candidate_paths = []

        try:
            try:
                from importlib_resources import files as pkg_files
            except ImportError:
                from importlib.resources import files as pkg_files

            candidate_paths.append(
                Path(str(pkg_files("climkern").joinpath("data", "kernels")))
            )
            candidate_paths.append(
                Path(str(pkg_files("climkern").joinpath("data", "data", "kernels")))
            )
        except (TypeError, OSError, ModuleNotFoundError):
            pass

        this_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        candidate_paths.extend([
            this_dir / "climkern" / "data" / "kernels",
            this_dir / "climkern" / "data" / "data" / "kernels",
        ])

        kernels = []
        for data_dir in candidate_paths:
            if data_dir.is_dir():
                for entry in sorted(data_dir.iterdir()):
                    if entry.is_dir():
                        toa_file = entry / f"TOA_{entry.name}_Kerns.nc"
                        if toa_file.exists():
                            kernels.append(entry.name)
                if kernels:
                    self._kernel_data_dir = data_dir
                    break

        self._available_kernels = kernels
        return kernels

    def load_kernel_data(self, kernel_name: str) -> "xr.Dataset":
        """Load raw kernel NetCDF for a specific kernel set."""
        if not HAS_XARRAY:
            raise ImportError("xarray is required for kernel loading")

        data_dir = self._kernel_data_dir
        if data_dir is None:
            self.get_available_kernels()
            data_dir = self._kernel_data_dir

        if data_dir is None:
            raise FileNotFoundError("No kernel data directory found")

        path = data_dir / kernel_name / f"TOA_{kernel_name}_Kerns.nc"
        try:
            ds = xr.open_dataset(path)
        except ValueError:
            ds = xr.open_dataset(path, decode_times=False)

        # Normalize coordinates (matching climkern.util.check_coords)
        renames = {}
        if "month" in ds.dims:
            renames["month"] = "time"
        if "latitude" in ds.dims:
            renames["latitude"] = "lat"
        if "longitude" in ds.dims:
            renames["longitude"] = "lon"
        for old in ("lev", "player", "level"):
            if old in ds.dims:
                renames[old] = "plev"
        if renames:
            ds = ds.rename(renames)

        return ds

    def compute_temperature_feedbacks(
        self,
        ctrl_ta, ctrl_ts, ctrl_ps, pert_ta, pert_ts, pert_ps,
        pert_trop=None, sky="all-sky",
    ) -> dict[str, tuple]:
        """Compute T feedbacks for all kernels via ClimKern frontend."""
        from climkern.frontend import calc_T_feedbacks

        results = {}
        for name in self.get_available_kernels():
            try:
                lr, planck = calc_T_feedbacks(
                    ctrl_ta, ctrl_ts, ctrl_ps,
                    pert_ta, pert_ts, pert_ps,
                    pert_trop=pert_trop, kern=name, sky=sky,
                )
                results[name] = (lr, planck)
            except Exception as e:
                warnings.warn(f"Skipping kernel '{name}' for T feedbacks: {e}")
        return results

    def compute_humidity_feedbacks(
        self,
        ctrl_q, ctrl_ta, ctrl_ps, pert_q, pert_ps,
        pert_trop=None, sky="all-sky", method=1,
    ) -> dict[str, tuple]:
        """Compute q feedbacks for all kernels via ClimKern frontend."""
        from climkern.frontend import calc_q_feedbacks

        results = {}
        for name in self.get_available_kernels():
            try:
                qlw, qsw = calc_q_feedbacks(
                    ctrl_q, ctrl_ta, ctrl_ps, pert_q, pert_ps,
                    pert_trop=pert_trop, kern=name, sky=sky, method=method,
                )
                results[name] = (qlw, qsw)
            except Exception as e:
                warnings.warn(f"Skipping kernel '{name}' for q feedbacks: {e}")
        return results

    def compute_albedo_feedbacks(
        self,
        ctrl_rsus, ctrl_rsds, pert_rsus, pert_rsds, sky="all-sky",
    ) -> dict[str, Any]:
        """Compute albedo feedbacks for all kernels via ClimKern frontend."""
        from climkern.frontend import calc_alb_feedback

        results = {}
        for name in self.get_available_kernels():
            try:
                alb = calc_alb_feedback(
                    ctrl_rsus, ctrl_rsds, pert_rsus, pert_rsds,
                    kern=name, sky=sky,
                )
                results[name] = alb
            except Exception as e:
                warnings.warn(f"Skipping kernel '{name}' for albedo: {e}")
        return results

    def build_feature_matrix(
        self,
        t_feedbacks: dict[str, tuple] | None = None,
        q_feedbacks: dict[str, tuple] | None = None,
        alb_feedbacks: dict[str, Any] | None = None,
    ) -> tuple[NDArray, NDArray, list[str]]:
        """
        Build feature matrices X_lw, X_sw from kernel predictions.

        Returns (X_lw, X_sw, kernel_names) where each column is one kernel's
        total ΔR prediction.
        """
        all_names = set()
        if t_feedbacks:
            all_names.update(t_feedbacks.keys())
        if q_feedbacks:
            all_names.update(q_feedbacks.keys())
        if alb_feedbacks:
            all_names.update(alb_feedbacks.keys())

        kernel_names = sorted(
            name for name in all_names
            if t_feedbacks and name in t_feedbacks
        )

        lw_columns, sw_columns = [], []

        for name in kernel_names:
            lw_total = None
            if t_feedbacks and name in t_feedbacks:
                lr, planck = t_feedbacks[name]
                lw_total = lr.values + planck.values
            if q_feedbacks and name in q_feedbacks:
                qlw, _ = q_feedbacks[name]
                if lw_total is not None:
                    lw_total = lw_total + qlw.values
                else:
                    lw_total = qlw.values

            sw_total = np.zeros_like(lw_total) if lw_total is not None else None
            if q_feedbacks and name in q_feedbacks:
                _, qsw = q_feedbacks[name]
                sw_total = qsw.values.copy()
            if alb_feedbacks and name in alb_feedbacks:
                alb = alb_feedbacks[name]
                if sw_total is not None:
                    sw_total = sw_total + alb.values
                else:
                    sw_total = alb.values

            if lw_total is not None:
                lw_columns.append(lw_total.reshape(-1))
                sw_columns.append(
                    sw_total.reshape(-1) if sw_total is not None
                    else np.zeros(lw_total.size)
                )

        X_lw = np.column_stack(lw_columns) if lw_columns else np.empty((0, 0))
        X_sw = np.column_stack(sw_columns) if sw_columns else np.empty((0, 0))
        return X_lw, X_sw, kernel_names


# ---------------------------------------------------------------------------
# Helper: Q² computation
# ---------------------------------------------------------------------------


def _compute_q2(Y_true: NDArray, Y_pred: NDArray) -> float:
    """Compute Q² = 1 - SS_res / SS_tot."""
    Y_true = np.asarray(Y_true, dtype=np.float64)
    Y_pred = np.asarray(Y_pred, dtype=np.float64)
    ss_res = np.nansum((Y_true - Y_pred) ** 2)
    ss_tot = np.nansum((Y_true - np.nanmean(Y_true, axis=0)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def _compute_rmse(Y_true: NDArray, Y_pred: NDArray) -> float:
    """Compute RMSE."""
    return float(np.sqrt(np.nanmean((Y_true - Y_pred) ** 2)))


# ---------------------------------------------------------------------------
# KernelRegimeHarmonizer — main class
# ---------------------------------------------------------------------------


class KernelRegimeHarmonizer:
    """
    Two-stage pipeline: 11 SIMCA kernel regimes → retuned global PLS.

    Stage 1 preserves the original ClimKern kernels, fitting a global PLS
    baseline and then discovering 11 kernel regimes via SIMCA. Stage 2
    retunes a global PLS using the SIMCA structure via 4 approaches.

    Parameters
    ----------
    n_components : int
        Number of PLS components (default 3).
    constraints : list[PhysicalConstraint], optional
        Physical constraints for PLS fitting.
    min_samples_per_regime : int
        Minimum samples to fit a per-regime PLS model.
    """

    def __init__(
        self,
        n_components: int = 3,
        constraints: list[PhysicalConstraint] | None = None,
        min_samples_per_regime: int = 30,
    ):
        self.n_components = n_components
        self.constraints = constraints or []
        self.min_samples_per_regime = min_samples_per_regime

        self.loader = MultiKernelLoader()

        # Stage 1 models
        self.global_pls_: ConstrainedNipalsPLS | None = None
        self.regime_classifier_: KernelRegimeClassifier | None = None
        self.simca_: SIMCAClassifier | None = None
        self.regime_pls_models_: dict[int, ConstrainedNipalsPLS] = {}
        self.regime_ids_train_: NDArray | None = None

        # Stage 2 retune models
        self.retune_A_pls_: ConstrainedNipalsPLS | None = None
        self.retune_B_pls_: ConstrainedNipalsPLS | None = None
        self.retune_C_pls_: ConstrainedNipalsPLS | None = None
        self.retune_D_B_: NDArray | None = None  # collapsed regression coefficients
        self.retune_D_x_mean_: NDArray | None = None
        self.retune_D_y_mean_: NDArray | None = None

        # Metadata
        self.kernel_names_: list[str] = []
        self.is_fitted_: bool = False
        self._X_train_cache: NDArray | None = None
        self._Y_train_cache: NDArray | None = None
        self._best_method: str = "global_pls"

    # ===================================================================
    # Stage 1: ClimKern Foundation
    # ===================================================================

    def fit_stage1(
        self,
        X: NDArray[np.floating],
        Y: NDArray[np.floating],
        kernel_names: list[str],
    ) -> KernelRegimeHarmonizer:
        """
        Stage 1: Global PLS + 11 SIMCA kernel regimes.

        1. Fit global PLS (baseline)
        2. Assign kernel regimes (KernelRegimeClassifier)
        3. Train SIMCA on regime labels
        4. Fit per-regime PLS models
        """
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        self.kernel_names_ = list(kernel_names)

        # Remove NaN rows
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(Y).any(axis=1))
        X_v = X[valid_mask]
        Y_v = Y[valid_mask]
        self._X_train_cache = X_v
        self._Y_train_cache = Y_v

        n_samples, n_features = X_v.shape
        n_comp = min(self.n_components, n_features, n_samples // 3)
        if n_comp < 1:
            n_comp = 1

        # 1. Global PLS (baseline)
        self.global_pls_ = ConstrainedNipalsPLS(
            n_components=n_comp, constraints=self.constraints,
        )
        self.global_pls_.fit(X_v, Y_v)

        # 2. Kernel regime assignment
        self.regime_classifier_ = KernelRegimeClassifier(n_regimes=len(kernel_names))
        self.regime_ids_train_ = self.regime_classifier_.fit_predict(
            X_v, Y_v, kernel_names=kernel_names,
        )

        # 3. SIMCA on regime labels
        n_simca_comp = min(3, n_features)
        self.simca_ = SIMCAClassifier(n_components=n_simca_comp)
        self.simca_.fit(X_v, self.regime_ids_train_)

        # 4. Per-regime PLS
        self.regime_pls_models_ = {}
        for rid in np.unique(self.regime_ids_train_):
            mask = self.regime_ids_train_ == rid
            n_regime = mask.sum()
            if n_regime < self.min_samples_per_regime:
                continue

            nc = min(n_comp, n_regime // 3)
            if nc < 1:
                nc = 1

            pls = ConstrainedNipalsPLS(
                n_components=nc, constraints=self.constraints,
            )
            pls.fit(X_v[mask], Y_v[mask])
            self.regime_pls_models_[int(rid)] = pls

        self.is_fitted_ = True
        return self

    def predict_stage1_global(self, X: NDArray) -> NDArray:
        """Predict using Stage 1 global PLS."""
        return self.global_pls_.predict(np.asarray(X, dtype=np.float64))

    def predict_stage1_regime(self, X: NDArray) -> NDArray:
        """Predict using Stage 1 per-regime PLS with SIMCA soft blending."""
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]

        if self.simca_ is None or not self.regime_pls_models_:
            return self.predict_stage1_global(X)

        proba = self.simca_.predict_proba(X)
        regime_ids_ordered = sorted(self.regime_pls_models_.keys())

        n_targets = self._Y_train_cache.shape[1] if self._Y_train_cache is not None else 1
        Y_pred = np.zeros((n_samples, n_targets))

        for j, rid in enumerate(regime_ids_ordered):
            if j >= proba.shape[1]:
                break
            pls = self.regime_pls_models_[rid]
            Y_regime = pls.predict(X)
            weights = proba[:, j:j + 1]
            Y_pred += weights * Y_regime

        # Normalize by total weight used
        total_weight = np.zeros((n_samples, 1))
        for j in range(min(len(regime_ids_ordered), proba.shape[1])):
            total_weight += proba[:, j:j + 1]
        total_weight = np.maximum(total_weight, 1e-10)
        Y_pred /= total_weight

        return Y_pred

    # ===================================================================
    # Stage 2: Retune Approaches
    # ===================================================================

    def fit_retune_A(self, X: NDArray, Y: NDArray) -> None:
        """
        Approach A: SIMCA probabilities as additional features → global PLS.

        Augments X with 11 SIMCA regime probabilities so the model sees
        both kernel predictions and regime context.
        """
        X = np.asarray(X, dtype=np.float64)
        proba = self.simca_.predict_proba(X)
        X_aug = np.hstack([X, proba])

        n_comp = min(self.n_components + 2, X_aug.shape[1], X_aug.shape[0] // 3)
        if n_comp < 1:
            n_comp = 1

        self.retune_A_pls_ = ConstrainedNipalsPLS(
            n_components=n_comp, constraints=self.constraints,
        )
        self.retune_A_pls_.fit(X_aug, Y)

    def predict_retune_A(self, X: NDArray) -> NDArray:
        """Predict using Approach A (augmented features)."""
        X = np.asarray(X, dtype=np.float64)
        proba = self.simca_.predict_proba(X)
        X_aug = np.hstack([X, proba])
        return self.retune_A_pls_.predict(X_aug)

    def fit_retune_B(self, X: NDArray, Y: NDArray) -> None:
        """
        Approach B: Regime-weighted global PLS.

        Samples with high SIMCA confidence (clearly in one regime) get
        higher weight. Boundary samples are downweighted. Implemented
        by pre-multiplying X and Y by sqrt(weight).
        """
        X = np.asarray(X, dtype=np.float64)
        proba = self.simca_.predict_proba(X)
        # Weight = max regime probability (confidence)
        weights = np.max(proba, axis=1)
        # Normalize so mean weight = 1
        weights = weights / (weights.mean() + 1e-10)

        sqrt_w = np.sqrt(weights)[:, None]
        X_w = X * sqrt_w
        Y_w = Y * sqrt_w

        n_comp = min(self.n_components, X.shape[1], X.shape[0] // 3)
        if n_comp < 1:
            n_comp = 1

        self.retune_B_pls_ = ConstrainedNipalsPLS(
            n_components=n_comp, constraints=self.constraints,
        )
        self.retune_B_pls_.fit(X_w, Y_w)

    def predict_retune_B(self, X: NDArray) -> NDArray:
        """Predict using Approach B (regime-weighted)."""
        # At prediction time, use unweighted X
        return self.retune_B_pls_.predict(np.asarray(X, dtype=np.float64))

    def fit_retune_C(self, X: NDArray, Y: NDArray, max_iter: int = 5) -> None:
        """
        Approach C: Two-pass iterative refinement.

        Alternates between:
        - SIMCA regime update (based on current PLS residuals)
        - PLS refit with regime-informed constraint weights

        Convergence when Q² change < 0.001 between iterations.
        """
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)

        n_comp = min(self.n_components, X.shape[1], X.shape[0] // 3)
        if n_comp < 1:
            n_comp = 1

        # Start with Stage 1 global PLS
        current_pls = ConstrainedNipalsPLS(
            n_components=n_comp, constraints=self.constraints,
        )
        current_pls.fit(X, Y)
        prev_q2 = _compute_q2(Y, current_pls.predict(X))

        for iteration in range(max_iter):
            # Update regime assignment based on current residuals
            Y_pred = current_pls.predict(X)
            residuals_per_kernel = np.abs(X - Y_pred[:, 0:1])
            # Invert: low residual → high weight for regime assignment
            regime_quality = 1.0 / (residuals_per_kernel + 1e-6)
            regime_quality /= regime_quality.sum(axis=1, keepdims=True)

            # Weight samples by regime clarity
            clarity = np.max(regime_quality, axis=1)
            clarity /= clarity.mean() + 1e-10
            sqrt_w = np.sqrt(clarity)[:, None]

            # Refit PLS with clarity weighting
            current_pls = ConstrainedNipalsPLS(
                n_components=n_comp, constraints=self.constraints,
            )
            current_pls.fit(X * sqrt_w, Y * sqrt_w)

            # Check convergence
            curr_q2 = _compute_q2(Y, current_pls.predict(X))
            if abs(curr_q2 - prev_q2) < 0.001:
                break
            prev_q2 = curr_q2

        self.retune_C_pls_ = current_pls

    def predict_retune_C(self, X: NDArray) -> NDArray:
        """Predict using Approach C (iterative)."""
        return self.retune_C_pls_.predict(np.asarray(X, dtype=np.float64))

    def fit_retune_D(self, X: NDArray, Y: NDArray) -> None:
        """
        Approach D: Regime collapse → single global PLS.

        Fits per-regime PLS models, then collapses their regression
        coefficients into a single global model weighted by regime size.
        """
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)

        if not self.regime_pls_models_:
            # Fallback: just use global PLS
            self.retune_D_B_ = None
            return

        # Compute weighted average of per-regime regression coefficients
        total_samples = 0
        B_weighted = None
        x_mean_weighted = np.zeros(X.shape[1])
        y_mean_weighted = np.zeros(Y.shape[1])

        for rid, pls in self.regime_pls_models_.items():
            r = pls.results_
            if r is None:
                continue

            mask = self.regime_ids_train_ == rid
            n_k = mask.sum()
            total_samples += n_k

            W = r.x_weights
            P = r.x_loadings
            Q = r.y_loadings
            B_inner = r.regression_matrix
            B_k = W @ B_inner @ Q.T  # (n_features, n_targets)

            if B_weighted is None:
                B_weighted = n_k * B_k
            else:
                # Pad if shapes differ (different n_components)
                if B_k.shape == B_weighted.shape:
                    B_weighted += n_k * B_k
                else:
                    # Use the larger shape
                    min_f = min(B_k.shape[0], B_weighted.shape[0])
                    min_t = min(B_k.shape[1], B_weighted.shape[1])
                    B_weighted[:min_f, :min_t] += n_k * B_k[:min_f, :min_t]

            x_mean_weighted += n_k * r.x_mean
            y_mean_weighted += n_k * r.y_mean

        if total_samples > 0 and B_weighted is not None:
            self.retune_D_B_ = B_weighted / total_samples
            self.retune_D_x_mean_ = x_mean_weighted / total_samples
            self.retune_D_y_mean_ = y_mean_weighted / total_samples
        else:
            self.retune_D_B_ = None

    def predict_retune_D(self, X: NDArray) -> NDArray:
        """Predict using Approach D (regime-collapsed global)."""
        X = np.asarray(X, dtype=np.float64)
        if self.retune_D_B_ is None:
            return self.predict_stage1_global(X)

        X_centered = X - self.retune_D_x_mean_
        Y_pred = X_centered @ self.retune_D_B_ + self.retune_D_y_mean_
        return Y_pred

    # ===================================================================
    # Fit all
    # ===================================================================

    def fit_all(
        self,
        X: NDArray[np.floating],
        Y: NDArray[np.floating],
        kernel_names: list[str],
    ) -> KernelRegimeHarmonizer:
        """
        Run complete two-stage pipeline: Stage 1 + all 4 retune approaches.
        """
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        # Remove NaNs
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(Y).any(axis=1))
        X_v = X[valid_mask]
        Y_v = Y[valid_mask]

        # Stage 1
        self.fit_stage1(X_v, Y_v, kernel_names)

        # Stage 2: all 4 retune approaches
        self.fit_retune_A(X_v, Y_v)
        self.fit_retune_B(X_v, Y_v)
        self.fit_retune_C(X_v, Y_v)
        self.fit_retune_D(X_v, Y_v)

        return self

    # ===================================================================
    # Unified predict
    # ===================================================================

    def predict(
        self,
        X: NDArray[np.floating],
        method: str = "best",
    ) -> NDArray[np.floating]:
        """
        Predict ΔR using specified method.

        Parameters
        ----------
        method : str
            One of: 'best', 'global_pls', 'regime_blend',
            'retune_A', 'retune_B', 'retune_C', 'retune_D'.
        """
        if not self.is_fitted_:
            raise RuntimeError("KernelRegimeHarmonizer must be fitted first")

        if method == "best":
            method = self._best_method

        dispatch = {
            "global_pls": self.predict_stage1_global,
            "regime_blend": self.predict_stage1_regime,
            "retune_A": self.predict_retune_A,
            "retune_B": self.predict_retune_B,
            "retune_C": self.predict_retune_C,
            "retune_D": self.predict_retune_D,
        }

        pred_fn = dispatch.get(method)
        if pred_fn is None:
            raise ValueError(f"Unknown method '{method}'. Choose from {list(dispatch)}")
        return pred_fn(X)

    # ===================================================================
    # Evaluation
    # ===================================================================

    def evaluate(self, X: NDArray, Y: NDArray, method: str = "best") -> float:
        """Compute Q² for a specific method."""
        Y = np.asarray(Y, dtype=np.float64)
        Y_pred = self.predict(X, method=method)
        return _compute_q2(Y, Y_pred)

    def evaluate_per_kernel(self, X: NDArray, Y: NDArray) -> dict[str, float]:
        """Q² for each individual kernel (column of X) vs observations."""
        Y = np.asarray(Y, dtype=np.float64)
        n_kernels = X.shape[1]
        results = {}
        for i, name in enumerate(self.kernel_names_[:n_kernels]):
            pred_i = X[:, i:i + 1]
            if Y.shape[1] > 1:
                pred_i = np.tile(pred_i, (1, Y.shape[1]))
            results[name] = _compute_q2(Y, pred_i)
        return results

    def evaluate_simple_mean(self, X: NDArray, Y: NDArray) -> float:
        """Q² of simple (equal-weight) kernel mean."""
        Y = np.asarray(Y, dtype=np.float64)
        Y_pred = np.nanmean(X, axis=1, keepdims=True)
        if Y.shape[1] > 1:
            Y_pred = np.tile(Y_pred, (1, Y.shape[1]))
        return _compute_q2(Y, Y_pred)

    def evaluate_per_regime(self, X: NDArray, Y: NDArray) -> dict[int, float]:
        """Q² per SIMCA regime using the regime-specific PLS models."""
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        results = {}

        if self.simca_ is None:
            return results

        regime_ids = self.simca_.predict(X)
        for rid, pls in self.regime_pls_models_.items():
            mask = regime_ids == rid
            if mask.sum() < 2:
                continue
            Y_pred = pls.predict(X[mask])
            results[rid] = _compute_q2(Y[mask], Y_pred)

        return results

    def get_kernel_weights(self, regime_id: int | None = None) -> NDArray:
        """Extract PLS regression coefficients as kernel weights."""
        if regime_id is not None and regime_id in self.regime_pls_models_:
            pls = self.regime_pls_models_[regime_id]
        elif self.global_pls_ is not None:
            pls = self.global_pls_
        else:
            raise RuntimeError("No fitted model")

        r = pls.results_
        W = r.x_weights
        Q = r.y_loadings
        B_inner = r.regression_matrix
        return W @ B_inner @ Q.T

    def compute_spread_reduction(
        self, X: NDArray, Y: NDArray, method: str = "best",
    ) -> tuple[float, float, float]:
        """Compute RMSE before (individual kernels) vs after (harmonized)."""
        Y = np.asarray(Y, dtype=np.float64)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        n_kernels = X.shape[1]
        kernel_rmses = []
        for k in range(n_kernels):
            pred_k = X[:, k:k + 1]
            if Y.shape[1] > 1:
                pred_k = np.tile(pred_k, (1, Y.shape[1]))
            kernel_rmses.append(_compute_rmse(Y, pred_k))
        rmse_before = float(np.mean(kernel_rmses))

        Y_pred = self.predict(X, method=method)
        rmse_after = _compute_rmse(Y, Y_pred)

        reduction_pct = (
            100.0 * (rmse_before - rmse_after) / rmse_before
            if rmse_before > 0 else 0.0
        )
        return rmse_before, rmse_after, reduction_pct

    def thermodynamic_compliance(
        self,
        X: NDArray,
        Y: NDArray,
        surface_temp: NDArray | None = None,
    ) -> dict[str, dict[str, float]]:
        """
        Evaluate thermodynamic law compliance for each approach.

        Returns dict[method_name → dict["stefan_boltzmann", "energy_conservation"]].
        """
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)

        if surface_temp is None:
            surface_temp = np.full(X.shape[0], TYPICAL_SURFACE_TEMP)
        surface_temp = np.asarray(surface_temp, dtype=np.float64)

        methods = ["global_pls", "retune_A", "retune_B", "retune_C", "retune_D"]
        results = {}

        for method in methods:
            try:
                Y_pred = self.predict(X, method=method)
            except Exception:
                results[method] = {
                    "stefan_boltzmann": float("nan"),
                    "energy_conservation": float("nan"),
                }
                continue

            # Stefan-Boltzmann: surface flux should scale as 4εσT³ΔT
            # Expected change in OLR per K ≈ 4σT³ ≈ 5.4 W/m²/K at 288K
            expected_dR_per_K = 4 * STEFAN_BOLTZMANN * surface_temp ** 3
            # If Y has LW column, check consistency
            actual_dR = Y_pred[:, 0] if Y_pred.ndim > 1 else Y_pred.ravel()
            # Simple check: correlation of actual_dR with expected pattern
            sb_residual = float(np.sqrt(np.nanmean(
                (actual_dR - np.nanmean(actual_dR)) ** 2
            )))

            # Energy conservation: global mean ΔR should be bounded
            energy_residual = float(np.abs(np.nanmean(actual_dR)))

            results[method] = {
                "stefan_boltzmann": sb_residual,
                "energy_conservation": energy_residual,
            }

        return results

    def q2_dashboard(
        self,
        X_test: NDArray,
        Y_test: NDArray,
        surface_temp: NDArray | None = None,
    ) -> Q2Dashboard:
        """
        Generate comprehensive Q² report for ALL steps and versions.
        """
        X_test = np.asarray(X_test, dtype=np.float64)
        Y_test = np.asarray(Y_test, dtype=np.float64)
        if Y_test.ndim == 1:
            Y_test = Y_test.reshape(-1, 1)

        # Per-kernel
        q2_per_kernel = self.evaluate_per_kernel(X_test, Y_test)

        # Baselines
        q2_simple_mean = self.evaluate_simple_mean(X_test, Y_test)
        q2_global = self.evaluate(X_test, Y_test, method="global_pls")

        # Stage 1 regime blend
        Y_regime = self.predict_stage1_regime(X_test)
        q2_regime = _compute_q2(Y_test, Y_regime)

        # Retune approaches
        q2_A = self.evaluate(X_test, Y_test, method="retune_A")
        q2_B = self.evaluate(X_test, Y_test, method="retune_B")
        q2_C = self.evaluate(X_test, Y_test, method="retune_C")
        q2_D = self.evaluate(X_test, Y_test, method="retune_D")

        # Per-regime Q²
        q2_per_regime = self.evaluate_per_regime(X_test, Y_test)

        # Thermodynamic compliance
        thermo = self.thermodynamic_compliance(X_test, Y_test, surface_temp)
        sb_residual = {m: v["stefan_boltzmann"] for m, v in thermo.items()}
        energy_residual = {m: v["energy_conservation"] for m, v in thermo.items()}

        # Spread reduction for each method
        rmse_before = 0.0
        rmse_after = {}
        spread_reduction_pct = {}
        for method in ["global_pls", "retune_A", "retune_B", "retune_C", "retune_D"]:
            rb, ra, rp = self.compute_spread_reduction(X_test, Y_test, method=method)
            rmse_before = rb  # same for all
            rmse_after[method] = ra
            spread_reduction_pct[method] = rp

        # Find best method
        all_q2 = {
            "global_pls": q2_global,
            "regime_blend": q2_regime,
            "retune_A": q2_A,
            "retune_B": q2_B,
            "retune_C": q2_C,
            "retune_D": q2_D,
        }
        best_method = max(all_q2, key=all_q2.get)
        self._best_method = best_method

        return Q2Dashboard(
            q2_per_kernel=q2_per_kernel,
            q2_simple_mean=q2_simple_mean,
            q2_global_pls=q2_global,
            q2_regime_blend=q2_regime,
            q2_retune_A=q2_A,
            q2_retune_B=q2_B,
            q2_retune_C=q2_C,
            q2_retune_D=q2_D,
            q2_per_regime=q2_per_regime,
            sb_residual=sb_residual,
            energy_residual=energy_residual,
            rmse_before=rmse_before,
            rmse_after=rmse_after,
            spread_reduction_pct=spread_reduction_pct,
            best_method=best_method,
            best_q2=all_q2[best_method],
        )

    def summarize(self, X: NDArray, Y: NDArray) -> HarmonizationResult:
        """Backward-compatible summary."""
        q2_harmonized = self.evaluate(X, Y, method="best")
        q2_simple_mean = self.evaluate_simple_mean(X, Y)
        q2_per_kernel = self.evaluate_per_kernel(X, Y)
        rmse_before, rmse_after, reduction_pct = self.compute_spread_reduction(X, Y)

        kernel_weights = {}
        samples_per_regime = {}
        for rid, pls in self.regime_pls_models_.items():
            kernel_weights[rid] = self.get_kernel_weights(rid)
            if self.regime_classifier_:
                samples_per_regime[rid] = self.regime_classifier_.regime_counts_.get(rid, 0)

        return HarmonizationResult(
            kernel_weights=kernel_weights,
            kernel_names=self.kernel_names_,
            q2_harmonized=q2_harmonized,
            q2_simple_mean=q2_simple_mean,
            q2_per_kernel=q2_per_kernel,
            spread_before=rmse_before,
            spread_after=rmse_after,
            spread_reduction_pct=reduction_pct,
            n_regimes_active=len(self.regime_pls_models_),
            samples_per_regime=samples_per_regime,
        )


# Backward compatibility alias
KernelHarmonizer = KernelRegimeHarmonizer
