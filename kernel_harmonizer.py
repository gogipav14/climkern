"""
Kernel Harmonization via Constrained PLS (Step 1)

Bridges Janoski et al. (2025) ClimKern's 11 discrete pre-computed kernel sets
with the SIMCA/PLS/PCA framework. Uses constrained NIPALS-PLS as a meta-learner
to find optimal state-dependent kernel combination weights.

Architecture:
    X = [ΔR_kernel1, ΔR_kernel2, ..., ΔR_kernel_N]  (n_samples, n_kernels)
    Y = [ΔR_observed_LW, ΔR_observed_SW]             (n_samples, 2)

    PLS finds: Y ≈ X @ W(P'W)⁻¹Q' + constraint_adjustment

This directly reuses ConstrainedNipalsPLS, ClimateStateClassifier, and
SIMCAClassifier — the same framework as the data-driven kernels (Step 2),
but with kernel predictions as features instead of raw atmospheric state.
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

from nipals_pls import ConstrainedNipalsPLS, PhysicalConstraint
from state_classifier import ClimateStateClassifier, SIMCAClassifier

try:
    import xarray as xr

    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class KernelPredictions:
    """Predictions from all available kernel sets for a given experiment."""

    kernel_names: list[str]
    # Per-kernel feedback fields: dict[kernel_name] -> dict[feedback_type] -> xr.DataArray
    # feedback_type in {"planck", "lapse_rate", "wv_lw", "wv_sw", "albedo"}
    feedbacks: dict[str, dict[str, Any]]
    # Aggregated total ΔR per kernel: dict[kernel_name] -> (ΔR_LW, ΔR_SW) xr.DataArrays
    total_lw: dict[str, Any]
    total_sw: dict[str, Any]


@dataclass
class HarmonizationResult:
    """Results from kernel harmonization."""

    # Per-regime kernel weights: regime_id -> array of shape (n_kernels,)
    kernel_weights: dict[int, NDArray[np.floating]]
    kernel_names: list[str]
    # Q² scores
    q2_harmonized: float
    q2_simple_mean: float
    q2_per_kernel: dict[str, float]
    # Spread metrics
    spread_before: float  # std across kernels before harmonization
    spread_after: float  # residual std after harmonization
    spread_reduction_pct: float
    # Regime info
    n_regimes_active: int
    samples_per_regime: dict[int, int]


# ---------------------------------------------------------------------------
# Multi-kernel loader
# ---------------------------------------------------------------------------


class MultiKernelLoader:
    """
    Load all available kernel sets and compute feedback predictions.

    Wraps climkern.util.get_kern() and climkern.frontend feedback functions
    to iterate over all kernel sets and compute ΔR predictions for each.
    """

    def __init__(self):
        self._available_kernels: list[str] | None = None

    def get_available_kernels(self) -> list[str]:
        """Discover which kernel sets are installed."""
        if self._available_kernels is not None:
            return self._available_kernels

        # Try multiple possible data paths
        candidate_paths = []

        # 1. Try importlib.resources (installed package)
        try:
            try:
                from importlib_resources import files as pkg_files
            except ImportError:
                from importlib.resources import files as pkg_files

            candidate_paths.append(
                Path(str(pkg_files("climkern").joinpath("data", "kernels")))
            )
            # Zenodo zip may extract to data/data/kernels/
            candidate_paths.append(
                Path(str(pkg_files("climkern").joinpath("data", "data", "kernels")))
            )
        except (TypeError, OSError, ModuleNotFoundError):
            pass

        # 2. Try relative to this file (for local development)
        this_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        candidate_paths.extend([
            this_dir / "climkern" / "data" / "kernels",
            this_dir / "climkern" / "data" / "data" / "kernels",
        ])

        # List subdirectories from first valid path
        kernels = []
        for data_dir in candidate_paths:
            if data_dir.is_dir():
                for entry in sorted(data_dir.iterdir()):
                    if entry.is_dir():
                        # Check that TOA kernel file exists
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

        from climkern.util import get_kern

        return get_kern(kernel_name)

    def compute_temperature_feedbacks(
        self,
        ctrl_ta: "xr.DataArray",
        ctrl_ts: "xr.DataArray",
        ctrl_ps: "xr.DataArray",
        pert_ta: "xr.DataArray",
        pert_ts: "xr.DataArray",
        pert_ps: "xr.DataArray",
        pert_trop: "xr.DataArray | None" = None,
        sky: str = "all-sky",
    ) -> dict[str, tuple["xr.DataArray", "xr.DataArray"]]:
        """
        Compute temperature feedbacks (lapse rate, Planck) for all available kernels.

        Returns dict mapping kernel_name -> (lr_feedback, planck_feedback).
        """
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
        ctrl_q: "xr.DataArray",
        ctrl_ta: "xr.DataArray",
        ctrl_ps: "xr.DataArray",
        pert_q: "xr.DataArray",
        pert_ps: "xr.DataArray",
        pert_trop: "xr.DataArray | None" = None,
        sky: str = "all-sky",
        method: int = 1,
    ) -> dict[str, tuple["xr.DataArray", "xr.DataArray"]]:
        """
        Compute water vapor feedbacks (LW, SW) for all available kernels.

        Returns dict mapping kernel_name -> (lw_q_feedback, sw_q_feedback).
        """
        from climkern.frontend import calc_q_feedbacks

        results = {}
        for name in self.get_available_kernels():
            try:
                qlw, qsw = calc_q_feedbacks(
                    ctrl_q, ctrl_ta, ctrl_ps,
                    pert_q, pert_ps,
                    pert_trop=pert_trop, kern=name, sky=sky, method=method,
                )
                results[name] = (qlw, qsw)
            except Exception as e:
                warnings.warn(f"Skipping kernel '{name}' for q feedbacks: {e}")
        return results

    def compute_albedo_feedbacks(
        self,
        ctrl_rsus: "xr.DataArray",
        ctrl_rsds: "xr.DataArray",
        pert_rsus: "xr.DataArray",
        pert_rsds: "xr.DataArray",
        sky: str = "all-sky",
    ) -> dict[str, "xr.DataArray"]:
        """
        Compute surface albedo feedbacks for all available kernels.

        Returns dict mapping kernel_name -> albedo_feedback.
        """
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


# ---------------------------------------------------------------------------
# Kernel Harmonizer
# ---------------------------------------------------------------------------


class KernelHarmonizer:
    """
    Harmonize multiple pre-computed radiative kernel sets using constrained PLS.

    For a given atmospheric perturbation experiment (ctrl vs. pert), each of the
    N kernel sets produces a different ΔR prediction. KernelHarmonizer uses
    constrained NIPALS-PLS to find optimal state-dependent combination weights
    that minimize residual against observed (or model-truth) radiative fluxes.

    Parameters
    ----------
    n_components : int
        Number of PLS components. With N kernels as features, 2-3 components
        is usually sufficient to capture the dominant modes of kernel variation.
    use_state_dependent : bool
        If True, fit separate PLS models per climate regime via SIMCA routing.
    constraints : list[PhysicalConstraint], optional
        Physical constraints to enforce. If None, fits unconstrained PLS.
    min_samples_per_regime : int
        Minimum samples required to fit a regime-specific model.
    """

    def __init__(
        self,
        n_components: int = 3,
        use_state_dependent: bool = True,
        constraints: list[PhysicalConstraint] | None = None,
        min_samples_per_regime: int = 50,
    ):
        self.n_components = n_components
        self.use_state_dependent = use_state_dependent
        self.constraints = constraints
        self.min_samples_per_regime = min_samples_per_regime

        self.loader = MultiKernelLoader()
        self.kernel_names_: list[str] = []
        self.pls_models_: dict[int, ConstrainedNipalsPLS] = {}
        self.classifier_: ClimateStateClassifier | None = None
        self.simca_: SIMCAClassifier | None = None
        self.global_pls_: ConstrainedNipalsPLS | None = None
        self.is_fitted_: bool = False

        # Cache for predictions
        self._X_feature_cache: NDArray | None = None
        self._Y_target_cache: NDArray | None = None

    def build_feature_matrix(
        self,
        t_feedbacks: dict[str, tuple] | None = None,
        q_feedbacks: dict[str, tuple] | None = None,
        alb_feedbacks: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating], list[str]]:
        """
        Build the feature matrix X from kernel predictions.

        For each kernel, total ΔR_LW = planck + lapse_rate + wv_lw,
        total ΔR_SW = wv_sw + albedo. Stack all kernels into columns.

        Parameters
        ----------
        t_feedbacks : dict
            kernel_name -> (lr_feedback, planck_feedback) from compute_temperature_feedbacks
        q_feedbacks : dict
            kernel_name -> (lw_q, sw_q) from compute_humidity_feedbacks
        alb_feedbacks : dict
            kernel_name -> albedo_feedback from compute_albedo_feedbacks

        Returns
        -------
        X_lw : array of shape (n_samples, n_kernels)
            LW ΔR predictions from each kernel, flattened over space and time.
        X_sw : array of shape (n_samples, n_kernels)
            SW ΔR predictions from each kernel.
        kernel_names : list[str]
            Ordered kernel names matching columns.
        """
        # Find kernels present in all feedback types
        all_names = set()
        if t_feedbacks:
            all_names.update(t_feedbacks.keys())
        if q_feedbacks:
            all_names.update(q_feedbacks.keys())
        if alb_feedbacks:
            all_names.update(alb_feedbacks.keys())

        # Only use kernels that have at least temperature feedbacks
        kernel_names = sorted(
            name for name in all_names
            if t_feedbacks and name in t_feedbacks
        )

        lw_columns = []
        sw_columns = []

        for name in kernel_names:
            # LW: planck + lapse_rate + wv_lw
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

            # SW: wv_sw + albedo
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
                # Flatten spatial + temporal dimensions
                lw_flat = lw_total.reshape(-1)
                sw_flat = sw_total.reshape(-1) if sw_total is not None else np.zeros_like(lw_flat)

                lw_columns.append(lw_flat)
                sw_columns.append(sw_flat)

        X_lw = np.column_stack(lw_columns) if lw_columns else np.empty((0, 0))
        X_sw = np.column_stack(sw_columns) if sw_columns else np.empty((0, 0))

        return X_lw, X_sw, kernel_names

    def fit(
        self,
        X: NDArray[np.floating],
        Y: NDArray[np.floating],
        kernel_names: list[str],
        latitude: NDArray[np.floating] | None = None,
        cloud_fraction: NDArray[np.floating] | None = None,
        lts: NDArray[np.floating] | None = None,
    ) -> KernelHarmonizer:
        """
        Fit optimal kernel weights against observed/target radiative fluxes.

        Parameters
        ----------
        X : array of shape (n_samples, n_kernels) or (n_samples, 2*n_kernels)
            Kernel predictions. If n_features = n_kernels, uses only LW.
            If n_features = 2*n_kernels, first n_kernels columns are LW,
            last n_kernels are SW.
        Y : array of shape (n_samples, n_targets)
            Observed/target radiative fluxes [ΔR_LW, ΔR_SW].
        kernel_names : list[str]
            Names of kernel sets (column labels for X).
        latitude : array of shape (n_samples,), optional
            Latitude for state classification. Required if use_state_dependent=True.
        cloud_fraction : array of shape (n_samples,), optional
            Cloud fraction for state classification.
        lts : array of shape (n_samples,), optional
            Lower tropospheric stability for state classification.

        Returns
        -------
        self : KernelHarmonizer
            Fitted harmonizer.
        """
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        self.kernel_names_ = list(kernel_names)
        n_samples = X.shape[0]

        # Remove NaN rows
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(Y).any(axis=1))
        X_valid = X[valid_mask]
        Y_valid = Y[valid_mask]

        # Cache for evaluation
        self._X_feature_cache = X_valid
        self._Y_target_cache = Y_valid

        if self.use_state_dependent and latitude is not None:
            lat_valid = latitude[valid_mask] if latitude is not None else None
            cf_valid = cloud_fraction[valid_mask] if cloud_fraction is not None else None
            lts_valid = lts[valid_mask] if lts is not None else None

            self._fit_state_dependent(X_valid, Y_valid, lat_valid, cf_valid, lts_valid)
        else:
            self._fit_global(X_valid, Y_valid)

        self.is_fitted_ = True
        return self

    def _fit_global(self, X: NDArray, Y: NDArray) -> None:
        """Fit a single global PLS model."""
        n_components = min(self.n_components, X.shape[1], X.shape[0] // 3)
        if n_components < 1:
            n_components = 1

        pls = ConstrainedNipalsPLS(
            n_components=n_components,
            constraints=self.constraints or [],
        )
        pls.fit(X, Y)
        self.global_pls_ = pls
        self.pls_models_[0] = pls

    def _fit_state_dependent(
        self,
        X: NDArray,
        Y: NDArray,
        latitude: NDArray | None,
        cloud_fraction: NDArray | None,
        lts: NDArray | None,
    ) -> None:
        """Fit per-regime PLS models with SIMCA routing."""
        self.classifier_ = ClimateStateClassifier()

        # Default LTS if not provided
        if lts is None:
            lts = np.full(X.shape[0], 15.0)  # moderate stability
        if cloud_fraction is None:
            cloud_fraction = np.full(X.shape[0], 0.5)

        regime_ids = self.classifier_.fit_predict(latitude, cloud_fraction, lts)

        # Fit per-regime models
        for rid in np.unique(regime_ids):
            mask = regime_ids == rid
            n_regime = mask.sum()

            if n_regime < self.min_samples_per_regime:
                continue

            n_components = min(
                self.n_components,
                X.shape[1],
                n_regime // 3,
            )
            if n_components < 1:
                n_components = 1

            pls = ConstrainedNipalsPLS(
                n_components=n_components,
                constraints=self.constraints or [],
            )
            pls.fit(X[mask], Y[mask])
            self.pls_models_[int(rid)] = pls

        # Also fit a global fallback
        self._fit_global(X, Y)

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Predict ΔR using optimally weighted kernel blend.

        Parameters
        ----------
        X : array of shape (n_samples, n_kernels)
            Kernel predictions for new data.

        Returns
        -------
        Y_pred : array of shape (n_samples, n_targets)
            Blended prediction.
        """
        if not self.is_fitted_:
            raise RuntimeError("KernelHarmonizer must be fitted first")

        X = np.asarray(X, dtype=np.float64)

        # Use global model (simplest path)
        if self.global_pls_ is not None:
            return self.global_pls_.predict(X)

        raise RuntimeError("No fitted model available")

    def evaluate(
        self,
        X: NDArray[np.floating],
        Y: NDArray[np.floating],
    ) -> float:
        """
        Compute Q² (predictive R²) of harmonized blend vs. observations.

        Parameters
        ----------
        X : array of shape (n_samples, n_kernels)
        Y : array of shape (n_samples, n_targets)

        Returns
        -------
        q2 : float
            1 - SS_res / SS_tot
        """
        Y = np.asarray(Y, dtype=np.float64)
        Y_pred = self.predict(X)

        ss_res = np.nansum((Y - Y_pred) ** 2)
        ss_tot = np.nansum((Y - np.nanmean(Y, axis=0)) ** 2)

        if ss_tot == 0:
            return 0.0
        return float(1.0 - ss_res / ss_tot)

    def evaluate_simple_mean(
        self,
        X: NDArray[np.floating],
        Y: NDArray[np.floating],
    ) -> float:
        """
        Compute Q² of the simple (equal-weight) kernel mean vs. observations.

        This is the baseline: just average all kernel predictions.
        """
        Y = np.asarray(Y, dtype=np.float64)
        # Simple mean across kernels
        Y_pred_mean = np.nanmean(X, axis=1, keepdims=True)
        if Y.shape[1] > 1:
            Y_pred_mean = np.tile(Y_pred_mean, (1, Y.shape[1]))

        ss_res = np.nansum((Y - Y_pred_mean) ** 2)
        ss_tot = np.nansum((Y - np.nanmean(Y, axis=0)) ** 2)

        if ss_tot == 0:
            return 0.0
        return float(1.0 - ss_res / ss_tot)

    def evaluate_per_kernel(
        self,
        X: NDArray[np.floating],
        Y: NDArray[np.floating],
    ) -> dict[str, float]:
        """
        Compute Q² for each individual kernel against observations.

        Each kernel's column in X is treated as a univariate prediction.
        """
        Y = np.asarray(Y, dtype=np.float64)
        n_kernels = X.shape[1]
        ss_tot = np.nansum((Y - np.nanmean(Y, axis=0)) ** 2)

        results = {}
        for i, name in enumerate(self.kernel_names_[:n_kernels]):
            pred_i = X[:, i : i + 1]
            if Y.shape[1] > 1:
                pred_i = np.tile(pred_i, (1, Y.shape[1]))
            ss_res = np.nansum((Y - pred_i) ** 2)
            results[name] = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        return results

    def get_kernel_weights(self, regime_id: int | None = None) -> NDArray[np.floating]:
        """
        Extract PLS-derived combination weights for each kernel.

        The PLS regression coefficients indicate how much each kernel
        contributes to the optimal blend. Larger positive weight =
        that kernel is more trusted in this regime.

        Parameters
        ----------
        regime_id : int, optional
            Specific regime to get weights for. If None, returns global weights.

        Returns
        -------
        weights : array of shape (n_kernels,) or (n_kernels, n_targets)
            Normalized kernel combination weights.
        """
        if not self.is_fitted_:
            raise RuntimeError("KernelHarmonizer must be fitted first")

        pls = self.pls_models_.get(regime_id if regime_id is not None else 0)
        if pls is None:
            pls = self.global_pls_
        if pls is None:
            raise RuntimeError("No model for the requested regime")

        # Extract regression coefficients from PLS
        # B = W @ (P'W)^-1 @ Q'
        r = pls.results_
        W = r.x_weights
        P = r.x_loadings
        Q = r.y_loadings
        B_inner = r.regression_matrix

        # Score-based: regression coefficients = W @ B_inner @ Q'
        B = W @ B_inner @ Q.T
        return B

    def compute_spread_reduction(
        self,
        X: NDArray[np.floating],
        Y: NDArray[np.floating],
    ) -> tuple[float, float, float]:
        """
        Compute prediction error before and after harmonization.

        Compares RMSE of individual kernel predictions against observations
        (before) vs RMSE of the harmonized blend (after).

        Returns
        -------
        rmse_before : float
            Mean RMSE across individual kernels vs observations.
        rmse_after : float
            RMSE of harmonized prediction vs observations.
        reduction_pct : float
            Percentage reduction in RMSE.
        """
        Y = np.asarray(Y, dtype=np.float64)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        # Before: mean RMSE of each individual kernel vs observations
        n_kernels = X.shape[1]
        kernel_rmses = []
        for k in range(n_kernels):
            pred_k = X[:, k:k+1]
            if Y.shape[1] > 1:
                pred_k = np.tile(pred_k, (1, Y.shape[1]))
            rmse_k = np.sqrt(np.nanmean((Y - pred_k) ** 2))
            kernel_rmses.append(rmse_k)
        rmse_before = float(np.mean(kernel_rmses))

        # After: RMSE of harmonized prediction
        Y_pred = self.predict(X)
        rmse_after = float(np.sqrt(np.nanmean((Y - Y_pred) ** 2)))

        reduction_pct = (
            100.0 * (rmse_before - rmse_after) / rmse_before
            if rmse_before > 0
            else 0.0
        )

        return rmse_before, rmse_after, reduction_pct

    def summarize(
        self,
        X: NDArray[np.floating],
        Y: NDArray[np.floating],
    ) -> HarmonizationResult:
        """
        Produce a complete harmonization summary.

        Parameters
        ----------
        X : array of shape (n_samples, n_kernels)
        Y : array of shape (n_samples, n_targets)

        Returns
        -------
        result : HarmonizationResult
        """
        q2_harmonized = self.evaluate(X, Y)
        q2_simple_mean = self.evaluate_simple_mean(X, Y)
        q2_per_kernel = self.evaluate_per_kernel(X, Y)
        spread_before, spread_after, reduction_pct = self.compute_spread_reduction(X, Y)

        # Per-regime weights
        kernel_weights = {}
        samples_per_regime = {}
        for rid, pls in self.pls_models_.items():
            kernel_weights[rid] = self.get_kernel_weights(rid)
            samples_per_regime[rid] = 0  # will be filled if state-dependent

        return HarmonizationResult(
            kernel_weights=kernel_weights,
            kernel_names=self.kernel_names_,
            q2_harmonized=q2_harmonized,
            q2_simple_mean=q2_simple_mean,
            q2_per_kernel=q2_per_kernel,
            spread_before=spread_before,
            spread_after=spread_after,
            spread_reduction_pct=reduction_pct,
            n_regimes_active=len(self.pls_models_),
            samples_per_regime=samples_per_regime,
        )


# ---------------------------------------------------------------------------
# Convenience: end-to-end harmonization from ClimKern tutorial data
# ---------------------------------------------------------------------------


def harmonize_from_tutorial(
    n_components: int = 3,
    sky: str = "all-sky",
    use_state_dependent: bool = False,
) -> tuple[KernelHarmonizer, HarmonizationResult]:
    """
    End-to-end kernel harmonization using ClimKern tutorial data.

    This demonstrates Step 1: loading all available kernel sets, computing
    their feedback predictions on the tutorial 2xCO2 experiment, and using
    constrained PLS to find optimal kernel weights.

    Parameters
    ----------
    n_components : int
        Number of PLS components for harmonization.
    sky : str
        "all-sky" or "clear-sky"
    use_state_dependent : bool
        Whether to use regime-dependent harmonization.

    Returns
    -------
    harmonizer : KernelHarmonizer
        Fitted harmonizer.
    result : HarmonizationResult
        Complete harmonization metrics.
    """
    if not HAS_XARRAY:
        raise ImportError("xarray is required for tutorial-based harmonization")

    from climkern.frontend import tutorial_data

    # Load tutorial data
    ctrl = tutorial_data("ctrl")
    pert = tutorial_data("pert")

    # Initialize loader and compute all kernel predictions
    loader = MultiKernelLoader()
    available = loader.get_available_kernels()
    if len(available) < 2:
        raise RuntimeError(
            f"Need at least 2 kernel sets, found {len(available)}. "
            "Run: python -c \"from climkern.download import download; download()\""
        )

    print(f"Computing feedbacks for {len(available)} kernel sets: {available}")

    # Compute feedbacks for all kernels
    t_fb = loader.compute_temperature_feedbacks(
        ctrl.T, ctrl.TS, ctrl.PS, pert.T, pert.TS, pert.PS, sky=sky,
    )
    q_fb = loader.compute_humidity_feedbacks(
        ctrl.Q, ctrl.T, ctrl.PS, pert.Q, pert.PS, sky=sky,
    )

    # Albedo feedbacks (need surface SW fluxes)
    alb_fb = {}
    if all(v in ctrl for v in ["FSUS", "FSDS"]) and all(v in pert for v in ["FSUS", "FSDS"]):
        alb_fb = loader.compute_albedo_feedbacks(
            ctrl.FSUS, ctrl.FSDS, pert.FSUS, pert.FSDS, sky=sky,
        )

    # Build feature matrix
    harmonizer = KernelHarmonizer(
        n_components=n_components,
        use_state_dependent=use_state_dependent,
    )
    X_lw, X_sw, kernel_names = harmonizer.build_feature_matrix(t_fb, q_fb, alb_fb)

    # For Y target: use the model's actual ΔR (net TOA flux change)
    # Since tutorial data is model output, we can compute Y from the model itself
    # ΔR_LW = -(pert.FLNT - ctrl_clim.FLNT), ΔR_SW = pert.FSNT - ctrl_clim.FSNT
    # For simplicity, use the total LW feedback from first available kernel as Y
    # (in real application, Y would be CERES observations)
    print(f"Built feature matrix: X_lw shape = {X_lw.shape}")

    # Use X_lw as features, and the mean of kernel LW as target (self-consistency)
    # In a real application, Y would be CERES ΔR observations
    Y_lw = np.nanmean(X_lw, axis=1, keepdims=True)

    # Fit harmonizer
    harmonizer.fit(X_lw, Y_lw, kernel_names)

    # Evaluate
    result = harmonizer.summarize(X_lw, Y_lw)
    return harmonizer, result
