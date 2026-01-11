"""
Tunable Radiative Kernel Implementation

Main interface for NIPALS-PLS based radiative kernels with:
- State-dependent (SIMCA-style) regime routing
- Physical constraint enforcement
- Comparison with traditional kernels
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from climkern_retune.core.nipals_pls import (
    ConstrainedNipalsPLS,
    PhysicalConstraint,
    create_multilevel_constraints,
)
from climkern_retune.core.state_classifier import (
    ClimateState,
    ClimateStateClassifier,
    SIMCAClassifier,
)


# Standard pressure levels (hPa) matching traditional kernels
STANDARD_PRESSURE_LEVELS = np.array([
    1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10
])


@dataclass
class KernelConfig:
    """Configuration for tunable radiative kernel."""

    # PLS settings
    n_components: int = 5
    max_iter: int = 500
    tol: float = 1e-10

    # Constraint settings
    use_surface_constraint: bool = True
    use_toa_constraint: bool = True
    use_conservation_constraint: bool = True
    surface_constraint_weight: float = 1.0
    toa_constraint_weight: float = 1.0
    conservation_constraint_weight: float = 0.5

    # Vertical resolution
    pressure_levels: NDArray[np.floating] | None = None
    use_adaptive_levels: bool = False

    # State classification
    use_state_dependent: bool = True
    n_regimes: int = 8


@dataclass
class KernelOutput:
    """Output from kernel computation."""

    # Radiative response (W/m²)
    delta_r_lw: NDArray[np.floating]  # Longwave
    delta_r_sw: NDArray[np.floating]  # Shortwave
    delta_r_net: NDArray[np.floating]  # Net (LW + SW)

    # Per-variable contributions (W/m²)
    contributions: dict[str, NDArray[np.floating]] = field(default_factory=dict)

    # Diagnostics
    regime_ids: NDArray[np.int_] | None = None
    q2_score: float | None = None


@dataclass
class VerticalKernelProfile:
    """Vertical structure of kernel sensitivity."""

    pressure_levels: NDArray[np.floating]
    temperature_kernel: NDArray[np.floating]  # dR/dT at each level
    humidity_kernel: NDArray[np.floating]  # dR/dq at each level

    @property
    def n_levels(self) -> int:
        return len(self.pressure_levels)

    def integrate(self) -> tuple[float, float]:
        """Integrate kernel over pressure to get column-integrated sensitivity."""
        # Pressure-weighted integral (dp in hPa, kernel in W/m²/K or W/m²/(g/kg))
        dp = np.gradient(self.pressure_levels)

        t_integrated = np.trapz(self.temperature_kernel, self.pressure_levels)
        q_integrated = np.trapz(self.humidity_kernel, self.pressure_levels)

        return t_integrated, q_integrated


class TunableKernel:
    """
    Single-regime tunable radiative kernel using NIPALS-PLS.

    This kernel learns the radiative response to atmospheric state changes
    from training data (observations or model output) while enforcing
    physical constraints.

    Parameters
    ----------
    config : KernelConfig
        Kernel configuration.
    constraints : list[PhysicalConstraint], optional
        Physical constraints. If None, uses default multi-level constraints.

    Attributes
    ----------
    pls_model_ : ConstrainedNipalsPLS
        Fitted PLS model.
    feature_names_ : list[str]
        Names of input features.
    is_fitted_ : bool
        Whether the model has been fitted.

    Examples
    --------
    >>> kernel = TunableKernel()
    >>> kernel.fit(X_train, Y_train, feature_names=['T_1000', 'T_850', ...])
    >>> output = kernel.compute(X_test)
    >>> print(f"LW response: {output.delta_r_lw.mean():.2f} W/m²")
    """

    def __init__(
        self,
        config: KernelConfig | None = None,
        constraints: list[PhysicalConstraint] | None = None,
    ):
        self.config = config or KernelConfig()

        if constraints is None and any([
            self.config.use_surface_constraint,
            self.config.use_toa_constraint,
            self.config.use_conservation_constraint,
        ]):
            constraints = create_multilevel_constraints(
                surface_weight=self.config.surface_constraint_weight,
                toa_weight=self.config.toa_constraint_weight,
                conservation_weight=self.config.conservation_constraint_weight,
            )

        self.constraints = constraints or []
        self.pls_model_: ConstrainedNipalsPLS | None = None
        self.feature_names_: list[str] = []
        self.target_names_: list[str] = ["delta_R_LW", "delta_R_SW"]
        self.is_fitted_: bool = False

    def fit(
        self,
        X: NDArray[np.floating],
        Y: NDArray[np.floating],
        feature_names: list[str] | None = None,
        target_names: list[str] | None = None,
        constraint_params: dict | None = None,
    ) -> TunableKernel:
        """
        Fit the tunable kernel to training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Predictor matrix containing atmospheric state changes:
            - Temperature changes at each pressure level: ΔT(p)
            - Humidity changes at each pressure level: Δq(p)
            - Surface albedo change: Δα
            - Cloud property changes: Δcloud

        Y : array-like of shape (n_samples, n_targets)
            Response matrix containing radiative flux changes:
            - Column 0: ΔR_LW (longwave, positive = more OLR)
            - Column 1: ΔR_SW (shortwave, positive = less absorbed)

        feature_names : list of str, optional
            Names for each feature column.

        target_names : list of str, optional
            Names for each target column.

        constraint_params : dict, optional
            Parameters for physical constraints (surface_temp, emissivity, etc.)

        Returns
        -------
        self : TunableKernel
            Fitted kernel.
        """
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)

        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        # Store feature/target names
        n_features = X.shape[1]
        n_targets = Y.shape[1]

        if feature_names is not None:
            self.feature_names_ = feature_names
        else:
            self.feature_names_ = [f"feature_{i}" for i in range(n_features)]

        if target_names is not None:
            self.target_names_ = target_names
        elif n_targets == 2:
            self.target_names_ = ["delta_R_LW", "delta_R_SW"]
        else:
            self.target_names_ = [f"target_{i}" for i in range(n_targets)]

        # Initialize PLS model
        self.pls_model_ = ConstrainedNipalsPLS(
            n_components=self.config.n_components,
            constraints=self.constraints,
            max_iter=self.config.max_iter,
            tol=self.config.tol,
        )

        # Set constraint parameters
        if constraint_params:
            self.pls_model_.set_constraint_params(**constraint_params)

        # Fit model
        self.pls_model_.fit(X, Y)
        self.is_fitted_ = True

        return self

    def compute(
        self,
        X: NDArray[np.floating],
    ) -> KernelOutput:
        """
        Compute radiative response for given atmospheric state changes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Atmospheric state changes (same format as training X).

        Returns
        -------
        output : KernelOutput
            Radiative response and per-variable contributions.
        """
        if not self.is_fitted_ or self.pls_model_ is None:
            raise ValueError("Kernel not fitted. Call fit() first.")

        X = np.asarray(X, dtype=np.float64)
        Y_pred = self.pls_model_.predict(X)

        # Parse outputs
        if Y_pred.ndim == 1:
            delta_r_lw = Y_pred
            delta_r_sw = np.zeros_like(Y_pred)
        else:
            delta_r_lw = Y_pred[:, 0] if Y_pred.shape[1] > 0 else np.zeros(len(X))
            delta_r_sw = Y_pred[:, 1] if Y_pred.shape[1] > 1 else np.zeros(len(X))

        delta_r_net = delta_r_lw + delta_r_sw

        # Get per-feature contributions
        contributions = self.pls_model_.get_kernel_contributions(self.feature_names_)

        return KernelOutput(
            delta_r_lw=delta_r_lw,
            delta_r_sw=delta_r_sw,
            delta_r_net=delta_r_net,
            contributions=contributions,
            q2_score=None,  # Computed separately with validation data
        )

    def evaluate(
        self,
        X: NDArray[np.floating],
        Y: NDArray[np.floating],
    ) -> float:
        """
        Evaluate kernel on test data using Q² score.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test predictor matrix.
        Y : array-like of shape (n_samples, n_targets)
            Test response matrix.

        Returns
        -------
        q2 : float
            Predictive R² (Q²) score.
        """
        if not self.is_fitted_ or self.pls_model_ is None:
            raise ValueError("Kernel not fitted. Call fit() first.")

        return self.pls_model_.q2_score(X, Y)

    def get_kernel_sensitivities(self) -> dict[str, NDArray[np.floating]]:
        """
        Get kernel sensitivities (regression coefficients).

        Returns
        -------
        sensitivities : dict
            Mapping from feature name to sensitivity vector [dR_LW/dx, dR_SW/dx].
        """
        if not self.is_fitted_ or self.pls_model_ is None:
            raise ValueError("Kernel not fitted.")

        return self.pls_model_.get_kernel_contributions(self.feature_names_)

    def get_vertical_profile(
        self,
        variable: str = "temperature",
        pressure_levels: NDArray[np.floating] | None = None,
    ) -> VerticalKernelProfile | None:
        """
        Extract vertical kernel profile for a variable.

        Parameters
        ----------
        variable : str
            Variable type: "temperature" or "humidity"
        pressure_levels : array-like, optional
            Pressure levels corresponding to features. If None, attempts
            to infer from feature names.

        Returns
        -------
        profile : VerticalKernelProfile or None
            Vertical structure of kernel, or None if not available.
        """
        if not self.is_fitted_:
            return None

        sensitivities = self.get_kernel_sensitivities()

        # Find features matching the variable
        prefix = "T_" if variable == "temperature" else "q_"
        matching_features = [
            (name, sens) for name, sens in sensitivities.items()
            if name.startswith(prefix)
        ]

        if not matching_features:
            return None

        # Try to parse pressure levels from feature names
        if pressure_levels is None:
            try:
                pressure_levels = np.array([
                    float(name.split("_")[1]) for name, _ in matching_features
                ])
            except (IndexError, ValueError):
                return None

        # Extract LW sensitivities (first target column)
        kernel_values = np.array([sens[0] for _, sens in matching_features])

        if variable == "temperature":
            return VerticalKernelProfile(
                pressure_levels=pressure_levels,
                temperature_kernel=kernel_values,
                humidity_kernel=np.zeros_like(kernel_values),
            )
        else:
            return VerticalKernelProfile(
                pressure_levels=pressure_levels,
                temperature_kernel=np.zeros_like(kernel_values),
                humidity_kernel=kernel_values,
            )


class MultiStateKernel:
    """
    Multi-regime tunable radiative kernel with SIMCA-style state routing.

    This kernel maintains separate PLS models for each climate regime
    and routes predictions based on the atmospheric state.

    Parameters
    ----------
    config : KernelConfig
        Kernel configuration.
    classifier : ClimateStateClassifier, optional
        State classifier for regime assignment.
    use_soft_assignment : bool
        If True, uses SIMCA soft classification for regime blending.
        If False, uses hard regime assignment.

    Attributes
    ----------
    regime_kernels_ : dict[int, TunableKernel]
        Fitted kernels for each regime.
    classifier_ : ClimateStateClassifier
        State classifier.
    simca_classifier_ : SIMCAClassifier
        SIMCA classifier for soft assignment (if enabled).
    """

    def __init__(
        self,
        config: KernelConfig | None = None,
        classifier: ClimateStateClassifier | None = None,
        use_soft_assignment: bool = True,
    ):
        self.config = config or KernelConfig()
        self.classifier_ = classifier or ClimateStateClassifier()
        self.use_soft_assignment = use_soft_assignment

        self.regime_kernels_: dict[int, TunableKernel] = {}
        self.simca_classifier_: SIMCAClassifier | None = None
        self.is_fitted_: bool = False
        self.feature_names_: list[str] = []

    def fit(
        self,
        X: NDArray[np.floating],
        Y: NDArray[np.floating],
        latitude: NDArray[np.floating] | None = None,
        cloud_fraction: NDArray[np.floating] | None = None,
        lts: NDArray[np.floating] | None = None,
        feature_names: list[str] | None = None,
        constraint_params: dict | None = None,
    ) -> MultiStateKernel:
        """
        Fit regime-specific kernels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Predictor matrix.
        Y : array-like of shape (n_samples, n_targets)
            Response matrix.
        latitude : array-like of shape (n_samples,), optional
            Latitude for state classification.
        cloud_fraction : array-like of shape (n_samples,), optional
            Cloud fraction for state classification.
        lts : array-like of shape (n_samples,), optional
            Lower tropospheric stability for state classification.
        feature_names : list of str, optional
            Feature names.
        constraint_params : dict, optional
            Parameters for physical constraints.

        Returns
        -------
        self : MultiStateKernel
            Fitted multi-state kernel.
        """
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)

        if feature_names is not None:
            self.feature_names_ = feature_names
        else:
            self.feature_names_ = [f"feature_{i}" for i in range(X.shape[1])]

        # Classify samples into regimes
        regime_ids = self.classifier_.fit_predict(
            latitude=latitude,
            cloud_fraction=cloud_fraction,
            lts=lts,
        )

        # Fit SIMCA classifier if using soft assignment
        if self.use_soft_assignment:
            self.simca_classifier_ = SIMCAClassifier(
                base_classifier=self.classifier_,
                n_components=min(3, X.shape[1]),
            )
            self.simca_classifier_.fit(X, regime_ids)

        # Fit kernel for each regime
        active_regimes = self.classifier_.get_active_regimes()

        for regime_id in active_regimes:
            mask = regime_ids == regime_id

            # Skip regimes with too few samples
            n_samples = np.sum(mask)
            if n_samples < self.config.n_components * 2:
                continue

            # Create and fit regime-specific kernel
            regime_kernel = TunableKernel(config=self.config)
            regime_kernel.fit(
                X[mask],
                Y[mask],
                feature_names=self.feature_names_,
                constraint_params=constraint_params,
            )

            self.regime_kernels_[regime_id] = regime_kernel

        self.is_fitted_ = True
        return self

    def compute(
        self,
        X: NDArray[np.floating],
        latitude: NDArray[np.floating] | None = None,
        cloud_fraction: NDArray[np.floating] | None = None,
        lts: NDArray[np.floating] | None = None,
    ) -> KernelOutput:
        """
        Compute radiative response using regime-appropriate kernels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Atmospheric state changes.
        latitude : array-like, optional
            Latitude for regime assignment.
        cloud_fraction : array-like, optional
            Cloud fraction for regime assignment.
        lts : array-like, optional
            LTS for regime assignment.

        Returns
        -------
        output : KernelOutput
            Radiative response with regime information.
        """
        if not self.is_fitted_:
            raise ValueError("Kernel not fitted. Call fit() first.")

        X = np.asarray(X, dtype=np.float64)
        n_samples = len(X)

        # Initialize outputs
        delta_r_lw = np.zeros(n_samples)
        delta_r_sw = np.zeros(n_samples)

        if self.use_soft_assignment and self.simca_classifier_ is not None:
            # Soft assignment: blend predictions weighted by regime probability
            proba = self.simca_classifier_.predict_proba(X)
            regime_ids_ordered = sorted(self.regime_kernels_.keys())

            for j, regime_id in enumerate(regime_ids_ordered):
                if regime_id not in self.regime_kernels_:
                    continue

                kernel = self.regime_kernels_[regime_id]
                output = kernel.compute(X)

                # Weight by regime probability
                weights = proba[:, j] if j < proba.shape[1] else np.zeros(n_samples)
                delta_r_lw += weights * output.delta_r_lw
                delta_r_sw += weights * output.delta_r_sw

            regime_assignments = self.simca_classifier_.predict(X)

        else:
            # Hard assignment: use single regime per sample
            regime_ids = self.classifier_.fit_predict(
                latitude=latitude,
                cloud_fraction=cloud_fraction,
                lts=lts,
            )

            for regime_id, kernel in self.regime_kernels_.items():
                mask = regime_ids == regime_id
                if not mask.any():
                    continue

                output = kernel.compute(X[mask])
                delta_r_lw[mask] = output.delta_r_lw
                delta_r_sw[mask] = output.delta_r_sw

            regime_assignments = regime_ids

        return KernelOutput(
            delta_r_lw=delta_r_lw,
            delta_r_sw=delta_r_sw,
            delta_r_net=delta_r_lw + delta_r_sw,
            regime_ids=regime_assignments,
        )

    def evaluate(
        self,
        X: NDArray[np.floating],
        Y: NDArray[np.floating],
        latitude: NDArray[np.floating] | None = None,
        cloud_fraction: NDArray[np.floating] | None = None,
        lts: NDArray[np.floating] | None = None,
    ) -> dict[str, float]:
        """
        Evaluate kernel on test data.

        Returns
        -------
        metrics : dict
            Q² scores for overall and per-regime performance.
        """
        if not self.is_fitted_:
            raise ValueError("Kernel not fitted.")

        output = self.compute(X, latitude, cloud_fraction, lts)
        Y_pred = np.column_stack([output.delta_r_lw, output.delta_r_sw])

        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        # Overall Q²
        ss_res = np.nansum((Y - Y_pred[:, : Y.shape[1]]) ** 2)
        ss_tot = np.nansum((Y - np.nanmean(Y, axis=0)) ** 2)
        overall_q2 = 1.0 - ss_res / ss_tot

        metrics = {"overall_q2": overall_q2}

        # Per-regime Q²
        if output.regime_ids is not None:
            for regime_id in np.unique(output.regime_ids):
                mask = output.regime_ids == regime_id
                if mask.sum() < 2:
                    continue

                ss_res_r = np.nansum((Y[mask] - Y_pred[mask, : Y.shape[1]]) ** 2)
                ss_tot_r = np.nansum((Y[mask] - np.nanmean(Y[mask], axis=0)) ** 2)

                if ss_tot_r > 0:
                    metrics[f"regime_{regime_id}_q2"] = 1.0 - ss_res_r / ss_tot_r

        return metrics

    def get_regime_statistics(self) -> dict[int, dict[str, Any]]:
        """
        Get statistics for each regime kernel.

        Returns
        -------
        stats : dict
            Per-regime statistics including sample count and kernel info.
        """
        stats = {}

        for regime_id, kernel in self.regime_kernels_.items():
            state = ClimateState.from_regime_id(regime_id)
            stats[regime_id] = {
                "name": state.name,
                "latitude_band": state.latitude_band.name,
                "cloud_state": state.cloud_state.name,
                "stability_state": state.stability_state.name,
                "n_components": kernel.config.n_components,
                "is_fitted": kernel.is_fitted_,
            }

        return stats


def compare_vertical_resolutions(
    X: NDArray[np.floating],
    Y: NDArray[np.floating],
    feature_names: list[str],
    pressure_levels_standard: NDArray[np.floating],
    n_folds: int = 5,
    random_state: int = 42,
) -> dict[str, float]:
    """
    Compare Q² performance between standard and adaptive vertical resolution.

    Parameters
    ----------
    X : array-like
        Full feature matrix including all vertical levels.
    Y : array-like
        Response matrix.
    feature_names : list of str
        Names of features (used to identify level-specific features).
    pressure_levels_standard : array-like
        Standard 17 pressure levels for comparison.
    n_folds : int
        Number of cross-validation folds.
    random_state : int
        Random seed.

    Returns
    -------
    results : dict
        Q² scores for each configuration.
    """
    from sklearn.model_selection import KFold

    rng = np.random.default_rng(random_state)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    results = {
        "standard_17_level_q2": [],
        "adaptive_pls_q2": [],
    }

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]

        # Standard 17-level kernel
        kernel_standard = TunableKernel(
            config=KernelConfig(n_components=5, use_adaptive_levels=False)
        )
        kernel_standard.fit(X_train, Y_train, feature_names=feature_names)
        q2_standard = kernel_standard.evaluate(X_test, Y_test)
        results["standard_17_level_q2"].append(q2_standard)

        # Adaptive (let PLS determine importance)
        kernel_adaptive = TunableKernel(
            config=KernelConfig(n_components=10, use_adaptive_levels=True)
        )
        kernel_adaptive.fit(X_train, Y_train, feature_names=feature_names)
        q2_adaptive = kernel_adaptive.evaluate(X_test, Y_test)
        results["adaptive_pls_q2"].append(q2_adaptive)

    # Compute mean and std
    return {
        "standard_17_level_q2_mean": np.mean(results["standard_17_level_q2"]),
        "standard_17_level_q2_std": np.std(results["standard_17_level_q2"]),
        "adaptive_pls_q2_mean": np.mean(results["adaptive_pls_q2"]),
        "adaptive_pls_q2_std": np.std(results["adaptive_pls_q2"]),
    }
