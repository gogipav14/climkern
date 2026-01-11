"""
Constrained NIPALS-PLS for Tunable Radiative Kernels

Extends open_nipals.NipalsPLS with:
- Physical constraint integration (Stefan-Boltzmann, energy conservation)
- Multi-level radiative constraints
- Interpretability tools for kernel decomposition
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np
from numpy.typing import NDArray
from scipy import linalg

try:
    from open_nipals.nipalsPLS import NipalsPLS as BaseNipalsPLS
    HAS_OPEN_NIPALS = True
except ImportError:
    HAS_OPEN_NIPALS = False
    BaseNipalsPLS = object


# Physical constants
STEFAN_BOLTZMANN = 5.670374419e-8  # W m⁻² K⁻⁴


class ConstraintFunction(Protocol):
    """Protocol for constraint evaluation functions."""

    def __call__(
        self,
        Y_pred: NDArray[np.floating],
        X: NDArray[np.floating],
        params: dict,
    ) -> NDArray[np.floating]: ...


@dataclass
class PhysicalConstraint:
    """
    Physical constraint for constrained NIPALS-PLS.

    Parameters
    ----------
    name : str
        Constraint identifier (e.g., "stefan_boltzmann", "energy_conservation")
    evaluate : ConstraintFunction
        Function that computes constraint residual: f(Y_pred, X, params) -> residual
    weight : float
        Regularization weight (λ) for this constraint in the loss function.
    """

    name: str
    evaluate: ConstraintFunction
    weight: float = 1.0


@dataclass
class ConstrainedPLSResults:
    """Results from constrained NIPALS-PLS fitting."""

    # Core PLS components (from base NipalsPLS)
    x_scores: NDArray[np.floating]  # T matrix
    y_scores: NDArray[np.floating]  # U matrix
    x_loadings: NDArray[np.floating]  # P matrix
    y_loadings: NDArray[np.floating]  # Q matrix
    x_weights: NDArray[np.floating]  # W matrix
    regression_matrix: NDArray[np.floating]  # B matrix

    # Variance explained
    x_variance_explained: NDArray[np.floating]
    y_variance_explained: NDArray[np.floating]

    # Constraint diagnostics
    constraint_residuals: dict[str, float] = field(default_factory=dict)

    # Preprocessing (for transform/predict)
    x_mean: NDArray[np.floating] | None = None
    y_mean: NDArray[np.floating] | None = None

    @property
    def coefficients(self) -> NDArray[np.floating]:
        """Full regression coefficients X -> Y."""
        return self.x_weights @ self.regression_matrix @ self.y_loadings.T


class ConstrainedNipalsPLS:
    """
    NIPALS-PLS with physical constraints for radiative kernel estimation.

    Extends open_nipals.NipalsPLS with soft physical constraints enforced
    through iterative adjustment of loadings after initial PLS fit.

    Loss function:
        L = ||Y - XB||² + Σᵢ λᵢ ||Cᵢ(Y_pred, X)||²

    Parameters
    ----------
    n_components : int
        Number of PLS latent variables.
    constraints : list[PhysicalConstraint]
        Physical constraints to enforce.
    constraint_iter : int
        Max iterations for constraint optimization after PLS fit.
    constraint_tol : float
        Convergence tolerance for constraint residuals.
    max_iter : int
        Max NIPALS iterations per component (passed to base).
    tol : float
        Convergence tolerance for NIPALS (passed to base).

    Attributes
    ----------
    base_pls_ : NipalsPLS
        Underlying open_nipals PLS model.
    results_ : ConstrainedPLSResults
        Fitted model results.
    """

    def __init__(
        self,
        n_components: int = 5,
        constraints: list[PhysicalConstraint] | None = None,
        constraint_iter: int = 20,
        constraint_tol: float = 1e-6,
        max_iter: int = 500,
        tol: float = 1e-10,
    ):
        if not HAS_OPEN_NIPALS:
            raise ImportError(
                "open_nipals is required. Install via: "
                "pip install git+https://github.com/gogipav14/open_nipals.git"
            )

        self.n_components = n_components
        self.constraints = constraints or []
        self.constraint_iter = constraint_iter
        self.constraint_tol = constraint_tol
        self.max_iter = max_iter
        self.tol = tol

        self._constraint_params: dict = {}
        self.base_pls_: BaseNipalsPLS | None = None
        self.results_: ConstrainedPLSResults | None = None

    def set_constraint_params(self, **params) -> None:
        """
        Set parameters used by constraint functions.

        Common parameters:
        - surface_temp: Surface temperature field T_s (K)
        - emissivity: Surface emissivity ε (scalar or array)
        - delta_temp_idx: Index of ΔT_surface in X columns
        - area_weights: Grid cell area weights for global integrals
        - expected_imbalance: Expected global energy imbalance (W/m²)
        """
        self._constraint_params.update(params)

    def fit(
        self,
        X: NDArray[np.floating],
        Y: NDArray[np.floating],
    ) -> ConstrainedNipalsPLS:
        """
        Fit constrained NIPALS-PLS model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Predictor matrix. For radiative kernels:
            [ΔT(p₁), ΔT(p₂), ..., Δq(p₁), Δq(p₂), ..., Δα, Δcloud, ...]
        Y : array-like of shape (n_samples, n_targets)
            Response matrix. For radiative kernels:
            [ΔR_TOA_LW, ΔR_TOA_SW] or [ΔR_LW(levels), ΔR_SW(levels)]

        Returns
        -------
        self : ConstrainedNipalsPLS
            Fitted estimator.
        """
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)

        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        # Store means for later transform/predict
        x_mean = np.nanmean(X, axis=0)
        y_mean = np.nanmean(Y, axis=0)

        # Fit base PLS model
        self.base_pls_ = BaseNipalsPLS(
            n_components=self.n_components,
            max_iter=self.max_iter,
            tol_criteria=self.tol,
        )
        self.base_pls_.fit(X, Y)

        # Extract results from base model
        x_scores, y_scores = self.base_pls_.transform(X, Y)

        # Build initial results
        self.results_ = ConstrainedPLSResults(
            x_scores=x_scores,
            y_scores=y_scores,
            x_loadings=self.base_pls_.loadings_x,
            y_loadings=self.base_pls_.loadings_y,
            x_weights=self.base_pls_.weights_x,
            regression_matrix=self.base_pls_.regression_matrix,
            x_variance_explained=self._compute_variance_explained(
                X, x_scores, self.base_pls_.loadings_x
            ),
            y_variance_explained=self._compute_variance_explained(
                Y, x_scores, self.base_pls_.loadings_y
            ),
            x_mean=x_mean,
            y_mean=y_mean,
        )

        # Apply constraint adjustments if any constraints defined
        if self.constraints:
            self._apply_constraints(X, Y)

        return self

    def _compute_variance_explained(
        self,
        Z: NDArray[np.floating],
        scores: NDArray[np.floating],
        loadings: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Compute per-component variance explained."""
        total_var = np.nansum(Z**2)
        var_explained = np.zeros(self.n_components)

        Z_residual = Z.copy()
        for a in range(self.n_components):
            t_a = scores[:, a : a + 1]
            p_a = loadings[:, a : a + 1]
            Z_residual = Z_residual - t_a @ p_a.T
            var_explained[a] = 1.0 - np.nansum(Z_residual**2) / total_var

        # Convert cumulative to per-component
        return np.diff(np.concatenate([[0], var_explained]))

    def _apply_constraints(
        self,
        X: NDArray[np.floating],
        Y: NDArray[np.floating],
    ) -> None:
        """Iteratively adjust loadings to satisfy physical constraints."""
        if self.results_ is None:
            return

        for iteration in range(self.constraint_iter):
            # Get current predictions
            Y_pred = self.predict(X)

            # Compute constraint residuals
            total_residual_norm = 0.0
            constraint_residuals = {}

            for constraint in self.constraints:
                residual = constraint.evaluate(Y_pred, X, self._constraint_params)
                residual_norm = np.sqrt(np.nanmean(residual**2))
                constraint_residuals[constraint.name] = residual_norm
                total_residual_norm += constraint.weight * residual_norm

            self.results_.constraint_residuals = constraint_residuals

            # Check convergence
            if total_residual_norm < self.constraint_tol:
                break

            # Compute adjustment direction
            adjustment = self._compute_constraint_gradient(X, Y_pred)

            # Apply damped update to Y loadings
            damping = 0.1 / (1 + iteration * 0.1)  # Decreasing step size
            self.results_.y_loadings = (
                self.results_.y_loadings - damping * adjustment
            )

    def _compute_constraint_gradient(
        self,
        X: NDArray[np.floating],
        Y_pred: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Compute gradient of constraint loss w.r.t. Y loadings."""
        if self.results_ is None:
            return np.zeros((Y_pred.shape[1], self.n_components))

        gradient = np.zeros_like(self.results_.y_loadings)

        for constraint in self.constraints:
            # Numerical gradient via finite differences
            eps = 1e-6
            for j in range(self.results_.y_loadings.shape[0]):
                for a in range(self.results_.y_loadings.shape[1]):
                    # Perturb loading
                    original = self.results_.y_loadings[j, a]

                    self.results_.y_loadings[j, a] = original + eps
                    Y_plus = self._predict_internal(X)
                    loss_plus = np.nanmean(
                        constraint.evaluate(Y_plus, X, self._constraint_params) ** 2
                    )

                    self.results_.y_loadings[j, a] = original - eps
                    Y_minus = self._predict_internal(X)
                    loss_minus = np.nanmean(
                        constraint.evaluate(Y_minus, X, self._constraint_params) ** 2
                    )

                    self.results_.y_loadings[j, a] = original

                    gradient[j, a] += (
                        constraint.weight * (loss_plus - loss_minus) / (2 * eps)
                    )

        return gradient

    def _predict_internal(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """Internal prediction using current loadings (no mean restoration)."""
        if self.results_ is None:
            raise ValueError("Model not fitted.")

        scores = self.base_pls_.transform(X)
        if isinstance(scores, tuple):
            scores = scores[0]

        return scores @ self.results_.regression_matrix @ self.results_.y_loadings.T

    def transform(
        self,
        X: NDArray[np.floating],
        Y: NDArray[np.floating] | None = None,
    ) -> NDArray[np.floating] | tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        Transform X (and optionally Y) to latent space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Predictor matrix.
        Y : array-like of shape (n_samples, n_targets), optional
            Response matrix.

        Returns
        -------
        X_scores : ndarray
            X projected onto latent variables.
        Y_scores : ndarray, optional
            Y projected onto latent variables (if Y provided).
        """
        if self.base_pls_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        return self.base_pls_.transform(X, Y)

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Predict Y from X using the fitted (and constraint-adjusted) model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Predictor matrix.

        Returns
        -------
        Y_pred : ndarray of shape (n_samples, n_targets)
            Predicted response.
        """
        if self.base_pls_ is None or self.results_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Use base PLS prediction (which uses adjusted loadings via our results)
        return self.base_pls_.predict(X)

    def fit_transform(
        self,
        X: NDArray[np.floating],
        Y: NDArray[np.floating],
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Fit model and return scores."""
        self.fit(X, Y)
        return self.transform(X, Y)

    def get_kernel_contributions(
        self,
        feature_names: list[str] | None = None,
    ) -> dict[str, NDArray[np.floating]]:
        """
        Decompose regression coefficients into per-feature contributions.

        This recovers interpretable kernel-like sensitivities from the PLS
        loadings, analogous to ∂R/∂x_j from traditional radiative kernels.

        Parameters
        ----------
        feature_names : list of str, optional
            Names for each feature (e.g., ["T_1000", "T_850", ..., "q_1000", ...])

        Returns
        -------
        contributions : dict
            Mapping from feature name to coefficient vector for each target.
        """
        if self.results_ is None:
            raise ValueError("Model not fitted.")

        coeffs = self.results_.coefficients
        n_features = coeffs.shape[0]

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]

        return {
            name: coeffs[i, :] for i, name in enumerate(feature_names)
        }

    def q2_score(
        self,
        X: NDArray[np.floating],
        Y: NDArray[np.floating],
    ) -> float:
        """
        Compute Q² (predictive R²) score.

        Q² = 1 - SS_res / SS_tot

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test predictor matrix.
        Y : array-like of shape (n_samples, n_targets)
            Test response matrix.

        Returns
        -------
        q2 : float
            Predictive R² score (higher is better, max 1.0).
        """
        Y_pred = self.predict(X)
        ss_res = np.nansum((Y - Y_pred) ** 2)
        ss_tot = np.nansum((Y - np.nanmean(Y, axis=0)) ** 2)
        return 1.0 - ss_res / ss_tot


# =============================================================================
# Pre-defined Constraint Functions
# =============================================================================


def stefan_boltzmann_constraint(
    Y_pred: NDArray[np.floating],
    X: NDArray[np.floating],
    params: dict,
) -> NDArray[np.floating]:
    """
    Stefan-Boltzmann constraint for surface upwelling flux.

    Enforces linearized S-B law: ΔF_sfc ≈ 4 ε σ T³ ΔT

    Required params:
    - surface_temp: Surface temperature T_s (K), shape (n_samples,)
    - emissivity: Surface emissivity ε (scalar or array)
    - delta_temp_idx: Column index of ΔT_surface in X
    - surface_flux_idx: Column index of surface flux in Y (default 0)
    """
    T_s = params.get("surface_temp")
    eps = params.get("emissivity", 1.0)
    dt_idx = params.get("delta_temp_idx", 0)
    flux_idx = params.get("surface_flux_idx", 0)

    if T_s is None:
        return np.zeros((Y_pred.shape[0], 1))

    T_s = np.asarray(T_s)
    delta_T = X[:, dt_idx]

    # Expected flux change from linearized Stefan-Boltzmann
    expected_delta_F = 4 * eps * STEFAN_BOLTZMANN * T_s**3 * delta_T

    # Actual predicted flux change
    if Y_pred.ndim > 1:
        actual_delta_F = Y_pred[:, flux_idx]
    else:
        actual_delta_F = Y_pred

    return (actual_delta_F - expected_delta_F).reshape(-1, 1)


def energy_conservation_constraint(
    Y_pred: NDArray[np.floating],
    X: NDArray[np.floating],
    params: dict,
) -> NDArray[np.floating]:
    """
    Global energy conservation constraint.

    Enforces: Σ(ΔR × area_weight) ≈ expected_imbalance

    Required params:
    - area_weights: Grid cell area weights, shape (n_samples,)
    - expected_imbalance: Expected global mean imbalance change (W/m²)
    """
    area_weights = params.get("area_weights")
    expected = params.get("expected_imbalance", 0.0)

    if area_weights is None:
        return np.zeros((Y_pred.shape[0], 1))

    area_weights = np.asarray(area_weights)

    # Sum all flux components
    if Y_pred.ndim > 1:
        total_flux = Y_pred.sum(axis=1)
    else:
        total_flux = Y_pred

    # Compute area-weighted global mean
    global_mean = np.sum(total_flux * area_weights) / np.sum(area_weights)

    # Residual is deviation from expected
    residual = global_mean - expected

    # Distribute residual across all samples (uniform correction needed)
    return np.full((Y_pred.shape[0], 1), residual)


def toa_emissivity_constraint(
    Y_pred: NDArray[np.floating],
    X: NDArray[np.floating],
    params: dict,
) -> NDArray[np.floating]:
    """
    TOA effective emissivity constraint.

    Enforces: OLR ≈ ε_eff σ T_eff⁴

    Required params:
    - effective_temp: Effective emission temperature T_eff (K)
    - effective_emissivity: Effective TOA emissivity ε_eff
    - olr_idx: Column index of OLR in Y (default 0)
    """
    T_eff = params.get("effective_temp")
    eps_eff = params.get("effective_emissivity", 1.0)
    olr_idx = params.get("olr_idx", 0)

    if T_eff is None:
        return np.zeros((Y_pred.shape[0], 1))

    T_eff = np.asarray(T_eff)

    # Expected OLR from effective S-B
    expected_olr = eps_eff * STEFAN_BOLTZMANN * T_eff**4

    # Actual OLR (note: Y_pred is anomaly, need baseline)
    baseline_olr = params.get("baseline_olr", 240.0)  # Typical Earth OLR
    if Y_pred.ndim > 1:
        predicted_olr = baseline_olr + Y_pred[:, olr_idx]
    else:
        predicted_olr = baseline_olr + Y_pred

    return (predicted_olr - expected_olr).reshape(-1, 1)


# =============================================================================
# Factory Functions
# =============================================================================


def create_surface_constraint(
    weight: float = 1.0,
) -> PhysicalConstraint:
    """Create Stefan-Boltzmann surface constraint."""
    return PhysicalConstraint(
        name="stefan_boltzmann",
        evaluate=stefan_boltzmann_constraint,
        weight=weight,
    )


def create_conservation_constraint(
    weight: float = 1.0,
) -> PhysicalConstraint:
    """Create energy conservation constraint."""
    return PhysicalConstraint(
        name="energy_conservation",
        evaluate=energy_conservation_constraint,
        weight=weight,
    )


def create_toa_constraint(
    weight: float = 1.0,
) -> PhysicalConstraint:
    """Create TOA effective emissivity constraint."""
    return PhysicalConstraint(
        name="toa_emissivity",
        evaluate=toa_emissivity_constraint,
        weight=weight,
    )


def create_multilevel_constraints(
    surface_weight: float = 1.0,
    toa_weight: float = 1.0,
    conservation_weight: float = 0.5,
) -> list[PhysicalConstraint]:
    """
    Create the recommended multi-level constraint set.

    Returns constraints for:
    1. Surface Stefan-Boltzmann (λ = surface_weight)
    2. TOA effective emissivity (λ = toa_weight)
    3. Global energy conservation (λ = conservation_weight)
    """
    return [
        create_surface_constraint(weight=surface_weight),
        create_toa_constraint(weight=toa_weight),
        create_conservation_constraint(weight=conservation_weight),
    ]
