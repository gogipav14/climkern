"""
Constrained NIPALS-PLS for Tunable Radiative Kernels

Extends open_nipals.NipalsPLS with:
- Physical constraint integration (Stefan-Boltzmann, energy conservation)
- Multi-level radiative constraints
- Interpretability tools for kernel decomposition
- JAX autodiff for constraint gradient computation (with NumPy fallback)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from backend import HAS_JAX, get_array_module, get_nipals_pls_class, to_numpy

if HAS_JAX:
    import jax
    import jax.numpy as jnp

BaseNipalsPLS = get_nipals_pls_class()

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
    y_loadings: NDArray[np.floating]  # Q matrix (may be adjusted by constraints)
    x_weights: NDArray[np.floating]  # W matrix
    regression_matrix: NDArray[np.floating]  # Inner (P'W)^-1 matrix from base PLS

    # Marker indicating y_loadings have been adjusted by constraints.
    # We don't store a precomputed regression vector due to numerical stability.
    # Instead, predict() uses score-based computation: T @ B_inner @ Q'
    regression_vector: str | None = None  # "adjusted" if constraints applied, None otherwise

    # Variance explained
    x_variance_explained: NDArray[np.floating] = field(default_factory=lambda: np.array([]))
    y_variance_explained: NDArray[np.floating] = field(default_factory=lambda: np.array([]))

    # Constraint diagnostics
    constraint_residuals: dict[str, float] = field(default_factory=dict)

    # Preprocessing (for transform/predict)
    x_mean: NDArray[np.floating] | None = None
    y_mean: NDArray[np.floating] | None = None

    @property
    def coefficients(self) -> NDArray[np.floating]:
        """
        Full regression coefficients X -> Y.

        Note: For numerical stability, we compute via W @ R @ Q' where R is the
        inner regression matrix. This may have numerical issues for ill-conditioned
        data; use predict() for actual predictions.
        """
        return self.x_weights @ self.regression_matrix @ self.y_loadings.T


class ConstrainedNipalsPLS:
    """
    NIPALS-PLS with physical constraints for radiative kernel estimation.

    Extends open_nipals.NipalsPLS with soft physical constraints enforced
    through iterative adjustment of loadings after initial PLS fit.

    Uses JAX autodiff for exact constraint gradients when available,
    falling back to finite differences with NumPy.

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
        if BaseNipalsPLS is None:
            raise ImportError(
                "open_nipals is required. Install via: "
                "pip install git+https://github.com/gogipav14/open_nipals.git@claude/convert-to-jax-FRO4Q"
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

        # Fit base PLS model (uses JAX or NumPy backend via open_nipals)
        self.base_pls_ = BaseNipalsPLS(
            n_components=self.n_components,
            max_iter=self.max_iter,
            tol_criteria=self.tol,
        )
        self.base_pls_.fit(X, Y)

        # Extract results from base model
        x_scores, y_scores = self.base_pls_.transform(X, Y)

        # Ensure results are NumPy arrays for storage
        x_scores = to_numpy(x_scores)
        y_scores = to_numpy(y_scores)

        # Build initial results
        self.results_ = ConstrainedPLSResults(
            x_scores=x_scores,
            y_scores=y_scores,
            x_loadings=to_numpy(self.base_pls_.loadings_x),
            y_loadings=to_numpy(self.base_pls_.loadings_y),
            x_weights=to_numpy(self.base_pls_.weights_x),
            regression_matrix=to_numpy(self.base_pls_.regression_matrix),
            x_variance_explained=self._compute_variance_explained(
                X, x_scores, to_numpy(self.base_pls_.loadings_x)
            ),
            y_variance_explained=self._compute_variance_explained(
                Y, x_scores, to_numpy(self.base_pls_.loadings_y)
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

    def _recompute_regression_vector(self) -> None:
        """
        Mark that y_loadings have been adjusted and predictions should use them.

        Note: We don't precompute a regression vector B = W @ (P'W)^-1 @ B_inner @ Q'
        because the (P'W)^-1 term can be numerically unstable for ill-conditioned data.
        Instead, we use a score-based prediction path that matches open_nipals.predict().
        """
        if self.results_ is None:
            return
        self.results_.regression_vector = "adjusted"

    def _get_scores(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """Get X scores from base PLS model, handling tuple return."""
        scores = self.base_pls_.transform(X)
        if isinstance(scores, tuple):
            scores = scores[0]
        return to_numpy(scores)

    def _apply_constraints(
        self,
        X: NDArray[np.floating],
        Y: NDArray[np.floating],
    ) -> None:
        """Iteratively adjust loadings to satisfy physical constraints.

        This method is numerically robust to ill-conditioned data by:
        1. Only adjusting loadings for components with significant score variance
        2. Using gradient normalization to prevent large updates
        3. Monitoring prediction stability during optimization

        Uses JAX autodiff for exact gradients when available, falling back
        to finite differences with NumPy.
        """
        if self.results_ is None or self.base_pls_ is None:
            return

        # Mark that constraints are being applied
        self._recompute_regression_vector()

        # Determine which components contribute significantly to predictions
        scores = self._get_scores(X)
        score_vars = np.var(scores, axis=0)
        max_var = np.max(score_vars) if np.max(score_vars) > 0 else 1.0
        active_components = score_vars > 1e-10 * max_var

        # Cache scores and B_inner for gradient computation
        self._cached_scores = scores
        self._cached_B_inner = self.results_.regression_matrix.copy()

        # Get initial prediction quality for stability check
        Y_pred_initial = self._predict_with_adjusted_loadings(X)
        initial_pred_range = np.max(np.abs(Y_pred_initial))

        for iteration in range(self.constraint_iter):
            Y_pred = self._predict_with_adjusted_loadings(X)

            # Stability check
            pred_range = np.max(np.abs(Y_pred))
            if pred_range > 100 * initial_pred_range or np.isnan(pred_range):
                break

            # Compute constraint residuals
            total_residual_norm = 0.0
            constraint_residuals = {}

            for constraint in self.constraints:
                residual = constraint.evaluate(Y_pred, X, self._constraint_params)
                residual_norm = np.sqrt(np.nanmean(to_numpy(residual) ** 2))
                constraint_residuals[constraint.name] = residual_norm
                total_residual_norm += constraint.weight * residual_norm

            self.results_.constraint_residuals = constraint_residuals

            if total_residual_norm < self.constraint_tol:
                break

            # Compute gradient (autodiff or finite differences)
            adjustment = self._compute_constraint_gradient(X, Y_pred)

            # Zero out adjustments for inactive components
            adjustment[:, ~active_components] = 0.0

            # Apply damped update with gradient normalization
            grad_norm = np.sqrt(np.sum(adjustment**2))
            if grad_norm > 1e-10:
                active_loadings = self.results_.y_loadings[:, active_components]
                loading_scale = np.sqrt(np.sum(active_loadings**2))
                if loading_scale > 1e-10:
                    max_step = 0.01 * loading_scale
                    step_size = min(0.1 / (1 + iteration * 0.1), max_step / grad_norm)
                    self.results_.y_loadings = (
                        self.results_.y_loadings - step_size * adjustment
                    )

            self._recompute_regression_vector()

        # Clean up cached data
        del self._cached_scores
        del self._cached_B_inner

    def _compute_constraint_gradient(
        self,
        X: NDArray[np.floating],
        Y_pred: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Compute gradient of constraint loss w.r.t. Y loadings.

        Uses JAX autodiff when available for exact gradients in a single
        backward pass. Falls back to finite differences with NumPy.
        """
        if self.results_ is None:
            return np.zeros((Y_pred.shape[1], self.n_components))

        if HAS_JAX:
            return self._compute_constraint_gradient_autodiff(X)
        return self._compute_constraint_gradient_fd(X, Y_pred)

    def _compute_constraint_gradient_autodiff(
        self,
        X: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Compute exact gradient via JAX autodiff.

        Defines a scalar loss function L(Q) = Σᵢ λᵢ mean(Cᵢ(T @ B @ Q', X)²)
        and uses jax.grad to compute dL/dQ in a single backward pass.
        """
        scores_jax = jnp.array(self._cached_scores)
        B_inner_jax = jnp.array(self._cached_B_inner)
        X_jax = jnp.array(X)
        q_shape = self.results_.y_loadings.shape

        # Prepare constraint params as JAX arrays where needed
        jax_params = {}
        for key, val in self._constraint_params.items():
            if isinstance(val, np.ndarray):
                jax_params[key] = jnp.array(val)
            else:
                jax_params[key] = val

        constraints = self.constraints

        def constraint_loss(y_loadings_flat: jnp.ndarray) -> jnp.ndarray:
            """Scalar loss: sum of weighted constraint residuals."""
            Q = y_loadings_flat.reshape(q_shape)
            Y_pred = scores_jax @ B_inner_jax @ Q.T

            total_loss = jnp.float32(0.0)
            for constraint in constraints:
                residual = constraint.evaluate(Y_pred, X_jax, jax_params)
                total_loss = total_loss + constraint.weight * jnp.nanmean(residual**2)
            return total_loss

        grad_fn = jax.grad(constraint_loss)
        y_loadings_flat = jnp.array(self.results_.y_loadings.flatten())
        gradient_flat = grad_fn(y_loadings_flat)

        return np.array(gradient_flat.reshape(q_shape))

    def _compute_constraint_gradient_fd(
        self,
        X: NDArray[np.floating],
        Y_pred: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Compute gradient via finite differences (NumPy fallback)."""
        gradient = np.zeros_like(self.results_.y_loadings)

        for constraint in self.constraints:
            eps = 1e-6
            for j in range(self.results_.y_loadings.shape[0]):
                for a in range(self.results_.y_loadings.shape[1]):
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

    def _predict_with_adjusted_loadings(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Predict using adjusted y_loadings via score-based computation.

        This is numerically stable because it matches open_nipals.predict():
            Y_pred = scores @ B_inner @ Q'
        where scores = transform(X) and Q may have been adjusted by constraints.
        """
        if self.results_ is None or self.base_pls_ is None:
            raise ValueError("Model not fitted.")

        scores = self._get_scores(X)
        B_inner = self.results_.regression_matrix
        Q = self.results_.y_loadings

        return scores @ B_inner @ Q.T

    def _predict_internal(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """Internal prediction using current loadings (for gradient computation)."""
        if self.results_ is None:
            raise ValueError("Model not fitted.")
        return self._predict_with_adjusted_loadings(X)

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

        result = self.base_pls_.transform(X, Y)
        if isinstance(result, tuple):
            return to_numpy(result[0]), to_numpy(result[1])
        return to_numpy(result)

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

        Note
        ----
        If constraints were applied during fitting, predictions use the
        adjusted y_loadings via score-based computation:
            Y_pred = scores @ B_inner @ Q'
        where Q has been adjusted to satisfy physical constraints.
        This is numerically stable for ill-conditioned data.
        """
        if self.base_pls_ is None or self.results_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if self.results_.regression_vector is not None:
            return self._predict_with_adjusted_loadings(X)
        else:
            return to_numpy(self.base_pls_.predict(X))

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
        loadings, analogous to dR/dx_j from traditional radiative kernels.

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
        Compute Q2 (predictive R2) score.

        Q2 = 1 - SS_res / SS_tot

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test predictor matrix.
        Y : array-like of shape (n_samples, n_targets)
            Test response matrix.

        Returns
        -------
        q2 : float
            Predictive R2 score (higher is better, max 1.0).
        """
        Y_pred = self.predict(X)
        ss_res = np.nansum((Y - Y_pred) ** 2)
        ss_tot = np.nansum((Y - np.nanmean(Y, axis=0)) ** 2)
        return 1.0 - ss_res / ss_tot


# =============================================================================
# Pre-defined Constraint Functions
# =============================================================================
# These functions use get_array_module() to work with both JAX and NumPy.
# They must be pure functions (no side effects) to support jax.grad.


def stefan_boltzmann_constraint(
    Y_pred,
    X,
    params: dict,
):
    """
    Stefan-Boltzmann constraint for surface upwelling flux.

    Enforces linearized S-B law: dF_sfc ~ 4 e s T^3 dT

    Required params:
    - surface_temp: Surface temperature T_s (K), shape (n_samples,)
    - emissivity: Surface emissivity e (scalar or array)
    - delta_temp_idx: Column index of dT_surface in X
    - surface_flux_idx: Column index of surface flux in Y (default 0)
    """
    xp = get_array_module()

    T_s = params.get("surface_temp")
    eps = params.get("emissivity", 1.0)
    dt_idx = params.get("delta_temp_idx", 0)
    flux_idx = params.get("surface_flux_idx", 0)

    if T_s is None:
        return xp.zeros((Y_pred.shape[0], 1))

    T_s = xp.asarray(T_s)
    delta_T = X[:, dt_idx]

    expected_delta_F = 4 * eps * STEFAN_BOLTZMANN * T_s**3 * delta_T

    if Y_pred.ndim > 1:
        actual_delta_F = Y_pred[:, flux_idx]
    else:
        actual_delta_F = Y_pred

    return (actual_delta_F - expected_delta_F).reshape(-1, 1)


def energy_conservation_constraint(
    Y_pred,
    X,
    params: dict,
):
    """
    Global energy conservation constraint.

    Enforces: sum(dR * area_weight) ~ expected_imbalance

    Required params:
    - area_weights: Grid cell area weights, shape (n_samples,)
    - expected_imbalance: Expected global mean imbalance change (W/m2)
    """
    xp = get_array_module()

    area_weights = params.get("area_weights")
    expected = params.get("expected_imbalance", 0.0)

    if area_weights is None:
        return xp.zeros((Y_pred.shape[0], 1))

    area_weights = xp.asarray(area_weights)

    if Y_pred.ndim > 1:
        total_flux = Y_pred.sum(axis=1)
    else:
        total_flux = Y_pred

    global_mean = xp.sum(total_flux * area_weights) / xp.sum(area_weights)
    residual = global_mean - expected

    return xp.full((Y_pred.shape[0], 1), residual)


def toa_emissivity_constraint(
    Y_pred,
    X,
    params: dict,
):
    """
    TOA effective emissivity constraint.

    Enforces: OLR ~ e_eff * sigma * T_eff^4

    Required params:
    - effective_temp: Effective emission temperature T_eff (K)
    - effective_emissivity: Effective TOA emissivity e_eff
    - olr_idx: Column index of OLR in Y (default 0)
    """
    xp = get_array_module()

    T_eff = params.get("effective_temp")
    eps_eff = params.get("effective_emissivity", 1.0)
    olr_idx = params.get("olr_idx", 0)

    if T_eff is None:
        return xp.zeros((Y_pred.shape[0], 1))

    T_eff = xp.asarray(T_eff)

    expected_olr = eps_eff * STEFAN_BOLTZMANN * T_eff**4

    baseline_olr = params.get("baseline_olr", 240.0)
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
    1. Surface Stefan-Boltzmann (weight = surface_weight)
    2. TOA effective emissivity (weight = toa_weight)
    3. Global energy conservation (weight = conservation_weight)
    """
    return [
        create_surface_constraint(weight=surface_weight),
        create_toa_constraint(weight=toa_weight),
        create_conservation_constraint(weight=conservation_weight),
    ]
