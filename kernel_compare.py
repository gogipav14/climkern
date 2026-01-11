"""
Comparison utilities for tunable vs traditional radiative kernels.

Provides tools to:
- Compare kernel sensitivities (dR/dT, dR/dq, etc.)
- Compare climate feedback estimates
- Validate against IPCC AR6 assessed ranges
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


# IPCC AR6 assessed feedback ranges (W/m²/K)
# From Forster et al. (2021), Chapter 7
IPCC_AR6_FEEDBACKS = {
    "planck": {"mean": -3.2, "likely_range": (-3.4, -3.0)},
    "water_vapor": {"mean": 1.8, "likely_range": (1.5, 2.1)},
    "lapse_rate": {"mean": -0.5, "likely_range": (-0.8, -0.2)},
    "surface_albedo": {"mean": 0.35, "likely_range": (0.25, 0.45)},
    "cloud": {"mean": 0.45, "likely_range": (-0.1, 1.0)},  # Large uncertainty
    "net": {"mean": -1.1, "likely_range": (-1.5, -0.8)},
}


@dataclass
class KernelComparison:
    """Results from kernel comparison."""

    # Kernel names
    kernel_names: list[str]

    # Sensitivity comparison (per variable)
    sensitivity_correlation: dict[str, float]
    sensitivity_rmse: dict[str, float]

    # Feedback estimates (W/m²/K)
    feedback_estimates: dict[str, dict[str, float]]

    # Summary statistics
    overall_correlation: float
    overall_rmse: float

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "kernel_names": self.kernel_names,
            "sensitivity_correlation": self.sensitivity_correlation,
            "sensitivity_rmse": self.sensitivity_rmse,
            "feedback_estimates": self.feedback_estimates,
            "overall_correlation": self.overall_correlation,
            "overall_rmse": self.overall_rmse,
        }


def compare_to_traditional(
    tunable_kernel,
    traditional_kernel: dict[str, NDArray[np.floating]],
    feature_names: list[str] | None = None,
) -> KernelComparison:
    """
    Compare tunable kernel sensitivities to traditional kernel.

    Parameters
    ----------
    tunable_kernel : TunableKernel
        Fitted tunable kernel.
    traditional_kernel : dict
        Traditional kernel sensitivities. Keys are variable names
        (e.g., "T_1000", "q_700"), values are sensitivity arrays.
    feature_names : list of str, optional
        Feature names in tunable kernel.

    Returns
    -------
    comparison : KernelComparison
        Comparison results.
    """
    # Get tunable kernel sensitivities
    tunable_sens = tunable_kernel.get_kernel_sensitivities()

    if feature_names is None:
        feature_names = list(tunable_sens.keys())

    # Compare each variable
    correlations = {}
    rmses = {}
    all_tunable = []
    all_traditional = []

    for var_name in feature_names:
        if var_name not in tunable_sens or var_name not in traditional_kernel:
            continue

        t_sens = tunable_sens[var_name]
        trad_sens = traditional_kernel[var_name]

        # Ensure same shape
        if isinstance(t_sens, np.ndarray) and len(t_sens) > 0:
            t_val = t_sens[0] if len(t_sens) > 0 else t_sens
        else:
            t_val = t_sens

        if isinstance(trad_sens, np.ndarray):
            trad_val = np.mean(trad_sens)  # Average over space/time
        else:
            trad_val = trad_sens

        all_tunable.append(t_val)
        all_traditional.append(trad_val)

    # Compute overall statistics
    all_tunable = np.array(all_tunable)
    all_traditional = np.array(all_traditional)

    if len(all_tunable) > 1:
        overall_corr = float(np.corrcoef(all_tunable, all_traditional)[0, 1])
        overall_rmse = float(np.sqrt(np.mean((all_tunable - all_traditional) ** 2)))
    else:
        overall_corr = np.nan
        overall_rmse = np.nan

    # Per-variable comparison
    for var_name in feature_names:
        if var_name not in tunable_sens or var_name not in traditional_kernel:
            continue

        t_sens = tunable_sens[var_name]
        trad_sens = traditional_kernel[var_name]

        # Flatten arrays
        t_flat = np.asarray(t_sens).flatten()
        trad_flat = np.asarray(trad_sens).flatten()

        # Match sizes (take minimum)
        n = min(len(t_flat), len(trad_flat))
        if n > 1:
            correlations[var_name] = float(np.corrcoef(t_flat[:n], trad_flat[:n])[0, 1])
            rmses[var_name] = float(np.sqrt(np.mean((t_flat[:n] - trad_flat[:n]) ** 2)))
        else:
            correlations[var_name] = np.nan
            rmses[var_name] = float(np.abs(t_flat[0] - trad_flat[0])) if n == 1 else np.nan

    return KernelComparison(
        kernel_names=["tunable", "traditional"],
        sensitivity_correlation=correlations,
        sensitivity_rmse=rmses,
        feedback_estimates={},  # Computed separately
        overall_correlation=overall_corr,
        overall_rmse=overall_rmse,
    )


def compute_feedback(
    kernel_sensitivities: dict[str, NDArray[np.floating]],
    climate_change: dict[str, NDArray[np.floating]],
    delta_T_global: float,
    area_weights: NDArray[np.floating] | None = None,
) -> dict[str, float]:
    """
    Compute climate feedbacks from kernel sensitivities and climate change.

    Feedback λ_x = ∫∫ (∂R/∂x) × Δx dA / ΔT_global

    Parameters
    ----------
    kernel_sensitivities : dict
        Kernel sensitivities by variable (W/m² per unit change).
    climate_change : dict
        Climate change by variable (e.g., ΔT(p), Δq(p)).
    delta_T_global : float
        Global mean surface temperature change (K).
    area_weights : array-like, optional
        Area weights for spatial averaging.

    Returns
    -------
    feedbacks : dict
        Feedback parameters (W/m²/K) for each variable and total.
    """
    feedbacks = {}
    total_feedback = 0.0

    for var_name, sensitivity in kernel_sensitivities.items():
        if var_name not in climate_change:
            continue

        delta_x = climate_change[var_name]
        sensitivity = np.asarray(sensitivity)
        delta_x = np.asarray(delta_x)

        # Compute flux change: ΔR = kernel × Δx
        if sensitivity.shape == delta_x.shape:
            delta_R = sensitivity * delta_x
        else:
            # Broadcast if shapes differ
            delta_R = sensitivity.flatten()[:len(delta_x.flatten())] * delta_x.flatten()

        # Area-weighted global mean
        if area_weights is not None:
            global_delta_R = np.average(delta_R, weights=area_weights.flatten()[:len(delta_R)])
        else:
            global_delta_R = np.mean(delta_R)

        # Feedback = ΔR / ΔT_global
        if abs(delta_T_global) > 1e-10:
            feedback = float(global_delta_R / delta_T_global)
        else:
            feedback = 0.0

        feedbacks[var_name] = feedback
        total_feedback += feedback

    feedbacks["total"] = total_feedback

    return feedbacks


def compare_feedback_estimates(
    feedback_estimates: dict[str, float],
    reference: str = "IPCC_AR6",
    feedback_mapping: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Compare feedback estimates to reference values.

    Parameters
    ----------
    feedback_estimates : dict
        Computed feedbacks (W/m²/K) by variable.
    reference : str
        Reference to compare against: "IPCC_AR6" or custom.
    feedback_mapping : dict, optional
        Mapping from computed variable names to reference names.

    Returns
    -------
    comparison : dict
        Comparison results including agreement with assessed ranges.
    """
    if reference == "IPCC_AR6":
        ref_feedbacks = IPCC_AR6_FEEDBACKS
    else:
        ref_feedbacks = {}

    if feedback_mapping is None:
        # Default mapping from variable names to feedback types
        feedback_mapping = {
            "T_surface": "planck",
            "T_1000": "planck",
            "q_1000": "water_vapor",
            "q_700": "water_vapor",
            "albedo": "surface_albedo",
            "cloud_fraction": "cloud",
        }

    comparison = {}

    for var_name, estimate in feedback_estimates.items():
        fb_type = feedback_mapping.get(var_name)

        if fb_type and fb_type in ref_feedbacks:
            ref = ref_feedbacks[fb_type]
            ref_mean = ref["mean"]
            ref_range = ref["likely_range"]

            within_range = ref_range[0] <= estimate <= ref_range[1]
            deviation = estimate - ref_mean
            relative_error = deviation / abs(ref_mean) if ref_mean != 0 else np.nan

            comparison[var_name] = {
                "estimate": estimate,
                "reference_mean": ref_mean,
                "reference_range": ref_range,
                "within_likely_range": within_range,
                "deviation": deviation,
                "relative_error": relative_error,
            }
        else:
            comparison[var_name] = {
                "estimate": estimate,
                "reference_mean": None,
                "reference_range": None,
                "within_likely_range": None,
                "deviation": None,
                "relative_error": None,
            }

    return comparison


def compute_inter_kernel_spread(
    kernel_list: list,
    X: NDArray[np.floating],
    Y: NDArray[np.floating],
    variable: str | None = None,
) -> dict[str, float]:
    """
    Compute spread across multiple kernel estimates.

    Useful for quantifying structural uncertainty from kernel choice.

    Parameters
    ----------
    kernel_list : list
        List of fitted kernel objects.
    X : array-like
        Feature matrix for computing predictions.
    Y : array-like
        True responses (for reference).
    variable : str, optional
        Specific variable to compute spread for.

    Returns
    -------
    spread : dict
        Statistics on inter-kernel spread.
    """
    predictions = []

    for kernel in kernel_list:
        if hasattr(kernel, 'predict'):
            Y_pred = kernel.predict(X)
        elif hasattr(kernel, 'compute'):
            output = kernel.compute(X)
            Y_pred = np.column_stack([output.delta_r_lw, output.delta_r_sw])
        else:
            continue

        predictions.append(Y_pred)

    if len(predictions) < 2:
        return {"std": np.nan, "range": np.nan, "cv": np.nan}

    predictions = np.array(predictions)  # (n_kernels, n_samples, n_targets)

    # Compute spread statistics
    std = float(np.nanstd(predictions, axis=0).mean())
    pred_range = float((np.nanmax(predictions, axis=0) - np.nanmin(predictions, axis=0)).mean())
    mean = float(np.nanmean(predictions))
    cv = std / abs(mean) if abs(mean) > 1e-10 else np.nan

    return {
        "std": std,
        "range": pred_range,
        "cv": cv,
        "mean": mean,
    }


def validate_against_4xCO2(
    kernel,
    X_4xCO2: NDArray[np.floating],
    Y_4xCO2_expected: NDArray[np.floating],
    feature_names: list[str] | None = None,
) -> dict[str, float]:
    """
    Validate kernel extrapolation to 4×CO2 scenario.

    This tests whether kernels trained on observed variability
    can accurately predict responses to larger forcing.

    Parameters
    ----------
    kernel : TunableKernel
        Fitted kernel (trained on observations).
    X_4xCO2 : array-like
        Atmospheric state changes in 4×CO2 scenario.
    Y_4xCO2_expected : array-like
        Expected radiative response (from GCM or other source).
    feature_names : list of str, optional
        Feature names.

    Returns
    -------
    validation : dict
        Validation metrics for extrapolation.
    """
    # Predict using tunable kernel
    Y_pred = kernel.predict(X_4xCO2)

    # Compute metrics
    Y_exp = np.asarray(Y_4xCO2_expected)
    Y_pred = np.asarray(Y_pred)

    if Y_exp.ndim == 1:
        Y_exp = Y_exp.reshape(-1, 1)
    if Y_pred.ndim == 1:
        Y_pred = Y_pred.reshape(-1, 1)

    # Ensure same number of columns
    n_targets = min(Y_exp.shape[1], Y_pred.shape[1])
    Y_exp = Y_exp[:, :n_targets]
    Y_pred = Y_pred[:, :n_targets]

    # R² for extrapolation
    ss_res = np.sum((Y_exp - Y_pred) ** 2)
    ss_tot = np.sum((Y_exp - Y_exp.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # RMSE
    rmse = float(np.sqrt(np.mean((Y_exp - Y_pred) ** 2)))

    # Bias
    bias = float(np.mean(Y_pred - Y_exp))

    # Mean absolute error
    mae = float(np.mean(np.abs(Y_pred - Y_exp)))

    return {
        "r2_extrapolation": float(r2),
        "rmse": rmse,
        "bias": bias,
        "mae": mae,
        "expected_mean": float(Y_exp.mean()),
        "predicted_mean": float(Y_pred.mean()),
    }
