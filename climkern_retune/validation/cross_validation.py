"""
Cross-validation framework for tunable kernel selection.

Implements Q² (predictive R²) based model selection with:
- K-fold cross-validation
- Time-series aware splits (for autocorrelated climate data)
- Component number selection
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator

import numpy as np
from numpy.typing import NDArray


@dataclass
class CVResult:
    """Cross-validation results."""

    q2_scores: NDArray[np.floating]  # Q² for each fold
    q2_mean: float
    q2_std: float

    # Per-target scores
    q2_per_target: dict[str, float] | None = None

    # Optional: per-fold predictions for analysis
    predictions: list[NDArray] | None = None
    actual: list[NDArray] | None = None

    @property
    def q2_cv(self) -> float:
        """Cross-validated Q² (mean across folds)."""
        return self.q2_mean


class CrossValidator(ABC):
    """Abstract base class for cross-validation strategies."""

    @abstractmethod
    def split(
        self,
        X: NDArray[np.floating],
        Y: NDArray[np.floating] | None = None,
    ) -> Iterator[tuple[NDArray[np.int_], NDArray[np.int_]]]:
        """
        Generate train/test indices.

        Yields
        ------
        train_idx : ndarray
            Training indices.
        test_idx : ndarray
            Test indices.
        """
        pass

    @abstractmethod
    def get_n_splits(self) -> int:
        """Return number of splits."""
        pass


class KFoldCV(CrossValidator):
    """
    K-Fold cross-validation.

    Parameters
    ----------
    n_splits : int
        Number of folds.
    shuffle : bool
        Whether to shuffle before splitting.
    random_state : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int = 42,
    ):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(
        self,
        X: NDArray[np.floating],
        Y: NDArray[np.floating] | None = None,
    ) -> Iterator[tuple[NDArray[np.int_], NDArray[np.int_]]]:
        """Generate K-fold train/test splits."""
        n_samples = len(X)
        indices = np.arange(n_samples)

        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            rng.shuffle(indices)

        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits)
        fold_sizes[: n_samples % self.n_splits] += 1

        current = 0
        for fold_size in fold_sizes:
            test_idx = indices[current : current + fold_size]
            train_idx = np.concatenate([
                indices[:current],
                indices[current + fold_size:]
            ])
            yield train_idx, test_idx
            current += fold_size

    def get_n_splits(self) -> int:
        return self.n_splits


class TimeSeriesCV(CrossValidator):
    """
    Time-series aware cross-validation.

    Uses expanding window or rolling window to respect temporal ordering
    and avoid data leakage from future to past.

    Parameters
    ----------
    n_splits : int
        Number of splits.
    test_size : int or float
        Size of test set (absolute or fraction).
    gap : int
        Gap between train and test to account for autocorrelation.
    expanding : bool
        If True, use expanding window. If False, use rolling window.
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_size: int | float = 0.2,
        gap: int = 0,
        expanding: bool = True,
    ):
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.expanding = expanding

    def split(
        self,
        X: NDArray[np.floating],
        Y: NDArray[np.floating] | None = None,
    ) -> Iterator[tuple[NDArray[np.int_], NDArray[np.int_]]]:
        """Generate time-series train/test splits."""
        n_samples = len(X)

        # Determine test size
        if isinstance(self.test_size, float):
            test_size = int(n_samples * self.test_size)
        else:
            test_size = self.test_size

        # Calculate split points
        if self.expanding:
            # Expanding window: train grows, test fixed size
            min_train = n_samples // (self.n_splits + 1)

            for i in range(self.n_splits):
                train_end = min_train + i * (n_samples - min_train - test_size) // self.n_splits
                test_start = train_end + self.gap
                test_end = min(test_start + test_size, n_samples)

                train_idx = np.arange(train_end)
                test_idx = np.arange(test_start, test_end)

                yield train_idx, test_idx

        else:
            # Rolling window: train and test both slide
            window_size = (n_samples - test_size - self.gap) // self.n_splits

            for i in range(self.n_splits):
                train_start = i * window_size
                train_end = train_start + window_size
                test_start = train_end + self.gap
                test_end = test_start + test_size

                train_idx = np.arange(train_start, train_end)
                test_idx = np.arange(test_start, min(test_end, n_samples))

                yield train_idx, test_idx

    def get_n_splits(self) -> int:
        return self.n_splits


def compute_q2_score(
    Y_true: NDArray[np.floating],
    Y_pred: NDArray[np.floating],
    Y_train_mean: NDArray[np.floating] | None = None,
) -> float:
    """
    Compute Q² (predictive R²) score.

    Q² = 1 - SS_res / SS_tot

    where SS_tot is computed using the training set mean (not test set mean)
    for proper out-of-sample evaluation.

    Parameters
    ----------
    Y_true : array-like
        True target values.
    Y_pred : array-like
        Predicted target values.
    Y_train_mean : array-like, optional
        Mean of training targets for SS_tot calculation.
        If None, uses mean of Y_true.

    Returns
    -------
    q2 : float
        Predictive R² score.
    """
    Y_true = np.asarray(Y_true)
    Y_pred = np.asarray(Y_pred)

    if Y_train_mean is None:
        Y_train_mean = np.nanmean(Y_true, axis=0)

    ss_res = np.nansum((Y_true - Y_pred) ** 2)
    ss_tot = np.nansum((Y_true - Y_train_mean) ** 2)

    if ss_tot < 1e-10:
        return 0.0

    return 1.0 - ss_res / ss_tot


def cross_validate(
    model,
    X: NDArray[np.floating],
    Y: NDArray[np.floating],
    cv: CrossValidator | None = None,
    return_predictions: bool = False,
) -> CVResult:
    """
    Perform cross-validation on a tunable kernel model.

    Parameters
    ----------
    model : TunableKernel or similar
        Model with fit() and predict() methods.
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    Y : array-like of shape (n_samples, n_targets)
        Target matrix.
    cv : CrossValidator, optional
        Cross-validation strategy. Default: 5-fold.
    return_predictions : bool
        Whether to return predictions for each fold.

    Returns
    -------
    result : CVResult
        Cross-validation results.
    """
    if cv is None:
        cv = KFoldCV(n_splits=5)

    X = np.asarray(X)
    Y = np.asarray(Y)

    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    q2_scores = []
    predictions = [] if return_predictions else None
    actuals = [] if return_predictions else None

    for train_idx, test_idx in cv.split(X, Y):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]

        # Clone and fit model
        model_clone = _clone_model(model)
        model_clone.fit(X_train, Y_train)

        # Predict
        Y_pred = model_clone.predict(X_test)

        # Compute Q² using training mean
        Y_train_mean = np.nanmean(Y_train, axis=0)
        q2 = compute_q2_score(Y_test, Y_pred, Y_train_mean)
        q2_scores.append(q2)

        if return_predictions:
            predictions.append(Y_pred)
            actuals.append(Y_test)

    q2_scores = np.array(q2_scores)

    return CVResult(
        q2_scores=q2_scores,
        q2_mean=float(np.mean(q2_scores)),
        q2_std=float(np.std(q2_scores)),
        predictions=predictions,
        actual=actuals,
    )


def _clone_model(model):
    """Create a fresh copy of a model with same parameters."""
    # Try to use model's config if available
    if hasattr(model, 'config'):
        model_class = type(model)
        return model_class(config=model.config)
    else:
        # Fallback: create new instance with default params
        model_class = type(model)
        return model_class()


def select_n_components(
    X: NDArray[np.floating],
    Y: NDArray[np.floating],
    model_class,
    max_components: int = 15,
    cv: CrossValidator | None = None,
    verbose: bool = False,
) -> dict:
    """
    Select optimal number of PLS components using cross-validation.

    Parameters
    ----------
    X : array-like
        Feature matrix.
    Y : array-like
        Target matrix.
    model_class : type
        Model class to use (e.g., TunableKernel).
    max_components : int
        Maximum number of components to test.
    cv : CrossValidator, optional
        Cross-validation strategy.
    verbose : bool
        Print progress.

    Returns
    -------
    results : dict
        Contains:
        - optimal_n: Best number of components
        - q2_by_n: Q² for each n_components
        - q2_std_by_n: Std of Q² for each n_components
    """
    if cv is None:
        cv = KFoldCV(n_splits=5)

    max_possible = min(max_components, X.shape[1], X.shape[0] // cv.get_n_splits() - 1)

    q2_by_n = []
    q2_std_by_n = []

    for n_comp in range(1, max_possible + 1):
        if verbose:
            print(f"Testing n_components={n_comp}...")

        # Create model with this n_components
        if hasattr(model_class, 'config'):
            from climkern_retune.core import KernelConfig
            config = KernelConfig(n_components=n_comp)
            model = model_class(config=config)
        else:
            model = model_class(n_components=n_comp)

        result = cross_validate(model, X, Y, cv)
        q2_by_n.append(result.q2_mean)
        q2_std_by_n.append(result.q2_std)

        if verbose:
            print(f"  Q² = {result.q2_mean:.4f} ± {result.q2_std:.4f}")

    # Find optimal (within 1 std of max, prefer simpler)
    q2_by_n = np.array(q2_by_n)
    q2_std_by_n = np.array(q2_std_by_n)

    max_q2 = np.max(q2_by_n)
    max_idx = np.argmax(q2_by_n)

    # One-standard-error rule: choose simplest model within 1 SE of best
    threshold = max_q2 - q2_std_by_n[max_idx]
    optimal_n = np.where(q2_by_n >= threshold)[0][0] + 1

    return {
        "optimal_n": optimal_n,
        "q2_by_n": q2_by_n.tolist(),
        "q2_std_by_n": q2_std_by_n.tolist(),
        "max_q2": float(max_q2),
        "max_n": int(max_idx + 1),
    }


def compare_vertical_resolutions_cv(
    X_standard: NDArray[np.floating],
    X_adaptive: NDArray[np.floating],
    Y: NDArray[np.floating],
    model_class,
    cv: CrossValidator | None = None,
) -> dict:
    """
    Compare Q² performance between standard and adaptive vertical resolution.

    Parameters
    ----------
    X_standard : array-like
        Features with standard 17 pressure levels.
    X_adaptive : array-like
        Features with all available levels (adaptive).
    Y : array-like
        Target matrix.
    model_class : type
        Model class to use.
    cv : CrossValidator, optional
        Cross-validation strategy.

    Returns
    -------
    comparison : dict
        Q² statistics for each configuration.
    """
    if cv is None:
        cv = KFoldCV(n_splits=5)

    # Standard resolution
    model_std = model_class()
    result_std = cross_validate(model_std, X_standard, Y, cv)

    # Adaptive resolution
    model_adp = model_class()
    result_adp = cross_validate(model_adp, X_adaptive, Y, cv)

    return {
        "standard_17_level": {
            "q2_mean": result_std.q2_mean,
            "q2_std": result_std.q2_std,
            "q2_scores": result_std.q2_scores.tolist(),
        },
        "adaptive": {
            "q2_mean": result_adp.q2_mean,
            "q2_std": result_adp.q2_std,
            "q2_scores": result_adp.q2_scores.tolist(),
        },
        "difference": result_adp.q2_mean - result_std.q2_mean,
        "better": "adaptive" if result_adp.q2_mean > result_std.q2_mean else "standard",
    }
