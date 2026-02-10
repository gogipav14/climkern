"""
Climate State Classification for SIMCA-style Regime-Dependent Kernels

Implements combined latitude band + cloud/stability classification
to enable state-dependent radiative kernel estimation.

Regime taxonomy (8-12 classes):
- Latitude bands: {tropical, subtropical, midlatitude, polar}
- Cloud/stability: {clear, cloudy} × {convective, stable}
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from backend import HAS_JAX, get_array_module, to_numpy


class LatitudeBand(Enum):
    """Latitude band classification."""

    TROPICAL = auto()  # |lat| < 15°
    SUBTROPICAL = auto()  # 15° <= |lat| < 35°
    MIDLATITUDE = auto()  # 35° <= |lat| < 60°
    POLAR = auto()  # |lat| >= 60°


class CloudState(Enum):
    """Cloud cover classification."""

    CLEAR = auto()  # Cloud fraction < threshold
    CLOUDY = auto()  # Cloud fraction >= threshold


class StabilityState(Enum):
    """Atmospheric stability classification."""

    CONVECTIVE = auto()  # Low stability / active convection
    STABLE = auto()  # High stability / suppressed convection


@dataclass
class ClimateState:
    """Complete climate state classification."""

    latitude_band: LatitudeBand
    cloud_state: CloudState
    stability_state: StabilityState

    @property
    def regime_id(self) -> int:
        """Unique integer ID for this regime (0-15)."""
        lat_idx = self.latitude_band.value - 1  # 0-3
        cloud_idx = 0 if self.cloud_state == CloudState.CLEAR else 1  # 0-1
        stab_idx = 0 if self.stability_state == StabilityState.CONVECTIVE else 1  # 0-1
        return lat_idx * 4 + cloud_idx * 2 + stab_idx

    @property
    def name(self) -> str:
        """Human-readable regime name."""
        return (
            f"{self.latitude_band.name.lower()}_"
            f"{self.cloud_state.name.lower()}_"
            f"{self.stability_state.name.lower()}"
        )

    @classmethod
    def from_regime_id(cls, regime_id: int) -> ClimateState:
        """Reconstruct ClimateState from regime ID."""
        lat_idx = regime_id // 4
        cloud_idx = (regime_id % 4) // 2
        stab_idx = regime_id % 2

        return cls(
            latitude_band=LatitudeBand(lat_idx + 1),
            cloud_state=CloudState.CLEAR if cloud_idx == 0 else CloudState.CLOUDY,
            stability_state=(
                StabilityState.CONVECTIVE if stab_idx == 0 else StabilityState.STABLE
            ),
        )


@dataclass
class ClassifierConfig:
    """Configuration for climate state classifier."""

    # Latitude band boundaries (degrees)
    tropical_bound: float = 15.0
    subtropical_bound: float = 35.0
    polar_bound: float = 60.0

    # Cloud fraction threshold (0-1)
    cloud_threshold: float = 0.5

    # Stability metric: Lower Tropospheric Stability (LTS) threshold (K)
    # LTS = θ_700 - θ_surface, where θ is potential temperature
    lts_threshold: float = 18.0  # K, typical boundary between regimes

    # Alternative stability: Estimated Inversion Strength (EIS) threshold
    eis_threshold: float = 5.0  # K


class ClimateStateClassifier:
    """
    Classify climate states using latitude + cloud + stability taxonomy.

    This classifier assigns each sample to one of up to 16 regimes based on:
    1. Latitude band (4 classes)
    2. Cloud state (2 classes: clear/cloudy)
    3. Stability state (2 classes: convective/stable)

    Parameters
    ----------
    config : ClassifierConfig
        Classification thresholds and boundaries.
    use_latitude : bool
        Whether to use latitude-based classification.
    use_cloud : bool
        Whether to use cloud-based classification.
    use_stability : bool
        Whether to use stability-based classification.

    Attributes
    ----------
    regime_counts_ : dict
        Count of samples in each regime after fitting.
    regime_names_ : list
        Names of all regimes with at least one sample.
    """

    def __init__(
        self,
        config: ClassifierConfig | None = None,
        use_latitude: bool = True,
        use_cloud: bool = True,
        use_stability: bool = True,
    ):
        self.config = config or ClassifierConfig()
        self.use_latitude = use_latitude
        self.use_cloud = use_cloud
        self.use_stability = use_stability

        self.regime_counts_: dict[int, int] = {}
        self.regime_names_: list[str] = []

    def classify_latitude(
        self,
        latitude: NDArray[np.floating],
    ) -> NDArray[np.int_]:
        """Classify samples by latitude band."""
        abs_lat = np.abs(latitude)
        result = np.zeros(len(latitude), dtype=np.int_)

        result[abs_lat < self.config.tropical_bound] = LatitudeBand.TROPICAL.value
        result[
            (abs_lat >= self.config.tropical_bound)
            & (abs_lat < self.config.subtropical_bound)
        ] = LatitudeBand.SUBTROPICAL.value
        result[
            (abs_lat >= self.config.subtropical_bound)
            & (abs_lat < self.config.polar_bound)
        ] = LatitudeBand.MIDLATITUDE.value
        result[abs_lat >= self.config.polar_bound] = LatitudeBand.POLAR.value

        return result

    def classify_cloud(
        self,
        cloud_fraction: NDArray[np.floating],
    ) -> NDArray[np.int_]:
        """Classify samples by cloud state."""
        return np.where(
            cloud_fraction < self.config.cloud_threshold,
            CloudState.CLEAR.value,
            CloudState.CLOUDY.value,
        ).astype(np.int_)

    def classify_stability(
        self,
        lts: NDArray[np.floating] | None = None,
        eis: NDArray[np.floating] | None = None,
    ) -> NDArray[np.int_]:
        """
        Classify samples by atmospheric stability.

        Parameters
        ----------
        lts : array-like, optional
            Lower Tropospheric Stability (θ_700 - θ_surface) in K.
        eis : array-like, optional
            Estimated Inversion Strength in K.

        At least one of lts or eis must be provided.
        """
        if lts is not None:
            stability_metric = lts
            threshold = self.config.lts_threshold
        elif eis is not None:
            stability_metric = eis
            threshold = self.config.eis_threshold
        else:
            raise ValueError("Either lts or eis must be provided")

        return np.where(
            stability_metric < threshold,
            StabilityState.CONVECTIVE.value,
            StabilityState.STABLE.value,
        ).astype(np.int_)

    def fit_predict(
        self,
        latitude: NDArray[np.floating] | None = None,
        cloud_fraction: NDArray[np.floating] | None = None,
        lts: NDArray[np.floating] | None = None,
        eis: NDArray[np.floating] | None = None,
    ) -> NDArray[np.int_]:
        """
        Classify all samples and return regime IDs.

        Parameters
        ----------
        latitude : array-like of shape (n_samples,)
            Latitude in degrees (-90 to 90).
        cloud_fraction : array-like of shape (n_samples,)
            Total cloud fraction (0 to 1).
        lts : array-like of shape (n_samples,), optional
            Lower Tropospheric Stability in K.
        eis : array-like of shape (n_samples,), optional
            Estimated Inversion Strength in K.

        Returns
        -------
        regime_ids : ndarray of shape (n_samples,)
            Integer regime ID for each sample (0-15).
        """
        # Determine number of samples from first available array
        n_samples = None
        for arr in [latitude, cloud_fraction, lts, eis]:
            if arr is not None:
                n_samples = len(arr)
                break

        if n_samples is None:
            raise ValueError("At least one classification array must be provided")

        # Initialize base regime components
        lat_class = np.ones(n_samples, dtype=np.int_)  # Default: tropical (1)
        cloud_class = np.ones(n_samples, dtype=np.int_)  # Default: clear (1)
        stab_class = np.ones(n_samples, dtype=np.int_)  # Default: convective (1)

        # Classify each dimension
        if self.use_latitude and latitude is not None:
            lat_class = self.classify_latitude(latitude)

        if self.use_cloud and cloud_fraction is not None:
            cloud_class = self.classify_cloud(cloud_fraction)

        if self.use_stability and (lts is not None or eis is not None):
            stab_class = self.classify_stability(lts=lts, eis=eis)

        # Combine into regime ID
        # lat_class is 1-4, cloud_class is 1-2, stab_class is 1-2
        regime_ids = (
            (lat_class - 1) * 4  # 0, 4, 8, 12
            + (cloud_class - 1) * 2  # 0, 2
            + (stab_class - 1)  # 0, 1
        )

        # Track regime statistics
        unique, counts = np.unique(regime_ids, return_counts=True)
        self.regime_counts_ = dict(zip(unique.tolist(), counts.tolist()))
        self.regime_names_ = [
            ClimateState.from_regime_id(rid).name for rid in unique
        ]

        return regime_ids

    def get_regime_mask(
        self,
        regime_ids: NDArray[np.int_],
        regime_id: int,
    ) -> NDArray[np.bool_]:
        """Get boolean mask for samples in a specific regime."""
        return regime_ids == regime_id

    def get_active_regimes(self) -> list[int]:
        """Return list of regime IDs that have at least one sample."""
        return list(self.regime_counts_.keys())


class SIMCAClassifier:
    """
    SIMCA-style classifier using PCA models per regime.

    Soft Independent Modeling of Class Analogy (SIMCA) builds a separate
    PCA model for each class/regime and classifies new samples based on
    their distance to each class model.

    This is useful for:
    1. Handling regime overlap (samples near boundaries)
    2. Detecting novel/outlier climate states
    3. Probabilistic regime assignment

    Parameters
    ----------
    base_classifier : ClimateStateClassifier
        Initial hard classifier for training regime assignment.
    n_components : int
        Number of PCA components per regime model.
    alpha : float
        Significance level for class membership (0.01-0.10).

    Attributes
    ----------
    regime_models_ : dict
        PCA models for each regime.
    regime_limits_ : dict
        Q-residual limits for each regime.
    """

    def __init__(
        self,
        base_classifier: ClimateStateClassifier | None = None,
        n_components: int = 3,
        alpha: float = 0.05,
    ):
        self.base_classifier = base_classifier or ClimateStateClassifier()
        self.n_components = n_components
        self.alpha = alpha

        self.regime_models_: dict[int, dict] = {}
        self.regime_limits_: dict[int, float] = {}
        self.scaler_ = StandardScaler()

    def fit(
        self,
        X: NDArray[np.floating],
        regime_ids: NDArray[np.int_],
    ) -> SIMCAClassifier:
        """
        Fit SIMCA models for each regime.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix (atmospheric profiles, etc.).
        regime_ids : array-like of shape (n_samples,)
            Regime assignment from base classifier.

        Returns
        -------
        self : SIMCAClassifier
            Fitted classifier.
        """
        X = np.asarray(X)
        regime_ids = np.asarray(regime_ids)

        # Scale features
        X_scaled = self.scaler_.fit_transform(X)

        # Build PCA model for each regime
        xp = get_array_module()
        unique_regimes = np.unique(regime_ids)

        for regime_id in unique_regimes:
            mask = regime_ids == regime_id
            X_regime = xp.asarray(X_scaled[mask])

            if len(X_regime) < self.n_components + 1:
                continue

            # Fit PCA via SVD (accelerated by JAX when available)
            mean = xp.mean(X_regime, axis=0)
            X_centered = X_regime - mean
            U, S, Vt = xp.linalg.svd(X_centered, full_matrices=False)

            n_comp = min(self.n_components, len(S))
            self.regime_models_[int(regime_id)] = {
                "mean": to_numpy(mean),
                "loadings": to_numpy(Vt[:n_comp].T),  # (n_features, n_components)
                "singular_values": to_numpy(S[:n_comp]),
                "n_samples": len(X_regime),
            }

            # Compute Q-residual limit (based on eigenvalue distribution)
            eigenvalues = to_numpy(S**2) / (len(X_regime) - 1)
            theta1 = np.sum(eigenvalues[n_comp:])
            theta2 = np.sum(eigenvalues[n_comp:] ** 2)
            theta3 = np.sum(eigenvalues[n_comp:] ** 3)

            if theta1 > 0:
                h0 = 1 - 2 * theta1 * theta3 / (3 * theta2**2)
                from scipy import stats

                c_alpha = stats.norm.ppf(1 - self.alpha)
                q_limit = theta1 * (
                    c_alpha * np.sqrt(2 * theta2 * h0**2) / theta1
                    + 1
                    + theta2 * h0 * (h0 - 1) / theta1**2
                ) ** (1 / h0)
            else:
                q_limit = np.inf

            self.regime_limits_[int(regime_id)] = q_limit

        return self

    def predict_proba(
        self,
        X: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        Compute probability of belonging to each regime.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_regimes)
            Probability for each regime (columns ordered by regime_id).
        """
        xp = get_array_module()
        X = np.asarray(X)
        X_scaled = xp.asarray(self.scaler_.transform(X))

        regime_ids = sorted(self.regime_models_.keys())
        n_samples = len(X)
        n_regimes = len(regime_ids)

        distances = xp.zeros((n_samples, n_regimes))

        for j, regime_id in enumerate(regime_ids):
            model = self.regime_models_[regime_id]
            limit = self.regime_limits_[regime_id]

            X_centered = X_scaled - xp.asarray(model["mean"])
            loadings = xp.asarray(model["loadings"])
            scores = X_centered @ loadings
            X_reconstructed = scores @ loadings.T
            residuals = X_centered - X_reconstructed

            q_residuals = xp.sum(residuals**2, axis=1)

            if HAS_JAX:
                distances = distances.at[:, j].set(
                    q_residuals / limit if limit > 0 else q_residuals
                )
            else:
                distances[:, j] = q_residuals / limit if limit > 0 else q_residuals

        inv_distances = 1.0 / (1.0 + distances)
        proba = inv_distances / inv_distances.sum(axis=1, keepdims=True)

        return to_numpy(proba)

    def predict(
        self,
        X: NDArray[np.floating],
    ) -> NDArray[np.int_]:
        """
        Predict regime ID for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        regime_ids : ndarray of shape (n_samples,)
            Predicted regime ID (highest probability).
        """
        proba = self.predict_proba(X)
        regime_ids = sorted(self.regime_models_.keys())
        best_idx = np.argmax(proba, axis=1)
        return np.array([regime_ids[i] for i in best_idx])

    def is_outlier(
        self,
        X: NDArray[np.floating],
    ) -> NDArray[np.bool_]:
        """
        Identify samples that don't fit any regime model well.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        outlier : ndarray of shape (n_samples,)
            True if sample exceeds Q-limit for all regimes.
        """
        X = np.asarray(X)
        X_scaled = self.scaler_.transform(X)

        n_samples = len(X)
        all_exceed = np.ones(n_samples, dtype=bool)

        for regime_id, model in self.regime_models_.items():
            limit = self.regime_limits_[regime_id]

            X_centered = X_scaled - model["mean"]
            scores = X_centered @ model["loadings"]
            X_reconstructed = scores @ model["loadings"].T
            residuals = X_centered - X_reconstructed
            q_residuals = np.sum(residuals**2, axis=1)

            # Sample is not outlier if it fits this regime
            all_exceed &= q_residuals > limit

        return all_exceed


class DataDrivenClassifier:
    """
    Data-driven climate state classification using k-means clustering.

    Alternative to rule-based classification when explicit thresholds
    are unknown or when the data suggests different natural groupings.

    Parameters
    ----------
    n_clusters : int
        Number of climate regimes to identify.
    features : list of str
        Which features to use for clustering.
        Options: "temperature", "humidity", "stability", "cloud", "latitude"
    random_state : int
        Random seed for reproducibility.

    Attributes
    ----------
    kmeans_ : KMeans
        Fitted k-means model.
    cluster_centers_ : ndarray
        Centroid of each cluster.
    cluster_labels_ : list of str
        Descriptive labels for each cluster.
    """

    FEATURE_SETS = {
        "temperature": ["T_surface", "T_700", "T_300"],
        "humidity": ["q_surface", "q_700", "RH_700"],
        "stability": ["LTS", "EIS", "CAPE"],
        "cloud": ["cloud_fraction", "cloud_top_pressure"],
        "latitude": ["latitude"],
    }

    def __init__(
        self,
        n_clusters: int = 8,
        features: list[str] | None = None,
        random_state: int = 42,
    ):
        self.n_clusters = n_clusters
        self.features = features or ["temperature", "stability", "cloud"]
        self.random_state = random_state

        self.kmeans_: KMeans | None = None
        self.scaler_ = StandardScaler()
        self.cluster_centers_: NDArray | None = None
        self.cluster_labels_: list[str] = []
        self.feature_names_: list[str] = []

    def fit(
        self,
        X: NDArray[np.floating],
        feature_names: list[str] | None = None,
    ) -> DataDrivenClassifier:
        """
        Fit k-means clustering on climate state features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        feature_names : list of str, optional
            Names of features in X (for interpretability).

        Returns
        -------
        self : DataDrivenClassifier
            Fitted classifier.
        """
        X = np.asarray(X)

        if feature_names is not None:
            self.feature_names_ = feature_names

        # Scale features
        X_scaled = self.scaler_.fit_transform(X)

        # Fit k-means
        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
        )
        self.kmeans_.fit(X_scaled)

        # Store cluster centers in original scale
        self.cluster_centers_ = self.scaler_.inverse_transform(
            self.kmeans_.cluster_centers_
        )

        # Generate descriptive labels
        self._generate_cluster_labels()

        return self

    def _generate_cluster_labels(self) -> None:
        """Generate human-readable labels for clusters based on centroids."""
        if self.cluster_centers_ is None:
            return

        labels = []
        for i in range(self.n_clusters):
            labels.append(f"regime_{i}")

        self.cluster_labels_ = labels

    def predict(
        self,
        X: NDArray[np.floating],
    ) -> NDArray[np.int_]:
        """
        Predict cluster assignment for samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels (0 to n_clusters-1).
        """
        if self.kmeans_ is None:
            raise ValueError("Classifier not fitted. Call fit() first.")

        X_scaled = self.scaler_.transform(X)
        return self.kmeans_.predict(X_scaled)

    def fit_predict(
        self,
        X: NDArray[np.floating],
        feature_names: list[str] | None = None,
    ) -> NDArray[np.int_]:
        """Fit classifier and return predictions."""
        self.fit(X, feature_names)
        return self.predict(X)


def compute_lts(
    T_surface: NDArray[np.floating],
    T_700: NDArray[np.floating],
    P_surface: NDArray[np.floating] | float = 1013.25,
) -> NDArray[np.floating]:
    """
    Compute Lower Tropospheric Stability (LTS).

    LTS = θ_700 - θ_surface

    where θ is potential temperature: θ = T * (1000/P)^(R/cp)

    Parameters
    ----------
    T_surface : array-like
        Surface temperature (K).
    T_700 : array-like
        Temperature at 700 hPa (K).
    P_surface : array-like or float
        Surface pressure (hPa). Default 1013.25.

    Returns
    -------
    lts : ndarray
        Lower Tropospheric Stability (K).
    """
    R_cp = 0.286  # R/cp for dry air

    theta_surface = T_surface * (1000.0 / np.asarray(P_surface)) ** R_cp
    theta_700 = T_700 * (1000.0 / 700.0) ** R_cp

    return theta_700 - theta_surface


def compute_eis(
    T_surface: NDArray[np.floating],
    T_700: NDArray[np.floating],
    z_700: NDArray[np.floating] | float = 3000.0,
    z_lcl: NDArray[np.floating] | float = 1000.0,
) -> NDArray[np.floating]:
    """
    Compute Estimated Inversion Strength (EIS).

    EIS = LTS - Γ_m * (z_700 - z_LCL)

    where Γ_m is the moist adiabatic lapse rate (~6.5 K/km).

    Parameters
    ----------
    T_surface : array-like
        Surface temperature (K).
    T_700 : array-like
        Temperature at 700 hPa (K).
    z_700 : array-like or float
        Height of 700 hPa level (m). Default 3000.
    z_lcl : array-like or float
        Lifting condensation level height (m). Default 1000.

    Returns
    -------
    eis : ndarray
        Estimated Inversion Strength (K).
    """
    lts = compute_lts(T_surface, T_700)

    # Moist adiabatic lapse rate (K/m)
    gamma_m = 6.5 / 1000.0

    eis = lts - gamma_m * (np.asarray(z_700) - np.asarray(z_lcl))

    return eis
