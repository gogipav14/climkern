"""
Data preprocessing for tunable kernel training.

Converts raw observational data into feature matrices suitable for PLS regression.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from climkern_retune.data.loaders import ObservationalData, STANDARD_PRESSURE_LEVELS


@dataclass
class FeatureMatrix:
    """Feature matrix for PLS regression."""

    X: NDArray[np.floating]  # (n_samples, n_features)
    Y: NDArray[np.floating]  # (n_samples, n_targets)
    feature_names: list[str]
    target_names: list[str]

    # Metadata
    latitude: NDArray[np.floating] | None = None
    longitude: NDArray[np.floating] | None = None
    time: NDArray | None = None
    pressure_levels: NDArray[np.floating] | None = None

    @property
    def n_samples(self) -> int:
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    @property
    def n_targets(self) -> int:
        return self.Y.shape[1] if self.Y.ndim > 1 else 1

    def train_test_split(
        self,
        test_fraction: float = 0.2,
        random_state: int = 42,
    ) -> tuple[FeatureMatrix, FeatureMatrix]:
        """Split into training and test sets."""
        rng = np.random.default_rng(random_state)
        n = self.n_samples
        indices = rng.permutation(n)

        n_test = int(n * test_fraction)
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

        train = FeatureMatrix(
            X=self.X[train_idx],
            Y=self.Y[train_idx],
            feature_names=self.feature_names,
            target_names=self.target_names,
            latitude=self.latitude[train_idx] if self.latitude is not None else None,
            longitude=self.longitude[train_idx] if self.longitude is not None else None,
            time=self.time[train_idx] if self.time is not None else None,
            pressure_levels=self.pressure_levels,
        )

        test = FeatureMatrix(
            X=self.X[test_idx],
            Y=self.Y[test_idx],
            feature_names=self.feature_names,
            target_names=self.target_names,
            latitude=self.latitude[test_idx] if self.latitude is not None else None,
            longitude=self.longitude[test_idx] if self.longitude is not None else None,
            time=self.time[test_idx] if self.time is not None else None,
            pressure_levels=self.pressure_levels,
        )

        return train, test


class DataPreprocessor:
    """
    Preprocess observational data for kernel training.

    Handles:
    - Computing anomalies from climatology
    - Interpolating to standard pressure levels
    - Flattening gridded data to sample vectors
    - Feature normalization

    Parameters
    ----------
    pressure_levels : array-like, optional
        Target pressure levels for interpolation. Default: standard 17 levels.
    normalize : bool
        Whether to normalize features to zero mean, unit variance.
    """

    def __init__(
        self,
        pressure_levels: NDArray[np.floating] | None = None,
        normalize: bool = True,
    ):
        self.pressure_levels = (
            pressure_levels if pressure_levels is not None
            else STANDARD_PRESSURE_LEVELS
        )
        self.normalize = normalize

        # Normalization parameters (computed during fit)
        self._feature_mean: NDArray | None = None
        self._feature_std: NDArray | None = None
        self._target_mean: NDArray | None = None
        self._target_std: NDArray | None = None

    def create_training_data(
        self,
        flux_data: ObservationalData,
        profile_data: ObservationalData,
        flux_climatology: ObservationalData | None = None,
        profile_climatology: ObservationalData | None = None,
    ) -> FeatureMatrix:
        """
        Create feature matrix from flux and profile data.

        Parameters
        ----------
        flux_data : ObservationalData
            CERES-like flux data (OLR, OSR, etc.)
        profile_data : ObservationalData
            AIRS-like profile data (T, q at levels)
        flux_climatology : ObservationalData, optional
            Climatology for computing flux anomalies.
        profile_climatology : ObservationalData, optional
            Climatology for computing profile anomalies.

        Returns
        -------
        data : FeatureMatrix
            Training data ready for PLS.
        """
        # Compute anomalies if climatology provided
        if flux_climatology is not None:
            flux_anomalies = flux_data.to_anomalies(flux_climatology)
        else:
            flux_anomalies = flux_data

        if profile_climatology is not None:
            profile_anomalies = profile_data.to_anomalies(profile_climatology)
        else:
            profile_anomalies = profile_data

        # Build feature names
        feature_names = []
        feature_list = []

        # Temperature at each level
        if profile_anomalies.temperature is not None:
            T = profile_anomalies.temperature
            levels = profile_anomalies.pressure_levels
            if levels is None:
                levels = self.pressure_levels

            for i, p in enumerate(levels):
                feature_names.append(f"T_{int(p)}")
                if T.ndim == 4:  # (time, level, lat, lon)
                    feature_list.append(T[:, i, :, :])
                else:
                    feature_list.append(T[:, i] if T.ndim == 2 else T)

        # Specific humidity at each level
        if profile_anomalies.specific_humidity is not None:
            q = profile_anomalies.specific_humidity
            levels = profile_anomalies.pressure_levels
            if levels is None:
                levels = self.pressure_levels

            for i, p in enumerate(levels):
                feature_names.append(f"q_{int(p)}")
                if q.ndim == 4:
                    feature_list.append(q[:, i, :, :])
                else:
                    feature_list.append(q[:, i] if q.ndim == 2 else q)

        # Surface temperature
        if profile_anomalies.surface_temperature is not None:
            feature_names.append("T_surface")
            feature_list.append(profile_anomalies.surface_temperature)

        # Cloud fraction
        if profile_anomalies.cloud_fraction is not None:
            feature_names.append("cloud_fraction")
            feature_list.append(profile_anomalies.cloud_fraction)

        # Surface albedo
        if profile_anomalies.surface_albedo is not None:
            feature_names.append("albedo")
            feature_list.append(profile_anomalies.surface_albedo)

        # Build target (Y) matrix
        target_names = []
        target_list = []

        if flux_anomalies.olr is not None:
            target_names.append("delta_OLR")
            target_list.append(flux_anomalies.olr)

        if flux_anomalies.osr is not None:
            target_names.append("delta_OSR")
            target_list.append(flux_anomalies.osr)

        # Flatten spatial dimensions
        X, lat, lon, time = self._flatten_features(feature_list, flux_data)
        Y, _, _, _ = self._flatten_features(target_list, flux_data)

        # Normalize if requested
        if self.normalize:
            X, Y = self._normalize(X, Y, fit=True)

        return FeatureMatrix(
            X=X,
            Y=Y,
            feature_names=feature_names,
            target_names=target_names,
            latitude=lat,
            longitude=lon,
            time=time,
            pressure_levels=self.pressure_levels,
        )

    def _flatten_features(
        self,
        feature_list: list[NDArray],
        reference_data: ObservationalData,
    ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """Flatten gridded features to (n_samples, n_features)."""
        if not feature_list:
            return np.array([]), np.array([]), np.array([]), np.array([])

        # Assume all features have same shape
        first = feature_list[0]

        if first.ndim == 3:  # (time, lat, lon)
            n_time, n_lat, n_lon = first.shape
            n_samples = n_time * n_lat * n_lon

            # Flatten each feature
            X = np.column_stack([f.reshape(-1) for f in feature_list])

            # Create coordinate arrays
            time_idx = np.repeat(np.arange(n_time), n_lat * n_lon)
            lat_idx = np.tile(np.repeat(np.arange(n_lat), n_lon), n_time)
            lon_idx = np.tile(np.arange(n_lon), n_time * n_lat)

            lat = reference_data.latitude[lat_idx]
            lon = reference_data.longitude[lon_idx]
            time = reference_data.time[time_idx]

        elif first.ndim == 2:  # (time, space) or (lat, lon)
            X = np.column_stack([f.reshape(-1) for f in feature_list])
            n_samples = X.shape[0]
            lat = np.zeros(n_samples)
            lon = np.zeros(n_samples)
            time = np.arange(n_samples)

        else:  # 1D
            X = np.column_stack(feature_list)
            n_samples = X.shape[0]
            lat = np.zeros(n_samples)
            lon = np.zeros(n_samples)
            time = np.arange(n_samples)

        return X, lat, lon, time

    def _normalize(
        self,
        X: NDArray[np.floating],
        Y: NDArray[np.floating],
        fit: bool = True,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Normalize features and targets."""
        if fit:
            self._feature_mean = np.nanmean(X, axis=0)
            self._feature_std = np.nanstd(X, axis=0)
            self._feature_std[self._feature_std < 1e-10] = 1.0

            self._target_mean = np.nanmean(Y, axis=0)
            self._target_std = np.nanstd(Y, axis=0)
            self._target_std[self._target_std < 1e-10] = 1.0

        X_norm = (X - self._feature_mean) / self._feature_std
        Y_norm = (Y - self._target_mean) / self._target_std

        return X_norm, Y_norm

    def transform(
        self,
        X: NDArray[np.floating],
        Y: NDArray[np.floating] | None = None,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating] | None]:
        """Apply fitted normalization to new data."""
        if self._feature_mean is None:
            raise ValueError("Preprocessor not fitted. Call create_training_data first.")

        X_norm = (X - self._feature_mean) / self._feature_std

        if Y is not None:
            Y_norm = (Y - self._target_mean) / self._target_std
            return X_norm, Y_norm

        return X_norm, None

    def inverse_transform_targets(
        self,
        Y_norm: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Convert normalized targets back to original scale."""
        if self._target_mean is None:
            return Y_norm

        return Y_norm * self._target_std + self._target_mean


def compute_anomalies(
    data: NDArray[np.floating],
    time_axis: int = 0,
    method: Literal["monthly", "annual", "rolling"] = "monthly",
    window: int = 12,
) -> NDArray[np.floating]:
    """
    Compute anomalies from time series.

    Parameters
    ----------
    data : array-like
        Input data with time as first dimension.
    time_axis : int
        Axis corresponding to time.
    method : str
        Anomaly method:
        - "monthly": subtract monthly climatology
        - "annual": subtract annual mean
        - "rolling": subtract rolling mean
    window : int
        Window size for rolling mean (months).

    Returns
    -------
    anomalies : ndarray
        Anomaly time series.
    """
    data = np.asarray(data)

    if method == "annual":
        climatology = np.nanmean(data, axis=time_axis, keepdims=True)
        return data - climatology

    elif method == "monthly":
        # Group by month and compute mean
        n_time = data.shape[time_axis]
        n_years = n_time // 12

        if n_years < 1:
            return compute_anomalies(data, time_axis, method="annual")

        # Reshape to (years, months, ...)
        shape = list(data.shape)
        new_shape = [n_years, 12] + shape[1:]
        data_reshaped = data[: n_years * 12].reshape(new_shape)

        # Monthly climatology
        monthly_clim = np.nanmean(data_reshaped, axis=0)  # (12, ...)

        # Tile climatology to match original data
        n_tiles = (n_time + 11) // 12
        climatology = np.tile(monthly_clim, (n_tiles,) + (1,) * (data.ndim - 1))
        climatology = climatology[:n_time]

        return data - climatology

    elif method == "rolling":
        # Simple rolling mean
        from scipy.ndimage import uniform_filter1d
        climatology = uniform_filter1d(data, size=window, axis=time_axis, mode='nearest')
        return data - climatology

    else:
        raise ValueError(f"Unknown method: {method}")


def compute_interannual_variability(
    data: NDArray[np.floating],
    time_axis: int = 0,
) -> dict[str, float]:
    """
    Compute interannual variability statistics.

    Parameters
    ----------
    data : array-like
        Time series data.
    time_axis : int
        Time axis.

    Returns
    -------
    stats : dict
        Statistics including std, range, trend.
    """
    data = np.asarray(data)

    # Compute annual means
    n_time = data.shape[time_axis]
    n_years = n_time // 12

    if n_years < 2:
        return {
            "std": float(np.nanstd(data)),
            "range": float(np.nanmax(data) - np.nanmin(data)),
            "trend": 0.0,
        }

    # Annual means
    shape = list(data.shape)
    new_shape = [n_years, 12] + shape[1:]
    annual_means = np.nanmean(
        data[: n_years * 12].reshape(new_shape),
        axis=1
    )

    # Statistics
    std = float(np.nanstd(annual_means))
    data_range = float(np.nanmax(annual_means) - np.nanmin(annual_means))

    # Linear trend
    years = np.arange(n_years)
    flat_means = np.nanmean(annual_means, axis=tuple(range(1, annual_means.ndim)))
    valid = ~np.isnan(flat_means)
    if valid.sum() > 1:
        trend = float(np.polyfit(years[valid], flat_means[valid], 1)[0])
    else:
        trend = 0.0

    return {
        "std": std,
        "range": data_range,
        "trend": trend,  # per year
    }


def create_feature_matrix(
    temperature: NDArray[np.floating] | None = None,
    humidity: NDArray[np.floating] | None = None,
    surface_temp: NDArray[np.floating] | None = None,
    cloud_fraction: NDArray[np.floating] | None = None,
    albedo: NDArray[np.floating] | None = None,
    pressure_levels: NDArray[np.floating] | None = None,
) -> tuple[NDArray[np.floating], list[str]]:
    """
    Create feature matrix from individual variables.

    Convenience function for building X matrix from separate arrays.

    Parameters
    ----------
    temperature : array-like, optional
        Temperature profiles (n_samples, n_levels).
    humidity : array-like, optional
        Humidity profiles (n_samples, n_levels).
    surface_temp : array-like, optional
        Surface temperature (n_samples,).
    cloud_fraction : array-like, optional
        Cloud fraction (n_samples,).
    albedo : array-like, optional
        Surface albedo (n_samples,).
    pressure_levels : array-like, optional
        Pressure levels for T and q.

    Returns
    -------
    X : ndarray
        Feature matrix (n_samples, n_features).
    feature_names : list of str
        Names for each feature column.
    """
    features = []
    names = []

    if pressure_levels is None:
        pressure_levels = STANDARD_PRESSURE_LEVELS

    if temperature is not None:
        temperature = np.asarray(temperature)
        if temperature.ndim == 1:
            features.append(temperature.reshape(-1, 1))
            names.append("T")
        else:
            n_levels = temperature.shape[1]
            for i in range(n_levels):
                features.append(temperature[:, i : i + 1])
                p = pressure_levels[i] if i < len(pressure_levels) else i
                names.append(f"T_{int(p)}")

    if humidity is not None:
        humidity = np.asarray(humidity)
        if humidity.ndim == 1:
            features.append(humidity.reshape(-1, 1))
            names.append("q")
        else:
            n_levels = humidity.shape[1]
            for i in range(n_levels):
                features.append(humidity[:, i : i + 1])
                p = pressure_levels[i] if i < len(pressure_levels) else i
                names.append(f"q_{int(p)}")

    if surface_temp is not None:
        surface_temp = np.asarray(surface_temp)
        features.append(surface_temp.reshape(-1, 1))
        names.append("T_surface")

    if cloud_fraction is not None:
        cloud_fraction = np.asarray(cloud_fraction)
        features.append(cloud_fraction.reshape(-1, 1))
        names.append("cloud_fraction")

    if albedo is not None:
        albedo = np.asarray(albedo)
        features.append(albedo.reshape(-1, 1))
        names.append("albedo")

    if not features:
        return np.array([]).reshape(0, 0), []

    X = np.hstack(features)
    return X, names


def extract_state_variables(
    data: ObservationalData,
) -> dict[str, NDArray[np.floating]]:
    """
    Extract state variables needed for classification.

    Parameters
    ----------
    data : ObservationalData
        Loaded observational data.

    Returns
    -------
    state_vars : dict
        Dictionary with latitude, cloud_fraction, lts, etc.
    """
    state_vars = {}

    if data.latitude is not None:
        # Expand latitude to match data shape
        if data.olr is not None:
            n_time = data.olr.shape[0]
            n_lat = len(data.latitude)
            n_lon = data.olr.shape[2] if data.olr.ndim > 2 else 1

            # Create latitude grid matching flattened data
            lat_grid = np.broadcast_to(
                data.latitude[np.newaxis, :, np.newaxis],
                (n_time, n_lat, n_lon)
            )
            state_vars["latitude"] = lat_grid.flatten()
        else:
            state_vars["latitude"] = data.latitude

    if data.cloud_fraction is not None:
        cf = data.cloud_fraction
        state_vars["cloud_fraction"] = cf.flatten() if cf.ndim > 1 else cf

    # Compute LTS if temperature profile available
    if data.temperature is not None and data.surface_temperature is not None:
        from climkern_retune.core.state_classifier import compute_lts

        T = data.temperature
        T_sfc = data.surface_temperature

        # Find 700 hPa level
        if data.pressure_levels is not None:
            idx_700 = np.argmin(np.abs(data.pressure_levels - 700))
            if T.ndim == 4:  # (time, level, lat, lon)
                T_700 = T[:, idx_700, :, :]
            else:
                T_700 = T[:, idx_700] if T.ndim == 2 else T

            lts = compute_lts(T_sfc.flatten(), T_700.flatten())
            state_vars["lts"] = lts

    return state_vars
