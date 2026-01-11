"""
Tests for core ClimKern-Retune functionality.

Tests cover:
- State classification
- Constraint functions
- Cross-validation utilities
"""

import numpy as np
import pytest

from climkern_retune.core.state_classifier import (
    ClimateStateClassifier,
    ClassifierConfig,
    ClimateState,
    LatitudeBand,
    CloudState,
    StabilityState,
    compute_lts,
    compute_eis,
)
from climkern_retune.core.nipals_pls import (
    stefan_boltzmann_constraint,
    energy_conservation_constraint,
    STEFAN_BOLTZMANN,
)
from climkern_retune.validation.cross_validation import (
    KFoldCV,
    TimeSeriesCV,
    compute_q2_score,
)


class TestClimateStateClassifier:
    """Tests for climate state classification."""

    def test_latitude_classification(self):
        """Test latitude band classification."""
        classifier = ClimateStateClassifier()

        # Test different latitude bands
        lats = np.array([0, 10, 20, 40, 70, -30, -80])
        expected_bands = [
            LatitudeBand.TROPICAL,  # 0°
            LatitudeBand.TROPICAL,  # 10°
            LatitudeBand.SUBTROPICAL,  # 20°
            LatitudeBand.MIDLATITUDE,  # 40°
            LatitudeBand.POLAR,  # 70°
            LatitudeBand.SUBTROPICAL,  # -30°
            LatitudeBand.POLAR,  # -80°
        ]

        lat_class = classifier.classify_latitude(lats)

        for i, expected in enumerate(expected_bands):
            assert lat_class[i] == expected.value, f"Failed for latitude {lats[i]}"

    def test_cloud_classification(self):
        """Test cloud fraction classification."""
        classifier = ClimateStateClassifier()

        cloud_fractions = np.array([0.0, 0.3, 0.5, 0.7, 1.0])

        cloud_class = classifier.classify_cloud(cloud_fractions)

        assert cloud_class[0] == CloudState.CLEAR.value
        assert cloud_class[1] == CloudState.CLEAR.value
        assert cloud_class[2] == CloudState.CLOUDY.value
        assert cloud_class[3] == CloudState.CLOUDY.value
        assert cloud_class[4] == CloudState.CLOUDY.value

    def test_stability_classification(self):
        """Test atmospheric stability classification."""
        classifier = ClimateStateClassifier()
        config = classifier.config

        # LTS values below and above threshold
        lts = np.array([10.0, 15.0, 20.0, 25.0])

        stab_class = classifier.classify_stability(lts=lts)

        # Default threshold is 18 K
        assert stab_class[0] == StabilityState.CONVECTIVE.value
        assert stab_class[1] == StabilityState.CONVECTIVE.value
        assert stab_class[2] == StabilityState.STABLE.value
        assert stab_class[3] == StabilityState.STABLE.value

    def test_combined_classification(self):
        """Test combined regime classification."""
        classifier = ClimateStateClassifier()

        n_samples = 100
        rng = np.random.default_rng(42)

        latitude = rng.uniform(-90, 90, n_samples)
        cloud_fraction = rng.uniform(0, 1, n_samples)
        lts = rng.uniform(5, 30, n_samples)

        regime_ids = classifier.fit_predict(
            latitude=latitude,
            cloud_fraction=cloud_fraction,
            lts=lts,
        )

        # Check output shape
        assert len(regime_ids) == n_samples

        # Check valid range (0-15 for 4×2×2 regimes)
        assert regime_ids.min() >= 0
        assert regime_ids.max() <= 15

    def test_climate_state_from_regime_id(self):
        """Test reconstructing ClimateState from regime ID."""
        for regime_id in range(16):
            state = ClimateState.from_regime_id(regime_id)
            reconstructed_id = state.regime_id

            assert reconstructed_id == regime_id, f"Failed for regime_id {regime_id}"

    def test_compute_lts(self):
        """Test Lower Tropospheric Stability calculation."""
        T_surface = np.array([290.0, 280.0, 300.0])
        T_700 = np.array([270.0, 260.0, 275.0])

        lts = compute_lts(T_surface, T_700)

        # LTS should be positive when atmosphere is stable
        # (700 hPa θ > surface θ)
        assert len(lts) == 3
        assert all(np.isfinite(lts))

    def test_compute_eis(self):
        """Test Estimated Inversion Strength calculation."""
        T_surface = np.array([290.0, 280.0])
        T_700 = np.array([270.0, 260.0])

        eis = compute_eis(T_surface, T_700)

        assert len(eis) == 2
        assert all(np.isfinite(eis))


class TestConstraintFunctions:
    """Tests for physical constraint functions."""

    def test_stefan_boltzmann_constraint(self):
        """Test Stefan-Boltzmann surface constraint."""
        n_samples = 10
        T_surface = np.full(n_samples, 288.0)  # K
        delta_T = np.linspace(-2, 2, n_samples)  # K change

        # Expected flux change: 4 ε σ T³ ΔT
        expected_delta_F = 4 * 1.0 * STEFAN_BOLTZMANN * 288.0**3 * delta_T

        # Create Y_pred that matches expected (zero residual)
        Y_pred = expected_delta_F.reshape(-1, 1)

        # Create X with delta_T in first column
        X = delta_T.reshape(-1, 1)

        params = {
            "surface_temp": T_surface,
            "emissivity": 1.0,
            "delta_temp_idx": 0,
            "surface_flux_idx": 0,
        }

        residual = stefan_boltzmann_constraint(Y_pred, X, params)

        # Residual should be near zero
        assert np.allclose(residual, 0, atol=1e-10)

    def test_stefan_boltzmann_nonzero_residual(self):
        """Test S-B constraint with non-matching prediction."""
        n_samples = 5
        T_surface = np.full(n_samples, 288.0)
        delta_T = np.ones(n_samples)

        # Wrong prediction
        Y_pred = np.zeros((n_samples, 1))
        X = delta_T.reshape(-1, 1)

        params = {
            "surface_temp": T_surface,
            "emissivity": 1.0,
            "delta_temp_idx": 0,
        }

        residual = stefan_boltzmann_constraint(Y_pred, X, params)

        # Residual should be non-zero (the expected S-B flux change)
        expected = -4 * STEFAN_BOLTZMANN * 288.0**3 * 1.0
        assert np.allclose(residual.flatten(), expected, rtol=1e-6)

    def test_energy_conservation_constraint(self):
        """Test energy conservation constraint."""
        n_samples = 100
        rng = np.random.default_rng(42)

        # Predictions that average to expected imbalance
        Y_pred = rng.normal(0.5, 0.1, (n_samples, 1))
        X = rng.randn(n_samples, 5)
        area_weights = np.ones(n_samples) / n_samples

        params = {
            "area_weights": area_weights,
            "expected_imbalance": np.mean(Y_pred),
        }

        residual = energy_conservation_constraint(Y_pred, X, params)

        # Residual should be near zero when prediction matches expected
        assert np.abs(residual.mean()) < 0.01


class TestCrossValidation:
    """Tests for cross-validation utilities."""

    def test_kfold_cv_splits(self):
        """Test K-fold CV generates correct number of splits."""
        cv = KFoldCV(n_splits=5, shuffle=True, random_state=42)

        X = np.random.randn(100, 10)
        Y = np.random.randn(100, 2)

        splits = list(cv.split(X, Y))

        assert len(splits) == 5

        # Check all samples are used
        all_test = np.concatenate([test for _, test in splits])
        assert len(np.unique(all_test)) == 100

    def test_kfold_cv_no_overlap(self):
        """Test K-fold CV test sets don't overlap."""
        cv = KFoldCV(n_splits=5)

        X = np.random.randn(50, 5)
        splits = list(cv.split(X))

        for i, (_, test_i) in enumerate(splits):
            for j, (_, test_j) in enumerate(splits):
                if i != j:
                    overlap = set(test_i) & set(test_j)
                    assert len(overlap) == 0

    def test_timeseries_cv_ordering(self):
        """Test time-series CV respects temporal ordering."""
        cv = TimeSeriesCV(n_splits=3, test_size=10, expanding=True)

        X = np.arange(60).reshape(-1, 1)
        splits = list(cv.split(X))

        for train_idx, test_idx in splits:
            # Test indices should come after train indices
            assert train_idx.max() < test_idx.min()

    def test_compute_q2_score(self):
        """Test Q² score computation."""
        # Perfect predictions
        Y_true = np.array([1, 2, 3, 4, 5])
        Y_pred = np.array([1, 2, 3, 4, 5])

        q2 = compute_q2_score(Y_true, Y_pred)
        assert np.isclose(q2, 1.0)

        # Zero predictions (predict mean)
        Y_pred_mean = np.full_like(Y_true, float(Y_true.mean()))
        q2_zero = compute_q2_score(Y_true, Y_pred_mean)
        assert np.isclose(q2_zero, 0.0, atol=1e-10)

    def test_q2_with_train_mean(self):
        """Test Q² score with separate training mean."""
        Y_true = np.array([10, 11, 12])
        Y_pred = np.array([10, 11, 12])
        Y_train_mean = np.array([5.0])  # Different from test mean

        q2 = compute_q2_score(Y_true, Y_pred, Y_train_mean)

        # Should still be 1.0 for perfect predictions
        assert np.isclose(q2, 1.0)


class TestDataPreprocessing:
    """Tests for data preprocessing utilities."""

    def test_compute_anomalies_annual(self):
        """Test annual anomaly computation."""
        from climkern_retune.data.preprocessors import compute_anomalies

        # Create synthetic data with trend
        n_months = 120  # 10 years
        time = np.arange(n_months)
        data = np.sin(2 * np.pi * time / 12) + 0.01 * time  # Seasonal + trend

        anomalies = compute_anomalies(data, method="annual")

        # Anomalies should have zero mean
        assert np.abs(anomalies.mean()) < 0.1

    def test_compute_anomalies_monthly(self):
        """Test monthly climatology anomaly computation."""
        from climkern_retune.data.preprocessors import compute_anomalies

        # Create data with strong seasonal cycle
        n_months = 48  # 4 years
        time = np.arange(n_months)
        seasonal = 10 * np.sin(2 * np.pi * time / 12)
        noise = np.random.randn(n_months)
        data = seasonal + noise

        anomalies = compute_anomalies(data, method="monthly")

        # Should remove seasonal cycle
        seasonal_amplitude = np.std(anomalies)
        assert seasonal_amplitude < 5  # Much reduced

    def test_create_feature_matrix(self):
        """Test feature matrix creation."""
        from climkern_retune.data.preprocessors import create_feature_matrix

        n_samples = 50
        n_levels = 5

        temperature = np.random.randn(n_samples, n_levels)
        humidity = np.random.randn(n_samples, n_levels)
        surface_temp = np.random.randn(n_samples)

        X, names = create_feature_matrix(
            temperature=temperature,
            humidity=humidity,
            surface_temp=surface_temp,
        )

        assert X.shape == (n_samples, n_levels * 2 + 1)
        assert len(names) == n_levels * 2 + 1
        assert "T_surface" in names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
