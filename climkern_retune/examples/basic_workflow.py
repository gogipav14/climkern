#!/usr/bin/env python
"""
Basic Workflow: Training and Evaluating Tunable Radiative Kernels

This example demonstrates:
1. Creating synthetic training data (in place of CERES+AIRS)
2. Training a single-regime tunable kernel
3. Training a multi-state kernel with regime classification
4. Comparing vertical resolution approaches via Q²
5. Validating against traditional kernel format

For real usage, replace synthetic data with actual CERES/AIRS observations.
"""

import numpy as np

# Note: This example uses synthetic data that doesn't require open_nipals
# For full functionality, install: pip install git+https://github.com/gogipav14/open_nipals.git


def generate_synthetic_climate_data(
    n_samples: int = 1000,
    n_pressure_levels: int = 17,
    random_state: int = 42,
) -> dict:
    """
    Generate synthetic climate data for demonstration.

    In real usage, this would be replaced with:
    - CERES-EBAF TOA fluxes
    - AIRS L3 temperature/humidity profiles
    - ERA5 reanalysis data
    """
    rng = np.random.default_rng(random_state)

    # Pressure levels (hPa)
    pressure_levels = np.array([
        1000, 925, 850, 700, 600, 500, 400, 300, 250, 200,
        150, 100, 70, 50, 30, 20, 10
    ])[:n_pressure_levels]

    # Coordinates
    latitude = rng.uniform(-90, 90, n_samples)
    longitude = rng.uniform(-180, 180, n_samples)

    # Temperature anomalies (K) - stronger near surface
    temp_anomalies = np.zeros((n_samples, n_pressure_levels))
    base_anomaly = rng.normal(0, 1, n_samples)
    for i, p in enumerate(pressure_levels):
        # Lapse rate effect: upper levels warm less
        scale = 1.0 - 0.3 * (1000 - p) / 1000
        temp_anomalies[:, i] = base_anomaly * scale + rng.normal(0, 0.2, n_samples)

    # Humidity anomalies (relative change)
    # Water vapor increases ~7%/K (Clausius-Clapeyron)
    humidity_anomalies = np.zeros((n_samples, n_pressure_levels))
    for i, p in enumerate(pressure_levels):
        if p > 200:  # Only below 200 hPa
            humidity_anomalies[:, i] = 0.07 * temp_anomalies[:, i] + rng.normal(0, 0.01, n_samples)

    # Surface temperature anomaly
    surface_temp_anomaly = base_anomaly + rng.normal(0, 0.3, n_samples)

    # Cloud fraction anomaly (-0.1 to 0.1)
    cloud_anomaly = rng.normal(0, 0.03, n_samples)

    # Surface albedo anomaly (small)
    albedo_anomaly = rng.normal(0, 0.005, n_samples)

    # Stability (LTS)
    T_surface = 288 + surface_temp_anomaly
    T_700_idx = np.argmin(np.abs(pressure_levels - 700))
    T_700 = 270 + temp_anomalies[:, T_700_idx]
    lts = (T_700 * (1000/700)**0.286) - (T_surface * (1000/1013.25)**0.286)

    # Cloud fraction (for classification)
    cloud_fraction = np.clip(0.5 + cloud_anomaly + 0.1 * rng.randn(n_samples), 0, 1)

    # Radiative response (simplified kernel model)
    # OLR change ≈ Planck + water vapor + lapse rate
    planck_response = -3.2 * surface_temp_anomaly  # W/m²/K
    wv_response = 1.8 * surface_temp_anomaly  # Coupled to T
    lapse_rate_response = -0.5 * (temp_anomalies[:, 0] - temp_anomalies[:, -1])
    albedo_response = -50 * albedo_anomaly * np.cos(np.radians(latitude))  # SW

    delta_olr = planck_response + wv_response + lapse_rate_response + rng.normal(0, 0.5, n_samples)
    delta_osr = albedo_response + rng.normal(0, 0.3, n_samples)

    return {
        "pressure_levels": pressure_levels,
        "latitude": latitude,
        "longitude": longitude,
        "temp_anomalies": temp_anomalies,
        "humidity_anomalies": humidity_anomalies,
        "surface_temp_anomaly": surface_temp_anomaly,
        "cloud_anomaly": cloud_anomaly,
        "albedo_anomaly": albedo_anomaly,
        "lts": lts,
        "cloud_fraction": cloud_fraction,
        "delta_olr": delta_olr,
        "delta_osr": delta_osr,
    }


def build_feature_matrix(data: dict) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build X and Y matrices from synthetic data."""
    n_samples = len(data["latitude"])
    n_levels = len(data["pressure_levels"])

    # Features: T at each level, q at each level, surface T, cloud, albedo
    features = []
    names = []

    # Temperature at each level
    for i, p in enumerate(data["pressure_levels"]):
        features.append(data["temp_anomalies"][:, i])
        names.append(f"T_{int(p)}")

    # Humidity at each level
    for i, p in enumerate(data["pressure_levels"]):
        features.append(data["humidity_anomalies"][:, i])
        names.append(f"q_{int(p)}")

    # Surface and cloud
    features.append(data["surface_temp_anomaly"])
    names.append("T_surface")

    features.append(data["cloud_anomaly"])
    names.append("cloud_fraction")

    features.append(data["albedo_anomaly"])
    names.append("albedo")

    X = np.column_stack(features)
    Y = np.column_stack([data["delta_olr"], data["delta_osr"]])

    return X, Y, names


def example_single_kernel():
    """Example 1: Train a single-regime tunable kernel."""
    print("\n" + "="*60)
    print("Example 1: Single-Regime Tunable Kernel")
    print("="*60)

    # Generate data
    data = generate_synthetic_climate_data(n_samples=500)
    X, Y, feature_names = build_feature_matrix(data)

    print(f"Data shape: X={X.shape}, Y={Y.shape}")
    print(f"Features: {len(feature_names)}")

    # Split train/test
    n_train = int(0.8 * len(X))
    X_train, X_test = X[:n_train], X[n_train:]
    Y_train, Y_test = Y[:n_train], Y[n_train:]

    try:
        from climkern_retune import TunableKernel, KernelConfig

        # Create and fit kernel
        config = KernelConfig(n_components=5)
        kernel = TunableKernel(config=config)
        kernel.fit(X_train, Y_train, feature_names=feature_names)

        # Evaluate
        q2 = kernel.evaluate(X_test, Y_test)
        print(f"\nTest Q² score: {q2:.4f}")

        # Get kernel sensitivities
        sensitivities = kernel.get_kernel_sensitivities()
        print("\nTop 5 kernel sensitivities (LW):")
        sorted_sens = sorted(
            [(k, v[0] if len(v) > 0 else v) for k, v in sensitivities.items()],
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]
        for name, sens in sorted_sens:
            print(f"  {name}: {sens:.4f} W/m²/unit")

    except ImportError as e:
        print(f"\nNote: Full kernel training requires open_nipals.")
        print(f"Install via: pip install git+https://github.com/gogipav14/open_nipals.git")
        print(f"Error: {e}")


def example_multistate_kernel():
    """Example 2: Train a multi-state kernel with regime classification."""
    print("\n" + "="*60)
    print("Example 2: Multi-State Tunable Kernel")
    print("="*60)

    # Generate data
    data = generate_synthetic_climate_data(n_samples=1000)
    X, Y, feature_names = build_feature_matrix(data)

    # State classification variables
    latitude = data["latitude"]
    cloud_fraction = data["cloud_fraction"]
    lts = data["lts"]

    print(f"Data shape: X={X.shape}, Y={Y.shape}")

    try:
        from climkern_retune import (
            MultiStateKernel,
            ClimateStateClassifier,
            KernelConfig,
        )

        # Create classifier
        classifier = ClimateStateClassifier()
        regime_ids = classifier.fit_predict(
            latitude=latitude,
            cloud_fraction=cloud_fraction,
            lts=lts,
        )

        print(f"\nRegime distribution:")
        for regime_id, count in classifier.regime_counts_.items():
            from climkern_retune import ClimateState
            state = ClimateState.from_regime_id(regime_id)
            print(f"  {state.name}: {count} samples")

        # Split train/test
        n_train = int(0.8 * len(X))

        # Create and fit multi-state kernel
        config = KernelConfig(n_components=3)
        kernel = MultiStateKernel(config=config, classifier=classifier)
        kernel.fit(
            X[:n_train], Y[:n_train],
            latitude=latitude[:n_train],
            cloud_fraction=cloud_fraction[:n_train],
            lts=lts[:n_train],
            feature_names=feature_names,
        )

        # Evaluate
        metrics = kernel.evaluate(
            X[n_train:], Y[n_train:],
            latitude=latitude[n_train:],
            cloud_fraction=cloud_fraction[n_train:],
            lts=lts[n_train:],
        )

        print(f"\nMulti-state kernel evaluation:")
        print(f"  Overall Q²: {metrics['overall_q2']:.4f}")

        for key, value in metrics.items():
            if key.startswith("regime_"):
                print(f"  {key}: {value:.4f}")

    except ImportError as e:
        print(f"\nNote: Full kernel training requires open_nipals.")
        print(f"Error: {e}")


def example_cross_validation():
    """Example 3: Component selection via cross-validation."""
    print("\n" + "="*60)
    print("Example 3: Cross-Validation for Component Selection")
    print("="*60)

    # Generate data
    data = generate_synthetic_climate_data(n_samples=300)
    X, Y, _ = build_feature_matrix(data)

    print(f"Data shape: X={X.shape}, Y={Y.shape}")

    from climkern_retune.validation.cross_validation import (
        KFoldCV,
        compute_q2_score,
    )

    # Demonstrate K-fold CV
    cv = KFoldCV(n_splits=5, shuffle=True, random_state=42)

    print("\nK-Fold CV splits:")
    for i, (train_idx, test_idx) in enumerate(cv.split(X, Y)):
        print(f"  Fold {i+1}: train={len(train_idx)}, test={len(test_idx)}")

    # Demonstrate Q² computation
    Y_pred_perfect = Y.copy()
    Y_pred_noise = Y + np.random.randn(*Y.shape)
    Y_pred_mean = np.full_like(Y, Y.mean(axis=0))

    print("\nQ² score examples:")
    print(f"  Perfect predictions: Q² = {compute_q2_score(Y, Y_pred_perfect):.4f}")
    print(f"  Noisy predictions: Q² = {compute_q2_score(Y, Y_pred_noise):.4f}")
    print(f"  Mean predictions: Q² = {compute_q2_score(Y, Y_pred_mean):.4f}")


def example_state_classification():
    """Example 4: Climate state classification demo."""
    print("\n" + "="*60)
    print("Example 4: Climate State Classification")
    print("="*60)

    from climkern_retune.core.state_classifier import (
        ClimateStateClassifier,
        ClimateState,
        compute_lts,
    )

    # Generate varied climate states
    n_samples = 200
    rng = np.random.default_rng(42)

    # Mix of different regimes
    latitude = np.concatenate([
        rng.uniform(-10, 10, 50),   # Tropical
        rng.uniform(20, 35, 50),    # Subtropical
        rng.uniform(40, 60, 50),    # Midlatitude
        rng.uniform(70, 90, 50),    # Polar
    ])

    cloud_fraction = rng.uniform(0, 1, n_samples)
    lts = rng.uniform(5, 30, n_samples)

    # Classify
    classifier = ClimateStateClassifier()
    regime_ids = classifier.fit_predict(
        latitude=latitude,
        cloud_fraction=cloud_fraction,
        lts=lts,
    )

    print("Regime classification results:")
    print("-" * 40)

    for regime_id in classifier.get_active_regimes():
        state = ClimateState.from_regime_id(regime_id)
        count = classifier.regime_counts_[regime_id]
        print(f"{state.name:40s}: {count:4d} samples")

    print(f"\nTotal regimes active: {len(classifier.get_active_regimes())}")
    print(f"Maximum possible: 16 (4 lat × 2 cloud × 2 stability)")


def main():
    """Run all examples."""
    print("ClimKern-Retune: NIPALS-PLS Tunable Radiative Kernels")
    print("="*60)
    print("This demo uses synthetic data for illustration.")
    print("For real applications, use CERES/AIRS observations.")

    # Run examples
    example_state_classification()
    example_cross_validation()
    example_single_kernel()
    example_multistate_kernel()

    print("\n" + "="*60)
    print("Examples completed!")
    print("="*60)


if __name__ == "__main__":
    main()
