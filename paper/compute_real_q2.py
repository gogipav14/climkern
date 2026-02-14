"""
Compute real Q² values for all Table 4 rows using CERES+NCEP observational data.

Pipeline:
1. Load merged CERES+NCEP dataset (2003-2020, 36×36 grid)
2. Compute anomalies from 2003-2017 climatology
3. For each of 11 kernels: load kernel, regrid to data grid, compute ΔR_k
4. Build X matrix (n_samples, 11) of kernel predictions
5. Use CERES ΔOLR anomaly as target Y
6. Run KernelRegimeHarmonizer: global PLS, retune A/B/C/D
7. Report Q² for all methods

Also computes Step 2 (data-driven) Q² from atmospheric features.
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import xarray as xr


# ============================================================
# Kernel loading and prediction
# ============================================================

KERNEL_NAMES = [
    "BMRC", "CAM3", "CAM5", "CERES", "CloudSat",
    "ECHAM6", "ECMWF-RRTM", "ERA5", "GFDL", "HadGEM2", "HadGEM3-GA7.1",
]


def _extract_kernel_arrays(kernel_name: str, kernel_dir: Path) -> dict:
    """
    Load a kernel NetCDF and extract raw arrays with coordinate metadata.

    Bypasses xarray coordinate manipulation by extracting to numpy immediately.
    Returns dict with 'lw_t', 'lw_ts', 'lw_q' arrays and 'lat', 'lon', 'plev' coords,
    all normalized to consistent shapes: (12, n_plev, n_lat, n_lon) for 3D fields,
    (12, n_lat, n_lon) for surface fields.
    """
    path = kernel_dir / kernel_name / f"TOA_{kernel_name}_Kerns.nc"
    ds = xr.open_dataset(path, decode_times=False)

    # Identify dimension names (varies across kernels)
    dim_names = list(ds.dims)
    lat_dim = next((d for d in dim_names if d in ("lat", "latitude")), None)
    lon_dim = next((d for d in dim_names if d in ("lon", "longitude")), None)
    time_dim = next((d for d in dim_names if d in ("time", "month")), None)
    plev_dim = next((d for d in dim_names if d in ("plev", "lev", "player", "level")), None)

    # Extract coordinate values
    if lat_dim in ds.coords:
        k_lat = ds.coords[lat_dim].values.astype(float)
    elif lat_dim in ds.data_vars:
        k_lat = ds[lat_dim].values.astype(float)
    else:
        # Try common data var names
        for candidate in ["latitude", "lat"]:
            if candidate in ds.data_vars:
                k_lat = ds[candidate].values.astype(float)
                break

    if lon_dim in ds.coords:
        k_lon = ds.coords[lon_dim].values.astype(float)
    elif lon_dim in ds.data_vars:
        k_lon = ds[lon_dim].values.astype(float)
    else:
        for candidate in ["longitude", "lon"]:
            if candidate in ds.data_vars:
                k_lon = ds[candidate].values.astype(float)
                break

    k_plev = ds.coords[plev_dim].values.astype(float) if plev_dim in ds.coords else ds[plev_dim].values.astype(float)

    # Extract kernel arrays as numpy
    lw_t_raw = ds["lw_t"].values.astype(np.float64)
    lw_ts_raw = ds["lw_ts"].values.astype(np.float64)
    lw_q_raw = ds["lw_q"].values.astype(np.float64)

    # Determine dim order from the data variable's dims
    lw_t_dims = list(ds["lw_t"].dims)
    lw_ts_dims = list(ds["lw_ts"].dims)

    ds.close()

    # Transpose to standard order: (time, plev, lat, lon) for 3D, (time, lat, lon) for 2D
    def _get_axis_order(dims, target_dims):
        """Map dim names to target order indices."""
        dim_map = {}
        for d in dims:
            if d in ("time", "month"):
                dim_map[d] = "time"
            elif d in ("plev", "lev", "player", "level"):
                dim_map[d] = "plev"
            elif d in ("lat", "latitude"):
                dim_map[d] = "lat"
            elif d in ("lon", "longitude"):
                dim_map[d] = "lon"
        return [dims.index(next(k for k, v in dim_map.items() if v == t)) for t in target_dims]

    # 3D fields: (time, plev, lat, lon)
    order_3d = _get_axis_order(lw_t_dims, ["time", "plev", "lat", "lon"])
    lw_t = np.transpose(lw_t_raw, order_3d)   # (12, n_plev, n_lat, n_lon)
    lw_q = np.transpose(lw_q_raw, order_3d)

    # 2D field: (time, lat, lon)
    order_2d = _get_axis_order(lw_ts_dims, ["time", "lat", "lon"])
    lw_ts = np.transpose(lw_ts_raw, order_2d)  # (12, n_lat, n_lon)

    # Ensure lat is ascending
    if len(k_lat) > 1 and k_lat[0] > k_lat[-1]:
        k_lat = k_lat[::-1]
        lw_t = lw_t[:, :, ::-1, :]
        lw_ts = lw_ts[:, ::-1, :]
        lw_q = lw_q[:, :, ::-1, :]

    # Ensure lon is in [0, 360) range
    if np.any(k_lon < 0):
        k_lon = np.mod(k_lon + 360, 360)
        sort_idx = np.argsort(k_lon)
        k_lon = k_lon[sort_idx]
        lw_t = lw_t[:, :, :, sort_idx]
        lw_ts = lw_ts[:, :, sort_idx]
        lw_q = lw_q[:, :, :, sort_idx]

    return {
        "lw_t": lw_t,      # (12, n_plev, n_lat, n_lon)
        "lw_ts": lw_ts,    # (12, n_lat, n_lon)
        "lw_q": lw_q,      # (12, n_plev, n_lat, n_lon)
        "lat": k_lat,
        "lon": k_lon,
        "plev": k_plev,
    }


def _regrid_nearest(
    src_arr: np.ndarray,
    src_lat: np.ndarray,
    src_lon: np.ndarray,
    dst_lat: np.ndarray,
    dst_lon: np.ndarray,
) -> np.ndarray:
    """
    Regrid a 2D array using nearest-neighbor lookup.

    src_arr: (n_src_lat, n_src_lon)
    Returns: (n_dst_lat, n_dst_lon)
    """
    lat_idx = np.array([np.argmin(np.abs(src_lat - dl)) for dl in dst_lat])
    lon_idx = np.array([np.argmin(np.abs(src_lon - dl)) for dl in dst_lon])
    return src_arr[np.ix_(lat_idx, lon_idx)]


def compute_kernel_prediction(
    kernel_data: dict,
    dT: np.ndarray,          # (n_time, n_levels, n_lat, n_lon) temperature anomaly [K]
    dTs: np.ndarray,         # (n_time, n_lat, n_lon) surface temp anomaly [K]
    dq: np.ndarray,          # (n_time, n_q_levels, n_lat, n_lon) humidity anomaly [kg/kg]
    q_clim: np.ndarray,      # (n_q_levels, n_lat, n_lon) climatological humidity [kg/kg]
    data_levels: np.ndarray,  # pressure levels of data [hPa]
    q_level_idx: np.ndarray,  # indices of levels with humidity data
    data_lat: np.ndarray,     # data lat coordinates
    data_lon: np.ndarray,     # data lon coordinates
    month_idx: np.ndarray,    # month index (0-11) for each timestep
) -> np.ndarray:
    """
    Compute ΔR prediction from one kernel applied to observed anomalies.

    Returns (n_time, n_lat, n_lon) array of predicted ΔR.
    """
    n_time, n_levels, n_lat, n_lon = dT.shape
    delta_R = np.zeros((n_time, n_lat, n_lon))

    lw_t = kernel_data["lw_t"]     # (12, n_plev, n_lat_k, n_lon_k)
    lw_ts = kernel_data["lw_ts"]   # (12, n_lat_k, n_lon_k)
    lw_q = kernel_data["lw_q"]     # (12, n_plev, n_lat_k, n_lon_k)
    k_lat = kernel_data["lat"]
    k_lon = kernel_data["lon"]
    k_plev = kernel_data["plev"]

    # Find nearest kernel pressure level index for each data level
    data_to_kplev_idx = {}
    for i, p in enumerate(data_levels):
        data_to_kplev_idx[i] = int(np.argmin(np.abs(k_plev - p)))

    q_to_kplev_idx = {}
    for i in q_level_idx:
        q_to_kplev_idx[int(i)] = int(np.argmin(np.abs(k_plev - data_levels[i])))

    # Pre-regrid all monthly kernels to data grid
    # Temperature kernels: (12, n_levels, n_lat, n_lon)
    K_T_regrid = np.zeros((12, n_levels, n_lat, n_lon))
    for m in range(12):
        for data_lev_idx in range(n_levels):
            kp_idx = data_to_kplev_idx[data_lev_idx]
            K_T_regrid[m, data_lev_idx] = _regrid_nearest(
                lw_t[m, kp_idx], k_lat, k_lon, data_lat, data_lon,
            )

    # Surface temperature kernel: (12, n_lat, n_lon)
    K_Ts_regrid = np.zeros((12, n_lat, n_lon))
    for m in range(12):
        K_Ts_regrid[m] = _regrid_nearest(
            lw_ts[m], k_lat, k_lon, data_lat, data_lon,
        )

    # Humidity kernels: (12, n_q_levels, n_lat, n_lon)
    n_q = len(q_level_idx)
    K_q_regrid = np.zeros((12, n_q, n_lat, n_lon))
    for m in range(12):
        for qi, data_lev_idx in enumerate(q_level_idx):
            kp_idx = q_to_kplev_idx[int(data_lev_idx)]
            K_q_regrid[m, qi] = _regrid_nearest(
                lw_q[m, kp_idx], k_lat, k_lon, data_lat, data_lon,
            )

    # Replace NaNs with 0 in regridded kernels
    K_T_regrid = np.nan_to_num(K_T_regrid, nan=0.0)
    K_Ts_regrid = np.nan_to_num(K_Ts_regrid, nan=0.0)
    K_q_regrid = np.nan_to_num(K_q_regrid, nan=0.0)

    # Compute ΔR for each timestep (vectorized per month)
    for m in range(12):
        t_mask = month_idx == m
        if not np.any(t_mask):
            continue

        # Temperature contribution: Σ_p K_T(p) * ΔT(p) for all timesteps in month
        # dT[t_mask] shape: (n_t, n_levels, n_lat, n_lon)
        # K_T_regrid[m] shape: (n_levels, n_lat, n_lon)
        delta_R[t_mask] += np.sum(
            K_T_regrid[m][None, :, :, :] * dT[t_mask], axis=1,
        )

        # Surface temperature contribution
        delta_R[t_mask] += K_Ts_regrid[m][None, :, :] * dTs[t_mask]

        # Humidity contribution: Σ_p K_q(p) * Δq(p) / q_clim(p)
        safe_q = np.where(np.abs(q_clim) > 1e-10, q_clim, 1e-10)
        dq_frac = np.nan_to_num(dq[t_mask] / safe_q[None, :, :, :], nan=0.0)
        delta_R[t_mask] += np.sum(
            K_q_regrid[m][None, :, :, :] * dq_frac, axis=1,
        )

    return delta_R


# ============================================================
# Main computation
# ============================================================


def compute_all_kernel_predictions(
    data_path: Path,
    kernel_dir: Path,
) -> dict:
    """
    Compute 11-kernel ΔR predictions from CERES+NCEP observational data.

    Returns dict with X_total, Y, kernel_names, and metadata.
    """
    print("=" * 60)
    print("Computing Real Kernel Predictions")
    print("=" * 60)

    # 1. Load merged dataset
    print(f"\nLoading {data_path}...")
    ds = xr.open_dataset(data_path)

    T_profile = ds["temperature"].values + 273.15  # °C → K
    q_profile = ds["specific_humidity"].values      # kg/kg
    Ts = ds["surface_temperature"].values + 273.15  # °C → K
    olr = ds["olr"].values                          # W/m²
    osr = ds["osr"].values                          # W/m²
    lat = ds["lat"].values
    lon = ds["lon"].values
    levels = ds["level"].values
    times = ds["time"].values

    n_time, n_lat, n_lon = olr.shape
    n_levels = len(levels)

    # Month index for each timestep (0-11)
    import pandas as pd
    time_pd = pd.DatetimeIndex(times)
    month_idx = (time_pd.month - 1).values  # 0-indexed

    print(f"  Grid: {n_time} months × {n_lat} lat × {n_lon} lon")
    print(f"  Levels: {n_levels} pressure levels")
    print(f"  Period: {time_pd[0].strftime('%Y-%m')} to {time_pd[-1].strftime('%Y-%m')}")

    # 2. Compute climatology (2003-2017 = first 180 months)
    n_clim = 180
    print(f"\n  Computing climatology from first {n_clim} months...")

    T_clim = np.nanmean(T_profile[:n_clim], axis=0)   # (17, 36, 36)
    q_clim = np.nanmean(q_profile[:n_clim], axis=0)    # (17, 36, 36)
    Ts_clim = np.nanmean(Ts[:n_clim], axis=0)          # (36, 36)
    olr_clim = np.nanmean(olr[:n_clim], axis=0)        # (36, 36)
    osr_clim = np.nanmean(osr[:n_clim], axis=0)        # (36, 36)

    # Compute anomalies
    dT = T_profile - T_clim[None, :, :, :]
    dq = q_profile - q_clim[None, :, :, :]
    dTs = Ts - Ts_clim[None, :, :]
    d_olr = olr - olr_clim[None, :, :]
    d_osr = osr - osr_clim[None, :, :]

    # Humidity: identify levels with valid data
    q_available = np.array([
        not np.all(np.isnan(q_profile[0, i, :, :]))
        for i in range(n_levels)
    ])
    q_level_idx = np.where(q_available)[0]
    dq_valid = dq[:, q_level_idx, :, :]        # (n_time, n_q_levels, n_lat, n_lon)
    q_clim_valid = q_clim[q_level_idx, :, :]   # (n_q_levels, n_lat, n_lon)

    print(f"  Humidity data at {len(q_level_idx)} levels: {levels[q_level_idx]}")
    print(f"  ΔT range: [{np.nanmin(dT):.1f}, {np.nanmax(dT):.1f}] K")
    print(f"  ΔOLR range: [{np.nanmin(d_olr):.1f}, {np.nanmax(d_olr):.1f}] W/m²")

    # 3. Compute each kernel's prediction
    print(f"\nComputing 11-kernel predictions...")
    X_predictions = {}
    successful_kernels = []

    for k_name in KERNEL_NAMES:
        print(f"  Processing {k_name}...", end=" ", flush=True)
        t0 = time.time()

        try:
            kernel_data = _extract_kernel_arrays(k_name, kernel_dir)

            delta_R = compute_kernel_prediction(
                kernel_data=kernel_data,
                dT=dT,
                dTs=dTs,
                dq=dq_valid,
                q_clim=q_clim_valid,
                data_levels=levels,
                q_level_idx=q_level_idx,
                data_lat=lat,
                data_lon=lon,
                month_idx=month_idx,
            )

            X_predictions[k_name] = delta_R
            successful_kernels.append(k_name)
            elapsed = time.time() - t0
            print(f"done ({elapsed:.1f}s)")

        except Exception as e:
            elapsed = time.time() - t0
            print(f"FAILED ({elapsed:.1f}s): {e}")
            import traceback
            traceback.print_exc()

    print(f"\n  Successfully computed: {len(successful_kernels)}/{len(KERNEL_NAMES)} kernels")

    ds.close()

    # 4. Build X matrix
    # Flatten (time, lat, lon) → n_samples
    n_samples = n_time * n_lat * n_lon
    n_kernels = len(successful_kernels)

    X_total = np.zeros((n_samples, n_kernels))
    for i, k_name in enumerate(successful_kernels):
        X_total[:, i] = X_predictions[k_name].reshape(-1)

    Y_olr = d_olr.reshape(-1, 1)
    Y_osr = d_osr.reshape(-1, 1)
    Y = np.hstack([Y_olr, Y_osr])

    # Latitude array for each sample
    lat_grid = np.broadcast_to(lat[None, :, None], (n_time, n_lat, n_lon)).reshape(-1)

    # Surface temperature for each sample
    Ts_flat = Ts.reshape(-1)

    # Remove NaN rows
    valid = (
        ~np.any(np.isnan(X_total), axis=1)
        & ~np.any(np.isnan(Y), axis=1)
    )
    X_total = X_total[valid]
    Y = Y[valid]
    lat_grid = lat_grid[valid]
    Ts_flat = Ts_flat[valid]

    print(f"\n  Final matrix: X={X_total.shape}, Y={Y.shape}")
    print(f"  Valid samples: {valid.sum()} / {n_samples}")

    return {
        "X_total": X_total,
        "Y": Y,
        "kernel_names": successful_kernels,
        "latitude": lat_grid,
        "surface_temp": Ts_flat,
        "n_time": n_time,
        "n_lat": n_lat,
        "n_lon": n_lon,
    }


def compute_real_q2(data: dict) -> dict:
    """
    Run KernelRegimeHarmonizer on real data and compute Q² for all methods.
    """
    from kernel_harmonizer import KernelRegimeHarmonizer

    print("\n" + "=" * 60)
    print("Computing Real Q² Values")
    print("=" * 60)

    X = data["X_total"]
    kernel_names = data["kernel_names"]
    n = len(X)

    # Use LW (ΔOLR) as target
    Y = data["Y"][:, 0:1]

    # Train/test split: temporal (first 80% train, last 20% test)
    n_train = int(0.8 * n)
    X_train, X_test = X[:n_train], X[n_train:]
    Y_train, Y_test = Y[:n_train], Y[n_train:]

    print(f"\n  Training: {n_train} samples")
    print(f"  Testing:  {n - n_train} samples")
    print(f"  Kernels:  {len(kernel_names)}")

    # Fit two-stage pipeline
    print(f"\n  Fitting KernelRegimeHarmonizer...")
    harmonizer = KernelRegimeHarmonizer(
        n_components=3, min_samples_per_regime=30,
    )
    t0 = time.time()
    harmonizer.fit_all(X_train, Y_train, kernel_names)
    fit_time = time.time() - t0
    print(f"  Fit time: {fit_time:.2f}s")

    # Generate full Q² dashboard
    print(f"\n  Generating Q² dashboard...")
    dashboard = harmonizer.q2_dashboard(X_test, Y_test)
    dashboard.print_report()

    # Individual kernel Q² (real data)
    q2_per_kernel = harmonizer.evaluate_per_kernel(X_test, Y_test)

    # Best individual kernel
    best_kernel = max(q2_per_kernel, key=q2_per_kernel.get)
    best_kernel_q2 = q2_per_kernel[best_kernel]

    # Simple mean
    q2_simple_mean = harmonizer.evaluate_simple_mean(X_test, Y_test)

    # All methods
    q2_global = harmonizer.evaluate(X_test, Y_test, method="global_pls")
    q2_A = harmonizer.evaluate(X_test, Y_test, method="retune_A")
    q2_B = harmonizer.evaluate(X_test, Y_test, method="retune_B")
    q2_C = harmonizer.evaluate(X_test, Y_test, method="retune_C")
    q2_D = harmonizer.evaluate(X_test, Y_test, method="retune_D")

    retune_q2s = {"A": q2_A, "B": q2_B, "C": q2_C, "D": q2_D}
    best_retune = max(retune_q2s, key=retune_q2s.get)
    best_retune_q2 = retune_q2s[best_retune]

    # Spread reduction
    for method in ["global_pls", "retune_A", "retune_B", "retune_C", "retune_D"]:
        rb, ra, rp = harmonizer.compute_spread_reduction(X_test, Y_test, method=method)

    # Thermodynamic compliance
    thermo = harmonizer.thermodynamic_compliance(
        X_test, Y_test, surface_temp=data["surface_temp"][n_train:]
    )

    results = {
        "q2_per_kernel": q2_per_kernel,
        "best_kernel": best_kernel,
        "best_kernel_q2": best_kernel_q2,
        "q2_simple_mean": q2_simple_mean,
        "q2_global_pls": q2_global,
        "q2_retune_A": q2_A,
        "q2_retune_B": q2_B,
        "q2_retune_C": q2_C,
        "q2_retune_D": q2_D,
        "best_retune": best_retune,
        "best_retune_q2": best_retune_q2,
        "thermodynamic": thermo,
        "dashboard": dashboard,
    }

    # Print summary for paper
    print("\n" + "=" * 60)
    print("TABLE 4 VALUES (Real Q²)")
    print("=" * 60)

    print(f"\n  Individual kernel Q² values:")
    for name, q2 in sorted(q2_per_kernel.items(), key=lambda x: -x[1]):
        print(f"    {name:<20} {q2:>+8.4f}")

    print(f"\n  Best individual kernel: {best_kernel} (Q² = {best_kernel_q2:.3f})")
    print(f"  Simple mean of {len(kernel_names)}:     Q² = {q2_simple_mean:.3f}")
    print(f"  Step 1 Stage 1 (PLS):   Q² = {q2_global:.3f}")
    print(f"  Step 1 Stage 2 best:    Q² = {best_retune_q2:.3f} (retune {best_retune})")
    print(f"    Retune A:             Q² = {q2_A:.3f}")
    print(f"    Retune B:             Q² = {q2_B:.3f}")
    print(f"    Retune C:             Q² = {q2_C:.3f}")
    print(f"    Retune D:             Q² = {q2_D:.3f}")

    return results


def compute_step2_real_q2(data_dir: Path) -> float | None:
    """Compute Step 2 (data-driven) Q² on real data using atmospheric features."""
    sys.path.insert(0, str(ROOT))
    from validate_real_data import load_real_data
    from tunable_kernel import TunableKernel, KernelConfig

    real_data = load_real_data(data_dir)
    if real_data is None:
        return None

    X, Y = real_data["X"], real_data["Y"]
    n = len(X)
    n_train = int(0.8 * n)

    X_train, X_test = X[:n_train], X[n_train:]
    Y_train, Y_test = Y[:n_train], Y[n_train:]

    # Mean-center
    X_mean = X_train.mean(axis=0)
    Y_mean = Y_train.mean(axis=0)
    X_train_c = X_train - X_mean
    X_test_c = X_test - X_mean

    config = KernelConfig(
        n_components=10,
        use_surface_constraint=False,
        use_toa_constraint=False,
        use_conservation_constraint=False,
    )
    kernel = TunableKernel(config=config)
    kernel.fit(X_train_c, Y_train - Y_mean, feature_names=real_data["feature_names"])

    Y_pred = kernel.compute(X_test_c)
    Y_pred_arr = np.column_stack([Y_pred.delta_r_lw, Y_pred.delta_r_sw])
    ss_res = np.sum((Y_test - Y_mean - Y_pred_arr) ** 2)
    ss_tot = np.sum((Y_test - Y_test.mean(axis=0)) ** 2)
    q2 = 1.0 - ss_res / ss_tot

    return float(q2)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    data_path = ROOT / "data" / "merged_CERES_NCEP_2003-2020.nc"
    kernel_dir = ROOT / "climkern" / "data" / "data" / "kernels"

    if not data_path.exists():
        print(f"ERROR: Merged dataset not found at {data_path}")
        sys.exit(1)

    if not kernel_dir.exists():
        print(f"ERROR: Kernel directory not found at {kernel_dir}")
        sys.exit(1)

    # Step 1: Kernel Harmonization (real Q²)
    data = compute_all_kernel_predictions(data_path, kernel_dir)
    step1_results = compute_real_q2(data)

    # Step 2: Data-driven kernels (real Q²)
    print("\n" + "=" * 60)
    print("Step 2: Data-Driven Kernel (Real Q²)")
    print("=" * 60)

    step2_q2 = compute_step2_real_q2(ROOT / "data")
    if step2_q2 is not None:
        print(f"  Step 2 real Q² = {step2_q2:.3f}")

    # Final summary for paper
    print("\n" + "=" * 60)
    print("FINAL TABLE 4 VALUES")
    print("=" * 60)
    print(f"  Best individual kernel:  Real Q² = {step1_results['best_kernel_q2']:.3f}")
    print(f"  Simple mean of 11:       Real Q² = {step1_results['q2_simple_mean']:.3f}")
    print(f"  Step 1 Stage 1 (PLS):    Real Q² = {step1_results['q2_global_pls']:.3f}")
    print(f"  Step 1 Stage 2 best:     Real Q² = {step1_results['best_retune_q2']:.3f}")
    if step2_q2 is not None:
        print(f"  Step 2 Data-driven:      Real Q² = {step2_q2:.3f}")

    # Retune comparison (Table 1)
    print(f"\n  TABLE 1 VALUES (Retune Comparison):")
    print(f"    Global PLS:    Q² = {step1_results['q2_global_pls']:.3f}")
    print(f"    Retune A:      Q² = {step1_results['q2_retune_A']:.3f}")
    print(f"    Retune B:      Q² = {step1_results['q2_retune_B']:.3f}")
    print(f"    Retune C:      Q² = {step1_results['q2_retune_C']:.3f}")
    print(f"    Retune D:      Q² = {step1_results['q2_retune_D']:.3f}")

    for method in ["global_pls", "retune_A", "retune_B", "retune_C", "retune_D"]:
        sb = step1_results["thermodynamic"][method]["stefan_boltzmann"]
        en = step1_results["thermodynamic"][method]["energy_conservation"]
        print(f"    {method}: SB={sb:.4f}, Energy={en:.4f}")
