"""
Data loaders for observational and model datasets.

Supports:
- CERES-EBAF TOA and surface radiative fluxes
- AIRS L3 atmospheric temperature and humidity profiles
- ERA5 reanalysis for atmospheric state
- Traditional radiative kernel formats (CAM5, GFDL, etc.)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False


# Standard pressure levels for atmospheric profiles
STANDARD_PRESSURE_LEVELS = np.array([
    1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10
])


@dataclass
class ObservationalData:
    """Container for observational data."""

    # Coordinates
    time: NDArray  # datetime or float (years)
    latitude: NDArray[np.floating]
    longitude: NDArray[np.floating]
    pressure_levels: NDArray[np.floating] | None = None

    # TOA fluxes (W/m²)
    olr: NDArray[np.floating] | None = None  # Outgoing LW radiation
    osr: NDArray[np.floating] | None = None  # Outgoing SW radiation (reflected)
    incoming_sw: NDArray[np.floating] | None = None  # Incoming solar

    # Surface fluxes (W/m²)
    surface_lw_up: NDArray[np.floating] | None = None
    surface_lw_down: NDArray[np.floating] | None = None
    surface_sw_up: NDArray[np.floating] | None = None
    surface_sw_down: NDArray[np.floating] | None = None

    # Atmospheric profiles
    temperature: NDArray[np.floating] | None = None  # (time, level, lat, lon)
    specific_humidity: NDArray[np.floating] | None = None
    relative_humidity: NDArray[np.floating] | None = None

    # Surface variables
    surface_temperature: NDArray[np.floating] | None = None
    surface_albedo: NDArray[np.floating] | None = None

    # Cloud properties
    cloud_fraction: NDArray[np.floating] | None = None
    cloud_top_pressure: NDArray[np.floating] | None = None

    @property
    def shape(self) -> tuple:
        """Return shape of primary data array."""
        if self.olr is not None:
            return self.olr.shape
        if self.temperature is not None:
            return self.temperature.shape
        return ()

    def to_anomalies(self, climatology: ObservationalData) -> ObservationalData:
        """Compute anomalies relative to climatology."""
        anomaly_data = ObservationalData(
            time=self.time,
            latitude=self.latitude,
            longitude=self.longitude,
            pressure_levels=self.pressure_levels,
        )

        # Compute anomalies for each variable
        for attr in ['olr', 'osr', 'temperature', 'specific_humidity',
                     'surface_temperature', 'cloud_fraction']:
            self_val = getattr(self, attr)
            clim_val = getattr(climatology, attr)
            if self_val is not None and clim_val is not None:
                setattr(anomaly_data, attr, self_val - clim_val)

        return anomaly_data


class CERESLoader:
    """
    Load CERES-EBAF (Energy Balanced and Filled) data.

    CERES provides TOA and surface radiative fluxes from satellite observations.
    EBAF products are adjusted to remove inconsistencies between instruments.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing CERES NetCDF files.
    product : str
        CERES product type: "EBAF-TOA" or "EBAF-Surface"

    References
    ----------
    Loeb et al. (2018), J. Climate, doi:10.1175/JCLI-D-17-0208.1
    """

    # Variable name mappings for different CERES versions
    VAR_NAMES = {
        "toa_lw_all_mon": ["toa_lw_all_mon", "rlut"],
        "toa_sw_all_mon": ["toa_sw_all_mon", "rsut"],
        "toa_net_all_mon": ["toa_net_all_mon", "rtmt"],
        "solar_mon": ["solar_mon", "rsdt"],
    }

    def __init__(
        self,
        data_dir: str | Path,
        product: Literal["EBAF-TOA", "EBAF-Surface"] = "EBAF-TOA",
    ):
        if not HAS_XARRAY:
            raise ImportError("xarray is required for data loading")

        self.data_dir = Path(data_dir)
        self.product = product

    def load(
        self,
        start_year: int = 2001,
        end_year: int = 2023,
        months: list[int] | None = None,
    ) -> ObservationalData:
        """
        Load CERES data for specified time period.

        Parameters
        ----------
        start_year : int
            First year to load.
        end_year : int
            Last year to load.
        months : list of int, optional
            Specific months to load (1-12). If None, loads all months.

        Returns
        -------
        data : ObservationalData
            Loaded flux data.
        """
        # Find CERES files
        pattern = f"*{self.product}*.nc"
        files = sorted(self.data_dir.glob(pattern))

        if not files:
            raise FileNotFoundError(
                f"No CERES files found matching {pattern} in {self.data_dir}"
            )

        # Load and concatenate
        ds = xr.open_mfdataset(files, combine='by_coords')

        # Select time range
        ds = ds.sel(time=slice(f"{start_year}", f"{end_year}"))

        if months is not None:
            ds = ds.sel(time=ds.time.dt.month.isin(months))

        # Extract variables
        olr = self._get_variable(ds, "toa_lw_all_mon")
        osr = self._get_variable(ds, "toa_sw_all_mon")
        incoming_sw = self._get_variable(ds, "solar_mon")

        return ObservationalData(
            time=ds.time.values,
            latitude=ds.lat.values,
            longitude=ds.lon.values,
            olr=olr,
            osr=osr,
            incoming_sw=incoming_sw,
        )

    def _get_variable(
        self,
        ds: "xr.Dataset",
        var_name: str,
    ) -> NDArray[np.floating] | None:
        """Get variable with fallback names."""
        for name in self.VAR_NAMES.get(var_name, [var_name]):
            if name in ds:
                return ds[name].values
        return None

    def compute_climatology(
        self,
        data: ObservationalData,
        method: Literal["monthly", "annual"] = "monthly",
    ) -> ObservationalData:
        """
        Compute climatology from loaded data.

        Parameters
        ----------
        data : ObservationalData
            Input data.
        method : str
            "monthly" for monthly climatology, "annual" for annual mean.

        Returns
        -------
        climatology : ObservationalData
            Climatological mean values.
        """
        # This would compute time-mean over the record
        # For monthly climatology, group by month first

        clim = ObservationalData(
            time=np.array([0]),  # Single time point for climatology
            latitude=data.latitude,
            longitude=data.longitude,
        )

        if data.olr is not None:
            clim.olr = np.nanmean(data.olr, axis=0, keepdims=True)
        if data.osr is not None:
            clim.osr = np.nanmean(data.osr, axis=0, keepdims=True)

        return clim


class AIRSLoader:
    """
    Load AIRS (Atmospheric Infrared Sounder) Level 3 data.

    AIRS provides atmospheric temperature and humidity profiles
    with ~1° horizontal resolution and ~1 km vertical resolution.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing AIRS NetCDF files.
    version : str
        AIRS product version (e.g., "v7").

    References
    ----------
    Susskind et al. (2014), J. Geophys. Res., doi:10.1002/2013JD021091
    """

    PRESSURE_LEVELS = np.array([
        1000, 925, 850, 700, 600, 500, 400, 300, 250, 200,
        150, 100, 70, 50, 30, 20, 15, 10, 7, 5, 3, 2, 1.5, 1
    ])

    def __init__(
        self,
        data_dir: str | Path,
        version: str = "v7",
    ):
        if not HAS_XARRAY:
            raise ImportError("xarray is required for data loading")

        self.data_dir = Path(data_dir)
        self.version = version

    def load(
        self,
        start_year: int = 2003,
        end_year: int = 2023,
        variables: list[str] | None = None,
    ) -> ObservationalData:
        """
        Load AIRS atmospheric profile data.

        Parameters
        ----------
        start_year : int
            First year to load.
        end_year : int
            Last year to load.
        variables : list of str, optional
            Variables to load. Default: ["Temperature", "H2O_MMR"]

        Returns
        -------
        data : ObservationalData
            Loaded atmospheric profiles.
        """
        if variables is None:
            variables = ["Temperature", "H2O_MMR", "RelHum"]

        # Find AIRS files
        pattern = "*.nc"
        files = sorted(self.data_dir.glob(pattern))

        if not files:
            raise FileNotFoundError(f"No AIRS files found in {self.data_dir}")

        ds = xr.open_mfdataset(files, combine='by_coords')
        ds = ds.sel(time=slice(f"{start_year}", f"{end_year}"))

        # Extract coordinates
        time = ds.time.values
        lat = ds.lat.values if 'lat' in ds else ds.Latitude.values
        lon = ds.lon.values if 'lon' in ds else ds.Longitude.values

        # Try to get pressure levels
        if 'lev' in ds:
            pressure = ds.lev.values
        elif 'StdPressureLev' in ds:
            pressure = ds.StdPressureLev.values
        else:
            pressure = self.PRESSURE_LEVELS

        data = ObservationalData(
            time=time,
            latitude=lat,
            longitude=lon,
            pressure_levels=pressure,
        )

        # Load variables
        if "Temperature" in variables and "Temperature" in ds:
            data.temperature = ds.Temperature.values
        if "H2O_MMR" in variables and "H2O_MMR" in ds:
            data.specific_humidity = ds.H2O_MMR.values
        if "RelHum" in variables and "RelHum" in ds:
            data.relative_humidity = ds.RelHum.values
        if "SurfAirTemp" in ds:
            data.surface_temperature = ds.SurfAirTemp.values

        return data


class ERA5Loader:
    """
    Load ERA5 reanalysis data.

    ERA5 provides comprehensive atmospheric state data useful for
    computing stability indices and filling gaps in satellite observations.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing ERA5 NetCDF files.
    """

    def __init__(self, data_dir: str | Path):
        if not HAS_XARRAY:
            raise ImportError("xarray is required for data loading")

        self.data_dir = Path(data_dir)

    def load(
        self,
        start_year: int = 2001,
        end_year: int = 2023,
        variables: list[str] | None = None,
    ) -> ObservationalData:
        """
        Load ERA5 data.

        Parameters
        ----------
        start_year : int
            First year.
        end_year : int
            Last year.
        variables : list of str, optional
            Variables to load.

        Returns
        -------
        data : ObservationalData
            ERA5 atmospheric data.
        """
        if variables is None:
            variables = ["t", "q", "sp", "skt", "tcc"]

        files = sorted(self.data_dir.glob("*.nc"))
        if not files:
            raise FileNotFoundError(f"No ERA5 files found in {self.data_dir}")

        ds = xr.open_mfdataset(files, combine='by_coords')
        ds = ds.sel(time=slice(f"{start_year}", f"{end_year}"))

        data = ObservationalData(
            time=ds.time.values,
            latitude=ds.latitude.values if 'latitude' in ds else ds.lat.values,
            longitude=ds.longitude.values if 'longitude' in ds else ds.lon.values,
        )

        # Map ERA5 variable names
        if "t" in ds:
            data.temperature = ds.t.values
            data.pressure_levels = ds.level.values if 'level' in ds else None
        if "q" in ds:
            data.specific_humidity = ds.q.values
        if "skt" in ds:
            data.surface_temperature = ds.skt.values
        if "tcc" in ds:
            data.cloud_fraction = ds.tcc.values

        return data


class TraditionalKernelLoader:
    """
    Load traditional radiative kernels for comparison.

    Supports kernel formats from:
    - CAM5 (Shell et al., 2008)
    - GFDL (Soden et al., 2008)
    - ERA-Interim based kernels

    Parameters
    ----------
    kernel_dir : str or Path
        Directory containing kernel files.
    kernel_type : str
        Kernel source: "CAM5", "GFDL", "ERA-Interim"
    """

    def __init__(
        self,
        kernel_dir: str | Path,
        kernel_type: Literal["CAM5", "GFDL", "ERA-Interim"] = "CAM5",
    ):
        if not HAS_XARRAY:
            raise ImportError("xarray is required for kernel loading")

        self.kernel_dir = Path(kernel_dir)
        self.kernel_type = kernel_type

    def load_temperature_kernel(self) -> "xr.DataArray":
        """Load temperature kernel (dR/dT at each level)."""
        if self.kernel_type == "CAM5":
            file_pattern = "*t_kernel*.nc"
        elif self.kernel_type == "GFDL":
            file_pattern = "*ta_*.nc"
        else:
            file_pattern = "*temp_kernel*.nc"

        files = list(self.kernel_dir.glob(file_pattern))
        if not files:
            raise FileNotFoundError(f"No temperature kernel found: {file_pattern}")

        ds = xr.open_dataset(files[0])

        # Find the kernel variable
        for var in ['t_kernel', 'ta_kernel', 'lwkernel', 'kernel']:
            if var in ds:
                return ds[var]

        raise ValueError("Could not identify kernel variable in file")

    def load_humidity_kernel(self) -> "xr.DataArray":
        """Load water vapor kernel (dR/dq at each level)."""
        if self.kernel_type == "CAM5":
            file_pattern = "*q_kernel*.nc"
        else:
            file_pattern = "*wv_*.nc"

        files = list(self.kernel_dir.glob(file_pattern))
        if not files:
            raise FileNotFoundError(f"No humidity kernel found: {file_pattern}")

        ds = xr.open_dataset(files[0])

        for var in ['q_kernel', 'wv_kernel', 'hus_kernel', 'kernel']:
            if var in ds:
                return ds[var]

        raise ValueError("Could not identify kernel variable")

    def load_surface_albedo_kernel(self) -> "xr.DataArray":
        """Load surface albedo kernel (dR/dα)."""
        file_pattern = "*alb*.nc"
        files = list(self.kernel_dir.glob(file_pattern))

        if not files:
            raise FileNotFoundError("No albedo kernel found")

        ds = xr.open_dataset(files[0])

        for var in ['alb_kernel', 'albedo_kernel', 'swkernel']:
            if var in ds:
                return ds[var]

        raise ValueError("Could not identify albedo kernel variable")


# Convenience functions
def load_ceres_ebaf(
    data_dir: str | Path,
    start_year: int = 2001,
    end_year: int = 2023,
) -> ObservationalData:
    """Convenience function to load CERES-EBAF data."""
    loader = CERESLoader(data_dir)
    return loader.load(start_year, end_year)


def load_airs_l3(
    data_dir: str | Path,
    start_year: int = 2003,
    end_year: int = 2023,
) -> ObservationalData:
    """Convenience function to load AIRS L3 data."""
    loader = AIRSLoader(data_dir)
    return loader.load(start_year, end_year)
