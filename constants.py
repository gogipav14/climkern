"""Physical constants and standard values for climate calculations."""

import numpy as np

# Fundamental physical constants
STEFAN_BOLTZMANN = 5.670374419e-8  # W m⁻² K⁻⁴
GRAVITY = 9.80665  # m s⁻²
DRY_AIR_GAS_CONSTANT = 287.058  # J kg⁻¹ K⁻¹
SPECIFIC_HEAT_DRY_AIR = 1004.0  # J kg⁻¹ K⁻¹
LATENT_HEAT_VAPORIZATION = 2.501e6  # J kg⁻¹

# Derived constants
R_CP = DRY_AIR_GAS_CONSTANT / SPECIFIC_HEAT_DRY_AIR  # ~0.286

# Standard pressure levels (hPa) - matches traditional radiative kernels
STANDARD_PRESSURE_LEVELS = np.array([
    1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10
])

# Earth parameters
EARTH_RADIUS = 6.371e6  # m
EARTH_SURFACE_AREA = 5.1e14  # m²

# Typical climate values for reference
TYPICAL_OLR = 240.0  # W/m² (global mean outgoing longwave radiation)
TYPICAL_SURFACE_TEMP = 288.0  # K (global mean surface temperature)
