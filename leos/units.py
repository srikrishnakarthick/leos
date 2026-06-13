"""
leos.units
----------
Central definition of all astropy units and physical constants used
across the LEOS library. Every module imports from here — never
define units inline elsewhere.
"""

from astropy import units as u
from astropy.constants import L_sun, au

# ── Geometric ────────────────────────────────────────────────────────────────
ANGLE_UNIT          = u.deg
SOLID_ANGLE_UNIT    = u.sr
DISTANCE_UNIT       = u.au
ELEVATION_UNIT      = u.m

# ── Radiometric ──────────────────────────────────────────────────────────────
IRRADIANCE_UNIT     = u.W / u.m**2           # broadband irradiance
SPECTRAL_IRRAD_UNIT = u.W / u.m**2 / u.nm   # per-wavelength irradiance
WAVELENGTH_UNIT     = u.nm
FLUX_UNIT           = u.W / u.m**2

# ── Power / Energy ───────────────────────────────────────────────────────────
POWER_UNIT          = u.W
ENERGY_UNIT         = u.W * u.h                   # watt-hours, useful for mission budgets

# ── Uncertainty ──────────────────────────────────────────────────────────────
# All uncertain quantities carry sigma in the same unit as the value.
# Use relative_error() to get dimensionless fractional uncertainty.

# ── Physical constants (re-exported for convenience) ─────────────────────────
SOLAR_LUMINOSITY    = L_sun                  # 3.828e26 W
AU                  = au                     # 1.496e11 m
