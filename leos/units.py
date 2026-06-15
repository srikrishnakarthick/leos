"""
leos.units
----------
Central definition of all astropy units and physical constants.
Updated for Phase 2: Atmospheric Ingestion and Terrain Analysis.
"""

from astropy import units as u
from astropy.constants import L_sun, au, k_B, sigma_sb

# -- Geometric & Terrain --
ANGLE_UNIT          = u.deg
SOLID_ANGLE_UNIT    = u.sr
DISTANCE_UNIT       = u.au
ELEVATION_UNIT      = u.m
VELOCITY_UNIT       = u.m / u.s

# -- Radiometric --
IRRADIANCE_UNIT     = u.W / u.m**2
SPECTRAL_IRRAD_UNIT = u.W / u.m**2 / u.nm
WAVELENGTH_UNIT     = u.nm

# -- Atmospheric & Thermodynamic --
TEMPERATURE_UNIT    = u.K
PRESSURE_UNIT       = u.Pa
MASS_DENSITY_UNIT   = u.kg / u.m**3
NUMBER_DENSITY_UNIT = u.m**-3
VMR_UNIT            = u.dimensionless_unscaled  # mol/mol
MMR_UNIT            = u.kg / u.kg               # mass mixing ratio

# -- Columnar Quantities --
COLUMN_MASS_UNIT    = u.kg / u.m**2
COLUMN_DENSITY_UNIT = u.m**-2                   # molecules/m2
DOBSON_UNIT = u.def_unit('DU', 2.6867e20 * u.m**-2)       # Standard Earth O3 unit

# -- Constants --
SOLAR_LUMINOSITY    = L_sun
AU                  = au
BOLTZMANN_CONSTANT  = k_B
STEFAN_BOLTZMANN    = sigma_sb

