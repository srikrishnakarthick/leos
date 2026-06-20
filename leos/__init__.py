"""
LEOS: Light Environment Observatory for the Solar System
A multi-language scientific framework for high-fidelity solar illumination, 
spectral irradiance, and energy-availability computations across every body 
in the Solar System.

Master initialization file exposing core geometry, units, multi-planetary 
atmospheres, spectral sources, and advanced data ingestion engines.
"""

# 1. Structural Uncertainty Wrappers
from .uncertainty import UncertainQuantity, propagate

# 2. Central Units and Physical Constants
from .units import (
    ANGLE_UNIT, SOLID_ANGLE_UNIT, DISTANCE_UNIT, ELEVATION_UNIT, VELOCITY_UNIT,
    IRRADIANCE_UNIT, SPECTRAL_IRRAD_UNIT, WAVELENGTH_UNIT,
    TEMPERATURE_UNIT, PRESSURE_UNIT, MASS_DENSITY_UNIT, NUMBER_DENSITY_UNIT,
    VMR_UNIT, MMR_UNIT, COLUMN_MASS_UNIT, COLUMN_DENSITY_UNIT, DOBSON_UNIT,
    SOLAR_LUMINOSITY, AU, BOLTZMANN_CONSTANT, STEFAN_BOLTZMANN
)

# 3. Coordinates and Astro-Geometry
from .Coordinates.geometry import Geometry, SUPPORTED_BODIES
from .Coordinates.spice_utils import (
    discover_kernels, load_kernels, unload_kernels, kernel_sandbox,
    utc_to_et, et_to_utc, body_name_to_id, body_radii, sun_position,
    # ── High-Fidelity Geometry Engine Extensions ──
    get_sub_point, get_sub_solar_point, angular_separation,
    get_state_vector, transform_position, transform_state,
    get_spacecraft_attitude, get_surface_illumination,
    get_fov_intercept, check_occultation
)
from .Coordinates.solar_geometry import (
    sun_body_distance, solar_zenith_angle, illumination_fraction, subsolar_point, toa_irradiance
)

# 4. Atmosphere Submodule Suite (Multi-Planet Column Managers)
from .Atmosphere.atmosphere import AtmosphericProfile
from .Atmosphere.atmosphere_earth import AtmosphericColumn, AtmosphericSource, ATMO_REGISTRY
from .Atmosphere.atmosphere_mars import AtmosphericColumnMars, MarsAtmosphericSource, MARS_ATMO_REGISTRY
from .Atmosphere.atmosphere_moon import MoonSurfaceConditions, MoonAtmosphericSource, MOON_ATMO_REGISTRY

# 5. Kernels Submodule Suite
# FIX: Removed DATA_DIRS and KERNEL_ROOT imports
from .kernels.fetch_kernels import (
    fetch_remote_md5s, calculate_local_md5, get_dynamic_ephemeris_urls, fetch_kernels
)

# 6. Spectrum Submodule Suite
from .Spectrum.spectrum import Spectrum
from .Spectrum.spectral_sources import (
    SpectralSource, SpectralSourceInfo, REGISTRY, get_info, sources_valid_at, best_source_for_time
)
from .Spectrum.solar_spectrum import get_solar_spectrum

# 7. SATIRE Ingestion Tools
from .scripts.convert_satire import (
    MASTER_WL,
    _to_master_grid,
    _calibration_sigma,
    parse_satire_s,
    parse_satire_t,
    parse_satire_m,
    parse_pmip4,
    write_npz as write_satire_npz
)

# 8. MERRA-2 Reanalysis Ingestion Tools
from .scripts.convert_merra2 import (
    extract_column as extract_merra2_column,
    write_npz as write_merra2_npz,
    _pressure_to_altitude as _merra2_pressure_to_altitude,
    _number_density as _merra2_number_density,
    _mmr_to_vmr as _merra2_mmr_to_vmr,
    _column_density as _merra2_column_density,
    _compute_sigma as _merra2_compute_sigma
)

# 9. MCD Web API Ingestion Tools
from .scripts.convert_mcd_webapi import (
    MCD_VARIABLES,
    _PRESSURE_LEVELS_PA as MCD_PRESSURE_LEVELS_PA,
    _DUST_CODES as MCD_DUST_CODES,
    _query_mcd,
    _parse_mcd_response,
    build_profile as build_mcd_profile,
    write_npz as write_mcd_npz,
    generate_bundled_profiles as generate_mcd_bundled_profiles
)

# 10. LRO LOLA Lunar Topography Ingestion Tools
from .scripts.convert_lola import (
    _R_MOON_KM,
    _detect_format as _detect_lola_format,
    _geotiff_extract_patch,
    _netcdf_extract_patch,
    _compute_slope_aspect,
    _compute_horizon,
    _illumination_from_horizon,
    write_npz as write_lola_npz
)

# 11. ERA5 Reanalysis Ingestion Tools
from .scripts.convert_era5 import (
    extract_column as extract_era5_column,
    write_npz as write_era5_npz,
    _pressure_to_altitude as _era5_pressure_to_altitude,
    _number_density as _era5_number_density,
    _mmr_to_vmr as _era5_mmr_to_vmr,
    _column_density as _era5_column_density,
    _compute_sigma as _era5_compute_sigma
)

__all__ = [
    "ANGLE_UNIT",
    "ATMO_REGISTRY",
    "AU",
    "AtmosphericColumn",
    "AtmosphericColumnMars",
    "AtmosphericProfile",
    "AtmosphericSource",
    "BOLTZMANN_CONSTANT",
    "COLUMN_DENSITY_UNIT",
    "COLUMN_MASS_UNIT",
    # FIX: Removed "DATA_DIRS" string entry
    "DISTANCE_UNIT",             
    "DOBSON_UNIT",
    "ELEVATION_UNIT",
    "Geometry",
    "IRRADIANCE_UNIT",
    # FIX: Removed "KERNEL_ROOT" string entry
    "MARS_ATMO_REGISTRY",
    "MASS_DENSITY_UNIT",
    "MASTER_WL",
    "MCD_DUST_CODES",
    "MCD_PRESSURE_LEVELS_PA",
    "MCD_VARIABLES",
    "MMR_UNIT",
    "MOON_ATMO_REGISTRY",
    "MarsAtmosphericSource",
    "MoonAtmosphericSource",
    "MoonSurfaceConditions",
    "NUMBER_DENSITY_UNIT",
    "PRESSURE_UNIT",
    "REGISTRY",
    "SOLAR_LUMINOSITY",
    "SOLID_ANGLE_UNIT",
    "SPECTRAL_IRRAD_UNIT",
    "STEFAN_BOLTZMANN",
    "SUPPORTED_BODIES",
    "SpectralSource",
    "SpectralSourceInfo",
    "Spectrum",
    "TEMPERATURE_UNIT",
    "UncertainQuantity",
    "VELOCITY_UNIT",
    "VMR_UNIT",
    "WAVELENGTH_UNIT",
    "_R_MOON_KM",
    "_calibration_sigma",
    "_compute_horizon",
    "_compute_slope_aspect",
    "_detect_lola_format",
    "_era5_column_density",
    "_era5_compute_sigma",
    "_era5_mmr_to_vmr",
    "_era5_number_density",
    "_era5_pressure_to_altitude",
    "_geotiff_extract_patch",
    "_illumination_from_horizon",
    "_merra2_column_density",
    "_merra2_compute_sigma",
    "_merra2_mmr_to_vmr",
    "_merra2_number_density",
    "_merra2_pressure_to_altitude",
    "_netcdf_extract_patch",
    "_parse_mcd_response",
    "_query_mcd",
    "_to_master_grid",
    "angular_separation",
    "best_source_for_time",
    "body_name_to_id",
    "body_radii",
    "build_mcd_profile",
    "calculate_local_md5",
    "check_occultation",
    "discover_kernels",
    "et_to_utc",
    "extract_era5_column",
    "extract_merra2_column",
    "fetch_kernels",
    "fetch_remote_md5s",
    "generate_mcd_bundled_profiles",
    "get_dynamic_ephemeris_urls",
    "get_fov_intercept",
    "get_info",
    "get_solar_spectrum",
    "get_spacecraft_attitude",
    "get_state_vector",
    "get_sub_point",
    "get_sub_solar_point",
    "get_surface_illumination",
    "illumination_fraction",
    "kernel_sandbox",
    "load_kernels",
    "parse_pmip4",
    "parse_satire_m",
    "parse_satire_s",
    "parse_satire_t",
    "propagate",             # Added from Section 1 mapping requirements
    "solar_zenith_angle",
    "sources_valid_at",
    "subsolar_point",
    "sun_body_distance",
    "sun_position",
    "toa_irradiance",
    "transform_position",
    "transform_state",
    "unload_kernels",
    "utc_to_et",
    "write_era5_npz",
    "write_lola_npz",
    "write_mcd_npz",
    "write_merra2_npz",
    "write_satire_npz"
]
