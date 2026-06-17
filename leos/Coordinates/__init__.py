"""
LEOS: Coordinates and Astro-Geometry Submodule
Exposes SPICE ephemeris access utilities, reference frame transformations,
and high-fidelity solar illumination metrics.
"""

from .geometry import Geometry, SUPPORTED_BODIES
from .solar_geometry import (
    illumination_fraction,
    solar_zenith_angle,
    subsolar_point,
    sun_body_distance,
    toa_irradiance,
)
from .spice_utils import (
    body_name_to_id,
    body_radii,
    discover_kernels,
    et_to_utc,
    kernel_sandbox,
    load_kernels,
    sun_position,
    unload_kernels,
    utc_to_et,
)

__all__ = [
    "Geometry",
    "SUPPORTED_BODIES",
    "body_name_to_id",
    "body_radii",
    "discover_kernels",
    "et_to_utc",
    "illumination_fraction",
    "kernel_sandbox",
    "load_kernels",
    "solar_zenith_angle",
    "subsolar_point",
    "sun_body_distance",
    "sun_position",
    "toa_irradiance",
    "unload_kernels",
    "utc_to_et",
]
