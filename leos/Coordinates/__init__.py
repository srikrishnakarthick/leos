"""
LEOS: Coordinates and Astro-Geometry Submodule
Exposes SPICE ephemeris access utilities, reference frame transformations,
and high-fidelity solar illumination metrics.
"""

from .geometry import Geometry, SUPPORTED_BODIES
from .solar_geometry import (
    illumination_fraction,
    solar_zenith_angle,
    subsolar_point,  # Analytical/Analytical-hybrid model
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
    # ── High-Fidelity Geometry Wrappers ──
    get_sub_point,
    get_sub_solar_point,  # Native SPICE subslr_c backing
    angular_separation,
    get_state_vector,
    transform_position,
    transform_state,
    get_spacecraft_attitude,
    get_surface_illumination,
    get_fov_intercept,
    check_occultation,
)

__all__ = [
    "Geometry",
    "SUPPORTED_BODIES",
    "angular_separation",
    "body_name_to_id",
    "body_radii",
    "check_occultation",
    "discover_kernels",
    "et_to_utc",
    "get_fov_intercept",
    "get_spacecraft_attitude",
    "get_state_vector",
    "get_sub_point",
    "get_sub_solar_point",
    "get_surface_illumination",
    "illumination_fraction",
    "kernel_sandbox",
    "load_kernels",
    "solar_zenith_angle",
    "subsolar_point",
    "sun_body_distance",
    "sun_position",
    "transform_position",
    "transform_state",
    "unload_kernels",  # FIX: Moved up to preserve true alphabetical order
    "utc_to_et",
]
