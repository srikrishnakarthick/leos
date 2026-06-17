"""
tests/test_init.py
------------------
Rigorous verification suite for the LEOS top-level public API.
Validates structural composition, types, and aliased name bindings.
"""

import pytest
import numpy as np
import leos

def test_package_docstring_presence():
    """Ensure the top-level informative paragraphs are successfully loaded."""
    assert leos.__doc__ is not None
    assert "LEOS" in leos.__doc__
    assert "multi-planetary atmospheres" in leos.__doc__ or "advanced data ingestion engines" in leos.__doc__


def test_public_api_completeness():
    """Verify that every single string entry in __all__ exists in the module namespace."""
    missing_elements = [item for item in leos.__all__ if not hasattr(leos, item)]
    assert not missing_elements, f"Items listed in __all__ but missing from namespace: {missing_elements}"


def test_public_api_is_sorted():
    """Strict style guard checking if the API exposure list maintains alphabetical order."""
    assert leos.__all__ == sorted(leos.__all__), "__all__ list should be strictly sorted alphabetically."


@pytest.mark.parametrize("attr_name, expected_type", [
    # Units & Enums
    ("ANGLE_UNIT", object),
    ("AtmosphericSource", type),
    ("MarsAtmosphericSource", type),
    ("MoonAtmosphericSource", type),
    ("SpectralSource", type),
    
    # Classes
    ("UncertainQuantity", type),
    ("Geometry", type),
    ("AtmosphericProfile", type),
    ("AtmosphericColumn", type),
    ("AtmosphericColumnMars", type),
    ("MoonSurfaceConditions", type),
    ("Spectrum", type),
    ("SpectralSourceInfo", type),
    
    # Data Structures / Registry Dictionaries
    ("ATMO_REGISTRY", dict),
    ("MARS_ATMO_REGISTRY", dict),
    ("MOON_ATMO_REGISTRY", dict),
    ("REGISTRY", dict),
    ("SUPPORTED_BODIES", set),
    ("MASTER_WL", np.ndarray),
    ("MCD_VARIABLES", dict),
    ("MCD_DUST_CODES", dict),
    ("MCD_PRESSURE_LEVELS_PA", list),
])
def test_core_object_types(attr_name, expected_type):
    """Rigorous structural type validation for critical components."""
    obj = getattr(leos, attr_name)
    assert isinstance(obj, expected_type), f"Expected {attr_name} to be of type {expected_type}, got {type(obj)}"


def test_callable_functions():
    """Scan all expected functional interfaces to confirm they are callable."""
    expected_callables = [
        # Astro Geometry
        "discover_kernels", "load_kernels", "unload_kernels", "kernel_sandbox",
        "utc_to_et", "et_to_utc", "body_name_to_id", "body_radii", "sun_position",
        "sun_body_distance", "solar_zenith_angle", "illumination_fraction", "subsolar_point", "toa_irradiance",
        
        # Spectral Core
        "get_info", "sources_valid_at", "best_source_for_time", "get_solar_spectrum",
        
        # Ingestion Utilities
        "parse_satire_s", "parse_satire_t", "parse_satire_m", "parse_pmip4", "write_satire_npz",
        "extract_merra2_column", "write_merra2_npz",
        "build_mcd_profile", "write_mcd_npz", "generate_mcd_bundled_profiles",
        "write_lola_npz", "extract_era5_column", "write_era5_npz"
    ]
    
    for func_name in expected_callables:
        assert hasattr(leos, func_name), f"Missing expected function: {func_name}"
        assert callable(getattr(leos, func_name)), f"Object {func_name} is in the namespace but not callable."


def test_internal_aliasing_coexistence():
    """Verify both reanalysis extraction engines co-exist safely without collision."""
    # Assert unique references
    assert leos.extract_merra2_column is not leos.extract_era5_column
    assert leos._merra2_pressure_to_altitude is not leos._era5_pressure_to_altitude
    assert leos._merra2_number_density is not leos._era5_number_density
    
    # Assert specific mathematical conversion helpers match functional expectations
    assert callable(leos._era5_pressure_to_altitude)
    assert callable(leos._merra2_pressure_to_altitude)
