import pytest
from astropy import units as u
from astropy.constants import L_sun, au, k_B, sigma_sb
import leos

def test_geometric_and_terrain_units():
    """Verify geometric and terrain layout unit bindings."""
    assert leos.ANGLE_UNIT == u.deg
    assert leos.SOLID_ANGLE_UNIT == u.sr
    assert leos.DISTANCE_UNIT == u.au
    assert leos.ELEVATION_UNIT == u.m
    assert leos.VELOCITY_UNIT == (u.m / u.s)

def test_radiometric_units():
    """Verify radiometric framework unit bindings."""
    assert leos.IRRADIANCE_UNIT == (u.W / u.m**2)
    assert leos.SPECTRAL_IRRAD_UNIT == (u.W / u.m**2 / u.nm)
    assert leos.WAVELENGTH_UNIT == u.nm

def test_atmospheric_and_thermodynamic_units():
    """Verify thermodynamic environment unit bindings."""
    assert leos.TEMPERATURE_UNIT == u.K
    assert leos.PRESSURE_UNIT == u.Pa
    assert leos.MASS_DENSITY_UNIT == (u.kg / u.m**3)
    assert leos.NUMBER_DENSITY_UNIT == (u.m**-3)
    assert leos.VMR_UNIT == u.dimensionless_unscaled
    assert leos.MMR_UNIT == (u.kg / u.kg)

def test_columnar_quantities_and_dobson_conversion():
    """Verify columnar bounds and operational Dobson Unit scaling factor."""
    assert leos.COLUMN_MASS_UNIT == (u.kg / u.m**2)
    assert leos.COLUMN_DENSITY_UNIT == (u.m**-2)
    
    # Target conversion test for Dobson Units (1 DU = 2.6867e20 molecules / m^2)
    test_ozone = 300.0 * leos.DOBSON_UNIT
    converted_ozone = test_ozone.to(u.m**-2)
    
    expected_value = 300.0 * 2.6867e20
    assert pytest.approx(converted_ozone.value, rel=1e-5) == expected_value

def test_physical_constants():
    """Verify central physical constant assignments match expected values."""
    assert leos.SOLAR_LUMINOSITY == L_sun
    assert leos.AU == au
    assert leos.BOLTZMANN_CONSTANT == k_B
    assert leos.STEFAN_BOLTZMANN == sigma_sb

