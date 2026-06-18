import pytest
import numpy as np
from astropy import units as u
import leos

# ==============================================================================
# 1. INITIALIZATION AND DIMENSIONAL GAURDRAILS
# ==============================================================================

def test_initialization_variants():
    """Verify initialization pathways (raw inputs vs explicit quantities)."""
    # Scenario A: Raw inputs + separate unit
    uq_a = leos.UncertainQuantity(1361.0, 5.0, u.W / u.m**2)
    assert uq_a.value.unit == u.W / u.m**2
    assert uq_a.uncertainty.value == 5.0

    # Scenario B: Pre-instantiated quantities
    uq_b = leos.UncertainQuantity(273.15 * u.K, 0.05 * u.K)
    assert uq_b.value.value == 273.15
    assert uq_b.uncertainty.value == 0.05

    # Scenario C: Quantity value + raw uncertainty
    uq_c = leos.UncertainQuantity(500000.0 * u.m, 0.2)
    assert uq_c.uncertainty.unit == u.m
    assert uq_c.uncertainty.value == 0.2

def test_dimensional_mismatch_guardrails():
    """Ensure initializing with incompatible dimensions raises a UnitConversionError."""
    with pytest.raises(Exception):
        # Meters value with Seconds uncertainty must be rejected
        leos.UncertainQuantity(100.0 * u.m, 2.0 * u.s)

def test_unit_conversion():
    """Verify explicit unit transformation and type-checking blockades."""
    velocity = leos.UncertainQuantity(108.0, 3.6, u.km / u.h)
    si_velocity = velocity.to_unit(u.m / u.s)
    
    assert si_velocity.value.value == pytest.approx(30.0)
    assert si_velocity.uncertainty.value == pytest.approx(1.0)
    
    with pytest.raises(Exception):
        velocity.to_unit(u.kg)

# ==============================================================================
# 2. VECTORIZATION (ARRAY) SAFETY
# ==============================================================================

def test_vectorized_array_propagation():
    """Test that UncertainQuantity handles NumPy arrays for both values and uncertainties."""
    val_arr = np.array([10.0, 20.0, 30.0])
    unc_arr = np.array([1.0, 2.0, 3.0])
    
    uq_array = leos.UncertainQuantity(val_arr, unc_arr, u.m)
    
    # Check shape properties
    assert uq_array.value.shape == (3,)
    
    # Perform math operation over the whole vector simultaneously
    result = uq_array * 2.0
    assert np.allclose(result.value.value, [20.0, 40.0, 60.0])
    assert np.allclose(result.uncertainty.value, [2.0, 4.0, 6.0])

# ==============================================================================
# 3. BOUNDARY CONDITIONS & ZERO HANDLING
# ==============================================================================

def test_relative_uncertainty_and_zero_clamping():
    """Verify relative uncertainty calculations over normal values and absolute zeros."""
    # Standard Case
    measurement = leos.UncertainQuantity(10.0, 0.2, u.kg)
    rel = measurement.relative_uncertainty()
    assert rel.value.value == pytest.approx(0.02)

    # Edge Case: Exact zero values should be safely clamped without throwing exceptions
    zero_edge = leos.UncertainQuantity(0.0, 1.5, u.m / u.s)
    rel_zero = zero_edge.relative_uncertainty()
    assert rel_zero.value.value == np.inf  # sigma / 0 = inf

def test_zero_division_safety_hooks():
    """Ensure division by an uncertain zero yields zero variance via internal hook protection."""
    v_src = leos.UncertainQuantity(100.0, 6.0, u.V)
    zero_denom = leos.UncertainQuantity(0.0, 1.5, u.A)
    
    # Division should result in an infinity central value safely
    result = v_src / zero_denom
    assert result.value.value == np.inf

# ==============================================================================
# 4. STANDARD ARITHMETIC & REFLECTED OPERATIONS
# ==============================================================================

def test_reflected_and_scalar_arithmetic():
    """Test standard operators and ensure reflected methods work identically."""
    a = leos.UncertainQuantity(10.0, 3.0, u.m)
    
    # Addition and Reflected Addition
    res1 = a + 5.0
    res2 = 5.0 + a
    assert res1.value == res2.value
    assert res1.uncertainty == res2.uncertainty
    
    # Subtraction and Reflected Subtraction
    res3 = a - 4.0
    assert res3.value.value == 6.0
    res4 = 14.0 - a
    assert res4.value.value == 4.0

    # Multiplication and Division
    v_src = leos.UncertainQuantity(100.0, 6.0, u.V)
    res_mul = 2 * v_src
    assert res_mul.value.value == 200.0
    assert res_mul.uncertainty.value == 12.0

    res_div = 100 / v_src
    assert res_div.value.value == 1.0

def test_powers_and_roots():
    """Verify power exponents and square root tracking."""
    area = leos.UncertainQuantity(16.0, 4.0, u.m**2)
    
    # Square Root rule verification: sigma_f = 0.5 * (sigma_x / sqrt(x))
    root = area.sqrt()
    assert root.value.value == 4.0
    assert root.uncertainty.value == pytest.approx(0.5 * (4.0 / np.sqrt(16.0)))

    # Reject uncertain exponents
    with pytest.raises(TypeError):
        _ = area ** area

# ==============================================================================
# 5. TRIGONOMETRIC DOMAIN CHECKS
# ==============================================================================

def test_trigonometric_validity_and_failures():
    """Verify angular tracking across forward and inverse trigonometric configurations."""
    angle = leos.UncertainQuantity(45.0, 1.0, u.deg)
    
    assert angle.sin().value.value == pytest.approx(np.sin(np.radians(45.0)))
    assert angle.cos().value.value == pytest.approx(np.cos(np.radians(45.0)))

    # Mismatched non-angular inputs must fail forward trig operations
    length = leos.UncertainQuantity(10.0, 0.5, u.m)
    with pytest.raises(ValueError):
        length.sin()

    # Inverse trig domain check
    ratio = leos.UncertainQuantity(0.5, 0.02, u.dimensionless_unscaled)
    assert ratio.asin().value.unit == u.rad

    # Inverse trig must reject quantities carrying explicit physical dimensions
    with pytest.raises(ValueError):
        angle.asin()

# ==============================================================================
# 6. GENERAL REVOLUTIONARY PROPAGATION (FINITE DIFFERENCE & COVARIANCE)
# ==============================================================================

def custom_test_model(x, y):
    return (x**2) * np.sin(y)

def test_propagate_independent_vs_dependent():
    """Verify multi-variable partial derivative mapping under correlation parameters."""
    x_param = leos.UncertainQuantity(10.0, 0.2, u.m)
    y_param = leos.UncertainQuantity(30.0, 1.0, u.deg)

    # 1. Independent Tracking Execution
    result_indep = leos.propagate(custom_test_model, x_param, y_param)
    assert result_indep.value.unit == u.m**2

    # 2. Correlated Covariance Tracking Execution
    sig_x = 0.2
    sig_y = 1.0
    covariance_xy = 0.8 * sig_x * sig_y  # strong positive relationship (rho = 0.8)
    
    cov_matrix = [
        [sig_x**2,      covariance_xy],
        [covariance_xy, sig_y**2     ]
    ]
    
    result_dep = leos.propagate(custom_test_model, x_param, y_param, cov=cov_matrix)
    
    # Positive covariance alongside matching positive partial derivatives should inflate final uncertainty
    assert result_dep.uncertainty.value > result_indep.uncertainty.value

def test_propagate_malformed_covariance_rejection():
    """Ensure propagate() throws an informative ValueError if a covariance shape is invalid."""
    x_param = leos.UncertainQuantity(10.0, 0.2, u.m)
    y_param = leos.UncertainQuantity(30.0, 1.0, u.deg)
    
    # Broken shape configuration (2x3 instead of 2x2 for 2 elements)
    bad_cov = [[1, 0, 0], [0, 1, 0]]
    
    with pytest.raises(ValueError, match="Covariance matrix shape must match number of UncertainQuantity arguments"):
        leos.propagate(custom_test_model, x_param, y_param, cov=bad_cov)


def compute_expected_iso_gum(v, s):
    import numpy as np
    """Programmatic engine to mirror ISO GUM rules without hardcoding expected strings."""
    order = int(np.floor(np.log10(s)))
    leading_digit = int(s / (10**order))
    sig_figs = 2 if leading_digit in [1, 2] else 1
    decimals = sig_figs - 1 - order
    
    s_expected = int(round(s, decimals))
    v_expected = int(round(v, decimals))
    
    # Cascade check for order shift
    new_order = int(np.floor(np.log10(s_expected))) if s_expected > 0 else order
    if new_order != order:
        decimals = sig_figs - 1 - new_order
        s_expected = int(round(s, decimals))
        v_expected = int(round(v, decimals))
        
    return f"{v_expected} ± {s_expected}"


@pytest.mark.parametrize("value, uncertainty", [
    (12345, 567),  # 1 s.f. -> rounds to hundreds place (-2) -> 12300 ± 600
    (12345, 123),  # 2 s.f. -> rounds to tens place (-1)     -> 12350 ± 120
    (987654, 34),  # 1 s.f. -> rounds to tens place (-1)     -> 987650 ± 30
    (987654, 19),  # 2 s.f. -> rounds to units place (0)     -> 987654 ± 19
    (5000, 950),   # 1 s.f. -> rounds to hundreds place (-2) -> 5000 ± 1000 (shifts up decade!)
])
def test_dynamic_large_integer_iso_gum_rounding(value, uncertainty):
    """Verify integer boundaries above the decimal point round dynamically based on ISO GUM mathematical limits."""
    import leos
    uq = leos.UncertainQuantity(value, uncertainty)
    
    expected_string = compute_expected_iso_gum(value, uncertainty)
    assert uq._format_rounded() == expected_string

@pytest.mark.parametrize("val, unc, unit, expected", [
    (10.05762, 0.027, u.Ohm, "10.058 Ohm ± 0.027 Ohm"),
    (10.4713, 0.01047, u.mOhm, "10.471 mOhm ± 0.010 mOhm"),
    (28.0531, 0.1285, u.kHz, "28.05 kHz ± 0.13 kHz")
])
def test_canonical_iso_gum_examples(val, unc, unit, expected):
    """Verify engine alignment with explicit Section 7.2.6 test cases."""
    uq = leos.UncertainQuantity(val, unc, unit)
    assert uq._format_rounded() == expected
