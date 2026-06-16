import numpy as np
import astropy.units as u
from leos.uncertainty import UncertainQuantity, propagate

def run_test(name, test_func):
    """Helper engine to run tests with clean pass/fail visuals."""
    try:
        test_func()
        print(f"✅ PASSED: {name}")
    except Exception as e:
        print(f"❌ FAILED: {name}")
        import traceback
        traceback.print_exc()
        print("-" * 50)

# =====================================================================
# TEST CASES
# =====================================================================

def test_initialization_and_units():
    # Test initialization with explicit units vs implicit Quantity units
    q1 = UncertainQuantity(10.0, 0.5, u.m)
    assert q1.value.unit == u.m
    assert q1.uncertainty.unit == u.m
    
    # Passing an already existing astropy quantity
    q2 = UncertainQuantity(10.0 * u.km, 500.0 * u.m)
    assert q2.value.unit == u.km
    assert q2.uncertainty.unit == u.km
    assert np.isclose(q2.uncertainty.value, 0.5) # 500m converted to 0.5km

def test_basic_arithmetic():
    a = UncertainQuantity(10.0, 3.0, u.m)
    b = UncertainQuantity(5.0, 4.0, u.m)
    
    # Addition variance check: sig^2 = 3^2 + 4^2 = 25 -> sig = 5
    res_add = a + b
    assert np.isclose(res_add.value.value, 15.0)
    assert np.isclose(res_add.uncertainty.value, 5.0)
    
    # Multiplication relative check: (3/10)^2 + (4/5)^2 = 0.09 + 0.64 = 0.73
    # Total sig = 50.0 * sqrt(0.73) = 42.7200187266
    res_mul = a * b
    assert np.isclose(res_mul.value.value, 50.0)
    assert np.isclose(res_mul.uncertainty.value, 50.0 * np.sqrt(0.73))

def test_relative_uncertainty_and_zeros():
    # Regular relative uncertainty tracking
    a = UncertainQuantity(10.0, 2.0, u.m)
    rel = a.relative_uncertainty()
    assert np.isclose(rel.value.value, 0.2)
    assert np.isclose(rel.uncertainty.value, 0.04) # sig_rel = (sigma/x)^2
    
    # Exact zero division protection test
    z = UncertainQuantity(0.0, 0.0, u.m)
    assert UncertainQuantity._safe_rel(z.uncertainty, z.value) == 0.0

def test_transcendental_functions():
    # Exponential tracking (requires dimensionless)
    x = UncertainQuantity(2.0, 0.1, u.dimensionless_unscaled)
    res_exp = x.exp()
    assert np.isclose(res_exp.value.value, np.exp(2.0))
    assert np.isclose(res_exp.uncertainty.value, np.exp(2.0) * 0.1)
    
    # Power handling (exact exponent)
    y = UncertainQuantity(4.0, 0.5, u.m)
    res_pow = y**0.5 # sqrt
    assert np.isclose(res_pow.value.value, 2.0)
    # df/dx = 0.5 * x^(-0.5) -> 0.5 * 0.5 * 0.5 = 0.125
    assert np.isclose(res_pow.uncertainty.value, 0.125)

def test_analytic_trigonometry():
    # Test forward trig with auto-radian conversions
    angle_deg = UncertainQuantity(30.0, 1.0, u.deg)
    res_sin = angle_deg.sin()
    
    rad_val = np.deg2rad(30.0)
    rad_sig = np.deg2rad(1.0)
    
    assert np.isclose(res_sin.value.value, np.sin(rad_val))
    assert np.isclose(res_sin.uncertainty.value, np.cos(rad_val) * rad_sig)
    
    # Test inverse trig maps seamlessly to radians
    ratio = UncertainQuantity(0.5, 0.01, u.dimensionless_unscaled)
    res_asin = ratio.asin()
    assert res_asin.value.unit == u.rad
    assert np.isclose(res_asin.value.value, np.arcsin(0.5))
    # d/dx(asin) = 1 / sqrt(1 - x^2) -> 1 / sqrt(0.75)
    assert np.isclose(res_asin.uncertainty.value, 0.01 / np.sqrt(0.75))

def test_propagate_independent():
    # f(x, y) = x^2 * sin(y)
    def model(x, y):
        return (x**2) * np.sin(y)
        
    x = UncertainQuantity(10.0, 0.2, u.m)
    y = UncertainQuantity(30.0, 1.0, u.deg) # Engine evaluates dy in degrees natively!
    
    res = propagate(model, x, y)
    
    # Analytical partial derivatives (in degree-space units for y)
    y_rad = np.deg2rad(30.0)
    df_dx = 2.0 * 10.0 * np.sin(y_rad)                     # = 10.0
    df_dy = (10.0**2) * np.cos(y_rad) * (np.pi / 180.0)    # = 1.511499
    
    expected_var = (df_dx * 0.2)**2 + (df_dy * 1.0)**2
    assert np.isclose(res.uncertainty.value**2, expected_var, rtol=1e-4)

def test_propagate_covariance_and_cross_units():
    # Crucial validation check for your target unit scaling matrix logic
    def multi_unit_model(x, y):
        return x * y
        
    x = UncertainQuantity(2.0, 0.1, u.m)
    y = UncertainQuantity(5.0, 0.2, u.kg)
    
    # Positive correlation coefficient rho = 0.50
    # Covariance matrix provided cleanly in native parameters [m, kg]
    cov = [
        [0.1**2,    0.5 * 0.1 * 0.2],
        [0.5*0.1*0.2, 0.2**2]
    ]
    
    res = propagate(multi_unit_model, x, y, cov=cov)
    assert res.value.unit == (u.m * u.kg)
    
    # Analytical verification:
    # df/dx = y = 5.0; df/dy = x = 2.0
    # var = (5*0.1)^2 + (2*0.2)^2 + 2*(5)*(2)*(0.5 * 0.1 * 0.2)
    # var = 0.25 + 0.16 + 20*(0.01) = 0.25 + 0.16 + 0.20 = 0.61
    assert np.isclose(res.uncertainty.value**2, 0.61, rtol=1e-4)

def test_array_quantities():
    # Assures array processing elements don't collapse or throw dimensionality exceptions
    v_arr = np.array([10.0, 20.0]) * u.m
    s_arr = np.array([0.5, 1.0]) * u.m
    
    q_arr = UncertainQuantity(v_arr, s_arr)
    res_add = q_arr + q_arr
    
    assert np.allclose(res_add.value.value, [20.0, 40.0])
    assert np.allclose(res_add.uncertainty.value, [np.sqrt(0.5), np.sqrt(2.0)])

# =====================================================================
# EXECUTION ROUTINE
# =====================================================================
if __name__ == "__main__":
    print("=" * 50)
    print("STARTING RIGOROUS UNCERTAINTY FRAMEWORK TEST SUITE")
    print("=" * 50)
    
    run_test("Initialization & Dimensional Conformity", test_initialization_and_units)
    run_test("Standard Arithmetic Error Propagation", test_basic_arithmetic)
    run_test("Relative Uncertainty Mapping & Zero Safety", test_relative_uncertainty_and_zeros)
    run_test("Transcendental Exponentials and Power Expansions", test_transcendental_functions)
    run_test("Analytic Trigonometry Coordinate Arc-Mappings", test_analytic_trigonometry)
    run_test("Independent Finite-Difference Propagation Engine", test_propagate_independent)
    run_test("Cross-Coupling Covariance Scaling Operations", test_propagate_covariance_and_cross_units)
    run_test("Vectorized Array Quantities Processing Evaluation", test_array_quantities)
    
    print("=" * 50)
