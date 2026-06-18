import numpy as np
from astropy import units as u
import leos

print("--- Testing Refactored UncertainQuantity Base Namespace ---")

# Scenario A: Passing raw numbers alongside an explicit physical unit container
solar_flux = leos.UncertainQuantity(1361.0, 5.0, u.W / u.m**2)
print(f"Scenario A: {solar_flux}")

# Scenario B: Passing pre-instantiated Astropy Quantities directly
val_qty = 273.15 * u.K
sig_qty = 0.05 * u.K
sensor_temp = leos.UncertainQuantity(val_qty, sig_qty)
print(f"Scenario B: {sensor_temp}")

# Scenario C: Passing an Astropy Quantity value with a bare numerical uncertainty
orbital_alt = leos.UncertainQuantity(500000.0 * u.m, 0.2)
print(f"Scenario C: {orbital_alt}")

# Scenario D: Intentional dimensional mismatch to trigger dimensional guardrails
try:
    mismatched = leos.UncertainQuantity(100.0 * u.m, 2.0 * u.s)
except Exception as e:
    print(f"Scenario D: Caught expected error -> {type(e).__name__}: {e}")

# 1. Successful conversion: Transform a velocity metric into meters per second
velocity_kmh = leos.UncertainQuantity(108.0, 3.6, u.km / u.h)
print(f"Original Metric: {velocity_kmh}")

velocity_si = velocity_kmh.to_unit(u.m / u.s)
print(f"Converted to SI: {velocity_si}")

# 2. Invalid conversion: Attempt to convert velocity into mass (kilograms)
try:
    invalid_transform = velocity_kmh.to_unit(u.kg)
except Exception as e:
    print(f"Conversion Error: {type(e).__name__} -> {e}")

# Scenario 1: Standard Fractional Uncertainty Verification
measurement = leos.UncertainQuantity(10.0, 0.2, u.kg)
rel_measurement = measurement.relative_uncertainty()

print(f"Original: {measurement}")
print(f"Relative: {rel_measurement}")

# Scenario 2: Absolute Zero Boundary Safety Clamping
zero_edge = leos.UncertainQuantity(0.0, 1.5, u.m / u.s)
rel_zero_edge = zero_edge.relative_uncertainty()

print(f"\nZero Edge Original: {zero_edge}")
print(f"Zero Edge Relative: {rel_zero_edge}")


print("\n--- Testing Arithmetic Methods ---")
a = leos.UncertainQuantity(10.0, 3.0, u.m)
b = leos.UncertainQuantity(4.0, 4.0, u.m)

print(f"Addition (a + b): {a + b}")
print(f"Scalar Addition (a + 5): {a + 5}")
print(f"Subtraction (a - b): {a - b}")
print(f"Negation (-a): {-a}")

print("\n--- Testing Multiplication & Division ---")
v_src = leos.UncertainQuantity(100.0, 6.0, u.V)
i_src = leos.UncertainQuantity(20.0, 1.6, u.A)

print(f"Multiplication (v_src * i_src): {v_src * i_src}")
print(f"Division (v_src / i_src): {v_src / i_src}")

# _safe_rel() safety hook handling a zero boundary denominator
zero_denom = leos.UncertainQuantity(0.0, 1.5, u.A)
print(f"Zero Division Safety (v_src / zero_denom): {v_src / zero_denom}")

print("\n--- Testing Power and Roots ---")
area = leos.UncertainQuantity(16.0, 4.0, u.m**2)

print(f"Power (area**2): {area**2}")
print(f"Square Root (area.sqrt()): {area.sqrt()}")

print("\n--- Testing Trigonometric Operations ---")
angle = leos.UncertainQuantity(45.0, 1.0, u.deg)

print(f"Sine:      {angle.sin()}")
print(f"Cosine:    {angle.cos()}")
print(f"Tangent:   {angle.tan()}")
print(f"Secant:    {angle.sec()}")
print(f"Cosecant:  {angle.csc()}")
print(f"Cotangent: {angle.cot()}")

try:
    length = leos.UncertainQuantity(10.0, 0.5, u.m)
    length.sin()
except ValueError as e:
    print(f"Caught Expected Exception: {e}")

print("\n--- Testing Inverse Trigonometric Operations ---")
ratio = leos.UncertainQuantity(0.5, 0.02, u.dimensionless_unscaled)
large_ratio = leos.UncertainQuantity(2.0, 0.1, u.dimensionless_unscaled)

print(f"Arcsin:   {ratio.asin()}")
print(f"Arccos:   {ratio.acos()}")
print(f"Arctan:   {ratio.atan()}")
print(f"Arcsec:   {large_ratio.asec()}")
print(f"Arccsc:   {large_ratio.acsc()}")
print(f"Arccot:   {ratio.acot()}")

try:
    angle_input = leos.UncertainQuantity(45.0, 1.0, u.deg)
    angle_input.asin()
except ValueError as e:
    print(f"Caught Expected Exception: {e}")

print("\n--- Test 1: Independent Variation Tracking ---")
def custom_model(x, y):
    return (x**2) * np.sin(y)

x_param = leos.UncertainQuantity(10.0, 0.2, u.m)
y_param = leos.UncertainQuantity(30.0, 1.0, u.deg)

result_indep = leos.propagate(custom_model, x_param, y_param)
print(f"Independent Result: {result_indep}")

print("\n--- Test 2: Strongly Dependent Covariance Tracking ---")
sig_x = 0.2
sig_y = 1.0

# Cross-covariance tracking (Pearson Correlation coefficient rho = 0.8)
covariance_xy = 0.8 * sig_x * sig_y  

cov_matrix = [
    [sig_x**2,      covariance_xy],
    [covariance_xy, sig_y**2     ]
]

result_dep = leos.propagate(custom_model, x_param, y_param, cov=cov_matrix)
print(f"Dependent Result:   {result_dep}")
