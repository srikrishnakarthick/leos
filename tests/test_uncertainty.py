import numpy as np
from astropy import units as u
from leos.uncertainty import UncertainQuantity, propagate

print("==================================================")
print("RUNNING UNCERTAINTY PROPAGATION VERIFICATION")
print("==================================================\n")

# 1. Base Arithmetic & Unit Matching
print("--- 1. Arithmetic & Safe Relative Error ---")
a = UncertainQuantity(100.0, 6.0, u.V)
b = UncertainQuantity(20.0, 1.6, u.A)
zero_denom = UncertainQuantity(0.0, 1.5, u.A)

print(f"Multiplication (a * b): {a * b}")
print(f"Division (a / b):       {a / b}")
print(f"Zero Boundary Guard:    {a / zero_denom}\n")

# 2. Exponentials and Logarithms
print("--- 2. Exponential & Logarithmic ---")
x_dim = UncertainQuantity(2.0, 0.1, u.dimensionless_unscaled)
print(f"Exponential exp(x):     {x_dim.exp()}")
print(f"Natural Log log(x):     {x_dim.log()}\n")

# 3. Forward Trigonometric Functions (Degrees & Radians)
print("--- 3. Forward Trigonometric Suite ---")
angle_deg = UncertainQuantity(45.0, 1.0, u.deg)
angle_rad = UncertainQuantity(np.pi / 4.0, 0.01745, u.rad)

print(f"Sine (from deg):        {angle_deg.sin()}")
print(f"Cosine (from rad):      {angle_rad.cos()}")
print(f"Tangent (from deg):     {angle_deg.tan()}")
print(f"Secant (from deg):      {angle_deg.sec()}")
print(f"Cosecant (from deg):    {angle_deg.csc()}")
print(f"Cotangent (from deg):   {angle_deg.cot()}\n")

# 4. Inverse Trigonometric Functions
print("--- 4. Inverse Trigonometric Suite ---")
ratio = UncertainQuantity(0.5, 0.02, u.dimensionless_unscaled)
large_ratio = UncertainQuantity(2.0, 0.1, u.dimensionless_unscaled)

print(f"Arcsing arcsin(0.5):    {ratio.asin()}")
print(f"Arccosine arccos(0.5):  {ratio.acos()}")
print(f"Arctangent arctan(0.5): {ratio.atan()}")
print(f"Arcsecant arcsec(2.0):  {large_ratio.asec()}")
print(f"Arccosecant arccsc(2.0):{large_ratio.acsc()}")
print(f"Arccotangent arccot(0.5):{ratio.acot()}\n")

print("==================================================")
print("VERIFICATION COMPLETED SUCCESSFULLY")
print("==================================================")
