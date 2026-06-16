"""
leos.uncertainty
----------------
UncertainQuantity: an astropy Quantity with an associated 1-sigma
standard uncertainty. Arithmetic operations propagate uncertainty analytically
using standard first-order Law of Propagation of Uncertainty rules (ISO GUM compliant).

All LEOS modules that return physical results use this type
so that every output represents a value with its quantified standard uncertainty (value ± sigma)
band rather than a bare number.


In addition to +, -, *, /, and ** (power), this module provides:
  - sqrt(), exp(), log() with correct analytic derivative-based sigma
  - __neg__, __radd__, __rsub__, __rmul__, __rtruediv__ for symmetric
    operations with plain numbers/Quantities on the left
  - a zero-value-safe relative_error()
  - propagate(func, *args): general first-order error propagation via
    the partial-derivatives formula

        sigma_f^2 = sum_i (df/dxi)^2 * sigma_i^2

    for ARBITRARY functions f(x1, x2, ...) of UncertainQuantity (and/or
    plain) arguments, using numerical (central finite-difference)
    partial derivatives when no analytic derivative is supplied. This
    covers any function not expressible via the basic operators above
    (e.g. 1/d^2, sin, custom radiative transfer expressions), without
    needing a hand-written propagation rule for each one.

    for ARBITRARY functions f(x1, x2, ...) of UncertainQuantity arguments.
    Accepts an optional user-defined covariance matrix (`cov`) to handle
    statistically dependent/correlated variables cleanly.
"""

import numpy as np
from astropy import units as u


class UncertainQuantity:
    """
    A physical quantity with a 1-sigma uncertainty.

    Parameters
    ----------
    value : float, np.ndarray, or astropy Quantity
        Central value.
    uncertainty : float, np.ndarray, or astropy Quantity
        1-sigma uncertainty, same unit as value.
    unit : astropy Unit, optional
        Physical unit. Ignored if value is already a Quantity.

    Examples
    --------
    >>> from astropy import units as u
    >>> from leos.uncertainty import UncertainQuantity
    >>> I = UncertainQuantity(1361.0, 5.0, u.W / u.m**2)
    >>> print(I)
    1361.0 W / m2 ± 5.0 W / m2
    """
    __array_ufunc__ = None
    def __init__(self, value, uncertainty, unit=None):
        if isinstance(value, u.Quantity):
            self.value = value
            self.uncertainty = uncertainty if isinstance(uncertainty, u.Quantity) \
                else uncertainty * value.unit
        else:
            unit = unit or u.dimensionless_unscaled
            self.value       = value * unit
            self.uncertainty = uncertainty * unit

        # uncertainty must be expressible in the value's unit
        self.uncertainty = self.uncertainty.to(self.value.unit)

        # 1-sigma is non-negative by convention; store the magnitude
        self.uncertainty = np.abs(self.uncertainty)

    # ── Representation ───────────────────────────────────────────────────────

    def _format_rounded(self):
        """
        Helper to round the central value and uncertainty to significant figures
        based on the uncertainty's magnitude. Works for scalars and arrays.
        """
        val_raw = self.value.value
        sig_raw = self.uncertainty.to(self.value.unit).value
        unit_str = f" {self.value.unit}" if self.value.unit != u.dimensionless_unscaled else ""

        if np.isscalar(val_raw):
            if sig_raw == 0 or not np.isfinite(sig_raw):
                return f"{val_raw}{unit_str} ± {sig_raw}{unit_str}"
            
            decimals = int(1 - np.floor(np.log10(sig_raw)))
            decimals = max(0, decimals)
            return f"{val_raw:.{decimals}f}{unit_str} ± {sig_raw:.{decimals}f}{unit_str}"
        else:
            val_raw = np.asarray(val_raw)
            sig_raw = np.asarray(sig_raw)
            rounded_vals = []
            rounded_sigs = []
            for v, s in zip(val_raw.flat, sig_raw.flat):
                if s == 0 or not np.isfinite(s):
                    rounded_vals.append(str(v))
                    rounded_sigs.append(str(s))
                else:
                    decimals = max(0, int(1 - np.floor(np.log10(s))))
                    rounded_vals.append(f"{v:.{decimals}f}")
                    rounded_sigs.append(f"{s:.{decimals}f}")
            v_str = np.array(rounded_vals).reshape(val_raw.shape)
            s_str = np.array(rounded_sigs).reshape(sig_raw.shape)
            return f"{v_str}{unit_str} ± {s_str}{unit_str}"

    def __str__(self):
        return self._format_rounded()

    def __repr__(self):
        return f"UncertainQuantity({self._format_rounded()})"

    # ── Derived properties ───────────────────────────────────────────────────

    def relative_uncertainty(self):
        """
        Fractional uncertainty (dimensionless), sigma/|value|, returned 
        as an UncertainQuantity (Value ± Sigma) using first-order propagation.
        """
        val = self.value.value
        sig = self.uncertainty.to(self.value.unit).value

        with np.errstate(divide="ignore", invalid="ignore"):
            rel_val = np.where(val != 0, sig / np.abs(val), np.inf)
            # Correct first-order standard uncertainty: (sigma_x / x)^2
            rel_sig = np.where(np.isfinite(rel_val), rel_val**2, 0.0) 

        return UncertainQuantity(
            rel_val * u.dimensionless_unscaled, 
            rel_sig * u.dimensionless_unscaled  # Now correctly scaled as a 1-sigma property
        )

    def to_unit(self, new_unit):
        """Return a new UncertainQuantity in different units."""
        return UncertainQuantity(
            self.value.to(new_unit),
            self.uncertainty.to(new_unit)
        )

    # ── Arithmetic: +, -, *, / ────────────────────────────────────────────────
    # First-order error propagation:
    #   addition/subtraction : sigma_z = sqrt(sigma_a^2 + sigma_b^2)
    #   multiplication       : sigma_z/z = sqrt((sigma_a/a)^2 + (sigma_b/b)^2)
    #   division             : same as multiplication

    def __add__(self, other):
        if isinstance(other, UncertainQuantity):
            val = self.value + other.value
            sig = np.sqrt(self.uncertainty**2 + other.uncertainty**2)
            return UncertainQuantity(val, sig)
        return UncertainQuantity(self.value + other, self.uncertainty)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, UncertainQuantity):
            val = self.value - other.value
            sig = np.sqrt(self.uncertainty**2 + other.uncertainty**2)
            return UncertainQuantity(val, sig)
        return UncertainQuantity(self.value - other, self.uncertainty)

    def __rsub__(self, other):
        # other - self
        return UncertainQuantity(other - self.value, self.uncertainty)

    def __neg__(self):
        return UncertainQuantity(-self.value, self.uncertainty)

    def __mul__(self, other):
        if isinstance(other, UncertainQuantity):
            val = self.value * other.value
            rel_a = self._safe_rel(self.uncertainty, self.value)
            rel_b = self._safe_rel(other.uncertainty, other.value)
            sig = np.abs(val) * np.sqrt(rel_a**2 + rel_b**2)
            return UncertainQuantity(val, sig)
        return UncertainQuantity(self.value * other, self.uncertainty * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, UncertainQuantity):
            val = self.value / other.value
            rel_a = self._safe_rel(self.uncertainty, self.value)
            rel_b = self._safe_rel(other.uncertainty, other.value)
            sig = np.abs(val) * np.sqrt(rel_a**2 + rel_b**2)
            return UncertainQuantity(val, sig)
        return UncertainQuantity(self.value / other, self.uncertainty / other)

    def __rtruediv__(self, other):
        # other / self  ->  f(x) = other / x, sigma_f = f * (sigma_x/x)
        val = other / self.value
        rel = self._safe_rel(self.uncertainty, self.value)
        sig = np.abs(val) * rel
        return UncertainQuantity(val, sig)

    @staticmethod
    def _safe_rel(sigma, value):
        """sigma/|value|, returning 0 where value == 0 (treated as exact zero)."""
        v = value.value if isinstance(value, u.Quantity) else value
        s = sigma.value if isinstance(sigma, u.Quantity) else sigma
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(v != 0, s / np.abs(v), 0.0)

    # ── Power, sqrt, exp, log ────────────────────────────────────────────────
    # For f(x) = x^n:           sigma_f = |n| * |x|^(n-1) * sigma_x
    # For f(x) = exp(x):        sigma_f = exp(x) * sigma_x   (x dimensionless)
    # For f(x) = ln(x):         sigma_f = sigma_x / |x|       (x dimensionless)

    def __pow__(self, n):
        """
        f(x) = x^n via sigma_f = |n| * |x|^(n-1) * sigma_x.

        n must be a plain number (not an UncertainQuantity) — this is
        the standard "exact exponent" case (e.g. x**2, x**0.5).
        """
        if isinstance(n, UncertainQuantity):
            raise TypeError(
                "UncertainQuantity ** UncertainQuantity is not supported "
                "by __pow__; use propagate(lambda a, b: a**b, base, exponent) "
                "instead."
            )
        val = self.value ** n
        sig = np.abs(n) * np.abs(self.value) ** (n - 1) * self.uncertainty
        return UncertainQuantity(val, sig)

    def sqrt(self):
        """f(x) = sqrt(x), equivalent to x**0.5."""
        return self ** 0.5

    def exp(self):
        """
        f(x) = exp(x), sigma_f = exp(x) * sigma_x.

        Requires x to be dimensionless.
        """
        if self.value.unit != u.dimensionless_unscaled:
            raise ValueError("exp() requires a dimensionless quantity.")
        val = np.exp(self.value.value)
        sig = val * self.uncertainty.value
        return UncertainQuantity(val, sig, u.dimensionless_unscaled)

    def log(self):
        """
        f(x) = ln(x), sigma_f = sigma_x / |x|.

        Requires x to be dimensionless and positive.
        """
        if self.value.unit != u.dimensionless_unscaled:
            raise ValueError("log() requires a dimensionless quantity.")
        x = self.value.value
        if np.any(np.asarray(x) <= 0):
            raise ValueError("log() requires a positive value.")
        val = np.log(x)
        sig = self.uncertainty.value / np.abs(x)
        return UncertainQuantity(val, sig, u.dimensionless_unscaled)


# ══════════════════════════════════════════════════════════════════════════════
# General error propagation via partial derivatives
# ══════════════════════════════════════════════════════════════════════════════

def propagate(func, *args, cov=None, h_rel=1e-6):
    """
    Propagate uncertainty through an arbitrary function via the
    first-order partial-derivatives formula:

        sigma_f^2 = sum_i (df/dxi)^2 * sigma_i^2 + 2 * sum_{i<j} (df/dxi)*(df/dxj)*sigma_ij

    assuming the arguments may have statistical dependencies defined by the 
    covariance matrix `cov`.

    Parameters
    ----------
    func : callable
        func(*central_values) -> central_value of the result.
    *args : UncertainQuantity, astropy Quantity, or plain number
        Inputs. Only UncertainQuantity arguments contribute to the error tracking.
    cov : 2D array-like, optional
        Covariance matrix matching the order and number of UncertainQuantity 
        arguments passed in. If None, inputs are assumed to be independent.
    h_rel : float
        Relative step size for the central finite-difference derivative.

    Returns
    -------
    UncertainQuantity
        func evaluated at the central values, with the propagated sigma.
    """
    values = []
    sigmas = []
    is_uq  = []
    for a in args:
        if isinstance(a, UncertainQuantity):
            values.append(a.value)
            sigmas.append(a.uncertainty)
            is_uq.append(True)
        else:
            values.append(a)
            sigmas.append(None)
            is_uq.append(False)

    central = func(*values)
    
    # Extract derivatives for UncertainQuantity arguments
    derivs = []
    for i, (val, flag) in enumerate(zip(values, is_uq)):
        if not flag:
            continue

        if isinstance(val, u.Quantity):
            mag = val.value
            h_mag = np.where(mag != 0, np.abs(mag) * h_rel, h_rel)
            h = h_mag * val.unit
        else:
            mag = np.asarray(val)
            h_mag = np.where(mag != 0, np.abs(mag) * h_rel, h_rel)
            h = h_mag

        args_plus  = list(values)
        args_minus = list(values)
        args_plus[i]  = val + h
        args_minus[i] = val - h

        f_plus  = func(*args_plus)
        f_minus = func(*args_minus)

        deriv = (f_plus - f_minus) / (2.0 * h)
        derivs.append(deriv)

    if not derivs:
        return UncertainQuantity(central, central * 0.0)

    # 1. Independent variance contribution: sum_i (df/dxi * sigma_i)^2
    variance = None
    for deriv, sig in zip(derivs, sigmas):
        if sig is not None:
            term = (deriv * sig) ** 2
            variance = term if variance is None else variance + term

    # 2. Dependent covariance contribution: 2 * sum_{i<j} (df/dxi)*(df/dxj)*sigma_ij
    if cov is not None:
        cov = np.asarray(cov)
        num_uq = len(derivs)
        if cov.shape != (num_uq, num_uq):
            raise ValueError(f"Covariance matrix shape must match number of UncertainQuantity arguments ({num_uq}x{num_uq})")
        
        # Calculate cross-terms safely extracting numerical values out of quantities
        for i in range(num_uq):
            for j in range(i + 1, num_uq):
                sigma_ij = cov[i, j]
                if sigma_ij != 0:
                    # Strip out potential units from derivatives safely during cross multiplication
                    d_i = derivs[i].value if isinstance(derivs[i], u.Quantity) else derivs[i]
                    d_j = derivs[j].value if isinstance(derivs[j], u.Quantity) else derivs[j]
                    
                    cross_term = 2.0 * d_i * d_j * sigma_ij
                    
                    # Match units back to variance container
                    if isinstance(variance, u.Quantity):
                        cross_term = cross_term * (variance.unit)
                        
                    variance = variance + cross_term

    sigma_f = np.sqrt(variance)
    return UncertainQuantity(central, sigma_f)
