"""
leos.uncertainty
----------------
UncertainQuantity: an astropy Quantity with an associated 1-sigma
uncertainty. Arithmetic operations propagate uncertainty analytically
using standard first-order error propagation rules.

All other LEOS modules that return physical results use this type
so that every output is  value ± sigma  rather than a bare number.
"""

import numpy as np
from astropy import units as u


class UncertainQuantity:
    """
    A physical quantity with a 1-sigma uncertainty.

    Parameters
    ----------
    value : float or np.ndarray
        Central value (bare number or astropy Quantity).
    uncertainty : float or np.ndarray
        1-sigma uncertainty, same unit as value.
    unit : astropy Unit, optional
        Physical unit. Ignored if value is already a Quantity.

    Examples
    --------
    >>> from astropy import units as u
    >>> from leos.uncertainty import UncertainQuantity
    >>> I = UncertainQuantity(1361.0, 5.0, u.W / u.m**2)
    >>> print(I)
    1361.0 ± 5.0 W / m2
    """

    def __init__(self, value, uncertainty, unit=None):
        if isinstance(value, u.Quantity):
            self.value = value
            self.uncertainty = uncertainty if isinstance(uncertainty, u.Quantity) \
                else uncertainty * value.unit
        else:
            unit = unit or u.dimensionless_unscaled
            self.value       = value * unit
            self.uncertainty = uncertainty * unit

    # ── Representation ───────────────────────────────────────────────────────

    def __repr__(self):
        return (f"UncertainQuantity({self.value} ± {self.uncertainty})")

    def __str__(self):
        return f"{self.value} ± {self.uncertainty}"

    # ── Derived properties ───────────────────────────────────────────────────

    def relative_error(self):
        """Fractional uncertainty (dimensionless)."""
        return (self.uncertainty / self.value).decompose()

    def to_unit(self, new_unit):
        """Return a new UncertainQuantity in different units."""
        return UncertainQuantity(
            self.value.to(new_unit),
            self.uncertainty.to(new_unit)
        )

    # ── Arithmetic ───────────────────────────────────────────────────────────
    # First-order error propagation:
    #   addition/subtraction : σ_z = sqrt(σ_a² + σ_b²)
    #   multiplication       : σ_z/z = sqrt((σ_a/a)² + (σ_b/b)²)
    #   division             : same as multiplication

    def __add__(self, other):
        if isinstance(other, UncertainQuantity):
            val = self.value + other.value
            sig = np.sqrt(self.uncertainty**2 + other.uncertainty**2)
            return UncertainQuantity(val, sig)
        return UncertainQuantity(self.value + other, self.uncertainty)

    def __sub__(self, other):
        if isinstance(other, UncertainQuantity):
            val = self.value - other.value
            sig = np.sqrt(self.uncertainty**2 + other.uncertainty**2)
            return UncertainQuantity(val, sig)
        return UncertainQuantity(self.value - other, self.uncertainty)

    def __mul__(self, other):
        if isinstance(other, UncertainQuantity):
            val = self.value * other.value
            sig = val * np.sqrt(
                (self.uncertainty / self.value)**2 +
                (other.uncertainty / other.value)**2
            )
            return UncertainQuantity(val, sig)
        return UncertainQuantity(self.value * other, self.uncertainty * other)

    def __truediv__(self, other):
        if isinstance(other, UncertainQuantity):
            val = self.value / other.value
            sig = val * np.sqrt(
                (self.uncertainty / self.value)**2 +
                (other.uncertainty / other.value)**2
            )
            return UncertainQuantity(val, sig)
        return UncertainQuantity(self.value / other, self.uncertainty / other)

    def __rmul__(self, other):
        return self.__mul__(other)
