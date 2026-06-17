"""
leos.spectrum
-------------
Spectrum: core data container for spectral irradiance I(lambda) +- sigma(lambda).
All wavelengths and fluxes are astropy Quantities. No physics lives here —
this is a typed, validated container that the radiative transfer modules
populate and return.
"""

import numpy as np
from astropy import units as u
from leos.units import WAVELENGTH_UNIT, SPECTRAL_IRRAD_UNIT


class Spectrum:
    """
    Spectral irradiance as a function of wavelength, with optional uncertainty.

    Parameters
    ----------
    wavelengths : astropy Quantity (length units)
        Wavelength array, e.g. in nm.
    flux : astropy Quantity (spectral irradiance units)
        Spectral irradiance I(lambda) in W/m²/nm.
    uncertainty : astropy Quantity, optional
        1-sigma uncertainty on flux, same unit as flux.
        If None, uncertainty is set to zero.
    label : str, optional
        Human-readable label, e.g. 'Mars surface, tau=0.5'.

    Examples
    --------
    >>> import numpy as np
    >>> from astropy import units as u
    >>> from leos.spectrum import Spectrum
    >>> wl = np.linspace(200, 2000, 500) * u.nm
    >>> fl = np.ones(500) * u.W / u.m**2 / u.nm
    >>> s = Spectrum(wl, fl)
    >>> print(s)
    Spectrum: 200.0–2000.0 nm, 500 points, integrated flux=1800.00 W / m2
    """

    def __init__(self, wavelengths, flux, uncertainty=None, label=None):
        # ── Validate and store wavelengths ───────────────────────────────────
        if not isinstance(wavelengths, u.Quantity):
            raise TypeError("wavelengths must be an astropy Quantity with length units.")
        self.wavelengths = wavelengths.to(WAVELENGTH_UNIT)

        # ── Validate and store flux ──────────────────────────────────────────
        if not isinstance(flux, u.Quantity):
            raise TypeError("flux must be an astropy Quantity.")
        self.flux = flux.to(SPECTRAL_IRRAD_UNIT)

        # ── Uncertainty defaults to zero ─────────────────────────────────────
        if uncertainty is None:
            self.uncertainty = np.zeros_like(flux.value) * SPECTRAL_IRRAD_UNIT
        else:
            self.uncertainty = uncertainty.to(SPECTRAL_IRRAD_UNIT)

        self.label = label or "Spectrum"

        # ── Sanity checks ────────────────────────────────────────────────────
        if self.wavelengths.shape != self.flux.shape:
            raise ValueError(
                f"wavelengths and flux must have the same shape. "
                f"Got {self.wavelengths.shape} and {self.flux.shape}."
            )

    # ── Representation ───────────────────────────────────────────────────────

    def __repr__(self):
        return (
            f"Spectrum(label='{self.label}', "
            f"wl={self.wavelengths[0]:.1f}–{self.wavelengths[-1]:.1f}, "
            f"n={len(self.wavelengths)})"
        )

    def __str__(self):
        return (
            f"Spectrum: {self.wavelengths[0]:.1f}–{self.wavelengths[-1]:.1f}, "
            f"{len(self.wavelengths)} points, "
            f"integrated flux={self.integrate():.2f}"
        )

    # ── Core methods ─────────────────────────────────────────────────────────

    def integrate(self):
        """
        Integrate I(lambda) over all wavelengths using the trapezoidal rule.

        Returns
        -------
        astropy Quantity in W/m²
        """
        return np.trapezoid(self.flux, self.wavelengths).to(u.W / u.m**2)

    def slice(self, wl_min, wl_max):
        """
        Return a new Spectrum trimmed to [wl_min, wl_max].

        Parameters
        ----------
        wl_min, wl_max : astropy Quantity (length units)
        """
        wl_min = wl_min.to(WAVELENGTH_UNIT)
        wl_max = wl_max.to(WAVELENGTH_UNIT)
        mask = (self.wavelengths >= wl_min) & (self.wavelengths <= wl_max)
        return Spectrum(
            self.wavelengths[mask],
            self.flux[mask],
            self.uncertainty[mask],
            label=f"{self.label} [{wl_min:.0f}–{wl_max:.0f}]"
        )

    def interpolate(self, new_wavelengths):
        """
        Interpolate spectrum onto a new wavelength grid.

        Parameters
        ----------
        new_wavelengths : astropy Quantity (length units)

        Returns
        -------
        Spectrum on the new grid.
        """
        new_wl = new_wavelengths.to(WAVELENGTH_UNIT)
        new_flux = np.interp(
            new_wl.value,
            self.wavelengths.value,
            self.flux.value
        ) * SPECTRAL_IRRAD_UNIT
        new_unc = np.interp(
            new_wl.value,
            self.wavelengths.value,
            self.uncertainty.value
        ) * SPECTRAL_IRRAD_UNIT
        return Spectrum(new_wl, new_flux, new_unc, label=self.label)

    def to_dict(self):
        """Return wavelengths, flux, uncertainty as plain numpy arrays (no units)."""
        return {
            "wavelengths_nm": self.wavelengths.value,
            "flux_W_m2_nm":   self.flux.value,
            "uncertainty_W_m2_nm": self.uncertainty.value,
            "label": self.label,
        }
