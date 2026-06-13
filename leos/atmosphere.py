"""
leos.atmosphere
---------------
AtmosphericProfile: typed container for atmospheric properties of a
planetary body. Provides preset profiles for Earth, Mars, and Moon
based on published literature values.

These profiles feed directly into the radiative transfer module.
"""

from dataclasses import dataclass, field
from typing import Optional
from astropy import units as u


@dataclass
class AtmosphericProfile:
    """
    Atmospheric properties for a planetary body.

    Parameters
    ----------
    body : str
        Body name, e.g. 'mars'.
    surface_pressure : astropy Quantity (pressure)
        Mean surface pressure.
    scale_height : astropy Quantity (length)
        Atmospheric scale height H = kT/mg.
    dust_tau : float
        Visible-band dust optical depth (dimensionless).
        0.0 = no dust. Typical Mars: 0.3–1.0. Dust storm: >3.0.
    angstrom_exponent : float
        Wavelength dependence of dust opacity: τ(λ) = τ_vis*(λ/550nm)^-α.
        Typical Mars: 1.0. Earth aerosols: 1.3–1.6.
    single_scatter_albedo : float
        Fraction of extinction due to scattering (vs absorption).
        1.0 = pure scattering. Mars dust: ~0.93.
    composition : dict
        Major atmospheric constituents as volume fractions.
        e.g. {'CO2': 0.953, 'N2': 0.027, 'Ar': 0.016}
    has_atmosphere : bool
        False for airless bodies (Moon, Mercury).
    label : str
        Human-readable label.

    Examples
    --------
    >>> profile = AtmosphericProfile.mars(dust_tau=0.5)
    >>> print(profile)
    AtmosphericProfile: mars (dust τ=0.50, P=636.0 Pa)
    """

    body                 : str
    surface_pressure     : u.Quantity
    scale_height         : u.Quantity
    dust_tau             : float
    angstrom_exponent    : float
    single_scatter_albedo: float
    composition          : dict
    has_atmosphere       : bool = True
    label                : str  = ""

    def __post_init__(self):
        self.body = self.body.lower().strip()
        if not self.label:
            self.label = self.body

    def __str__(self):
        if not self.has_atmosphere:
            return f"AtmosphericProfile: {self.body} (airless)"
        return (
            f"AtmosphericProfile: {self.body} "
            f"(dust τ={self.dust_tau:.2f}, "
            f"P={self.surface_pressure:.1f})"
        )

    def __repr__(self):
        return (
            f"AtmosphericProfile(body='{self.body}', "
            f"dust_tau={self.dust_tau}, "
            f"P={self.surface_pressure})"
        )

    # ── Dust opacity at wavelength ────────────────────────────────────────────

    def dust_tau_at(self, wavelength_nm: float) -> float:
        """
        Compute dust optical depth at a given wavelength using
        the Ångström exponent.

            τ(λ) = τ_vis * (λ / 550 nm)^(-α)

        Parameters
        ----------
        wavelength_nm : float — wavelength in nm

        Returns
        -------
        float — optical depth at that wavelength
        """
        if not self.has_atmosphere or self.dust_tau == 0.0:
            return 0.0
        return self.dust_tau * (wavelength_nm / 550.0) ** (-self.angstrom_exponent)

    # ── Preset profiles ───────────────────────────────────────────────────────

    @classmethod
    def earth(cls, aod_550=0.1):
        """
        Standard Earth atmosphere (mid-latitude, clear sky).

        Parameters
        ----------
        aod_550 : float
            Aerosol optical depth at 550 nm. Default 0.1 (clean).
            Urban/polluted: 0.3–0.8.

        Reference: US Standard Atmosphere 1976.
        """
        return cls(
            body                  = "earth",
            surface_pressure      = 101325.0 * u.Pa,
            scale_height          = 8.5 * u.km,
            dust_tau              = aod_550,
            angstrom_exponent     = 1.3,
            single_scatter_albedo = 0.95,
            composition           = {
                "N2" : 0.7809,
                "O2" : 0.2095,
                "Ar" : 0.0093,
                "CO2": 0.0004,
            },
            has_atmosphere        = True,
            label                 = f"Earth standard (AOD={aod_550})",
        )

    @classmethod
    def mars(cls, dust_tau=0.5, season="average"):
        """
        Mars atmosphere.

        Parameters
        ----------
        dust_tau : float
            Visible dust optical depth at 550 nm.
            Clear: 0.3. Average: 0.5. Dusty: 1.0. Storm: >3.0.
        season : str
            'average' | 'clear' | 'dusty' | 'storm'
            Overrides dust_tau if set to non-average.

        Reference: Forget et al. 1999, Mars Climate Database v6.
        """
        season_tau = {
            "clear"  : 0.3,
            "average": 0.5,
            "dusty"  : 1.0,
            "storm"  : 4.0,
        }
        if season != "average":
            dust_tau = season_tau.get(season, dust_tau)

        return cls(
            body                  = "mars",
            surface_pressure      = 636.0 * u.Pa,
            scale_height          = 11.1 * u.km,
            dust_tau              = dust_tau,
            angstrom_exponent     = 1.0,
            single_scatter_albedo = 0.93,
            composition           = {
                "CO2": 0.953,
                "N2" : 0.027,
                "Ar" : 0.016,
                "O2" : 0.001,
                "CO" : 0.001,
            },
            has_atmosphere        = True,
            label                 = f"Mars ({season}, τ={dust_tau})",
        )

    @classmethod
    def moon(cls):
        """
        Moon — airless body, no atmosphere.

        Reference: Stern 1999, RoG.
        """
        return cls(
            body                  = "moon",
            surface_pressure      = 3e-10 * u.Pa,
            scale_height          = 0.0 * u.km,
            dust_tau              = 0.0,
            angstrom_exponent     = 0.0,
            single_scatter_albedo = 0.0,
            composition           = {},
            has_atmosphere        = False,
            label                 = "Moon (airless)",
        )
