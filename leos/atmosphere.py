"""
leos.atmosphere
---------------
AtmosphericProfile: typed container for atmospheric properties of a
planetary body. Provides preset profiles for Earth, Mars, and Moon
based on published literature values.

These profiles feed directly into the radiative transfer module.
"""

import numpy as np
from scipy.special import erfc, erfcx
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
        0.0 = no dust. Typical Mars: 0.3-1.0. Dust storm: >3.0.
    angstrom_exponent : float
        Wavelength dependence of dust opacity:
        tau(lambda) = tau_vis * (lambda/550nm)^-alpha.
        Typical Mars: 1.0. Earth aerosols: 1.3-1.6.
    single_scatter_albedo : float
        Fraction of extinction due to scattering vs absorption.
        1.0 = pure scattering. Mars dust: ~0.93.
    composition : dict
        Major atmospheric constituents as volume fractions.
        e.g. {'CO2': 0.953, 'N2': 0.027, 'Ar': 0.016}
    has_atmosphere : bool
        False for airless bodies (Moon, Mercury).
    column_densities : dict
        Pre-computed vertical column densities Ni in molecules/m^2.
        If empty, column_density_for() computes from composition
        and pressure.
    include_rayleigh : bool
        Compute Rayleigh scattering from first principles
        (sigma ~ lambda^-4). Set False for airless bodies.
    rayleigh_king_factor : float
        Anisotropy correction for Rayleigh scattering.
        Air/N2-dominated: 1.048.
        CO2-dominated: 1.15 (Mars, Venus).
        Pure N2: 1.034.
    effective_refractive_index : float
        Real part of refractive index of the gas mixture at STP
        at 550 nm.
        Air (Earth): 1.0002926.
        CO2 (Mars):  1.0004493.
        Vacuum/Moon: 1.0.
    label : str
        Human-readable label.

    Examples
    --------
    >>> profile = AtmosphericProfile.mars(dust_tau=0.5)
    >>> print(profile)
    AtmosphericProfile: mars (dust tau=0.50, P=636.0 Pa)
    """

    body                      : str
    surface_pressure          : u.Quantity
    scale_height              : u.Quantity
    dust_tau                  : float
    angstrom_exponent         : float
    single_scatter_albedo     : float
    composition               : dict
    has_atmosphere            : bool  = True
    column_densities          : dict  = field(default_factory=dict)
    include_rayleigh          : bool  = True
    rayleigh_king_factor      : float = 1.048
    effective_refractive_index: float = 1.0002926
    label                     : str   = ""

    def __post_init__(self):
        self.body = self.body.lower().strip()
        if not self.label:
            self.label = self.body

    def __str__(self):
        if not self.has_atmosphere:
            return f"AtmosphericProfile: {self.body} (airless)"
        return (
            f"AtmosphericProfile: {self.body} "
            f"(dust tau={self.dust_tau:.2f}, "
            f"P={self.surface_pressure:.1f})"
        )

    def __repr__(self):
        return (
            f"AtmosphericProfile(body='{self.body}', "
            f"dust_tau={self.dust_tau}, "
            f"P={self.surface_pressure})"
        )

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_airless(self) -> bool:
        """True for bodies with no meaningful atmosphere."""
        return not self.has_atmosphere

    @property
    def dust_tau_uncertainty_fraction(self) -> float:
        """
        Fractional 1-sigma uncertainty on dust optical depth tau.
        Used by radiative_transfer.py for Beer-Lambert error
        propagation.

        Sources
        -------
        Mars  : MEDA/MCS retrievals, 20-25% season-to-season.
        Earth : AERONET climatology, ~10%.
        Moon  : airless, zero.
        """
        _fractions = {
            "earth": 0.10,
            "mars" : 0.25,
            "moon" : 0.00,
        }
        return _fractions.get(self.body, 0.15)

    # ── Dust opacity ──────────────────────────────────────────────────────────

    def dust_tau_at(self, wavelength_nm: float) -> float:
        """
        Dust optical depth at a given wavelength via Angstrom law.

            tau(lambda) = tau_vis * (lambda / 550 nm)^(-alpha)

        Parameters
        ----------
        wavelength_nm : float
            Wavelength in nm.

        Returns
        -------
        float
            Optical depth at that wavelength.
        """
        if not self.has_atmosphere or self.dust_tau == 0.0:
            return 0.0
        return self.dust_tau * (wavelength_nm / 550.0) ** (
            -self.angstrom_exponent
        )

    # ── Column density ────────────────────────────────────────────────────────

    def column_density_for(self, species: str) -> float:
        """
        Vertical column density Ni (molecules/m^2) for a species.

        Uses pre-computed value from self.column_densities if
        available. Otherwise derives from surface pressure and
        volume mixing ratio assuming a well-mixed hydrostatic
        atmosphere:

            Ni = xi * P / (mi * g)

        Parameters
        ----------
        species : str
            e.g. 'CO2', 'O3', 'H2O'

        Returns
        -------
        float
            Column density in molecules/m^2.
        """
        if species in self.column_densities:
            return self.column_densities[species]

        xi = self.composition.get(species, 0.0)
        if xi == 0.0:
            return 0.0

        _molecular_mass_kg = {
            "CO2": 7.308e-26,
            "N2" : 4.652e-26,
            "O2" : 5.314e-26,
            "Ar" : 6.634e-26,
            "H2O": 2.991e-26,
            "O3" : 7.972e-26,
            "CO" : 4.652e-26,
            "CH4": 2.664e-26,
            "NO" : 4.983e-26,
            "NO2": 7.641e-26,
            "SO2": 1.062e-25,
        }
        _surface_gravity = {
            "earth" : 9.807,
            "mars"  : 3.721,
            "moon"  : 1.620,
            "venus" : 8.870,
            "titan" : 1.352,
            "europa": 1.315,
        }

        mi = _molecular_mass_kg.get(species)
        if mi is None:
            raise ValueError(
                f"Molecular mass not known for '{species}'. "
                f"Add to _molecular_mass_kg or supply "
                f"column_densities dict."
            )

        g = _surface_gravity.get(self.body, 9.807)
        P = self.surface_pressure.to(u.Pa).value

        return (xi * P) / (mi * g)

    # ── Rayleigh scattering ───────────────────────────────────────────────────

    def rayleigh_tau(self, wavelength_nm: float) -> float:
        """
        Total Rayleigh scattering optical depth at a given wavelength.

        Uses the King-factor corrected formula:

            sigma_ray(lambda) = (24 pi^3 / (N^2 lambda^4))
                                * ((n^2-1)/(n^2+2))^2
                                * F_k

        where N is the standard number density at STP, n is the
        effective refractive index, and F_k is the King factor.

        Then tau_ray = sigma_ray(lambda) * N_total_column.

        Parameters
        ----------
        wavelength_nm : float
            Wavelength in nm.

        Returns
        -------
        float
            Rayleigh optical depth (dimensionless).

        References
        ----------
        Bodhaine et al. (1999). On Rayleigh optical depth
        calculations. J. Atmos. Ocean. Tech. 16, 1854-1861.
        """
        if not self.has_atmosphere or not self.include_rayleigh:
            return 0.0

        wl_m  = wavelength_nm * 1e-9
        N_stp = 2.6867e25
        n     = self.effective_refractive_index
        n2    = n ** 2

        sigma_ray = (
            (24.0 * np.pi**3) / (N_stp**2 * wl_m**4)
            * ((n2 - 1.0) / (n2 + 2.0))**2
            * self.rayleigh_king_factor
        )

        _surface_gravity = {
            "earth": 9.807,
            "mars" : 3.721,
            "moon" : 1.620,
            "venus": 8.870,
            "titan": 1.352,
        }
        _molecular_mass_kg = {
            "CO2": 7.308e-26,
            "N2" : 4.652e-26,
            "O2" : 5.314e-26,
            "Ar" : 6.634e-26,
            "H2O": 2.991e-26,
        }

        g = _surface_gravity.get(self.body, 9.807)
        P = self.surface_pressure.to(u.Pa).value

        if self.composition:
            m_mean = sum(
                self.composition.get(sp, 0.0)
                * _molecular_mass_kg.get(sp, 4.8e-26)
                for sp in _molecular_mass_kg
            )
            if m_mean == 0.0:
                m_mean = 4.8e-26
        else:
            m_mean = 4.8e-26

        N_column = P / (m_mean * g)

        return sigma_ray * N_column

    # ── Chapman airmass ───────────────────────────────────────────────────────

    def chapman_airmass(self, sza_deg: float) -> float:
        """
        Chapman grazing-incidence function for a spherical
        exponential atmosphere.

        Parameters
        ----------
        sza_deg : float
            Solar zenith angle in degrees.

        Returns
        -------
        float
            Airmass factor m(chi). Returns inf for SZA >= 90 deg.

        Notes
        -----
        Reduces to sec(SZA) for small zenith angles and remains
        finite near the horizon via the erfc term.
        Returns 1.0 for airless bodies regardless of SZA.

        References
        ----------
        Chapman, S. (1931). Proc. Phys. Soc. 43, 483.
        Smith & Smith (1972). J. Geophys. Res. 77, 3592-3597.
        """
        if not self.has_atmosphere:
            return 1.0

        if sza_deg > 90.0:
            return np.inf

        _radii_km = {
            "earth": 6371.0,
            "mars" : 3390.0,
            "moon" : 1737.0,
            "venus": 6051.0,
            "titan": 2575.0,
        }
        R = _radii_km.get(self.body, 6371.0)
        H = self.scale_height.to(u.km).value

        if H <= 0.0:
            return 1.0

        chi = np.deg2rad(sza_deg)
        mu  = np.cos(chi)

        # Plane-parallel limit — exact below 75 deg
        if sza_deg < 75.0:
            return 1.0 / mu

        # Chapman function for spherical exponential atmosphere
        # (Smith & Smith 1972): Ch(X,chi) = sqrt(pi*X/2) * erfcx(mu*sqrt(X/2))
        # erfcx(z) = exp(z^2)*erfc(z), used for numerical stability —
        # avoids exp() overflow for large z (grazing incidence).
        # Ch -> 1 as chi -> 0, Ch -> sqrt(pi*X/2) (finite) as chi -> 90 deg,
        # unlike the diverging plane-parallel sec(chi).
        X = R / H
        z = mu * np.sqrt(X / 2.0)
        chapman = np.sqrt(np.pi * X / 2.0) * erfcx(z)

        return max(chapman, 1.0)

    # ── Preset profiles ───────────────────────────────────────────────────────

    @classmethod
    def earth(cls, aod_550=0.1):
        """
        Standard Earth atmosphere (mid-latitude, clear sky).

        Parameters
        ----------
        aod_550 : float
            Aerosol optical depth at 550 nm.
            Default 0.1 (clean). Urban/polluted: 0.3-0.8.

        Reference: US Standard Atmosphere 1976.
        """
        return cls(
            body                       = "earth",
            surface_pressure           = 101325.0 * u.Pa,
            scale_height               = 8.5 * u.km,
            dust_tau                   = aod_550,
            angstrom_exponent          = 1.3,
            single_scatter_albedo      = 0.95,
            composition                = {
                "N2" : 0.7809,
                "O2" : 0.2095,
                "Ar" : 0.0093,
                "CO2": 0.0004,
            },
            has_atmosphere             = True,
            column_densities           = {},
            include_rayleigh           = True,
            rayleigh_king_factor       = 1.048,
            effective_refractive_index = 1.0002926,
            label                      = f"Earth standard (AOD={aod_550})",
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
            Overrides dust_tau if not 'average'.

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
            body                       = "mars",
            surface_pressure           = 636.0 * u.Pa,
            scale_height               = 11.1 * u.km,
            dust_tau                   = dust_tau,
            angstrom_exponent          = 1.0,
            single_scatter_albedo      = 0.93,
            composition                = {
                "CO2": 0.953,
                "N2" : 0.027,
                "Ar" : 0.016,
                "O2" : 0.001,
                "CO" : 0.001,
            },
            has_atmosphere             = True,
            column_densities           = {
                "CO2": 2.22e27,
                "N2" : 6.30e25,
                "Ar" : 3.73e25,
            },
            include_rayleigh           = True,
            rayleigh_king_factor       = 1.15,
            effective_refractive_index = 1.0004493,
            label                      = f"Mars ({season}, tau={dust_tau})",
        )

    @classmethod
    def moon(cls):
        """
        Moon — airless body, no atmosphere.

        Reference: Stern 1999, RoG.
        """
        return cls(
            body                       = "moon",
            surface_pressure           = 3e-10 * u.Pa,
            scale_height               = 0.0 * u.km,
            dust_tau                   = 0.0,
            angstrom_exponent          = 0.0,
            single_scatter_albedo      = 0.0,
            composition                = {},
            has_atmosphere             = False,
            column_densities           = {},
            include_rayleigh           = False,
            rayleigh_king_factor       = 1.0,
            effective_refractive_index = 1.0,
            label                      = "Moon (airless)",
        )
