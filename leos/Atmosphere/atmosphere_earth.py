"""
leos.atmosphere_earth
----------------------
Earth-specific atmospheric column data: altitude-resolved T(z), P(z),
n(z), and per-species volume mixing ratios vmr_i(z).

Two tiers of data:

Tier 1 (always available, no download)
    AtmosphericColumn.us_standard_1976() — the 1976 US Standard
    Atmosphere (piecewise-linear T(z), hydrostatic P(z), analytically
    defined, public domain) for the bulk well-mixed gases (N2, O2,
    Ar, CO2), plus a bundled approximate McPeters/Labow-style O3
    mixing-ratio climatology (mid-latitude annual mean).

Tier 2 (user-supplied, real conditions for a specific place/time)
    AtmosphericColumn.from_npz(path) — loads the output of
    scripts/convert_merra2.py or scripts/convert_era5.py, giving the
    actual reanalysis-derived T(z), P(z), and vmr_H2O(z)/vmr_O3(z)
    for a specific lat/lon/date.

In both cases, AtmosphericColumn provides:
    column_density(species)     — Ni = integral(xi(z)*n(z) dz), no
                                   isothermal/single-T assumption
    scale_height_at(z_km)        — local H(z) = kB*T(z)/(m_bar(z)*g)
    effective_scale_height()     — density-weighted H, for Chapman
    to_profile(**overrides)      — AtmosphericProfile for the
                                   radiative transfer module

If neither tier applies (e.g. a body/time atmosphere_earth.py cannot
describe), callers should fall back to AtmosphericProfile.earth()
in atmosphere.py, which uses literature-constant scale heights and
the hydrostatic well-mixed approximation.
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
from astropy import units as u

from .atmosphere import AtmosphericProfile

# ── Physical constants ────────────────────────────────────────────────────────
_kB     = 1.380649e-23   # J/K
_g0     = 9.80665        # m/s^2, standard gravity
_R_EARTH_KM = 6371.0

# Molecular masses (kg), shared with AtmosphericProfile._MOLECULAR_MASS_KG
_MOLECULAR_MASS_KG = AtmosphericProfile._MOLECULAR_MASS_KG


# ══════════════════════════════════════════════════════════════════════════════
# Source registry
# ══════════════════════════════════════════════════════════════════════════════

class AtmosphericSource(Enum):
    """Provenance of an AtmosphericColumn."""
    US_STD_1976    = "us_std_1976"     # bundled, no download
    MCPETERS_LABOW = "mcpeters_labow"  # bundled O3 climatology component
    MERRA2         = "merra2"          # scripts/convert_merra2.py output
    ERA5           = "era5"            # scripts/convert_era5.py output
    CUSTOM         = "custom"          # user-supplied npz/arrays


ATMO_REGISTRY = {
    AtmosphericSource.US_STD_1976: {
        "description": "US Standard Atmosphere 1976 (piecewise-linear T(z), "
                        "hydrostatic P(z), 0-86 km). Well-mixed gases only "
                        "(N2, O2, Ar, CO2). Bundled, no download.",
        "reference": "NOAA-S/T 76-1562, U.S. Standard Atmosphere, 1976.",
        "requires_download": False,
    },
    AtmosphericSource.MCPETERS_LABOW: {
        "description": "Approximate mid-latitude annual-mean O3 volume "
                        "mixing ratio profile, bundled as a small table "
                        "and combined with US_STD_1976 number density to "
                        "give an O3 column ~300 DU. For accurate, "
                        "date/location-specific O3, use MERRA2 or ERA5 "
                        "instead.",
        "reference": "McPeters & Labow (2012), J. Geophys. Res. (style "
                      "climatology; values here are an illustrative "
                      "approximation, not the original dataset).",
        "requires_download": False,
    },
    AtmosphericSource.MERRA2: {
        "description": "NASA MERRA-2 reanalysis column, converted via "
                        "scripts/convert_merra2.py.",
        "reference": "Gelaro et al. (2017), J. Climate. "
                      "doi:10.1175/JCLI-D-16-0758.1",
        "requires_download": True,
    },
    AtmosphericSource.ERA5: {
        "description": "ECMWF ERA5 reanalysis column, converted via "
                        "scripts/convert_era5.py.",
        "reference": "Hersbach et al. (2020), Q. J. R. Meteorol. Soc. "
                      "doi:10.1002/qj.3803",
        "requires_download": True,
    },
    AtmosphericSource.CUSTOM: {
        "description": "User-supplied column arrays.",
        "reference": "User-supplied data.",
        "requires_download": False,
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# Bundled US Standard Atmosphere 1976 layer table
# ══════════════════════════════════════════════════════════════════════════════
# (base geopotential height [km], base temperature [K],
#  lapse rate [K/km], base pressure [Pa])
_US76_LAYERS = [
    (0.0,  288.15, -6.5,  101325.0),
    (11.0, 216.65,  0.0,   22632.06),
    (20.0, 216.65,  1.0,    5474.889),
    (32.0, 228.65,  2.8,     868.0187),
    (47.0, 270.65,  0.0,     110.9063),
    (51.0, 270.65, -2.8,      66.93887),
    (71.0, 214.65, -2.0,       3.956420),
]
_US76_TOP_KM = 86.0

# Universal gas constant and mean molar mass used by the 1976 standard
_R_STAR  = 8.31432    # J/(mol K)
_M0      = 0.0289644  # kg/mol

# Well-mixed bulk composition, constant with altitude below ~86 km
_US76_COMPOSITION = {
    "N2" : 0.78084,
    "O2" : 0.20946,
    "Ar" : 0.00934,
    "CO2": 0.000420,   # ~420 ppm, modern value
}

# ── Bundled approximate O3 volume mixing ratio profile (mid-latitude, ──────────
# ── annual-mean style; McPeters & Labow climatology approximation) ────────────
# Values in ppmv (volume mixing ratio x 1e6). Linearly interpolated; zero
# outside this range. Integrated against US76 number density this gives
# a column of order ~300 DU (see _verify_o3_column below).
_O3_VMR_TABLE_KM  = np.array(
    [0, 5, 10, 13, 15, 18, 20, 23, 25, 28, 30, 35, 40, 45, 50, 60, 70, 80, 86]
)
_O3_VMR_TABLE_PPM = np.array(
    [0.02, 0.03, 0.05, 0.15, 0.40, 1.80, 3.50, 6.50, 8.00, 8.30,
     8.00, 5.50, 3.00, 1.40, 0.60, 0.15, 0.04, 0.01, 0.00]
)


# ══════════════════════════════════════════════════════════════════════════════
# AtmosphericColumn
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AtmosphericColumn:
    """
    Altitude-resolved atmospheric column: T(z), P(z), n(z), vmr_i(z).

    Parameters
    ----------
    z_km : np.ndarray
        Altitude grid, ascending, km.
    T_K : np.ndarray
        Temperature, K.
    P_Pa : np.ndarray
        Pressure, Pa.
    vmr : dict[str, np.ndarray]
        Volume mixing ratio profiles, species -> array (same length
        as z_km). Species not present are treated as zero.
    n_total : np.ndarray, optional
        Total number density, molecules/m^3. Computed from P_Pa, T_K
        via the ideal gas law if not provided.
    sigma_T : np.ndarray, optional
        1-sigma temperature uncertainty, K.
    sigma_vmr : dict[str, np.ndarray], optional
        1-sigma VMR uncertainty per species.
    lat, lon : float, optional
        Location, degrees.
    time_iso : str, optional
        Observation time / averaging window, ISO-ish string.
    source : AtmosphericSource
        Provenance.
    body : str
        Body name, default 'earth'.

    Notes
    -----
    All quantities here are exact given the input table — no
    isothermal or single-temperature assumption is used. Column
    densities are trapezoidal integrals of xi(z)*n(z) over z; scale
    heights are computed locally from T(z) and the mixture's mean
    molecular mass at that altitude.
    """

    z_km      : np.ndarray
    T_K       : np.ndarray
    P_Pa      : np.ndarray
    vmr       : dict = field(default_factory=dict)
    n_total   : Optional[np.ndarray] = None
    sigma_T   : Optional[np.ndarray] = None
    sigma_vmr : dict = field(default_factory=dict)
    lat       : Optional[float] = None
    lon       : Optional[float] = None
    time_iso  : Optional[str] = None
    source    : AtmosphericSource = AtmosphericSource.CUSTOM
    body      : str = "earth"

    def __post_init__(self):
        self.z_km = np.asarray(self.z_km, dtype=float)
        self.T_K  = np.asarray(self.T_K,  dtype=float)
        self.P_Pa = np.asarray(self.P_Pa, dtype=float)

        # Sort by altitude ascending — downstream integration assumes this.
        order = np.argsort(self.z_km)
        if not np.all(order == np.arange(len(order))):
            self.z_km = self.z_km[order]
            self.T_K  = self.T_K[order]
            self.P_Pa = self.P_Pa[order]
            if self.n_total is not None:
                self.n_total = np.asarray(self.n_total)[order]
            if self.sigma_T is not None:
                self.sigma_T = np.asarray(self.sigma_T)[order]
            for k in list(self.vmr):
                self.vmr[k] = np.asarray(self.vmr[k])[order]
            for k in list(self.sigma_vmr):
                self.sigma_vmr[k] = np.asarray(self.sigma_vmr[k])[order]

        if self.n_total is None:
            self.n_total = self.P_Pa / (_kB * self.T_K)
        else:
            self.n_total = np.asarray(self.n_total, dtype=float)

        for k in list(self.vmr):
            self.vmr[k] = np.asarray(self.vmr[k], dtype=float)

    def __str__(self):
        return (
            f"AtmosphericColumn: {self.body} | {self.source.value} | "
            f"{len(self.z_km)} levels, "
            f"{self.z_km[0]:.1f}-{self.z_km[-1]:.1f} km | "
            f"T {self.T_K.min():.1f}-{self.T_K.max():.1f} K"
        )

    def __repr__(self):
        return self.__str__()

    # ── Loading ──────────────────────────────────────────────────────────────

    @classmethod
    def from_npz(cls, path: str, body: str = "earth") -> "AtmosphericColumn":
        """
        Load a column produced by scripts/convert_merra2.py or
        scripts/convert_era5.py.

        Parameters
        ----------
        path : str
            Path to the .npz file.
        body : str
            Body name (default 'earth').

        Returns
        -------
        AtmosphericColumn
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Atmospheric column file not found: {path}")

        data = np.load(path, allow_pickle=True)

        vmr = {}
        sigma_vmr = {}
        for key in data.files:
            if key.startswith("vmr_"):
                species = key[len("vmr_"):]
                vmr[species] = data[key]
            elif key.startswith("sigma_vmr_"):
                species = key[len("sigma_vmr_"):]
                sigma_vmr[species] = data[key]

        source_str = str(data["source"][0]) if "source" in data.files else "custom"
        try:
            source = AtmosphericSource(source_str)
        except ValueError:
            source = AtmosphericSource.CUSTOM

        def _scalar(key):
            if key not in data.files:
                return None
            val = data[key]
            return val.item() if hasattr(val, "item") else val

        return cls(
            z_km      = data["z_km"],
            T_K       = data["T_K"],
            P_Pa      = data["P_Pa"],
            vmr       = vmr,
            n_total   = data["n_total"] if "n_total" in data.files else None,
            sigma_T   = data["sigma_T"] if "sigma_T" in data.files else None,
            sigma_vmr = sigma_vmr,
            lat       = _scalar("lat"),
            lon       = _scalar("lon"),
            time_iso  = str(data["time_iso"][0]) if "time_iso" in data.files else None,
            source    = source,
            body      = body,
        )

    @classmethod
    def us_standard_1976(
        cls,
        z_max_km: float = 86.0,
        dz_km: float = 1.0,
        include_o3: bool = True,
    ) -> "AtmosphericColumn":
        """
        Build the 1976 US Standard Atmosphere column.

        Parameters
        ----------
        z_max_km : float
            Top of the column, km. Must be <= 86 (model validity limit).
        dz_km : float
            Altitude grid spacing, km.
        include_o3 : bool
            If True, attach the bundled approximate O3 VMR climatology
            (see module docstring — for accurate O3 prefer
            from_npz() with MERRA-2/ERA5 data).

        Returns
        -------
        AtmosphericColumn
            source = AtmosphericSource.US_STD_1976 (or MCPETERS_LABOW
            if include_o3=True, to flag the O3 provenance distinctly).
        """
        z_max_km = min(z_max_km, _US76_TOP_KM)
        z_km = np.arange(0.0, z_max_km + dz_km, dz_km)
        z_km = z_km[z_km <= z_max_km]

        T_K  = np.empty_like(z_km)
        P_Pa = np.empty_like(z_km)

        for i, z in enumerate(z_km):
            # Find the layer containing z
            layer = _US76_LAYERS[0]
            for lay in _US76_LAYERS:
                if z >= lay[0]:
                    layer = lay
                else:
                    break
            h_b, T_b, L_b, P_b = layer
            T_z = T_b + L_b * (z - h_b)
            if abs(L_b) < 1e-12:
                P_z = P_b * np.exp(-_g0 * _M0 * (z - h_b) * 1000.0 / (_R_STAR * T_b))
            else:
                L_b_per_m = L_b / 1000.0
                P_z = P_b * (T_b / T_z) ** (_g0 * _M0 / (_R_STAR * L_b_per_m))
            T_K[i]  = T_z
            P_Pa[i] = P_z

        vmr = {sp: np.full_like(z_km, frac) for sp, frac in _US76_COMPOSITION.items()}

        source = AtmosphericSource.US_STD_1976
        if include_o3:
            vmr["O3"] = np.interp(
                z_km, _O3_VMR_TABLE_KM, _O3_VMR_TABLE_PPM,
                left=0.0, right=0.0,
            ) * 1e-6
            source = AtmosphericSource.MCPETERS_LABOW

        return cls(
            z_km=z_km, T_K=T_K, P_Pa=P_Pa, vmr=vmr,
            source=source, body="earth",
        )

    # ── Column densities ─────────────────────────────────────────────────────

    def column_density(self, species: str) -> float:
        """
        Vertical column density Ni = integral(xi(z) * n(z) dz),
        molecules/m^2. Trapezoidal integration over the native z grid.
        No isothermal or single-T assumption.

        Returns 0.0 if species not present in self.vmr.
        """
        if species not in self.vmr:
            return 0.0
        z_m = self.z_km * 1e3
        return float(np.trapezoid(self.vmr[species] * self.n_total, z_m))

    # ── Scale heights ────────────────────────────────────────────────────────

    def _mean_molecular_mass_at(self, idx: int) -> float:
        """Mean molecular mass [kg] at level idx, from vmr at that level."""
        total_mass, total_frac = 0.0, 0.0
        for species, arr in self.vmr.items():
            mi = _MOLECULAR_MASS_KG.get(species)
            if mi is None:
                continue
            xi = arr[idx]
            total_mass += xi * mi
            total_frac += xi
        if total_frac == 0.0:
            # Fall back to dry-air mean mass if no recognised species
            return _M0 / 6.02214076e23
        return total_mass / total_frac

    def _gravity_at(self, z_km: float) -> float:
        """g(z) = g0 * (Re/(Re+z))^2."""
        return _g0 * (_R_EARTH_KM / (_R_EARTH_KM + z_km)) ** 2

    def scale_height_at(self, z_km: float) -> u.Quantity:
        """
        Local scale height H(z) = kB*T(z) / (m_bar(z)*g(z)).

        Parameters
        ----------
        z_km : float
            Altitude, km. Interpolated within the column's z grid.

        Returns
        -------
        astropy Quantity (km)
        """
        T_z = float(np.interp(z_km, self.z_km, self.T_K))

        # Interpolate vmr at z_km for each species, then mean mass
        total_mass, total_frac = 0.0, 0.0
        for species, arr in self.vmr.items():
            mi = _MOLECULAR_MASS_KG.get(species)
            if mi is None:
                continue
            xi = float(np.interp(z_km, self.z_km, arr))
            total_mass += xi * mi
            total_frac += xi
        m_bar = total_mass / total_frac if total_frac > 0 else _M0 / 6.02214076e23

        g_z = self._gravity_at(z_km)
        H_m = _kB * T_z / (m_bar * g_z)
        return (H_m * u.m).to(u.km)

    def effective_scale_height(self, weight: str = "density") -> u.Quantity:
        """
        Density-weighted effective scale height, for use as the single
        H in Chapman's grazing-incidence formula (atmosphere.py
        chapman_airmass).

            H_eff = integral(n(z)*H(z) dz) / integral(n(z) dz)

        Parameters
        ----------
        weight : str
            Currently only 'density' is implemented (weights by total
            number density n(z), appropriate for Rayleigh scattering
            and well-mixed-gas absorbers, which are concentrated near
            the surface). For absorbers with a different vertical
            distribution (e.g. ozone, peaked near 25 km), this single
            H is an approximation — see chapman_airmass docstring.

        Returns
        -------
        astropy Quantity (km)
        """
        if weight != "density":
            raise ValueError(f"Unknown weight scheme: {weight!r}")

        H_z = np.array([self.scale_height_at(z).to(u.km).value for z in self.z_km])
        z_m = self.z_km * 1e3

        numerator   = np.trapezoid(self.n_total * H_z, z_m)
        denominator = np.trapezoid(self.n_total, z_m)
        return (numerator / denominator) * u.km

    # ── Bridge to AtmosphericProfile ─────────────────────────────────────────

    def to_profile(self, **overrides) -> AtmosphericProfile:
        """
        Build an AtmosphericProfile for the radiative transfer module,
        using this column's surface pressure, effective scale height,
        surface composition, and integrated column densities.

        Parameters
        ----------
        **overrides
            Any AtmosphericProfile field may be overridden, e.g.
            dust_tau=0.2, angstrom_exponent=1.4, label="MERRA-2 Jezero".

        Returns
        -------
        AtmosphericProfile

        Notes
        -----
        column_densities is populated for every species present in
        self.vmr using this column's own integration (column_density()),
        which bypasses AtmosphericProfile's well-mixed restriction —
        appropriate here since these are genuine integrated columns,
        not hydrostatic estimates from a single surface value.
        """
        surface_idx = 0  # z_km is sorted ascending; index 0 = surface

        composition = {
            sp: float(arr[surface_idx])
            for sp, arr in self.vmr.items()
            if sp in AtmosphericProfile._WELL_MIXED_SPECIES
        }
        column_densities = {
            sp: self.column_density(sp) for sp in self.vmr
        }

        defaults = dict(
            body                       = self.body,
            surface_pressure           = self.P_Pa[surface_idx] * u.Pa,
            scale_height               = self.effective_scale_height(),
            dust_tau                   = 0.1,
            angstrom_exponent          = 1.3,
            single_scatter_albedo      = 0.95,
            composition                = composition,
            has_atmosphere             = True,
            column_densities           = column_densities,
            include_rayleigh           = True,
            rayleigh_king_factor       = 1.048,
            effective_refractive_index = 1.0002926,
            label                      = f"{self.body} ({self.source.value})",
        )
        defaults.update(overrides)
        return AtmosphericProfile(**defaults)
