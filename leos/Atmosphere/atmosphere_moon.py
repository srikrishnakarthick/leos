"""
leos.atmosphere_moon
---------------------
Moon-specific surface conditions: surface temperature T_surf, terrain
properties (elevation, slope, aspect), and illumination fraction.

The Moon has no bound atmosphere (surface pressure ~3e-10 Pa). There
is no Rayleigh scattering, no gas absorption, no dust optical depth.
The scientifically interesting quantities for solar irradiance at the
Moon are:

  1. Surface temperature         — controls thermal emission
  2. Solar zenith angle          — controls direct irradiance
  3. Terrain shadowing           — slope, aspect, horizon mask
  4. Albedo                      — controls reflected fraction

Three tiers:

Tier 1 — Analytical (always available, no download)
    MoonSurfaceConditions.from_analytical(lat, lon, sza, albedo)
    Bundled T(lat, SZA) model from Diviner/LRO climatology
    (Paige et al. 2010, Williams et al. 2017).
    Three regimes:
      Dayside  : T = T_ss(lat) * cos(SZA)^0.25
      Nightside: T = T_night(lat)  [from Diviner annual mean]
      PSR      : T = 40 K  [permanently shadowed regions, |lat|>85]

Tier 2 — LOLA GeoTIFF (user-supplied, requires download)
    MoonSurfaceConditions.from_npz(path)
    Loads output of scripts/convert_lola.py, which reads a LOLA
    GeoTIFF and computes elevation, slope, aspect, horizon mask,
    and illumination fraction for a given lat/lon point.

    Recommended source (GeoTIFF, pixel-registered, MOON_PA DE421):
      LOLA 64ppd Global Shape Model GeoTIFF (~683 MB)
      https://pgda.gsfc.nasa.gov/products/95
    Fallback (NetCDF, older PDS products, gridline-registered):
      LOLA 4ppd PDS product (ldam_4_float) via PDS Geosciences Node
      https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/

    NOTE: PDS3 .IMG binary format is NOT supported. Convert to
    GeoTIFF first using GDAL:
      gdal_translate -of GTiff LDEM_4.IMG LDEM_4.tif

Tier 3 — Custom (user-supplied arrays)
    MoonSurfaceConditions(tsurf_K=..., elevation_m=..., ...)

In all cases, MoonSurfaceConditions.to_profile() returns an
AtmosphericProfile configured for an airless body, with surface
temperature and terrain metadata attached for use by the radiative
transfer module.
"""

import os
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
from astropy import units as u

from .atmosphere import AtmosphericProfile

# ── Physical constants ────────────────────────────────────────────────────────
_STEFAN_BOLTZMANN = 5.670374419e-8   # W/m²/K⁴
_R_MOON_KM        = 1737.4           # km, reference radius


# ══════════════════════════════════════════════════════════════════════════════
# Source registry
# ══════════════════════════════════════════════════════════════════════════════

class MoonAtmosphericSource(Enum):
    ANALYTICAL    = "analytical"    # bundled Diviner climatology model
    LOLA_GEOTIFF  = "lola_geotiff"  # LOLA GeoTIFF via convert_lola.py
    LOLA_NETCDF   = "lola_netcdf"   # LOLA NetCDF via convert_lola.py
    CUSTOM        = "custom"        # user-supplied arrays


MOON_ATMO_REGISTRY = {
    MoonAtmosphericSource.ANALYTICAL: {
        "description": (
            "Bundled analytical surface temperature model from "
            "Diviner/LRO climatology. Three regimes: dayside "
            "T=T_ss(lat)*cos(SZA)^0.25, nightside T=T_night(lat) "
            "from Diviner annual mean tables, PSR T=40 K. "
            "No terrain shadowing. No download required."
        ),
        "suitability": [
            "Broadband irradiance at non-polar locations",
            "Quick estimates without terrain data",
            "Phase 1 validation",
        ],
        "limitations": (
            "No terrain slope or shadowing. Equatorial T accurate to "
            "~10-20 K; polar T less reliable without PSR mapping. "
            "Use LOLA GeoTIFF tier for polar/terrain work."
        ),
        "reference": (
            "Paige et al. (2010), Science 330, 479-482. "
            "Williams et al. (2017), Icarus 283, 300-325."
        ),
        "requires_download": False,
    },

    MoonAtmosphericSource.LOLA_GEOTIFF: {
        "description": (
            "LOLA elevation data from GeoTIFF, converted via "
            "scripts/convert_lola.py. Provides elevation, slope, "
            "aspect, and horizon mask for terrain shadowing. "
            "PRIMARY recommended format — pixel-registered, "
            "MOON_PA DE421 frame, rasterio-compatible."
        ),
        "suitability": [
            "Polar illumination and PSR mapping",
            "Terrain-corrected irradiance",
            "Landing site analysis",
            "Slope-dependent solar panel orientation",
        ],
        "limitations": (
            "Requires downloading LOLA GeoTIFF (~683 MB for 64ppd). "
            "Horizon mask computation is CPU-intensive for large radii. "
            "GeoTIFF only — PDS3 .IMG files not supported "
            "(convert with: gdal_translate -of GTiff input.IMG output.tif)."
        ),
        "reference": (
            "Smith et al. (2010), Science 329, 1072-1075. "
            "Barker et al. (2016), Icarus 273, 346-355. "
            "Source: https://pgda.gsfc.nasa.gov/products/95"
        ),
        "requires_download": True,
        "download_url": "https://pgda.gsfc.nasa.gov/products/95",
        "recommended_file": "LOLA 64ppd Global Shape Model GeoTIFF (~683 MB)",
    },

    MoonAtmosphericSource.LOLA_NETCDF: {
        "description": (
            "LOLA elevation data from NetCDF/GMT format, converted via "
            "scripts/convert_lola.py. FALLBACK format — gridline-registered, "
            "requires xarray. Use GeoTIFF (LOLA_GEOTIFF) when possible."
        ),
        "suitability": [
            "Same as LOLA_GEOTIFF",
            "Users who already have PDS NetCDF products",
        ],
        "limitations": (
            "Gridline-registered (vs pixel-registered GeoTIFF) — "
            "requires half-pixel correction for slope calculations. "
            "Requires xarray + netCDF4 packages. "
            "Older PDS products may have lower resolution."
        ),
        "reference": (
            "Same as LOLA_GEOTIFF. "
            "Source: https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/"
        ),
        "requires_download": True,
        "download_url": (
            "https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/"
        ),
    },

    MoonAtmosphericSource.CUSTOM: {
        "description": "User-supplied surface conditions.",
        "reference":   "User-supplied data.",
        "requires_download": False,
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# Bundled Diviner surface temperature climatology
# ══════════════════════════════════════════════════════════════════════════════
# Dayside subsolar temperature T_ss by latitude band [K]
# Source: Paige et al. (2010) Table 1, Williams et al. (2017) Fig. 3
# Latitude band centres: 0, 15, 30, 45, 60, 75, 85, 90 deg
_DIVINER_LAT_BANDS_DEG = np.array([0, 15, 30, 45, 60, 75, 85, 90],
                                   dtype=float)

# T_ss: peak subsolar temperature at each latitude band
# (subsolar point, SZA=0, equatorial noon)
_DIVINER_T_SS = np.array([395, 390, 375, 350, 300, 220, 120, 100],
                          dtype=float)   # K

# T_night: nightside mean temperature at each latitude band
# From Diviner annual mean (Paige et al. 2010, Williams et al. 2017)
_DIVINER_T_NIGHT = np.array([95, 93, 90, 85, 78, 68, 55, 40],
                              dtype=float)   # K

# Permanently shadowed region temperature (|lat| > 85 deg, PSR)
_T_PSR = 40.0   # K  (Paige et al. 2010 — coldest traps ~25-40 K)


def _diviner_tsurf(lat_deg: float, sza_deg: float,
                   albedo: float = 0.12) -> float:
    """
    Analytical surface temperature from Diviner climatology.

    Three regimes:
      Dayside  (SZA < 90°): T = T_ss(lat) * (1-A)^0.25 * cos(SZA)^0.25
      Nightside (SZA >= 90°): T = T_night(lat)
      PSR      (|lat| > 85°, always): T = T_PSR = 40 K

    Parameters
    ----------
    lat_deg : float   Planetocentric latitude [deg]
    sza_deg : float   Solar zenith angle [deg]
    albedo  : float   Bolometric Bond albedo. Default 0.12 (global mean,
                      Diviner; highlands ~0.12, maria ~0.08)

    Returns
    -------
    float   Surface temperature [K]
    """
    abs_lat = abs(lat_deg)

    # PSR override for deep polar regions
    if abs_lat >= 88.0 and sza_deg >= 90.0:
        return _T_PSR

    # Interpolate T_ss and T_night to this latitude
    t_ss    = float(np.interp(abs_lat, _DIVINER_LAT_BANDS_DEG, _DIVINER_T_SS))
    t_night = float(np.interp(abs_lat, _DIVINER_LAT_BANDS_DEG, _DIVINER_T_NIGHT))

    if sza_deg >= 90.0:
        return t_night

    # Dayside: radiative equilibrium scaling
    # T = T_ss * ((1-A)/0.88)^0.25 * cos(SZA)^0.25
    # (0.88 = 1 - 0.12, the reference albedo used to tabulate T_ss)
    albedo_factor = ((1.0 - albedo) / 0.88) ** 0.25
    cos_factor    = np.cos(np.deg2rad(sza_deg)) ** 0.25
    return t_ss * albedo_factor * cos_factor


# ══════════════════════════════════════════════════════════════════════════════
# MoonSurfaceConditions
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MoonSurfaceConditions:
    """
    Surface conditions at a lunar location.

    The Moon has no bound atmosphere. This class replaces
    AtmosphericColumn for the Moon — it carries surface temperature,
    terrain geometry, and illumination fraction rather than a vertical
    profile.

    Parameters
    ----------
    tsurf_K : float
        Surface temperature [K].
    lat, lon : float
        Location [deg].
    sza_deg : float, optional
        Solar zenith angle [deg]. None if not computed.
    albedo : float
        Bolometric Bond albedo. Default 0.12.
    elevation_m : float, optional
        Surface elevation above reference radius 1737.4 km [m].
    slope_deg : float, optional
        Local terrain slope [deg].
    aspect_deg : float, optional
        Local terrain aspect (azimuth of downslope direction) [deg].
    illumination_fraction : float, optional
        Fraction of hemisphere illuminated [0–1].
        1.0 = fully illuminated, 0.0 = in shadow.
        None = not computed (use SZA-based estimate instead).
    horizon_elevation_deg : np.ndarray, optional
        Horizon elevation angles at 360 azimuths [deg].
        Used for terrain shadowing.
    source : MoonAtmosphericSource
    label : str
    """
    tsurf_K               : float
    lat                   : float
    lon                   : float
    sza_deg               : Optional[float] = None
    albedo                : float = 0.12
    elevation_m           : Optional[float] = None
    slope_deg             : Optional[float] = None
    aspect_deg            : Optional[float] = None
    illumination_fraction : Optional[float] = None
    horizon_elevation_deg : Optional[np.ndarray] = None
    source                : MoonAtmosphericSource = MoonAtmosphericSource.CUSTOM
    label                 : str = ""

    def __post_init__(self):
        if not self.label:
            self.label = (
                f"Moon | {self.source.value} | "
                f"lat={self.lat:.1f} lon={self.lon:.1f} | "
                f"T={self.tsurf_K:.1f} K"
            )

    def __str__(self):
        parts = [
            f"MoonSurfaceConditions: {self.source.value}",
            f"lat={self.lat:.2f}° lon={self.lon:.2f}°",
            f"T_surf={self.tsurf_K:.1f} K",
        ]
        if self.sza_deg is not None:
            parts.append(f"SZA={self.sza_deg:.1f}°")
        if self.elevation_m is not None:
            parts.append(f"elev={self.elevation_m:.0f} m")
        if self.slope_deg is not None:
            parts.append(f"slope={self.slope_deg:.1f}°")
        if self.illumination_fraction is not None:
            parts.append(f"illum={self.illumination_fraction:.3f}")
        return " | ".join(parts)

    def __repr__(self):
        return self.__str__()

    # ── Loaders ──────────────────────────────────────────────────────────────

    @classmethod
    def from_analytical(
        cls,
        lat: float,
        lon: float,
        sza_deg: float,
        albedo: float = 0.12,
    ) -> "MoonSurfaceConditions":
        """
        Compute surface conditions from the bundled Diviner analytical
        temperature model.

        Parameters
        ----------
        lat : float       Latitude [deg N]
        lon : float       Longitude [deg E]
        sza_deg : float   Solar zenith angle [deg]
        albedo : float    Bolometric Bond albedo. Default 0.12.

        Returns
        -------
        MoonSurfaceConditions

        Notes
        -----
        No terrain shadowing is applied — illumination_fraction is set
        from SZA only (1.0 if SZA < 90°, 0.0 if SZA >= 90°). For
        terrain-corrected illumination, use from_npz() with a LOLA
        GeoTIFF conversion.
        """
        tsurf = _diviner_tsurf(lat, sza_deg, albedo)
        illum = max(0.0, np.cos(np.deg2rad(sza_deg))) if sza_deg < 90.0 else 0.0

        return cls(
            tsurf_K               = tsurf,
            lat                   = float(lat),
            lon                   = float(lon),
            sza_deg               = float(sza_deg),
            albedo                = float(albedo),
            illumination_fraction = illum,
            source                = MoonAtmosphericSource.ANALYTICAL,
        )

    @classmethod
    def from_npz(cls, path: str) -> "MoonSurfaceConditions":
        """
        Load surface conditions from a LOLA converter output .npz file
        produced by scripts/convert_lola.py.

        Accepts both GeoTIFF-derived (PRIMARY) and NetCDF-derived
        (FALLBACK) converter outputs — the .npz schema is identical
        regardless of source format.

        Parameters
        ----------
        path : str   Path to .npz file from convert_lola.py.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"LOLA surface conditions file not found: {path}\n"
                f"Run: python scripts/convert_lola.py --help"
            )

        data = np.load(path, allow_pickle=True)

        def _get(key, default=None):
            if key not in data.files:
                return default
            v = data[key]
            if np.ndim(v) == 0:
                return float(v)
            if v.shape == (1,):
                try:
                    return float(v[0])
                except (TypeError, ValueError):
                    return str(v[0])
            return v

        source_str = str(_get("source", "custom"))
        if "geotiff" in source_str.lower():
            source = MoonAtmosphericSource.LOLA_GEOTIFF
        elif "netcdf" in source_str.lower():
            source = MoonAtmosphericSource.LOLA_NETCDF
        else:
            source = MoonAtmosphericSource.CUSTOM

        # Compute T_surf if not stored — use analytical model
        tsurf = _get("tsurf_K")
        if tsurf is None:
            sza = _get("sza_deg")
            lat = _get("lat", 0.0)
            alb = _get("albedo", 0.12)
            if sza is not None:
                tsurf = _diviner_tsurf(lat, sza, alb)
            else:
                tsurf = 250.0   # fallback
                warnings.warn(
                    "tsurf_K and sza_deg not found in npz — "
                    "using fallback T=250 K.",
                    UserWarning,
                )

        horizon = _get("horizon_elevation_deg")
        if isinstance(horizon, np.ndarray) and horizon.ndim == 0:
            horizon = None

        return cls(
            tsurf_K               = tsurf,
            lat                   = _get("lat", 0.0),
            lon                   = _get("lon", 0.0),
            sza_deg               = _get("sza_deg"),
            albedo                = _get("albedo", 0.12),
            elevation_m           = _get("elevation_m"),
            slope_deg             = _get("slope_deg"),
            aspect_deg            = _get("aspect_deg"),
            illumination_fraction = _get("illumination_fraction"),
            horizon_elevation_deg = horizon,
            source                = source,
        )

    # ── Science methods ───────────────────────────────────────────────────────

    def effective_sza(self) -> float:
        """
        Effective solar zenith angle accounting for terrain slope.

        For a tilted surface, the effective SZA is modified by the
        slope and the azimuth angle between the Sun and the slope
        downhill direction.

        Returns SZA if slope is unknown. Returns 90° if in shadow.
        """
        if self.sza_deg is None:
            return 90.0
        if self.slope_deg is None or self.slope_deg == 0.0:
            return self.sza_deg

        # cos(effective_SZA) = cos(SZA)*cos(slope)
        #   + sin(SZA)*sin(slope)*cos(sun_azimuth - aspect)
        # Without sun azimuth, return flat-surface SZA as conservative estimate
        warnings.warn(
            "Sun azimuth not available — effective_sza() returning "
            "flat-surface SZA. Provide sun_azimuth_deg for terrain correction.",
            UserWarning,
        )
        return self.sza_deg

    def is_illuminated(self) -> bool:
        """True if the surface point is illuminated (not in shadow)."""
        if self.illumination_fraction is not None:
            return self.illumination_fraction > 0.0
        if self.sza_deg is not None:
            return self.sza_deg < 90.0
        return True   # assume illuminated if unknown

    def column_density(self, species: str) -> float:
        """Always returns 0.0 — Moon has no bound atmosphere."""
        return 0.0

    # ── Bridge to AtmosphericProfile ─────────────────────────────────────────

    def to_profile(self, **overrides) -> AtmosphericProfile:
        """
        Return an AtmosphericProfile for the Moon (airless body).

        The profile has has_atmosphere=False, zero dust, zero Rayleigh.
        Surface temperature is attached as a note in the label.
        Slope and illumination metadata are passed through overrides
        if the caller wants to attach them.

        Parameters
        ----------
        **overrides
            Any AtmosphericProfile field.
        """
        defaults = dict(
            body                       = "moon",
            surface_pressure           = 3e-10 * u.Pa,
            scale_height               = 0.0   * u.km,
            dust_tau                   = 0.0,
            angstrom_exponent          = 0.0,
            single_scatter_albedo      = 0.0,
            composition                = {},
            has_atmosphere             = False,
            column_densities           = {},
            include_rayleigh           = False,
            rayleigh_king_factor       = 1.0,
            effective_refractive_index = 1.0,
            label                      = (
                f"Moon (airless) | T_surf={self.tsurf_K:.1f} K | "
                f"{self.source.value}"
            ),
        )
        defaults.update(overrides)
        return AtmosphericProfile(**defaults)
