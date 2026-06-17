"""
leos.spectral_sources
---------------------
Registry of solar spectral irradiance sources available in LEOS.

Each source is a SpectralSourceInfo dataclass describing:
  - what it is
  - what time window it covers
  - what wavelength range it covers
  - its absolute calibration uncertainty
  - how to fetch it (URL or local path)
  - which sources it cross-calibrates against

No fetching happens here. This is pure metadata.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from astropy.time import Time
from astropy import units as u


class SpectralSource(Enum):
    """
    All solar spectral irradiance sources supported by LEOS.
    Value is the string key used in cache filenames and logs.
    """
    ASTM_E490  = "astm_e490"   # Static standard reference, ~2000 epoch
    UARS_SUSIM = "uars_susim"  # Measured, 1991-2005, UV only
    SORCE_SIM  = "sorce_sim"   # Measured, 2003-2020
    TSIS1_SIM  = "tsis1_sim"   # Measured, 2018-present, live-fetchable
    NRLSSI2    = "nrlssi2"     # Empirical model, 1610-present
    SATIRE_S   = "satire_s"    # Physics model, ~1974-present
    ANALYTIC   = "analytic"    # Offline Planck fallback, any time
    CUSTOM     = "custom"      # User-supplied irradiance profile


@dataclass
class SpectralSourceInfo:
    """
    Metadata for one solar spectral irradiance source.

    Parameters
    ----------
    source : SpectralSource
    description : str
        One-line human description.
    time_start : astropy Time or None
        Start of valid time window. None = no lower bound.
    time_end : astropy Time or None
        End of valid time window. None = ongoing / no upper bound.
    time_note : str
        Human note about temporal coverage, e.g. "Solar cycles 22-23"
    wl_min_nm : float
        Minimum wavelength in nm.
    wl_max_nm : float
        Maximum wavelength in nm.
    is_time_resolved : bool
        True if source provides daily/hourly spectra varying in time.
        False if it is a single static reference spectrum.
    is_live : bool
        True if the source can be auto-fetched for today's date.
    calibration_reference : SpectralSource or None
        Which source this is calibrated against (for cross-cal chain).
    sigma_abs_percent : dict
        Absolute calibration uncertainty by wavelength band.
        Keys are band labels, values are percent (1-sigma).
        e.g. {"UV_200_300": 5.0, "VIS_400_700": 1.0}
    fetch_url : str or None
        Base URL for data access. None for locally bundled sources.
    version : str
        Data product version string.
    reference : str
        Citable reference for this data product.
    """
    source               : SpectralSource
    description          : str
    time_start           : Optional[Time]
    time_end             : Optional[Time]
    time_note            : str
    wl_min_nm            : float
    wl_max_nm            : float
    is_time_resolved     : bool
    is_live              : bool
    calibration_reference: Optional[SpectralSource]
    sigma_abs_percent    : dict
    fetch_url            : Optional[str]
    version              : str
    reference            : str

    def covers_time(self, time: Time) -> bool:
        """Return True if this source is valid at the given time."""
        if self.time_start is not None and time < self.time_start:
            return False
        if self.time_end is not None and time > self.time_end:
            return False
        return True

    def covers_wavelength(self, wl_nm: float) -> bool:
        return self.wl_min_nm <= wl_nm <= self.wl_max_nm

    def __str__(self):
        live = " [LIVE]" if self.is_live else ""
        return (
            f"{self.source.value}{live}: "
            f"{self.wl_min_nm}–{self.wl_max_nm} nm, "
            f"{self.time_note}"
        )


# ── Source registry ───────────────────────────────────────────────────────────

REGISTRY: dict[SpectralSource, SpectralSourceInfo] = {

    SpectralSource.ASTM_E490: SpectralSourceInfo(
        source                = SpectralSource.ASTM_E490,
        description           = "ASTM E-490 Air Mass Zero standard solar spectrum",
        time_start            = None,
        time_end              = None,
        time_note             = "Static reference, ~solar maximum cycle 23 (~2000)",
        wl_min_nm             = 119.5,
        wl_max_nm             = 1_000_000.0,
        is_time_resolved      = False,
        is_live               = False,
        calibration_reference = None,
        sigma_abs_percent     = {
            "UV_120_200" : 10.0,
            "UV_200_300" :  5.0,
            "UV_300_400" :  3.0,
            "VIS_400_700":  2.0,
            "NIR_700_2400": 2.0,
        },
        fetch_url  = "https://www.nrel.gov/grid/solar-resource/assets/data/astmg173.csv",
        version    = "ASTM E-490-00a (2000)",
        reference  = "ASTM International (2000). E490-00a Standard Solar Constant.",
    ),

    SpectralSource.UARS_SUSIM: SpectralSourceInfo(
        source                = SpectralSource.UARS_SUSIM,
        description           = "UARS SUSIM daily solar UV irradiance, 115-410 nm",
        time_start            = Time("1991-10-11", format="iso"),
        time_end              = Time("2005-07-31", format="iso"),
        time_note             = "Solar cycles 22–23 (1991–2005)",
        wl_min_nm             = 115.0,
        wl_max_nm             = 410.0,
        is_time_resolved      = True,
        is_live               = False,
        calibration_reference = SpectralSource.TSIS1_SIM,
        sigma_abs_percent     = {
            "UV_115_170" : 10.0,   # noisier, fewer photons
            "UV_170_300" :  5.0,   # after v22 degradation correction
            "UV_300_410" :  3.0,
        },
        fetch_url  = "https://lasp.colorado.edu/lisird/api/uars_susim_ssi.csv",
        version    = "v22",
        reference  = (
            "Brueckner et al. (1993). The Solar Ultraviolet Spectral Irradiance "
            "Monitor (SUSIM) experiment on board UARS. J. Geophys. Res."
        ),
    ),

    SpectralSource.SORCE_SIM: SpectralSourceInfo(
        source                = SpectralSource.SORCE_SIM,
        description           = "SORCE SIM daily solar spectral irradiance, 240-2400 nm",
        time_start            = Time("2003-04-14", format="iso"),
        time_end              = Time("2020-02-25", format="iso"),
        time_note             = "Solar cycles 23–24 (2003–2020)",
        wl_min_nm             = 240.0,
        wl_max_nm             = 2400.0,
        is_time_resolved      = True,
        is_live               = False,
        calibration_reference = SpectralSource.TSIS1_SIM,
        sigma_abs_percent     = {
            "UV_240_300" :  5.0,   # prism degradation corrected in v27
            "UV_300_400" :  2.0,
            "VIS_400_700":  1.5,
            "NIR_700_2400": 1.5,
        },
        fetch_url  = "https://lasp.colorado.edu/lisird/api/sorce_sim_ssi_24hr.csv",
        version    = "v27",
        reference  = (
            "Harder et al. (2005). The Spectral Irradiance Monitor: "
            "Science requirements and instrument design. Sol. Phys."
        ),
    ),

    SpectralSource.TSIS1_SIM: SpectralSourceInfo(
        source                = SpectralSource.TSIS1_SIM,
        description           = "TSIS-1 SIM daily solar spectral irradiance, 200-2400 nm",
        time_start            = Time("2018-01-11", format="iso"),
        time_end              = None,   # ongoing
        time_note             = "Solar cycles 24–25 (2018–present), updated daily",
        wl_min_nm             = 200.0,
        wl_max_nm             = 2400.0,
        is_time_resolved      = True,
        is_live               = True,   # auto-fetch today's data
        calibration_reference = None,   # primary calibration reference
        sigma_abs_percent     = {
            "UV_200_300" :  1.0,   # best UV calibration of the three
            "UV_300_400" :  0.7,
            "VIS_400_700":  0.5,
            "NIR_700_2400": 0.5,
        },
        fetch_url  = "https://lasp.colorado.edu/lisird/api/tsis1_ssi_24hr.csv",
        version    = "v03",
        reference  = (
            "Richard et al. (2020). The TSIS-1 Spectral Irradiance Monitor. "
            "Geophys. Res. Lett."
        ),
    ),

    SpectralSource.NRLSSI2: SpectralSourceInfo(
        source                = SpectralSource.NRLSSI2,
        description           = "NRL Solar Spectral Irradiance model v2, 1610-present",
        time_start            = Time("1610-01-01", format="iso"),
        time_end              = None,
        time_note             = "Empirical model 1610–present (sunspot/facular proxies)",
        wl_min_nm             = 1.0,
        wl_max_nm             = 100_000.0,
        is_time_resolved      = True,
        is_live               = False,
        calibration_reference = SpectralSource.TSIS1_SIM,
        sigma_abs_percent     = {
            "UV_120_200" : 15.0,   # model extrapolation, larger uncertainty
            "UV_200_300" :  8.0,
            "UV_300_400" :  4.0,
            "VIS_400_700":  2.0,
            "NIR_700_2400": 2.0,
        },
        fetch_url  = (
            "https://lasp.colorado.edu/lisird/api/nrl2_solar_irradiance_daily.csv"
        ),
        version    = "v2",
        reference  = (
            "Coddington et al. (2016). A solar irradiance climate data record. "
            "BAMS. doi:10.1175/BAMS-D-14-00265.1"
        ),
    ),

    SpectralSource.SATIRE_S: SpectralSourceInfo(
        source                = SpectralSource.SATIRE_S,
        description           = "SATIRE-S physics-based solar irradiance reconstruction",
        time_start            = Time("1974-05-27", format="iso"),
        time_end              = None,
        time_note             = "Physics model from solar magnetograms, 1974–present",
        wl_min_nm             = 115.0,
        wl_max_nm             = 160_000.0,
        is_time_resolved      = True,
        is_live               = False,
        calibration_reference = SpectralSource.TSIS1_SIM,
        sigma_abs_percent     = {
            "UV_115_200" : 10.0,
            "UV_200_300" :  5.0,
            "UV_300_400" :  3.0,
            "VIS_400_700":  2.0,
            "NIR_700_2400": 2.0,
        },
        fetch_url  = "https://www2.mps.mpg.de/projects/sun-climate/data.html",
        version    = "SATIRE-S 2021",
        reference  = (
            "Yeo et al. (2014). Solar irradiance variability in cycles 21 to 23 "
            "based on SATIRE-S. A&A. doi:10.1051/0004-6361/201423628"
        ),
    ),

    SpectralSource.CUSTOM: SpectralSourceInfo(
        source                = SpectralSource.CUSTOM,
        description           = "User-supplied irradiance profile (JD, wavelength, flux, uncertainty)",
        time_start            = None,
        time_end              = None,
        time_note             = "as provided by user",
        wl_min_nm             = 0.0,
        wl_max_nm             = 1e9,
        is_time_resolved      = True,
        is_live               = False,
        calibration_reference = None,
        sigma_abs_percent     = {},
        fetch_url             = None,
        version               = "user-supplied",
        reference             = "User-supplied data.",
    ),

    SpectralSource.ANALYTIC: SpectralSourceInfo(
        source                = SpectralSource.ANALYTIC,
        description           = "Analytic Planck blackbody T=5778K, offline fallback",
        time_start            = None,
        time_end              = None,
        time_note             = "No time dependence — offline fallback only",
        wl_min_nm             = 200.0,
        wl_max_nm             = 4000.0,
        is_time_resolved      = False,
        is_live               = False,
        calibration_reference = None,
        sigma_abs_percent     = {
            "ALL": 5.0,   # model uncertainty vs real Sun
        },
        fetch_url  = None,
        version    = "leos-internal",
        reference  = "Planck (1901), normalized to solar constant 1361 W/m².",
    ),
}


def get_info(source: SpectralSource) -> SpectralSourceInfo:
    """Return the SpectralSourceInfo for a given source."""
    return REGISTRY[source]


def sources_valid_at(time: Time) -> list[SpectralSourceInfo]:
    """Return all sources whose time window covers the given time."""
    return [info for info in REGISTRY.values() if info.covers_time(time)]


def best_source_for_time(time: Time) -> SpectralSource:
    """
    Return the highest-fidelity measured source for a given time.
    Preference: TSIS1 > SORCE > UARS > NRLSSI2 > ASTM > ANALYTIC.
    """
    priority = [
        SpectralSource.TSIS1_SIM,
        SpectralSource.SORCE_SIM,
        SpectralSource.UARS_SUSIM,
        SpectralSource.NRLSSI2,
        SpectralSource.ASTM_E490,
        SpectralSource.ANALYTIC,
    ]
    for src in priority:
        if REGISTRY[src].covers_time(time):
            return src
    return SpectralSource.ANALYTIC
