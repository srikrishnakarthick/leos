"""
leos.solar_variability
----------------------
Compute data-driven per-wavelength solar irradiance variability profiles
from measured and modelled spectral sources.

The variability profile replaces the flat calibration-percentage σ(λ)
used by static sources (ASTM E-490) with a physically motivated uncertainty
derived from real spectral variability over a user-specified time window.

Algorithm: three-level cascade
  Level 1 — Solar cycle anchors (~10 spectra, hardcoded turning points)
             Finds approximate global min/max per wavelength band.
  Level 2 — Monthly bracketing (~20 spectra around each L1 extreme)
             Narrows each band's min/max epoch to within a month.
  Level 3 — Weekly refinement (~8 spectra around each L2 extreme)
             Pins min/max to within a week.

  Stop condition: if L1 band variability < tolerance, skip L2/L3 for
  that band (variability is negligible).

Source selection by time window:
  2003–present  : SORCE SIM + TSIS-1 SIM (high fidelity, full spectrum)
  1991–2005 UV  : UARS SUSIM (115–410 nm), else user choice
  1974–present  : NRLSSI2 or SATIRE-S (lower fidelity, warn user)
  pre-1974      : NRLSSI2 / SATIRE-T / SATIRE-M (model only, warn clearly)

Output: VariabilityProfile object with per-wavelength statistics.
"""

import os
import warnings
import numpy as np
from astropy import units as u
from astropy.time import Time
from typing import Optional, Tuple

from .spectral_sources import SpectralSource, REGISTRY, get_info
from .spectrum import Spectrum


# ══════════════════════════════════════════════════════════════════════════════
# Hardcoded solar cycle turning points (Level 1 anchors)
# Source: SIDC, Hathaway 2015 Living Reviews, WDC-SILSO
# Format: (label, date_string, type)  type = 'min' | 'max'
# ══════════════════════════════════════════════════════════════════════════════

_SOLAR_CYCLE_ANCHORS = [
    # Cycle 20
    ("SC20_min", "1976-03-01", "min"),
    ("SC20_max", "1979-12-01", "max"),
    # Cycle 21
    ("SC21_min", "1986-09-01", "min"),
    ("SC21_max", "1989-11-01", "max"),
    # Cycle 22
    ("SC22_min", "1996-05-01", "min"),
    ("SC22_max", "2001-11-01", "max"),
    # Cycle 23
    ("SC23_min", "2008-12-01", "min"),
    ("SC23_max", "2014-04-01", "max"),
    # Cycle 24
    ("SC24_min", "2019-12-01", "min"),
    ("SC24_max", "2022-10-01", "max"),
    # Cycle 25 (ongoing)
    ("SC25_max_est", "2025-07-01", "max"),
]

# Wavelength bands for per-band cascade
_BANDS = {
    "UV":  (115.0,  400.0),
    "VIS": (400.0,  700.0),
    "NIR": (700.0, 2400.0),
}

# Source time windows for auto-selection
_SOURCE_WINDOWS = {
    SpectralSource.TSIS1_SIM  : ("2018-01-11", None),
    SpectralSource.SORCE_SIM  : ("2003-04-14", "2020-02-25"),
    SpectralSource.UARS_SUSIM : ("1991-10-11", "2005-07-31"),
    SpectralSource.NRLSSI2    : ("1610-01-01", None),
}


# ══════════════════════════════════════════════════════════════════════════════
# VariabilityProfile
# ══════════════════════════════════════════════════════════════════════════════

class VariabilityProfile:
    """
    Per-wavelength solar irradiance variability statistics.

    Attributes
    ----------
    wl_nm : np.ndarray
        Wavelength array in nm (native source grid).
    flux_min : np.ndarray
        Per-wavelength minimum flux [W/m²/nm].
    flux_max : np.ndarray
        Per-wavelength maximum flux [W/m²/nm].
    flux_mean : np.ndarray
        Per-wavelength mean flux [W/m²/nm].
    sigma_variability : np.ndarray
        Per-wavelength std dev of sampled spectra [W/m²/nm].
    sigma_calibration : np.ndarray
        Per-wavelength calibration uncertainty [W/m²/nm].
    sigma : np.ndarray
        Quadrature combination of variability + calibration [W/m²/nm].
        This is the recommended σ(λ) for use with ASTM E-490.
    epoch_min : dict
        {'UV': Time, 'VIS': Time, 'NIR': Time} — epoch of band minimum.
    epoch_max : dict
        {'UV': Time, 'VIS': Time, 'NIR': Time} — epoch of band maximum.
    source : SpectralSource
        Primary source used.
    wl_range : tuple
        (wl_min_nm, wl_max_nm) used for cascade integration.
    n_spectra : int
        Total number of spectra fetched across all cascade levels.
    time_start : astropy Time
    time_end : astropy Time
    """

    def __init__(
        self,
        wl_nm, flux_min, flux_max, flux_mean,
        sigma_variability, sigma_calibration,
        epoch_min, epoch_max,
        source, wl_range, n_spectra,
        time_start, time_end,
    ):
        self.wl_nm              = wl_nm
        self.flux_min           = flux_min
        self.flux_max           = flux_max
        self.flux_mean          = flux_mean
        self.sigma_variability  = sigma_variability
        self.sigma_calibration  = sigma_calibration
        self.sigma              = np.sqrt(sigma_variability**2 + sigma_calibration**2)
        self.epoch_min          = epoch_min
        self.epoch_max          = epoch_max
        self.source             = source
        self.wl_range           = wl_range
        self.n_spectra          = n_spectra
        self.time_start         = time_start
        self.time_end           = time_end

    def __str__(self):
        return (
            f"VariabilityProfile: {self.source.value} | "
            f"{self.time_start.iso[:10]}–{self.time_end.iso[:10]} | "
            f"{len(self.wl_nm)} wavelength points | "
            f"{self.n_spectra} spectra sampled"
        )

    def __repr__(self):
        return self.__str__()

    def to_spectrum(self) -> Spectrum:
        """
        Return the mean flux as a Spectrum with sigma as uncertainty.
        Useful for direct comparison with get_solar_spectrum() output.
        """
        return Spectrum(
            self.wl_nm  * u.nm,
            self.flux_mean * u.W / u.m**2 / u.nm,
            uncertainty=self.sigma * u.W / u.m**2 / u.nm,
            label=f"Variability mean | {self.source.value}",
        )

    def save(self, path: str):
        """
        Save profile to .npz. User is responsible for storing this file.

        Parameters
        ----------
        path : str — output path, e.g. 'my_variability.npz'
        """
        np.savez(
            path,
            wl_nm              = self.wl_nm,
            flux_min           = self.flux_min,
            flux_max           = self.flux_max,
            flux_mean          = self.flux_mean,
            sigma_variability  = self.sigma_variability,
            sigma_calibration  = self.sigma_calibration,
            sigma              = self.sigma,
            source             = np.array([self.source.value], dtype=object),
            wl_range           = np.array(self.wl_range),
            n_spectra          = np.array([self.n_spectra]),
            time_start         = np.array([self.time_start.iso]),
            time_end           = np.array([self.time_end.iso]),
            epoch_min_UV       = np.array([self.epoch_min.get("UV", "unknown")],  dtype=object),
            epoch_min_VIS      = np.array([self.epoch_min.get("VIS", "unknown")], dtype=object),
            epoch_min_NIR      = np.array([self.epoch_min.get("NIR", "unknown")], dtype=object),
            epoch_max_UV       = np.array([self.epoch_max.get("UV", "unknown")],  dtype=object),
            epoch_max_VIS      = np.array([self.epoch_max.get("VIS", "unknown")], dtype=object),
            epoch_max_NIR      = np.array([self.epoch_max.get("NIR", "unknown")], dtype=object),
        )
        size_mb = os.path.getsize(path) / 1e6
        print(f"Saved variability profile: {path} ({size_mb:.2f} MB)")

    @classmethod
    def load(cls, path: str) -> "VariabilityProfile":
        """
        Load a previously saved variability profile.

        Parameters
        ----------
        path : str — path to .npz file saved by VariabilityProfile.save()
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Variability profile not found: {path}")

        data = np.load(path, allow_pickle=True)

        def _ep(d, key):
            val = str(d[key][0])
            return val if val != "unknown" else "unknown"

        epoch_min = {
            "UV" : _ep(data, "epoch_min_UV"),
            "VIS": _ep(data, "epoch_min_VIS"),
            "NIR": _ep(data, "epoch_min_NIR"),
        }
        epoch_max = {
            "UV" : _ep(data, "epoch_max_UV"),
            "VIS": _ep(data, "epoch_max_VIS"),
            "NIR": _ep(data, "epoch_max_NIR"),
        }

        source_str = str(data["source"][0])
        source = next(
            (s for s in SpectralSource if s.value == source_str),
            SpectralSource.ANALYTIC
        )

        return cls(
            wl_nm             = data["wl_nm"],
            flux_min          = data["flux_min"],
            flux_max          = data["flux_max"],
            flux_mean         = data["flux_mean"],
            sigma_variability = data["sigma_variability"],
            sigma_calibration = data["sigma_calibration"],
            epoch_min         = epoch_min,
            epoch_max         = epoch_max,
            source            = source,
            wl_range          = tuple(data["wl_range"]),
            n_spectra         = int(data["n_spectra"][0]),
            time_start        = Time(str(data["time_start"][0])),
            time_end          = Time(str(data["time_end"][0])),
        )


# ══════════════════════════════════════════════════════════════════════════════
# Source selection
# ══════════════════════════════════════════════════════════════════════════════

def _select_source(
    time_start: Time,
    time_end: Time,
    wl_min_nm: float,
    wl_max_nm: float,
    user_source: Optional[SpectralSource] = None,
) -> SpectralSource:
    """
    Auto-select the highest fidelity source for the given time window.
    User can override with user_source.
    """
    if user_source is not None:
        info = get_info(user_source)
        if not info.covers_time(time_start) or not info.covers_time(time_end):
            warnings.warn(
                f"{user_source.value} does not fully cover "
                f"{time_start.iso[:10]}–{time_end.iso[:10]}. "
                f"Results may be incomplete.",
                UserWarning,
            )
        return user_source

    t_start_str = time_start.iso[:10]
    t_end_str   = time_end.iso[:10]

    # Pre-1974: model only
    if time_start < Time("1974-05-27"):
        warnings.warn(
            f"Time window starts before 1974. No measured spectral data available. "
            f"Using NRLSSI2 (empirical proxy model). "
            f"For pre-1610 windows, use SATIRE-T or SATIRE-M via SpectralSource.CUSTOM.",
            UserWarning,
        )
        return SpectralSource.NRLSSI2

    # UV-only window and UARS covers it
    uv_only = wl_max_nm <= 410.0
    uars_start = Time("1991-10-11")
    uars_end   = Time("2005-07-31")
    if uv_only and time_start >= uars_start and time_end <= uars_end:
        print("  Auto-selected: UARS SUSIM (UV window, 1991–2005)")
        return SpectralSource.UARS_SUSIM

    # 2018–present: TSIS-1
    if time_start >= Time("2018-01-11"):
        print("  Auto-selected: TSIS-1 SIM (2018–present, highest fidelity)")
        return SpectralSource.TSIS1_SIM

    # 2003–present: SORCE
    if time_start >= Time("2003-04-14"):
        print("  Auto-selected: SORCE SIM (2003–2020)")
        return SpectralSource.SORCE_SIM

    # 1974–2003: NRLSSI2 with warning
    warnings.warn(
        f"No high-fidelity measured spectral data for {t_start_str}–{t_end_str}. "
        f"Using NRLSSI2 (empirical model). Variability estimates will be less precise.",
        UserWarning,
    )
    return SpectralSource.NRLSSI2


# ══════════════════════════════════════════════════════════════════════════════
# Spectrum fetching with source chaining
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_spectrum_at(
    time: Time,
    primary_source: SpectralSource,
    wl_min_nm: float,
    wl_max_nm: float,
) -> Optional[Spectrum]:
    """
    Fetch spectrum at a given time using priority chaining:
    TSIS-1 → SORCE → UARS (UV only) → NRLSSI2 → ANALYTIC fallback.
    Returns None if all sources fail.
    """
    from leos.solar_spectrum import get_solar_spectrum

    priority = _build_priority_chain(primary_source, time, wl_min_nm, wl_max_nm)

    for source in priority:
        info = get_info(source)
        if not info.covers_time(time):
            continue
        try:
            s = get_solar_spectrum(source=source, time=time, resolution="high")
            return s
        except Exception as e:
            continue

    return None


def _build_priority_chain(
    primary: SpectralSource,
    time: Time,
    wl_min_nm: float,
    wl_max_nm: float,
) -> list:
    """Build ordered list of sources to try for a given time."""
    uv_only = wl_max_nm <= 410.0

    chain = [primary]

    # Fill gaps with lower-fidelity sources
    fallbacks = []
    if primary != SpectralSource.TSIS1_SIM:
        fallbacks.append(SpectralSource.TSIS1_SIM)
    if primary != SpectralSource.SORCE_SIM:
        fallbacks.append(SpectralSource.SORCE_SIM)
    if uv_only and primary != SpectralSource.UARS_SUSIM:
        fallbacks.append(SpectralSource.UARS_SUSIM)
    if primary != SpectralSource.NRLSSI2:
        fallbacks.append(SpectralSource.NRLSSI2)
    fallbacks.append(SpectralSource.ANALYTIC)

    return chain + fallbacks


# ══════════════════════════════════════════════════════════════════════════════
# Band integration
# ══════════════════════════════════════════════════════════════════════════════

def _band_flux(spectrum: Spectrum, wl_min_nm: float, wl_max_nm: float) -> float:
    """Integrate spectrum over [wl_min, wl_max] in nm. Returns float W/m²."""
    sliced = spectrum.slice(wl_min_nm * u.nm, wl_max_nm * u.nm)
    if len(sliced.wavelengths) < 2:
        return 0.0
    return float(sliced.integrate().value)


def _per_band_flux(spectrum: Spectrum) -> dict:
    """Return integrated flux in each of UV, VIS, NIR bands."""
    return {
        band: _band_flux(spectrum, lo, hi)
        for band, (lo, hi) in _BANDS.items()
    }


# ══════════════════════════════════════════════════════════════════════════════
# Date generation utilities
# ══════════════════════════════════════════════════════════════════════════════

def _monthly_dates(t_start: Time, t_end: Time) -> list:
    """Generate first-of-month dates between t_start and t_end."""
    from datetime import datetime, timedelta
    dates = []
    current = datetime(t_start.datetime.year, t_start.datetime.month, 1)
    end     = t_end.datetime
    while current <= end:
        dates.append(Time(current.strftime("%Y-%m-%d")))
        # Advance one month
        month = current.month + 1
        year  = current.year + (month - 1) // 12
        month = ((month - 1) % 12) + 1
        current = datetime(year, month, 1)
    return dates


def _weekly_dates(t_start: Time, t_end: Time) -> list:
    """Generate weekly dates between t_start and t_end."""
    from datetime import timedelta
    dates = []
    current = t_start.datetime
    while current <= t_end.datetime:
        dates.append(Time(current.strftime("%Y-%m-%d")))
        current += timedelta(weeks=1)
    return dates


def _anchor_dates_in_window(t_start: Time, t_end: Time) -> list:
    """Return solar cycle anchor dates within [t_start, t_end]."""
    dates = []
    for label, date_str, kind in _SOLAR_CYCLE_ANCHORS:
        t = Time(date_str)
        if t_start <= t <= t_end:
            dates.append((label, t, kind))
    return dates


# ══════════════════════════════════════════════════════════════════════════════
# Cascade
# ══════════════════════════════════════════════════════════════════════════════

def _run_cascade(
    source       : SpectralSource,
    time_start   : Time,
    time_end     : Time,
    wl_min_nm    : float,
    wl_max_nm    : float,
    tolerance    : float,
    verbose      : bool,
) -> Tuple[list, dict, dict]:
    """
    Run the three-level cascade.

    Returns
    -------
    spectra : list of Spectrum
        All fetched spectra across all levels.
    epoch_min : dict
        {'UV': Time or str, 'VIS': Time or str, 'NIR': Time or str}
    epoch_max : dict
        Same for maxima.
    """
    all_spectra  = []
    epoch_min    = {}
    epoch_max    = {}

    # ── Level 1: Solar cycle anchors ─────────────────────────────────────────
    if verbose:
        print("\n  [Cascade L1] Solar cycle anchors...")

    anchor_entries = _anchor_dates_in_window(time_start, time_end)

    # If fewer than 2 anchors in window, supplement with window endpoints
    # and midpoint to ensure minimum coverage
    anchor_dates = [t for _, t, _ in anchor_entries]
    if len(anchor_dates) < 2:
        mid = Time((time_start.jd + time_end.jd) / 2, format="jd")
        anchor_dates = [time_start, mid, time_end]
        if verbose:
            print("    Fewer than 2 cycle anchors in window — "
                  "using start/mid/end.")

    l1_spectra = []
    for t in anchor_dates:
        s = _fetch_spectrum_at(t, source, wl_min_nm, wl_max_nm)
        if s is not None:
            l1_spectra.append((t, s))
            if verbose:
                flux = _band_flux(s, wl_min_nm, wl_max_nm)
                print(f"    {t.iso[:10]}  integrated flux = {flux:.2f} W/m²")

    all_spectra.extend([s for _, s in l1_spectra])

    if len(l1_spectra) == 0:
        raise RuntimeError(
            "No spectra could be fetched at Level 1. "
            "Check source availability and network connection."
        )

    # Per-band L1 extremes
    l1_band_flux = {}
    for band, (lo, hi) in _BANDS.items():
        fluxes = [(t, _band_flux(s, lo, hi)) for t, s in l1_spectra]
        fluxes.sort(key=lambda x: x[1])
        l1_band_flux[band] = {
            "min_t": fluxes[0][0],  "min_f": fluxes[0][1],
            "max_t": fluxes[-1][0], "max_f": fluxes[-1][1],
        }

    # Check stop condition per band
    skip_l2 = {}
    for band in _BANDS:
        bf     = l1_band_flux[band]
        mean_f = (bf["min_f"] + bf["max_f"]) / 2.0
        if mean_f > 0:
            variability = (bf["max_f"] - bf["min_f"]) / mean_f
        else:
            variability = 0.0
        skip_l2[band] = variability < tolerance
        if verbose and skip_l2[band]:
            print(f"    [{band}] variability {variability*100:.3f}% < "
                  f"tolerance {tolerance*100:.3f}% — skipping L2/L3.")

    # ── Level 2: Monthly bracketing ───────────────────────────────────────────
    if verbose:
        print("\n  [Cascade L2] Monthly bracketing around L1 extremes...")

    l2_band_flux = {band: dict(l1_band_flux[band]) for band in _BANDS}

    for band, (lo, hi) in _BANDS.items():
        if skip_l2[band]:
            continue

        bf = l1_band_flux[band]
        for extreme_t, label in [(bf["min_t"], "min"), (bf["max_t"], "max")]:
            # ±1 year window around L1 extreme, clamped to overall window
            win_start = Time(max(
                (extreme_t.datetime.replace(year=extreme_t.datetime.year - 1)
                 ).strftime("%Y-%m-%d"),
                time_start.iso[:10]
            ))
            win_end = Time(min(
                (extreme_t.datetime.replace(year=extreme_t.datetime.year + 1)
                 ).strftime("%Y-%m-%d"),
                time_end.iso[:10]
            ))

            dates = _monthly_dates(win_start, win_end)
            if verbose:
                print(f"    [{band} {label}] {len(dates)} monthly samples "
                      f"around {extreme_t.iso[:10]}")

            for t in dates:
                s = _fetch_spectrum_at(t, source, lo, hi)
                if s is None:
                    continue
                all_spectra.append(s)
                f = _band_flux(s, lo, hi)
                if f < l2_band_flux[band]["min_f"]:
                    l2_band_flux[band]["min_f"] = f
                    l2_band_flux[band]["min_t"] = t
                if f > l2_band_flux[band]["max_f"]:
                    l2_band_flux[band]["max_f"] = f
                    l2_band_flux[band]["max_t"] = t

    # ── Level 3: Weekly refinement ────────────────────────────────────────────
    if verbose:
        print("\n  [Cascade L3] Weekly refinement around L2 extremes...")

    from datetime import timedelta as _td

    for band, (lo, hi) in _BANDS.items():
        if skip_l2[band]:
            epoch_min[band] = str(l1_band_flux[band]["min_t"].iso[:10])
            epoch_max[band] = str(l1_band_flux[band]["max_t"].iso[:10])
            continue

        bf = l2_band_flux[band]
        for extreme_t, label in [(bf["min_t"], "min"), (bf["max_t"], "max")]:
            # ±1 month window around L2 extreme
            win_start = Time(
                max((extreme_t.datetime - _td(days=30)).strftime("%Y-%m-%d"),
                    time_start.iso[:10])
            )
            win_end = Time(
                min((extreme_t.datetime + _td(days=30)).strftime("%Y-%m-%d"),
                    time_end.iso[:10])
            )

            dates = _weekly_dates(win_start, win_end)
            if verbose:
                print(f"    [{band} {label}] {len(dates)} weekly samples "
                      f"around {extreme_t.iso[:10]}")

            for t in dates:
                s = _fetch_spectrum_at(t, source, lo, hi)
                if s is None:
                    continue
                all_spectra.append(s)
                f = _band_flux(s, lo, hi)
                if label == "min" and f < bf["min_f"]:
                    bf["min_f"] = f
                    bf["min_t"] = t
                if label == "max" and f > bf["max_f"]:
                    bf["max_f"] = f
                    bf["max_t"] = t

        epoch_min[band] = str(l2_band_flux[band]["min_t"].iso[:10])
        epoch_max[band] = str(l2_band_flux[band]["max_t"].iso[:10])

    if verbose:
        print(f"\n  Cascade complete. Total spectra fetched: {len(all_spectra)}")
        for band in _BANDS:
            print(f"    [{band}] min epoch: {epoch_min[band]}  "
                  f"max epoch: {epoch_max[band]}")

    return all_spectra, epoch_min, epoch_max


# ══════════════════════════════════════════════════════════════════════════════
# Statistics
# ══════════════════════════════════════════════════════════════════════════════

def _compute_statistics(
    spectra      : list,
    source_info,
    wl_min_nm    : float,
    wl_max_nm    : float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-wavelength statistics from a list of Spectrum objects.

    Returns
    -------
    wl_nm, flux_min, flux_max, flux_mean, sigma_variability, sigma_calibration
    """
    # Use the first spectrum's wavelength grid as reference
    ref = spectra[0].slice(wl_min_nm * u.nm, wl_max_nm * u.nm)
    wl_nm = ref.wavelengths.value

    # Interpolate all spectra onto reference grid
    flux_matrix = []
    for s in spectra:
        try:
            s_sliced = s.interpolate(ref.wavelengths)
            flux_matrix.append(s_sliced.flux.value)
        except Exception:
            continue

    if len(flux_matrix) == 0:
        raise RuntimeError("No valid spectra to compute statistics from.")

    flux_matrix = np.array(flux_matrix)   # [n_spectra, n_wl]

    flux_min  = np.min(flux_matrix,  axis=0)
    flux_max  = np.max(flux_matrix,  axis=0)
    flux_mean = np.mean(flux_matrix, axis=0)
    sigma_var = np.std(flux_matrix,  axis=0, ddof=1)

    # Calibration sigma from registry
    sigma_cal = np.zeros_like(flux_mean)
    bands_cal = [
        (115,  200, source_info.sigma_abs_percent.get("UV_115_200",
               source_info.sigma_abs_percent.get("UV_120_200", 8.0))),
        (200,  300, source_info.sigma_abs_percent.get("UV_200_300", 5.0)),
        (300,  400, source_info.sigma_abs_percent.get("UV_300_400", 3.0)),
        (400,  700, source_info.sigma_abs_percent.get("VIS_400_700", 2.0)),
        (700, 1e6,  source_info.sigma_abs_percent.get("NIR_700_2400", 2.0)),
    ]
    for lo, hi, pct in bands_cal:
        mask = (wl_nm >= lo) & (wl_nm < hi)
        sigma_cal[mask] = flux_mean[mask] * (pct / 100.0)

    return wl_nm, flux_min, flux_max, flux_mean, sigma_var, sigma_cal


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def compute_variability(
    time_start   : Time,
    time_end     : Time,
    wl_min       : u.Quantity = 200 * u.nm,
    wl_max       : u.Quantity = 2400 * u.nm,
    source       : Optional[SpectralSource] = None,
    tolerance    : float = 0.001,
    verbose      : bool = True,
) -> VariabilityProfile:
    """
    Compute a data-driven solar variability profile over a time window.

    Parameters
    ----------
    time_start : astropy Time
        Start of time window.
    time_end : astropy Time
        End of time window.
    wl_min : astropy Quantity
        Minimum wavelength for cascade integration. Default 200 nm.
    wl_max : astropy Quantity
        Maximum wavelength for cascade integration. Default 2400 nm.
    source : SpectralSource, optional
        Override auto-selected source. If None, best available is chosen.
    tolerance : float
        Fractional variability threshold below which L2/L3 are skipped.
        Default 0.001 (0.1%).
    verbose : bool
        Print cascade progress. Default True.

    Returns
    -------
    VariabilityProfile
        Contains per-wavelength min, max, mean, sigma_variability,
        sigma_calibration, and combined sigma.

    Examples
    --------
    >>> from leos.solar_variability import compute_variability
    >>> from astropy.time import Time
    >>> from astropy import units as u
    >>>
    >>> profile = compute_variability(
    ...     time_start = Time("2003-01-01"),
    ...     time_end   = Time("2020-01-01"),
    ...     wl_min     = 200 * u.nm,
    ...     wl_max     = 2400 * u.nm,
    ...     verbose    = True,
    ... )
    >>> print(profile)
    >>> profile.save("my_variability.npz")
    """
    wl_min_nm = wl_min.to(u.nm).value
    wl_max_nm = wl_max.to(u.nm).value

    if time_end <= time_start:
        raise ValueError("time_end must be after time_start.")

    if verbose:
        print(f"\nLEOS Solar Variability Cascade")
        print(f"  Window  : {time_start.iso[:10]} – {time_end.iso[:10]}")
        print(f"  λ range : {wl_min_nm:.0f}–{wl_max_nm:.0f} nm")
        print(f"  Tolerance: {tolerance*100:.2f}%")

    # Auto-select source
    selected_source = _select_source(
        time_start, time_end, wl_min_nm, wl_max_nm, source
    )
    source_info = get_info(selected_source)

    if verbose:
        print(f"  Source  : {selected_source.value}")

    # Run cascade
    spectra, epoch_min, epoch_max = _run_cascade(
        source     = selected_source,
        time_start = time_start,
        time_end   = time_end,
        wl_min_nm  = wl_min_nm,
        wl_max_nm  = wl_max_nm,
        tolerance  = tolerance,
        verbose    = verbose,
    )

    # Compute statistics
    if verbose:
        print(f"\n  Computing per-wavelength statistics from "
              f"{len(spectra)} spectra...")

    wl_nm, flux_min, flux_max, flux_mean, sigma_var, sigma_cal = \
        _compute_statistics(spectra, source_info, wl_min_nm, wl_max_nm)

    profile = VariabilityProfile(
        wl_nm             = wl_nm,
        flux_min          = flux_min,
        flux_max          = flux_max,
        flux_mean         = flux_mean,
        sigma_variability = sigma_var,
        sigma_calibration = sigma_cal,
        epoch_min         = epoch_min,
        epoch_max         = epoch_max,
        source            = selected_source,
        wl_range          = (wl_min_nm, wl_max_nm),
        n_spectra         = len(spectra),
        time_start        = time_start,
        time_end          = time_end,
    )

    if verbose:
        print(f"\n{profile}")
        print(f"  σ range : {profile.sigma.min():.4f}–"
              f"{profile.sigma.max():.4f} W/m²/nm")

    return profile
