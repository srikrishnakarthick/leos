"""
leos.solar_spectrum
-------------------
Fetch, cache, and return solar spectral irradiance I(λ) ± σ(λ)
from any registered source in leos.spectral_sources.

Public API
----------
get_solar_spectrum(source, time, resolution, force_download) → Spectrum

Each source carries physically motivated σ(λ) from its own calibration
uncertainty. Time-resolved sources (TSIS-1, SORCE, UARS) return the
spectrum for the requested date. Static sources (ASTM) return their
fixed reference spectrum.

TSIS-1 is live-fetchable: if today's data is not cached, it is
downloaded automatically and cached locally.
"""

import os
import numpy as np
import requests
from datetime import datetime, timedelta
from astropy import units as u
from astropy.time import Time

from leos.spectrum import Spectrum
from leos.spectral_sources import SpectralSource, SpectralSourceInfo, REGISTRY, get_info

# ── Cache directory ───────────────────────────────────────────────────────────
_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(_CACHE_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def get_solar_spectrum(
    source: SpectralSource = SpectralSource.ASTM_E490,
    time: Time | None = None,
    resolution: str = "high",
    force_download: bool = False,
) -> Spectrum:
    """
    Return solar spectral irradiance I(λ) ± σ(λ) at 1 AU.

    Parameters
    ----------
    source : SpectralSource
        Which data source to use. Default ASTM_E490 (static reference).
        Use SpectralSource.TSIS1_SIM for live daily data.
    time : astropy Time, optional
        Observation date. Required for time-resolved sources
        (TSIS1_SIM, SORCE_SIM, UARS_SUSIM, NRLSSI2, SATIRE_S).
        Ignored for static sources (ASTM_E490, ANALYTIC).
    resolution : str
        'high' = native resolution. 'low' = 50-point coarse grid.
    force_download : bool
        Re-download even if cached.

    Returns
    -------
    Spectrum
        I(λ) in W/m²/nm with σ(λ) from calibration + variability.
        Spectrum.label includes source name and date.

    Raises
    ------
    ValueError
        If time is required but not provided.
    ValueError
        If requested time is outside the source's valid window.

    Examples
    --------
    >>> from leos.solar_spectrum import get_solar_spectrum
    >>> from leos.spectral_sources import SpectralSource
    >>> from astropy.time import Time
    >>>
    >>> # Static reference
    >>> s = get_solar_spectrum(SpectralSource.ASTM_E490)
    >>>
    >>> # Today's TSIS-1 spectrum (auto-downloaded)
    >>> s = get_solar_spectrum(SpectralSource.TSIS1_SIM, time=Time.now())
    """
    info = get_info(source)

    # ── Validate time ──────────────────────────────────────────────────────
    if info.is_time_resolved:
        if time is None:
            raise ValueError(
                f"{source.value} is time-resolved. "
                f"Provide time=astropy.Time(...). "
                f"Valid window: {info.time_note}"
            )
        if not info.covers_time(time):
            raise ValueError(
                f"{source.value} does not cover {time.iso}. "
                f"Valid window: {info.time_note}"
            )

    # ── Dispatch to source-specific fetcher ────────────────────────────────
    fetchers = {
        SpectralSource.ASTM_E490  : _get_astm,
        SpectralSource.UARS_SUSIM : _get_uars,
        SpectralSource.SORCE_SIM  : _get_sorce,
        SpectralSource.TSIS1_SIM  : _get_tsis1,
        SpectralSource.NRLSSI2    : _get_nrlssi2,
        SpectralSource.SATIRE_S   : _get_satire,
        SpectralSource.ANALYTIC   : _get_analytic,
    }
    spectrum = fetchers[source](info, time, force_download)
    return _resample(spectrum, resolution)


# ══════════════════════════════════════════════════════════════════════════════
# Private fetchers — one per source
# ══════════════════════════════════════════════════════════════════════════════

# ── ASTM E-490 ────────────────────────────────────────────────────────────────

def _get_astm(info: SpectralSourceInfo, time, force_download: bool) -> Spectrum:
    cache_path = os.path.join(_CACHE_DIR, "astm_e490.npz")
    if not force_download and os.path.exists(cache_path):
        return _load_npz(cache_path, info)

    try:
        print("Fetching ASTM E-490 from NREL...")
        r = requests.get(info.fetch_url, timeout=30)
        r.raise_for_status()
        wl, fl = [], []
        for line in r.text.splitlines():
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            try:
                w = float(parts[0])
                f = float(parts[1])
                if 119.5 <= w <= 100_000:
                    wl.append(w)
                    fl.append(f)
            except ValueError:
                continue
        if len(wl) < 10:
            raise ValueError("Fewer than 10 data points parsed.")

        wl_arr = np.array(wl)
        fl_arr = np.array(fl)
        # Absolute calibration σ: wavelength-dependent per registry
        sigma_arr = _astm_sigma(wl_arr, fl_arr, info)
        _save_npz(cache_path, wl_arr, fl_arr, sigma_arr)
        print(f"  cached to {cache_path}")

    except Exception as e:
        print(f"  ASTM download failed ({e}), using analytic fallback.")
        return _get_analytic(
            get_info(SpectralSource.ANALYTIC), time, force_download
        )

    return _build_spectrum(wl_arr, fl_arr, sigma_arr, info, time)


def _astm_sigma(wl_nm, flux, info: SpectralSourceInfo):
    """Wavelength-resolved σ for ASTM E-490 from registry calibration percents."""
    sigma = np.zeros_like(flux)
    bands = {
        (119.5, 200): info.sigma_abs_percent.get("UV_120_200", 10.0),
        (200,   300): info.sigma_abs_percent.get("UV_200_300",  5.0),
        (300,   400): info.sigma_abs_percent.get("UV_300_400",  3.0),
        (400,   700): info.sigma_abs_percent.get("VIS_400_700", 2.0),
        (700, 1e6)  : info.sigma_abs_percent.get("NIR_700_2400",2.0),
    }
    for (lo, hi), pct in bands.items():
        mask = (wl_nm >= lo) & (wl_nm < hi)
        sigma[mask] = flux[mask] * (pct / 100.0)
    return sigma


# ── UARS SUSIM ────────────────────────────────────────────────────────────────

def _get_uars(info: SpectralSourceInfo, time: Time, force_download: bool) -> Spectrum:
    date_str  = time.datetime.strftime("%Y-%m-%d")
    cache_path = os.path.join(_CACHE_DIR, f"uars_susim_{date_str}.npz")

    if not force_download and os.path.exists(cache_path):
        return _load_npz(cache_path, info)

    try:
        print(f"Fetching UARS SUSIM for {date_str}...")
        # LISIRD time-filtered request
        url = (
            f"{info.fetch_url}?"
            f"time>={date_str}T00:00:00"
            f"&time<={date_str}T23:59:59"
        )
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        wl, fl, unc = _parse_lisird_csv(r.text)

        if len(wl) == 0:
            raise ValueError(f"No UARS data found for {date_str}.")

        sigma = _instrument_sigma(wl, fl, unc, info)
        _save_npz(cache_path, wl, fl, sigma)

    except Exception as e:
        print(f"  UARS fetch failed ({e}). Check date is within 1991-2005.")
        raise

    return _build_spectrum(wl, fl, sigma, info, time)


# ── SORCE SIM ─────────────────────────────────────────────────────────────────

def _get_sorce(info: SpectralSourceInfo, time: Time, force_download: bool) -> Spectrum:
    date_str  = time.datetime.strftime("%Y-%m-%d")
    cache_path = os.path.join(_CACHE_DIR, f"sorce_sim_{date_str}.npz")

    if not force_download and os.path.exists(cache_path):
        return _load_npz(cache_path, info)

    try:
        print(f"Fetching SORCE SIM for {date_str}...")
        url = (
            f"{info.fetch_url}?"
            f"time>={date_str}T00:00:00"
            f"&time<={date_str}T23:59:59"
        )
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        wl, fl, unc = _parse_lisird_csv(r.text)

        if len(wl) == 0:
            raise ValueError(f"No SORCE data for {date_str}.")

        sigma = _instrument_sigma(wl, fl, unc, info)
        _save_npz(cache_path, wl, fl, sigma)

    except Exception as e:
        print(f"  SORCE fetch failed ({e}). Check date is within 2003-2020.")
        raise

    return _build_spectrum(wl, fl, sigma, info, time)


# ── TSIS-1 SIM (live) ─────────────────────────────────────────────────────────

def _get_tsis1(info: SpectralSourceInfo, time: Time, force_download: bool) -> Spectrum:
    date_str  = time.datetime.strftime("%Y-%m-%d")
    cache_path = os.path.join(_CACHE_DIR, f"tsis1_sim_{date_str}.npz")

    # For today's date: cache expires after 24 hours
    if not force_download and os.path.exists(cache_path):
        age_hours = (
            datetime.now().timestamp() - os.path.getmtime(cache_path)
        ) / 3600
        is_today = date_str == datetime.now().strftime("%Y-%m-%d")
        if not (is_today and age_hours > 24):
            return _load_npz(cache_path, info)

    try:
        print(f"Fetching TSIS-1 SIM for {date_str}...")
        url = (
            f"{info.fetch_url}?"
            f"time>={date_str}T00:00:00"
            f"&time<={date_str}T23:59:59"
        )
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        wl, fl, unc = _parse_lisird_csv(r.text)

        if len(wl) == 0:
            # TSIS might lag by 1-2 days — try yesterday
            yesterday = (time.datetime - timedelta(days=1)).strftime("%Y-%m-%d")
            print(f"  No data for {date_str}, trying {yesterday}...")
            url = (
                f"{info.fetch_url}?"
                f"time>={yesterday}T00:00:00"
                f"&time<={yesterday}T23:59:59"
            )
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            wl, fl, unc = _parse_lisird_csv(r.text)

        if len(wl) == 0:
            raise ValueError("No TSIS-1 data available for this date or yesterday.")

        sigma = _instrument_sigma(wl, fl, unc, info)
        _save_npz(cache_path, wl, fl, sigma)
        print(f"  cached: {cache_path}")

    except Exception as e:
        print(f"  TSIS-1 fetch failed ({e}), falling back to ASTM E-490.")
        return _get_astm(get_info(SpectralSource.ASTM_E490), time, False)

    return _build_spectrum(wl, fl, sigma, info, time)


# ── NRLSSI2 ───────────────────────────────────────────────────────────────────

def _get_nrlssi2(info: SpectralSourceInfo, time: Time, force_download: bool) -> Spectrum:
    date_str  = time.datetime.strftime("%Y-%m-%d")
    cache_path = os.path.join(_CACHE_DIR, f"nrlssi2_{date_str}.npz")

    if not force_download and os.path.exists(cache_path):
        return _load_npz(cache_path, info)

    try:
        print(f"Fetching NRLSSI2 for {date_str}...")
        url = (
            f"{info.fetch_url}?"
            f"time>={date_str}T00:00:00"
            f"&time<={date_str}T23:59:59"
        )
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        wl, fl, unc = _parse_lisird_csv(r.text)

        if len(wl) == 0:
            raise ValueError(f"No NRLSSI2 data for {date_str}.")

        sigma = _instrument_sigma(wl, fl, unc, info)
        _save_npz(cache_path, wl, fl, sigma)

    except Exception as e:
        print(f"  NRLSSI2 fetch failed ({e}).")
        raise

    return _build_spectrum(wl, fl, sigma, info, time)


# ── SATIRE-S ──────────────────────────────────────────────────────────────────

def _get_satire(info: SpectralSourceInfo, time: Time, force_download: bool) -> Spectrum:
    """
    SATIRE-S is distributed as bulk download files, not a day-queryable API.
    Users must download manually from MPS and place in leos/data/.
    This function checks for the local file and raises clearly if absent.
    """
    local_path = os.path.join(_CACHE_DIR, "satire_s.npz")
    if not os.path.exists(local_path):
        raise FileNotFoundError(
            "SATIRE-S data not found. Download from:\n"
            "  https://www2.mps.mpg.de/projects/sun-climate/data.html\n"
            "Convert to npz using scripts/convert_satire.py and place in leos/data/."
        )
    return _load_npz(local_path, info)


# ── Analytic fallback ─────────────────────────────────────────────────────────

def _get_analytic(info: SpectralSourceInfo, time, force_download: bool) -> Spectrum:
    wl_nm = np.linspace(200, 4000, 1000)
    h, c, kB, T = 6.626e-34, 3.0e8, 1.381e-23, 5778.0
    wl_m = wl_nm * 1e-9
    B = (2 * h * c**2 / wl_m**5) / (np.exp(h * c / (wl_m * kB * T)) - 1)
    flux_per_nm = B * 6.8e-5 * np.pi * 1e-9
    wl_q = wl_nm * u.nm
    fl_q = flux_per_nm * u.W / u.m**2 / u.nm
    from leos.spectrum import Spectrum as _Spec
    s_raw = _Spec(wl_q, fl_q)
    scale = 1361.0 / s_raw.integrate().value
    fl_scaled = flux_per_nm * scale
    sigma = fl_scaled * (info.sigma_abs_percent.get("ALL", 5.0) / 100.0)
    return _build_spectrum(wl_nm, fl_scaled, sigma, info, time)


# ══════════════════════════════════════════════════════════════════════════════
# Shared utilities
# ══════════════════════════════════════════════════════════════════════════════

def _parse_lisird_csv(text: str):
    """
    Parse LASP LISIRD CSV response into (wavelength, flux, uncertainty) arrays.
    Handles variable column orders by reading header line.
    """
    lines = text.strip().splitlines()
    if not lines:
        return np.array([]), np.array([]), np.array([])

    # Find header line
    header = None
    data_start = 0
    for i, line in enumerate(lines):
        if "wavelength" in line.lower() or "irradiance" in line.lower():
            header = [c.strip().lower() for c in line.split(",")]
            data_start = i + 1
            break

    if header is None:
        return np.array([]), np.array([]), np.array([])

    # Find column indices flexibly
    wl_col  = next((i for i, h in enumerate(header) if "wavelength" in h), None)
    fl_col  = next((i for i, h in enumerate(header)
                    if "irradiance" in h and "uncertainty" not in h), None)
    unc_col = next((i for i, h in enumerate(header) if "uncertainty" in h), None)

    if wl_col is None or fl_col is None:
        return np.array([]), np.array([]), np.array([])

    wl, fl, unc = [], [], []
    for line in lines[data_start:]:
        parts = line.split(",")
        try:
            w = float(parts[wl_col])
            f = float(parts[fl_col])
            u_val = float(parts[unc_col]) if unc_col is not None else 0.0
            wl.append(w)
            fl.append(f)
            unc.append(u_val)
        except (ValueError, IndexError):
            continue

    return np.array(wl), np.array(fl), np.array(unc)


def _instrument_sigma(wl_nm, flux, instrumental_unc, info: SpectralSourceInfo):
    """
    Combine instrumental uncertainty with absolute calibration uncertainty.
    σ_total² = σ_instrument² + σ_calibration²
    """
    # Absolute calibration component (wavelength-dependent)
    sigma_cal = np.zeros_like(flux)
    bands = {
        (115,  170): "UV_115_170",
        (170,  200): "UV_170_300",
        (200,  300): "UV_200_300",
        (300,  400): "UV_300_400",
        (400,  700): "VIS_400_700",
        (700, 1e6) : "NIR_700_2400",
    }
    for (lo, hi), key in bands.items():
        pct = info.sigma_abs_percent.get(key, 2.0)
        mask = (wl_nm >= lo) & (wl_nm < hi)
        sigma_cal[mask] = flux[mask] * (pct / 100.0)

    # Quadrature combination
    return np.sqrt(instrumental_unc**2 + sigma_cal**2)


def _build_spectrum(wl_nm, flux, sigma, info: SpectralSourceInfo, time) -> Spectrum:
    from leos.spectrum import Spectrum
    date_str = time.iso[:10] if time is not None else "static"
    label = f"{info.source.value} | {date_str}"
    return Spectrum(
        wl_nm * u.nm,
        flux  * u.W / u.m**2 / u.nm,
        uncertainty=sigma * u.W / u.m**2 / u.nm,
        label=label,
    )


def _save_npz(path, wl, flux, sigma):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, wavelengths=wl, flux=flux, uncertainty=sigma)


def _load_npz(path, info: SpectralSourceInfo) -> Spectrum:
    from leos.spectrum import Spectrum
    data = np.load(path)
    return Spectrum(
        data["wavelengths"] * u.nm,
        data["flux"]        * u.W / u.m**2 / u.nm,
        uncertainty=data["uncertainty"] * u.W / u.m**2 / u.nm,
        label=f"{info.source.value} (cached)",
    )


def _resample(spectrum: Spectrum, resolution: str) -> Spectrum:
    if resolution == "low":
        wl_new = np.linspace(
            spectrum.wavelengths[0].value,
            spectrum.wavelengths[-1].value,
            50
        ) * u.nm
        return spectrum.interpolate(wl_new)
    return spectrum
