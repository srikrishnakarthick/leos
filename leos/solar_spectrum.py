"""
leos.solar_spectrum
-------------------
Fetches, caches, and returns the reference solar spectrum I(lambda)
used as the top-of-atmosphere input for all radiative transfer.

Source: ASTM E-490 Air Mass Zero solar spectrum (standard reference).
Fallback: Kurucz 2005 synthetic spectrum via numpy generation.

The spectrum is cached locally after first download so subsequent
calls are instantaneous.
"""

import os
import numpy as np
import requests
from astropy import units as u
from leos.spectrum import Spectrum

# ── Cache location ────────────────────────────────────────────────────────────
_CACHE_DIR  = os.path.join(os.path.dirname(__file__), "..", "kernels", "data")
_CACHE_FILE = os.path.join(_CACHE_DIR, "solar_spectrum.npz")

# ── ASTM E-490 source ─────────────────────────────────────────────────────────
_ASTM_URL = (
    "https://www.nrel.gov/grid/solar-resource/assets/data/astmg173.csv"
)


def get_solar_spectrum(resolution="high", force_download=False) -> Spectrum:
    """
    Return the reference solar spectrum at 1 AU.

    Parameters
    ----------
    resolution : str
        'high' — full resolution (hundreds of points across UV-NIR)
        'low'  — coarse 50-point version for fast tests
    force_download : bool
        If True, re-download even if cache exists.

    Returns
    -------
    Spectrum
        Wavelengths in nm, flux in W/m²/nm, at 1 AU.

    Notes
    -----
    Total integrated flux should be ~1361 W/m² (solar constant at 1 AU).
    """
    if not force_download and os.path.exists(_CACHE_FILE):
        return _load_cached(resolution)

    # Try downloading ASTM E-490
    try:
        print("Downloading reference solar spectrum (ASTM E-490)...")
        spectrum = _download_astm()
        _save_cache(spectrum)
        print(f"  cached to {_CACHE_FILE}")
        return _resample(spectrum, resolution)
    except Exception as e:
        print(f"  download failed ({e}), using analytic fallback.")
        return _analytic_spectrum(resolution)


def _download_astm() -> Spectrum:
    """Download and parse the ASTM E-490 spectrum from NREL."""
    response = requests.get(_ASTM_URL, timeout=30)
    response.raise_for_status()

    lines = response.text.splitlines()

    wavelengths = []
    flux_toa    = []

    # Skip header rows — data starts after lines beginning with wavelength values
    for line in lines:
        parts = line.strip().split(",")
        if len(parts) < 2:
            continue
        try:
            wl = float(parts[0])
            fl = float(parts[1])   # W/m²/nm extraterrestrial
            if 200 <= wl <= 4000:
                wavelengths.append(wl)
                flux_toa.append(fl)
        except ValueError:
            continue

    if len(wavelengths) < 10:
        raise ValueError("Parsed fewer than 10 data points — format may have changed.")

    wl_arr = np.array(wavelengths) * u.nm
    fl_arr = np.array(flux_toa)    * u.W / u.m**2 / u.nm

    return Spectrum(wl_arr, fl_arr, label="ASTM E-490 (1 AU)")


def _analytic_spectrum(resolution="high") -> Spectrum:
    """
    Analytic blackbody fallback spectrum at T=5778K scaled to 1 AU.
    Used when ASTM download fails.
    Total flux integrates to ~1361 W/m².
    """
    if resolution == "low":
        wl_nm = np.linspace(200, 4000, 50)
    else:
        wl_nm = np.linspace(200, 4000, 1000)

    # Planck function in W/m²/nm/sr, scaled to solar disk solid angle
    h  = 6.626e-34   # J·s
    c  = 3.0e8       # m/s
    kB = 1.381e-23   # J/K
    T  = 5778.0      # K — solar effective temperature

    wl_m  = wl_nm * 1e-9
    B_lam = (2 * h * c**2 / wl_m**5) / (np.exp(h * c / (wl_m * kB * T)) - 1)

    # Convert W/m²/m/sr → W/m²/nm at 1 AU
    # Solar solid angle at 1 AU: Ω = π * (R_sun/d)² = 6.8e-5 sr
    omega_sun = 6.8e-5   # sr
    flux_per_nm = B_lam * omega_sun * np.pi * 1e-9  # per nm

    # Normalise so total flux = 1361 W/m²
    wl_q  = wl_nm * u.nm
    fl_q  = flux_per_nm * u.W / u.m**2 / u.nm
    s_raw = Spectrum(wl_q, fl_q, label="analytic")
    scale = (1361.0 / s_raw.integrate().value)
    fl_scaled = flux_per_nm * scale * u.W / u.m**2 / u.nm

    return Spectrum(wl_nm * u.nm, fl_scaled, label="Analytic blackbody 5778K (1 AU)")


def _save_cache(spectrum: Spectrum):
    """Save spectrum arrays to .npz cache."""
    os.makedirs(_CACHE_DIR, exist_ok=True)
    np.savez(
        _CACHE_FILE,
        wavelengths=spectrum.wavelengths.value,
        flux=spectrum.flux.value,
        uncertainty=spectrum.uncertainty.value,
    )


def _load_cached(resolution="high") -> Spectrum:
    """Load spectrum from .npz cache."""
    data = np.load(_CACHE_FILE)
    wl = data["wavelengths"] * u.nm
    fl = data["flux"]        * u.W / u.m**2 / u.nm
    uc = data["uncertainty"] * u.W / u.m**2 / u.nm
    s  = Spectrum(wl, fl, uc, label="ASTM E-490 (cached)")
    return _resample(s, resolution)


def _resample(spectrum: Spectrum, resolution="high") -> Spectrum:
    """Resample to low resolution if requested."""
    if resolution == "low":
        wl_new = np.linspace(
            spectrum.wavelengths[0].value,
            spectrum.wavelengths[-1].value,
            50
        ) * u.nm
        return spectrum.interpolate(wl_new)
    return spectrum
