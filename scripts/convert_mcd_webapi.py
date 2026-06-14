#!/usr/bin/env python3
"""
scripts/convert_mcd_webapi.py
------------------------------
Query the Mars Climate Database v6.1 web interface (LMD) to build
altitude-resolved atmospheric profiles for use in LEOS.

This script queries the MCD CGI one pressure level at a time, for any
combination of MCD variables (see --list-vars), assembles a full
profile, and writes a .npz file in the LEOS atmospheric column schema —
identical to what convert_merra2.py produces for Earth, so
AtmosphericColumn.from_npz() works identically.

MCD Web Interface
-----------------
  URL  : https://www-mars.lmd.jussieu.fr/mcd_python/cgi-bin/mcdcgi.py
  Docs : https://www-mars.lmd.jussieu.fr/mcd_python/
  Rate : ~1 request/second recommended (be a good citizen)

Variable catalog
----------------
  Run `python scripts/convert_mcd_webapi.py --list-vars` to see all
  ~85 queryable MCD variables (temperature, winds, composition VMRs,
  dust/water columns, radiative fluxes, RMS variability, etc.)

Dust scenario codes
-------------------
  1 = climatology average (default, recommended)
  2 = cold / minimum dust
  3 = warm / maximum dust
  (see --list-dust for the full set, including dust storm and
  per-Martian-Year scenarios)

Pressure levels queried (Pa) — 22 levels, 0–~120 km
-----------------------------------------------------
  700, 600, 500, 400, 300, 200, 150, 100, 70, 50, 30, 20,
  10, 5, 2, 1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01

Output schema (leos/data/mcd_Ls???_??lat_dust?.npz)
----------------------------------------------------
  z_km          float64[n]   altitude above areoid [km]
  <var_code>    float64[n]   one array per requested variable
                              (e.g. T_K, P_Pa, rho, vmr_co2, ...)
  n_total       float64[n]   number density [molecules/m3]
                              (only if T_K and P_Pa were requested)
  sigma_T       float64[n]   temperature uncertainty [K]
                              (only if T_K was requested)
  tsurf_K       float64      surface temperature [K]
  ps_Pa         float64      surface pressure [Pa]
  lat           float64      latitude [deg N]
  lon           float64      longitude [deg E]
  ls            float64      solar longitude [deg]
  localtime     float64      local solar time [hrs]
  dust_code     int          MCD dust scenario code
  dust_label    str          'avg'|'min'|'max'|...
  source        str          'mcd_webapi'
  mcd_version   str          'MCD_v6.1'

Usage
-----
  # Default profile (T, P, rho, altitude) — same as before
  python scripts/convert_mcd_webapi.py --lat 18.4 --lon 77.5 --ls 0

  # Custom variable set (any combination, any length)
  python scripts/convert_mcd_webapi.py --lat 18.4 --lon 77.5 --ls 0 \
      --vars t,p,rho,zareoid,vmr_co2,vmr_n2,vmr_ar,rmst

  # List all queryable variables
  python scripts/convert_mcd_webapi.py --list-vars

  # List all dust scenario codes
  python scripts/convert_mcd_webapi.py --list-dust

  # Polar, Ls=270, minimum dust
  python scripts/convert_mcd_webapi.py --lat 75 --lon 0 --ls 270 --dust 2

  # Generate all 36 bundled profiles (4 seasons x 3 dust x 3 lat bands)
  python scripts/convert_mcd_webapi.py --generate-bundled

  # Custom output path
  python scripts/convert_mcd_webapi.py --lat 0 --lon 0 --ls 90 --output my_mars.npz
"""

import argparse
import os
import sys
import time
import re
import numpy as np

try:
    import requests
except ImportError:
    sys.exit("ERROR: requests not installed. Run: pip install requests")

# ── MCD web API endpoint ──────────────────────────────────────────────────────
_MCD_URL = "https://www-mars.lmd.jussieu.fr/mcd_python/cgi-bin/mcdcgi.py"

# ── LEOS output directory ─────────────────────────────────────────────────────
_DEFAULT_OUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "leos", "data"
)

# ── Physical constants ────────────────────────────────────────────────────────
_kB   = 1.380649e-23   # J/K
_M_CO2 = 44.01e-3      # kg/mol  (Mars atmosphere ~95% CO2)
_NA    = 6.02214076e23
_m_CO2 = _M_CO2 / _NA  # kg per molecule

# ── Dust code mapping ─────────────────────────────────────────────────────────
# Short labels used in npz filenames / dust_label field.
_DUST_CODES = {
    1: "avg",       # climatology ave solar
    2: "min",       # climatology min solar
    3: "max",       # climatology max solar
    4: "stormmin",  # dust storm min solar
    5: "stormavg",  # dust storm ave solar
    6: "stormmax",  # dust storm max solar
    7: "warm",      # warm (dusty, max solar)
    8: "cold",      # cold (low dust, min solar)
}
# Martian-year-specific scenarios (24-35) are also valid dust codes on the
# MCD form; not given short labels here but usable via --dust directly.
_DUST_FULL_LABELS = {
    1: "climatology ave solar",
    2: "climatology min solar",
    3: "climatology max solar",
    4: "dust storm min solar",
    5: "dust storm ave solar",
    6: "dust storm max solar",
    7: "warm (dusty, max solar)",
    8: "cold (low dust, min solar)",
    **{my: f"Martian Year {my}" for my in range(24, 36)},
}

# ── Pressure levels to query [Pa] ────────────────────────────────────────────
_PRESSURE_LEVELS_PA = [
    700, 600, 500, 400, 300, 200, 150, 100,
    70, 50, 30, 20, 10, 5, 2, 1,
    0.5, 0.2, 0.1, 0.05, 0.02, 0.01,
]

# ── MCD T uncertainty fallback bands (used only if 'rmst' not requested) ────
# Based on MCD v6.1 validation against TES/MCS (Forget et al. 1999,
# Millour et al. 2015). Larger at high altitudes where GCM is less
# constrained. Prefer requesting --vars ...,rmst for real values.
_SIGMA_T_BANDS = [
    (0,   10,  5.0),   # surface–10 km: well-constrained, ~5 K
    (10,  30,  8.0),   # mid-troposphere
    (30,  60, 12.0),   # upper atmosphere
    (60, 200, 20.0),   # thermosphere: model uncertainty large
]

# ── 36-profile bundled set definition ────────────────────────────────────────
_BUNDLED_LS     = [0, 90, 180, 270]
_BUNDLED_LATS   = [0, 45, 75]
_BUNDLED_DUSTS  = [1, 2, 3]

# ── Default variable group (reproduces original script behavior) ────────────
_DEFAULT_VAR_GROUP = ["t", "p", "rho", "zareoid"]

# Output array names for vars that map onto the original LEOS schema fields.
_SCHEMA_ALIASES = {
    "t": "T_K",
    "p": "P_Pa",
}


# ══════════════════════════════════════════════════════════════════════════════
# Full MCD variable catalog (from listvar.js + default var1 option)
# code -> human-readable label, as shown on the MCD web form
# ══════════════════════════════════════════════════════════════════════════════
MCD_VARIABLES = {
    "t":              "Temperature (K)",
    "p":              "Pressure (Pa)",
    "rho":            "Density (kg/m3)",
    "u":              "W-E wind component (m/s)",
    "v":              "S-N wind component (m/s)",
    "wind":           "Horizontal wind speed (m/s)",
    "zradius":        "Radial distance from planet center (m)",
    "zareoid":        "Altitude above areoid (Mars geoid) (m)",
    "zsurface":       "Altitude above local surface (m)",
    "oroheight":      "Orographic height (m) (surface altitude above areoid)",
    "oro_gcm":        "GCM orography (m)",
    "theta_s":        "Local slope inclination (deg) (HR mode only)",
    "psi_s":          "Local slope orientation (deg) (HR mode only)",
    "marsau":         "Sun-Mars distance (in Astronomical Unit AU)",
    "ls":             "Ls, solar longitude of Mars (deg)",
    "loctime":        "LST:Local true solar time (hrs)",
    "lmeantime":      "LMT:Local mean time (hrs) at sought longitude",
    "utime":          "Universal solar time (LST at lon=0) (hrs)",
    "solzenang":      "Solar zenith angle (deg)",
    "tsurf":          "Surface temperature (K)",
    "ps":             "Surface pressure (Pa)",
    "ps_gcm":         "GCM surface pressure (Pa)",
    "potential_temp": "Potential temperature (K) (reference pressure=610Pa)",
    "w_l":            "Downward vertical wind component (m/s)",
    "zonal_slope_wind":  "Zonal slope wind component (m/s) (HR mode only)",
    "merid_slope_wind":  "Meridional slope wind component (m/s) (HR mode only)",
    "rmsps":          "Surface pressure RMS day to day variations (Pa)",
    "rmstsurf":       "Surface temperature RMS day to day variations (K)",
    "altrmsp":        "Atmospheric pressure RMS day to day variations (Pa)",
    "rmsrho":         "Density RMS day to day variations (kg/m^3)",
    "rmst":           "Temperature RMS day to day variations (K)",
    "rmsu":           "Zonal wind RMS day to day variations (m/s)",
    "rmsv":           "Meridional wind RMS day to day variations (m/s)",
    "rmsw":           "Vertical wind RMS day to day variations (m/s)",
    "fluxtop_dn_sw":  "Incident solar flux at top of the atmosphere (W/m2)",
    "fluxtop_up_sw":  "solar flux reflected to space (W/m2)",
    "fluxsurf_dn_sw": "Incident solar flux on horizontal surface (W/m2)",
    "fluxsurf_dn_sw_hr": "Incident solar flux on local slope (W/m2) (HR mode only)",
    "fluxsurf_up_sw": "Reflected solar flux on horizontal surface (W/m2)",
    "fluxtop_lw":     "thermal IR flux to space (W/m2)",
    "fluxsurf_lw":    "thermal IR flux on surface (W/m2)",
    "z_0":            "GCM surface roughness length z0 (m)",
    "thermal_inertia":"GCM surface thermal inertia",
    "ground_albedo":  "GCM surface bare ground albedo",
    "dod":            "Monthly mean dust column visible optical depth",
    "tauref":         "Daily mean dust column visible optical depth",
    "dust_mmr":       "Dust mass mixing ratio (kg/kg)",
    "dust_reff":      "Dust effective radius (m)",
    "dust_dep":       "Daily mean dust deposition rate (kg m-2 s-1)",
    "co2ice":         "Monthly mean surface CO2 ice layer (kg/m2)",
    "surf_h2o_ice":   "Monthly mean surface H2O layer (kg/m2)",
    "water_cap":      "GCM perennial surface water ice (0 or 1)",
    "col_h2ovapor":   "Water vapor column (kg/m2)",
    "vmr_h2o":        "Water vapor vol. mixing ratio (mol/mol)",
    "col_h2oice":     "Water ice column (kg/m2)",
    "vmr_h2oice":     "Water ice mixing ratio (mol/mol)",
    "h2oice_reff":    "Water ice effective radius (m)",
    "zmax":           "Convective Planetary Boundary Layer (PBL) height (m)",
    "wstar_up":       "Max. upward convective wind within the PBL (m/s)",
    "wstar_dn":       "Max. downward convective wind within the PBL (m/s)",
    "vvv":            "Convective vertical wind variance at level z (m2/s2)",
    "vhf":            "Convective eddy vertical heat flux at level z (m/s/K)",
    "surfstress":     "Surface wind stress (kg/m/s2)",
    "sensib_flux":    "Surface sensible heat flux (W/m2)",
    "Cp":             "Air heat capacity Cp (J kg-1 K-1)",
    "gamma":          "Ratio of specific heats Cp/Cv",
    "Rgas":           "Molecular gas constant R (J K-1 kg-1)",
    "viscosity":      "Air viscosity estimation (N s m-2)",
    "pscaleheight":   "Scale height H(p) (m)",
    "vmr_co2":        "[CO2] volume mixing ratio (mol/mol)",
    "vmr_n2":         "[N2] volume mixing ratio  (mol/mol)",
    "vmr_ar":         "[Ar] volume mixing ratio  (mol/mol)",
    "vmr_co":         "[CO] volume mixing ratio  (mol/mol)",
    "vmr_o":          "[O] volume mixing ratio   (mol/mol)",
    "vmr_o2":         "[O2] volume mixing ratio  (mol/mol)",
    "vmr_o3":         "[O3] volume mixing ratio  (mol/mol)",
    "vmr_h":          "[H] volume mixing ratio   (mol/mol)",
    "vmr_h2":         "[H2] volume mixing ratio  (mol/mol)",
    "vmr_he":         "[He] volume mixing ratio  (mol/mol)",
    "col_co2":        "CO2 column (kg/m2)",
    "col_n2":         "N2 column  (kg/m2)",
    "col_ar":         "Ar column  (kg/m2)",
    "col_co":         "CO column  (kg/m2)",
    "col_o":          "O column   (kg/m2)",
    "col_o2":         "O2 column  (kg/m2)",
    "col_o3":         "O3 column  (kg/m2)",
    "col_h":          "H column   (kg/m2)",
    "col_h2":         "H2 column  (kg/m2)",
    "col_he":         "He column  (kg/m2)",
    "vmr_elec":       "Electron number density (particules/cm3)",
    "col_elec":       "Total electonic content (TEC) (particules/m2)",
}

# Variables that only return data when HR (high-resolution topography)
# mode is enabled. We don't enable HR mode by default, so requesting
# these will generally yield NaN.
_HR_ONLY_VARS = {
    "theta_s", "psi_s", "zonal_slope_wind",
    "merid_slope_wind", "fluxsurf_dn_sw_hr",
}

# Variables that are constant with altitude — either true surface
# properties or column-integrated quantities — and should be queried
# once (at zkey=3, altitude=10m) rather than at every pressure level.
_SURFACE_VARS = {
    # true surface/terrain properties
    "tsurf", "ps", "ps_gcm", "theta_s", "psi_s",
    "z_0", "thermal_inertia", "ground_albedo",
    "oroheight", "oro_gcm",
    "zonal_slope_wind", "merid_slope_wind", "fluxsurf_dn_sw_hr",
    # surface fluxes / boundary-layer diagnostics
    "surfstress", "sensib_flux", "zmax", "wstar_up", "wstar_dn",
    "co2ice", "surf_h2o_ice", "water_cap",
    "dod", "tauref", "dust_dep",
    # column-integrated quantities (NOTE: assumed altitude-independent;
    # verify with --vars t,p,zareoid,col_co2 at two different pressure
    # levels if in doubt)
    "col_co2", "col_n2", "col_ar", "col_co", "col_o", "col_o2", "col_o3",
    "col_h", "col_h2", "col_he", "col_h2ovapor", "col_h2oice", "col_elec",
    # global/orbital scalars
    "marsau",
}

# ══════════════════════════════════════════════════════════════════════════════
# Response parsing
# ══════════════════════════════════════════════════════════════════════════════

_FLOAT_RE = r"([+\-]?\d*\.?\d+(?:[eE][+\-]?\d+)?)"


def _build_var_regex(code):
    """
    Build a regex that matches an MCD response line of the form:

        <Label> [(unit)] [(extra)] ......... <value>

    anchored to the start of a line, where <Label> is the portion of the
    variable's catalog label before its first parenthetical group. This
    disambiguates e.g. 'Temperature (K)' from
    'Temperature RMS day to day variations (K)'.
    """
    label = MCD_VARIABLES[code]
    prefix = label.split("(")[0].strip()
    pattern = (
        r"^\s*" + re.escape(prefix) +
        r"(?:\s*\([^)]*\))*\s*\.{2,}\s*" + _FLOAT_RE
    )
    return re.compile(pattern, re.MULTILINE)


# Cache compiled regexes
_VAR_REGEX_CACHE = {code: _build_var_regex(code) for code in MCD_VARIABLES}


def _parse_mcd_response(text, var_codes):
    """
    Parse an MCD CGI response for the requested variable codes.

    Returns dict {var_code: float or None}. 'none' codes are skipped.
    """
    result = {}
    if "Ooops" in text or "Error" in text or "error" in text.lower():
        for code in var_codes:
            if code != "none":
                result[code] = None
        return result

    for code in var_codes:
        if code == "none":
            continue
        m = _VAR_REGEX_CACHE[code].search(text)
        result[code] = float(m.group(1)) if m else None

    return result


# ══════════════════════════════════════════════════════════════════════════════
# MCD query
# ══════════════════════════════════════════════════════════════════════════════

def _query_mcd(ls, lat, lon, localtime, dust_code, altitude, zkey,
               var_codes, retries=3):
    """
    Query the MCD web API for up to 4 variables at a single point.

    Parameters
    ----------
    var_codes : list of up to 4 MCD variable codes (see MCD_VARIABLES).
                Padded with 'none' if fewer than 4 given.
    altitude, zkey : altitude value + its coordinate key
                      (zkey=4 -> altitude is a pressure level in Pa,
                       zkey=3 -> altitude is meters above surface, etc.)

    Returns
    -------
    dict {var_code: float or None}, for each non-'none' code in var_codes.
    """
    codes = (list(var_codes) + ["none"] * 4)[:4]

    params = {
        "var1"        : codes[0],
        "var2"        : codes[1],
        "var3"        : codes[2],
        "var4"        : codes[3],
        "datekeyhtml" : "1",
        "ls"          : str(float(ls)),
        "localtime"   : str(float(localtime)),
        "latitude"    : str(float(lat)),
        "longitude"   : str(float(lon)),
        "altitude"    : str(float(altitude)),
        "zkey"        : str(int(zkey)),
        "dust"        : str(int(dust_code)),
        "isfixedlt"   : "on",
        "iswind"      : "off",
        "averaging"   : "off",
        "zonmean"     : "off",
        "diumean"     : "off",
        "animation"   : "off",
        "islog"       : "off",
        "colorm"      : "jet",
        "proj"        : "cyl",
        "dpi"         : "80",
        "hrkey"       : "1",
        "istherepoint": "on",
        "plat"        : str(float(lat)),
        "plon"        : str(float(lon)),
        "palt"        : str(float(altitude)),
    }

    for attempt in range(retries):
        try:
            r = requests.get(_MCD_URL, params=params, timeout=30)
            r.raise_for_status()
            return _parse_mcd_response(r.text, codes)
        except requests.RequestException:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            return {c: None for c in codes if c != "none"}

    return {c: None for c in codes if c != "none"}


# ══════════════════════════════════════════════════════════════════════════════
# Profile builder
# ══════════════════════════════════════════════════════════════════════════════
def build_profile(ls, lat, lon=0.0, localtime=6.0, dust_code=1,
                  var_groups=None, verbose=True):
    """
    Build a full vertical profile by querying MCD at each pressure level
    for any number of variables.

    Per-level (altitude-resolved) variables are queried once per pressure
    level. Variables in _SURFACE_VARS (true surface properties and
    column-integrated quantities) are queried once at zkey=3, altitude=10m
    and stored as scalars in the output, regardless of which group the
    caller put them in.

    Parameters
    ----------
    ls, lat, lon, localtime, dust_code, var_groups, verbose : as before.

    Returns
    -------
    dict with array 'z_km' plus one array per requested per-level variable,
    plus scalar entries for any requested surface/column variables, plus
    n_total/sigma_T if T_K and P_Pa were requested, plus standard metadata
    (tsurf_K, ps_Pa, lat, lon, ls, localtime, dust_code, dust_label).
    Returns None if fewer than 3 levels have a valid altitude.
    """
    var_groups = [list(g) for g in (var_groups or [_DEFAULT_VAR_GROUP])]

    all_codes_flat = [c for g in var_groups for c in g if c != "none"]
    if "zareoid" not in all_codes_flat:
        all_codes_flat.append("zareoid")

    # de-duplicate while preserving order
    seen = set()
    all_codes = []
    for c in all_codes_flat:
        if c not in seen:
            all_codes.append(c)
            seen.add(c)

    # Split into per-level (profile) variables vs. scalar surface/column
    # variables. zareoid is always per-level (it builds the altitude grid).
    level_codes = [c for c in all_codes if c not in _SURFACE_VARS]
    surface_codes = [c for c in all_codes if c in _SURFACE_VARS]
    if "zareoid" not in level_codes:
        level_codes.append("zareoid")

    level_var_groups = [level_codes[i:i + 4] for i in range(0, len(level_codes), 4)]

    dust_label = _DUST_CODES.get(dust_code, str(dust_code))

    if verbose:
        print(f"  Querying MCD: Ls={ls}\u00b0 lat={lat}\u00b0 lon={lon}\u00b0 "
              f"LT={localtime}h dust={dust_label}")
        print(f"  Per-level variables: {', '.join(level_codes)}")
        if surface_codes:
            print(f"  Surface/column (scalar) variables: {', '.join(surface_codes)}")

    raw = {code: [None] * len(_PRESSURE_LEVELS_PA) for code in level_codes}

    for i, p in enumerate(_PRESSURE_LEVELS_PA):
        level_vals = {}
        for group in level_var_groups:
            res = _query_mcd(ls, lat, lon, localtime, dust_code,
                              altitude=p, zkey=4, var_codes=group)
            level_vals.update(res)
            time.sleep(0.5)   # be polite to LMD servers

        for code in level_codes:
            raw[code][i] = level_vals.get(code)

        if verbose:
            z = level_vals.get("zareoid")
            z_str = f"z={z/1000:6.1f} km" if z is not None else "z=  N/A   "
            preview_codes = [c for c in level_codes if c != "zareoid"][:3]
            preview = "  ".join(
                f"{c}={level_vals.get(c):.4g}" if level_vals.get(c) is not None
                else f"{c}=N/A"
                for c in preview_codes
            )
            print(f"    [{i+1:2d}/{len(_PRESSURE_LEVELS_PA)}] "
                  f"P={p:7.3f} Pa  {z_str}  {preview}")

    # Build altitude grid from zareoid, keep only levels with valid z
    z_m = raw["zareoid"]
    valid_idx = [i for i, z in enumerate(z_m) if z is not None]

    if len(valid_idx) < 3:
        print("  ERROR: Not enough valid altitude levels "
              f"({len(valid_idx)}/{len(_PRESSURE_LEVELS_PA)}).")
        return None

    valid_idx.sort(key=lambda i: z_m[i])
    z_km = np.array([z_m[i] / 1000.0 for i in valid_idx])

    profile = {"z_km": z_km}
    for code in level_codes:
        if code == "zareoid":
            continue
        arr = np.array([
            raw[code][i] if raw[code][i] is not None else np.nan
            for i in valid_idx
        ])
        out_name = _SCHEMA_ALIASES.get(code, code)
        profile[out_name] = arr

    # Derived quantities, only if T and P were both requested
    if "T_K" in profile and "P_Pa" in profile:
        T_K, P_Pa = profile["T_K"], profile["P_Pa"]
        with np.errstate(invalid="ignore", divide="ignore"):
            profile["n_total"] = P_Pa / (_kB * T_K)

        if "rmst" in profile:
            profile["sigma_T"] = profile["rmst"].copy()
        else:
            sigma_T = np.zeros_like(T_K)
            for z_lo, z_hi, sig in _SIGMA_T_BANDS:
                mask = (z_km >= z_lo) & (z_km < z_hi)
                sigma_T[mask] = sig
            profile["sigma_T"] = sigma_T

    # Surface/column (scalar) query: tsurf, ps, plus any requested
    # surface/column variables, batched into groups of up to 4.
    surf_query_codes = ["tsurf", "ps"] + surface_codes
    seen_s = set()
    surf_query_codes = [c for c in surf_query_codes
                         if not (c in seen_s or seen_s.add(c))]
    surf_groups = [surf_query_codes[i:i + 4] for i in range(0, len(surf_query_codes), 4)]

    surf_result = {}
    for group in surf_groups:
        res = _query_mcd(ls, lat, lon, localtime, dust_code,
                          altitude=10.0, zkey=3, var_codes=group)
        surf_result.update(res)
        time.sleep(0.5)

    if verbose:
        print(f"  Surface: Tsurf={surf_result.get('tsurf')} K  "
              f"Ps={surf_result.get('ps')} Pa")
        for code in surface_codes:
            print(f"    {code} = {surf_result.get(code)}")
        print(f"  Profile: {len(z_km)} levels  "
              f"z={z_km[0]:.1f}\u2013{z_km[-1]:.1f} km")
        nan_only = [name for name, v in profile.items()
                    if isinstance(v, np.ndarray) and np.isnan(v).all()]
        if nan_only:
            print(f"  WARNING: all-NaN variable(s): {', '.join(nan_only)} "
                  f"(check var code / HR-mode requirement)")

    profile["tsurf_K"] = surf_result.get("tsurf") if surf_result.get("tsurf") is not None else np.nan
    profile["ps_Pa"]   = surf_result.get("ps")    if surf_result.get("ps")    is not None else np.nan

    # Store any additional requested surface/column variables as scalars
    for code in surface_codes:
        out_name = _SCHEMA_ALIASES.get(code, code)
        val = surf_result.get(code)
        profile[out_name] = val if val is not None else np.nan

    profile["lat"]       = float(lat)
    profile["lon"]       = float(lon)
    profile["ls"]        = float(ls)
    profile["localtime"] = float(localtime)
    profile["dust_code"] = int(dust_code)
    profile["dust_label"] = dust_label

    return profile



# ══════════════════════════════════════════════════════════════════════════════
# Writer
# ══════════════════════════════════════════════════════════════════════════════

def write_npz(output_path, profile):
    """Write profile dict to LEOS .npz schema."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    save_dict = {}
    for k, v in profile.items():
        if isinstance(v, str):
            save_dict[k] = np.array([v], dtype=object)
        elif isinstance(v, (int, float, np.floating, np.integer)):
            save_dict[k] = np.array(v, dtype=np.float64)
        else:
            save_dict[k] = np.asarray(v)

    save_dict["source"]      = np.array(["mcd_webapi"], dtype=object)
    save_dict["mcd_version"] = np.array(["MCD_v6.1"],   dtype=object)

    np.savez(output_path, **save_dict)
    size_kb = os.path.getsize(output_path) / 1e3
    print(f"\n  Written: {output_path} ({size_kb:.1f} KB)")
    print(f"  Levels : {len(profile['z_km'])}")
    print(f"  z range: {profile['z_km'][0]:.1f}\u2013{profile['z_km'][-1]:.1f} km")
    array_vars = [k for k, v in profile.items()
                   if isinstance(v, np.ndarray) and k != "z_km"]
    print(f"  Variables saved: {', '.join(array_vars)}")


# ══════════════════════════════════════════════════════════════════════════════
# Bundled profile generator
# ══════════════════════════════════════════════════════════════════════════════

def generate_bundled_profiles(out_dir=None, var_groups=None):
    """
    Generate all 36 profiles (4 Ls x 3 lat x 3 dust) and save
    individually plus a combined mcd_bundled_profiles.npz.

    With the default variable group (t,p,rho,zareoid):
    36 profiles x 22 levels x 1 API call + 1 surface call = ~800 calls.
    At 0.5s delay each: ~7 minutes. Each extra 4-variable group
    multiplies the per-level cost. Be patient.
    """
    out_dir = out_dir or _DEFAULT_OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    all_profiles = {}
    total = len(_BUNDLED_LS) * len(_BUNDLED_LATS) * len(_BUNDLED_DUSTS)
    done  = 0

    for ls in _BUNDLED_LS:
        for lat in _BUNDLED_LATS:
            for dust in _BUNDLED_DUSTS:
                done += 1
                key = f"Ls{ls:03d}_lat{lat:+03d}_dust{dust}"
                print(f"\n[{done}/{total}] {key}")

                profile = build_profile(
                    ls=ls, lat=lat, lon=0.0,
                    localtime=6.0, dust_code=dust,
                    var_groups=var_groups,
                    verbose=True,
                )

                if profile is None:
                    print(f"  SKIPPED \u2014 no data returned")
                    continue

                fname = os.path.join(out_dir, f"mcd_{key}.npz")
                write_npz(fname, profile)
                all_profiles[key] = profile

    combined_path = os.path.join(out_dir, "mcd_bundled_profiles.npz")
    combined = {}
    keys_list = sorted(all_profiles.keys())

    for key in keys_list:
        p = all_profiles[key]
        for field, val in p.items():
            combined_key = f"{key}__{field}"
            if isinstance(val, str):
                combined[combined_key] = np.array([val], dtype=object)
            elif isinstance(val, (int, float, np.floating, np.integer)):
                combined[combined_key] = np.array(val)
            else:
                combined[combined_key] = np.asarray(val)

    combined["profile_keys"] = np.array(keys_list, dtype=object)
    combined["source"]       = np.array(["mcd_webapi_bundled"], dtype=object)
    combined["mcd_version"]  = np.array(["MCD_v6.1"], dtype=object)
    combined["n_profiles"]   = np.array([len(keys_list)])

    np.savez(combined_path, **combined)
    size_kb = os.path.getsize(combined_path) / 1e3
    print(f"\n{'='*55}")
    print(f"Bundled file: {combined_path} ({size_kb:.1f} KB)")
    print(f"Profiles saved: {len(keys_list)}/{total}")
    return combined_path


# ══════════════════════════════════════════════════════════════════════════════
# Default output path
# ══════════════════════════════════════════════════════════════════════════════

def _default_path(ls, lat, lon, dust_code):
    dust_label = _DUST_CODES.get(dust_code, str(dust_code))
    lat_str = f"{abs(lat):.0f}{'N' if lat >= 0 else 'S'}"
    lon_str = f"{abs(lon):.0f}{'E' if lon >= 0 else 'W'}"
    return os.path.join(
        _DEFAULT_OUT_DIR,
        f"mcd_Ls{ls:03.0f}_{lat_str}_{lon_str}_dust{dust_label}.npz"
    )


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--lat",   type=float, default=0.0,
                        help="Latitude [deg N]. Default 0.")
    parser.add_argument("--lon",   type=float, default=0.0,
                        help="Longitude [deg E]. Default 0.")
    parser.add_argument("--ls",    type=float, default=0.0,
                        help="Solar longitude [deg]. Default 0.")
    parser.add_argument("--localtime", type=float, default=6.0,
                        help="Local solar time [hrs]. Default 6.")
    parser.add_argument("--dust",  type=int, default=1,
                        help="MCD dust scenario code (1=avg, 2=min, 3=max, "
                             "4-8=storm/warm/cold, 24-35=Martian Year N). "
                             "Default 1. Run --list-dust for all options.")
    parser.add_argument("--vars", default=None,
                        help="Comma-separated MCD variable codes to query "
                             "(e.g. 't,p,rho,zareoid,vmr_co2,vmr_n2,vmr_ar,rmst'). "
                             "Automatically split into groups of 4 MCD queries. "
                             "'zareoid' is added automatically if missing. "
                             "Default: t,p,rho,zareoid. "
                             "Run --list-vars to see all options.")
    parser.add_argument("--list-vars", action="store_true",
                        help="List all queryable MCD variable codes and exit.")
    parser.add_argument("--list-dust", action="store_true",
                        help="List all MCD dust scenario codes and exit.")
    parser.add_argument("--output", default=None,
                        help="Output .npz path.")
    parser.add_argument("--generate-bundled", action="store_true",
                        help="Generate all 36 bundled profiles (~7 min with "
                             "default vars; longer with --vars).")
    parser.add_argument("--bundled-outdir", default=None,
                        help="Output directory for --generate-bundled.")

    args = parser.parse_args()

    if args.list_vars:
        print(f"\nMCD variable codes ({len(MCD_VARIABLES)} total):\n")
        for code, label in MCD_VARIABLES.items():
            hr_note = "  [HR mode only]" if code in _HR_ONLY_VARS else ""
            print(f"  {code:20s} {label}{hr_note}")
        print("\nUse with --vars, comma-separated, e.g.:")
        print("  --vars t,p,rho,zareoid,vmr_co2,vmr_n2,vmr_ar,rmst")
        return

    if args.list_dust:
        print("\nMCD dust scenario codes:\n")
        for code, label in _DUST_FULL_LABELS.items():
            short = _DUST_CODES.get(code, "")
            short_note = f"  (short label: {short})" if short else ""
            print(f"  {code:3d}  {label}{short_note}")
        return

    if args.vars:
        codes = [c.strip() for c in args.vars.split(",") if c.strip()]
        unknown = [c for c in codes if c not in MCD_VARIABLES]
        if unknown:
            sys.exit(f"ERROR: unknown variable code(s): {unknown}\n"
                     f"Run --list-vars to see all options.")
        var_groups = [codes[i:i + 4] for i in range(0, len(codes), 4)]
    else:
        var_groups = [_DEFAULT_VAR_GROUP]

    if args.generate_bundled:
        print("\nLEOS MCD Web API \u2014 generating 36 bundled profiles")
        print(f"Variable groups: {var_groups}")
        print("This queries the LMD server many times. Please be patient.\n")
        combined = generate_bundled_profiles(args.bundled_outdir, var_groups)
        print(f"\nLoad in LEOS with:")
        print(f"  from leos.atmosphere_mars import MarsAtmosphericColumn")
        print(f"  col = MarsAtmosphericColumn.from_bundled(ls=0, lat=0, dust='avg')")
        return

    print(f"\nLEOS MCD Web API converter")
    print(f"  Ls={args.ls}  lat={args.lat}  lon={args.lon}  "
          f"LT={args.localtime}  dust={args.dust} "
          f"({_DUST_FULL_LABELS.get(args.dust, 'unknown')})")

    profile = build_profile(
        ls=args.ls, lat=args.lat, lon=args.lon,
        localtime=args.localtime, dust_code=args.dust,
        var_groups=var_groups, verbose=True,
    )

    if profile is None:
        print("ERROR: No profile data returned. Check parameters.")
        sys.exit(1)

    output_path = args.output or _default_path(
        args.ls, args.lat, args.lon, args.dust
    )
    write_npz(output_path, profile)

    print(f"\nLoad in LEOS with:")
    print(f"  from leos.atmosphere_mars import MarsAtmosphericColumn")
    print(f"  col = MarsAtmosphericColumn.from_npz('{output_path}')")


if __name__ == "__main__":
    main()
