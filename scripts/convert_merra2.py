#!/usr/bin/env python3
"""
scripts/convert_merra2.py
--------------------------
IMPORTANT: Download format
---------------------------
This script requires NetCDF4 (.nc4 / .nc) input. If using the OPeNDAP
or GES DISC Subsetter interfaces, ensure "netCDF" (not "ASCII" or
"CSV") is selected as the output format — ASCII downloads will fail
to open with nc.Dataset().

Where to download (GES DISC, free NASA Earthdata login required):
  https://disc.gsfc.nasa.gov/datasets/M2I3NPASM_5.12.4/summary
  -> "Subset / Get Data" -> choose region/time/variables -> format: netCDF

Convert MERRA-2 M2I3NPASM (or any M2I3NP* inst3_3d_asm_Np) netCDF4 file
to the LEOS atmospheric column .npz format used by atmosphere_earth.py.

Handles all three MERRA-2 download options from GES DISC:
  1. Original files  (full global, all variables, 1-8 time steps, 42 levels)
  2. OPeNDAP subsets (cropped lat/lon/variables, netCDF or ASCII)
  3. GES DISC Subsetter (cropped lat/lon/time/variables, optional regrid)

All 13 available variables are handled if present; the converter
gracefully skips any that were not included in the user's download.

Available MERRA-2 variables in M2I3NPASM
-----------------------------------------
  T       air_temperature                     [K]
  U       eastward_wind                       [m/s]
  V       northward_wind                      [m/s]
  EPV     ertels_potential_vorticity          [K m^2 kg^-1 s^-1]
  CLOUD   mass_fraction_of_cloud_ice_water    [kg/kg]
  QL      mass_fraction_of_cloud_liquid_water [kg/kg]
  O3      ozone_mass_mixing_ratio             [kg/kg]
  RH      relative_humidity_after_moist       [1]
  SLP     sea_level_pressure                  [Pa]   (surface, no lev dim)
  QV      specific_humidity                   [kg/kg]
  PHIS    surface_geopotential_height         [m^2/s^2] (surface, no lev dim)
  PS      surface_pressure                    [Pa]   (surface, no lev dim)
  OMEGA   vertical_pressure_velocity          [Pa/s]

Spatial subsetting
------------------
  The GES DISC subsetter accepts a bounding box [W, S, E, N] in degrees.
  Default is [-180, -90, 180, 90] (global). This converter handles any
  spatial extent — the nearest-grid-point extraction works regardless of
  whether the file is global or a small regional crop.

Output schema (leos/data/merra2_YYYY-MM-DD_HH_LAT_LON.npz)
-----------------------------------------------------------
Always present (derived):
  z_km            float64[n_levels]   geometric altitude [km]
  P_Pa            float64[n_levels]   pressure [Pa]
  p_levels_hPa    float64[n_levels]   original pressure levels [hPa]
  n_total         float64[n_levels]   number density [molecules/m^3]
  lat, lon        float64             actual grid point used
  time_iso        str
  source          str  'merra2'
  product         str
  vars_present    str  comma-separated list of variables extracted

Present if variable was in the file:
  T_K             float64[n_levels]   temperature [K]
  sigma_T         float64[n_levels]   temperature uncertainty [K]
  U_ms            float64[n_levels]   eastward wind [m/s]
  V_ms            float64[n_levels]   northward wind [m/s]
  EPV             float64[n_levels]   Ertel PV [K m^2 kg^-1 s^-1]
  CLOUD           float64[n_levels]   cloud ice fraction [kg/kg]
  QL              float64[n_levels]   cloud liquid fraction [kg/kg]
  vmr_O3          float64[n_levels]   O3 volume mixing ratio
  col_O3          float64             O3 column [molecules/m^2]
  sigma_vmr_O3    float64[n_levels]   O3 VMR uncertainty
  RH              float64[n_levels]   relative humidity [0-1]
  vmr_H2O         float64[n_levels]   H2O VMR (from QV)
  col_H2O         float64             H2O column [molecules/m^2]
  sigma_vmr_H2O   float64[n_levels]   H2O VMR uncertainty
  SLP_Pa          float64             sea level pressure [Pa]
  PHIS_m2s2       float64             surface geopotential [m^2/s^2]
  PS_Pa           float64             surface pressure [Pa]
  OMEGA           float64[n_levels]   vertical pressure velocity [Pa/s]

Usage
-----
  # Nearest point to Jezero Crater, 12 UTC
  python scripts/convert_merra2.py path/to/MERRA2.nc4 \\
      --lat 18.4 --lon 77.5 --time 2026-01-01T12:00

  # Time-average all steps
  python scripts/convert_merra2.py path/to/MERRA2.nc4 \\
      --lat 18.4 --lon 77.5 --time-average

  # List available times
  python scripts/convert_merra2.py path/to/MERRA2.nc4 --list-times

  # List variables actually in the file
  python scripts/convert_merra2.py path/to/MERRA2.nc4 --list-vars

Download
--------
  Original files  : https://goldsmr5.gesdisc.eosdis.nasa.gov/data/MERRA2/M2I3NPASM.5.12.4/
  OPeNDAP         : https://goldsmr5.gesdisc.eosdis.nasa.gov/opendap/MERRA2/M2I3NPASM.5.12.4/
  GES DISC        : https://disc.gsfc.nasa.gov/datasets/M2I3NPASM_5.12.4/summary
"""

import argparse
import os
import sys
import warnings
import numpy as np
from datetime import datetime, timezone, timedelta

try:
    import netCDF4 as nc
except ImportError:
    print("ERROR: netCDF4 not installed. Run: pip install netCDF4", file=sys.stderr)
    sys.exit(1)

# ── LEOS output directory ─────────────────────────────────────────────────────
_DEFAULT_OUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "leos", "data"
)

# ── Physical constants ────────────────────────────────────────────────────────
_kB   = 1.380649e-23   # J/K
_g0   = 9.807          # m/s^2
_Rd   = 287.058        # J/(kg K) dry air
_Mair = 28.966e-3      # kg/mol
_M_H2O = 18.015e-3    # kg/mol
_M_O3  = 47.997e-3    # kg/mol

# ── Variable catalogue ────────────────────────────────────────────────────────
# Maps MERRA-2 short name → metadata
# dims: '3d' = (time,lev,lat,lon), 'sfc' = (time,lat,lon)
_VAR_CATALOGUE = {
    "T"    : {"long": "air_temperature",
               "units": "K",       "dims": "3d",  "out_key": "T_K"},
    "U"    : {"long": "eastward_wind",
               "units": "m s-1",   "dims": "3d",  "out_key": "U_ms"},
    "V"    : {"long": "northward_wind",
               "units": "m s-1",   "dims": "3d",  "out_key": "V_ms"},
    "EPV"  : {"long": "ertels_potential_vorticity",
               "units": "K m2 kg-1 s-1", "dims": "3d", "out_key": "EPV"},
    "CLOUD": {"long": "mass_fraction_of_cloud_ice_water",
               "units": "kg kg-1", "dims": "3d",  "out_key": "CLOUD"},
    "QL"   : {"long": "mass_fraction_of_cloud_liquid_water",
               "units": "kg kg-1", "dims": "3d",  "out_key": "QL"},
    "O3"   : {"long": "ozone_mass_mixing_ratio",
               "units": "kg kg-1", "dims": "3d",  "out_key": "vmr_O3",
               "mol_mass": _M_O3,  "vmr": True},
    "RH"   : {"long": "relative_humidity_after_moist",
               "units": "1",       "dims": "3d",  "out_key": "RH"},
    "QV"   : {"long": "specific_humidity",
               "units": "kg kg-1", "dims": "3d",  "out_key": "vmr_H2O",
               "mol_mass": _M_H2O, "vmr": True},
    "SLP"  : {"long": "sea_level_pressure",
               "units": "Pa",      "dims": "sfc", "out_key": "SLP_Pa"},
    "PHIS" : {"long": "surface_geopotential_height",
               "units": "m2 s-2",  "dims": "sfc", "out_key": "PHIS_m2s2"},
    "PS"   : {"long": "surface_pressure",
               "units": "Pa",      "dims": "sfc", "out_key": "PS_Pa"},
    "OMEGA": {"long": "vertical_pressure_velocity",
               "units": "Pa s-1",  "dims": "3d",  "out_key": "OMEGA"},
}

# ── Uncertainty estimates ─────────────────────────────────────────────────────
# Based on Gelaro et al. 2017 and reanalysis validation literature.
# Fractional (relative) uncertainties unless noted.
_SIGMA = {
    "T"    : {"type": "level", "values": {
                (100, 1000): 0.8,   # K, troposphere
                (10,   100): 1.5,   # K, lower stratosphere
                (0.0,   10): 2.5,   # K, upper stratosphere
              }},
    "QV"   : {"type": "frac", "value": 0.08},   # 8%
    "O3"   : {"type": "frac", "value": 0.10},   # 10%
    "U"    : {"type": "abs",  "value": 1.5},    # m/s
    "V"    : {"type": "abs",  "value": 1.5},    # m/s
    "RH"   : {"type": "abs",  "value": 0.05},   # 5 percentage points
    "CLOUD": {"type": "frac", "value": 0.30},   # 30% (poorly constrained)
    "QL"   : {"type": "frac", "value": 0.30},
    "OMEGA": {"type": "frac", "value": 0.20},
}


# ══════════════════════════════════════════════════════════════════════════════
# Time utilities
# ══════════════════════════════════════════════════════════════════════════════

def _decode_times(time_var):
    units_str = getattr(time_var, "units", "minutes since 2000-01-01 00:00:00")
    parts     = units_str.split(" since ")
    unit      = parts[0].strip().lower()
    epoch     = datetime.fromisoformat(parts[1].strip().replace(" ", "T"))
    epoch     = epoch.replace(tzinfo=timezone.utc)
    scale     = {"minutes": 60, "hours": 3600, "seconds": 1}.get(unit, 60)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = time_var[:]
    values = np.ma.filled(raw, fill_value=0).astype(np.float64).flatten().tolist()
    return [epoch + timedelta(seconds=int(v) * scale) for v in values]


def _nearest_time_index(times, target_str):
    target = datetime.fromisoformat(target_str)
    diffs  = [abs((t.replace(tzinfo=None) - target).total_seconds())
               for t in times]
    return int(np.argmin(diffs))


# ══════════════════════════════════════════════════════════════════════════════
# Grid utilities
# ══════════════════════════════════════════════════════════════════════════════

def _nearest_index(arr, value):
    return int(np.argmin(np.abs(np.asarray(arr) - value)))


def _check_coverage(actual, requested, coord_name, tolerance=1.0):
    diff = abs(actual - requested)
    if diff > tolerance:
        warnings.warn(
            f"Nearest {coord_name} grid point ({actual:.3f}) is {diff:.2f} deg "
            f"from requested ({requested}). "
            f"Use the GES DISC subsetter with a tighter bounding box for a "
            f"closer match.",
            UserWarning,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Derived quantities
# ══════════════════════════════════════════════════════════════════════════════

def _pressure_to_altitude(P_hPa, T_K):
    """
    Geometric altitude [km] from pressure [hPa] and temperature [K]
    via hypsometric equation, integrated layer by layer.
    Assumes surface is index of highest pressure.
    """
    P_Pa = np.asarray(P_hPa, dtype=float) * 100.0
    T    = np.asarray(T_K,    dtype=float)

    # Sort surface (high P) to top (low P)
    order   = np.argsort(P_Pa)[::-1]
    P_sort  = P_Pa[order]
    T_sort  = T[order]

    z = np.zeros(len(P_sort))
    for i in range(1, len(P_sort)):
        T_mean = 0.5 * (T_sort[i-1] + T_sort[i])
        dz     = -(_Rd * T_mean / _g0) * np.log(P_sort[i] / P_sort[i-1])
        z[i]   = z[i-1] + dz

    z_km = z / 1000.0

    # Restore original level order
    result = np.empty_like(z_km)
    result[order] = z_km
    return result


def _number_density(P_Pa, T_K):
    """Total number density [molecules/m^3] from ideal gas law."""
    return np.asarray(P_Pa) / (_kB * np.asarray(T_K))


def _mmr_to_vmr(mmr, mol_mass_species):
    """Mass mixing ratio [kg/kg] → volume mixing ratio."""
    return np.asarray(mmr) * (_Mair / mol_mass_species)


def _column_density(vmr, n_total, z_km):
    """Vertical column [molecules/m^2] by trapezoidal integration."""
    z_m = np.asarray(z_km) * 1e3
    return float(np.trapezoid(np.asarray(vmr) * np.asarray(n_total), z_m))


def _compute_sigma(var_name, values, P_hPa):
    """Compute per-level uncertainty for a given variable."""
    spec = _SIGMA.get(var_name)
    if spec is None:
        return None

    if spec["type"] == "level":
        sigma = np.zeros(len(P_hPa))
        for (lo, hi), val in spec["values"].items():
            mask = (np.asarray(P_hPa) >= lo) & (np.asarray(P_hPa) < hi)
            sigma[mask] = val
        return sigma
    elif spec["type"] == "frac":
        return np.abs(np.asarray(values)) * spec["value"]
    elif spec["type"] == "abs":
        return np.full(len(values), spec["value"])
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Core extractor
# ══════════════════════════════════════════════════════════════════════════════

def _extract_3d(var, time_indices, i_lat, i_lon):
    """Extract [lev] array, averaged over selected time steps. Fill→NaN."""
    import warnings
    slices = []
    for ti in time_indices:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = var[ti, :, i_lat, i_lon]
        arr = np.ma.filled(raw.astype(np.float64),
                           fill_value=np.nan)
        slices.append(arr)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(slices, axis=0)


def _extract_sfc(var, time_indices, i_lat, i_lon):
    """Extract scalar, averaged over selected time steps. Fill→NaN."""
    import warnings
    slices = []
    for ti in time_indices:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = var[ti, i_lat, i_lon]
        val = float(np.ma.filled(
            np.ma.array(raw, dtype=np.float64), fill_value=np.nan
        ))
        slices.append(val)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return float(np.nanmean(slices))


def extract_column(ds, lat_target, lon_target,
                   time_idx=None, time_average=False):
    """
    Extract atmospheric column from MERRA-2 dataset.

    Handles any subset of the 13 variables gracefully.
    Pressure (lev) and temperature (T) are required for altitude
    and number density — warns and uses fallback if T is absent.

    Parameters
    ----------
    ds : netCDF4.Dataset
    lat_target : float
    lon_target : float
    time_idx : int or None
    time_average : bool

    Returns
    -------
    dict — all output fields, only variables present in ds included
    """
    lats = ds.variables["lat"][:]
    lons = ds.variables["lon"][:]
    levs = ds.variables["lev"][:]   # hPa

    i_lat = _nearest_index(lats, lat_target)
    i_lon = _nearest_index(lons, lon_target)

    actual_lat = float(lats[i_lat])
    actual_lon = float(lons[i_lon])

    _check_coverage(actual_lat, lat_target, "latitude")
    _check_coverage(actual_lon, lon_target, "longitude")
    print(f"  Grid point: lat={actual_lat:.3f}  lon={actual_lon:.3f}")

    # ── Time selection ────────────────────────────────────────────────────────
    times   = _decode_times(ds.variables["time"])
    n_times = len(times)

    if time_average:
        time_indices = list(range(n_times))
        time_label   = (f"{times[0].strftime('%Y-%m-%dT%H:%M')}/"
                        f"{times[-1].strftime('%Y-%m-%dT%H:%M')}_avg")
        print(f"  Averaging {n_times} time steps")
    else:
        idx          = time_idx if time_idx is not None else 0
        time_indices = [idx]
        time_label   = times[idx].isoformat()
        print(f"  Time step [{idx}]: {time_label}")

    P_hPa = np.array(levs, dtype=np.float64)
    P_Pa  = P_hPa * 100.0
    n_lev = len(P_hPa)

    # ── Which variables are actually in this file? ────────────────────────────
    ds_vars    = set(ds.variables.keys())
    found_vars = [k for k in _VAR_CATALOGUE if k in ds_vars]
    missing    = [k for k in _VAR_CATALOGUE if k not in ds_vars]

    if missing:
        print(f"  Variables not in file (skipped): {', '.join(missing)}")
    print(f"  Variables extracting: {', '.join(found_vars)}")

    out = {
        "p_levels_hPa": P_hPa,
        "P_Pa"        : P_Pa,
        "lat"         : actual_lat,
        "lon"         : actual_lon,
        "time_iso"    : time_label,
        "source"      : "merra2",
        "product"     : getattr(ds, "Title", "") or "M2I3NPASM",
        "vars_present": ",".join(found_vars),
    }

    # ── Temperature (needed for altitude and n_total) ─────────────────────────
    if "T" in ds_vars:
        T_K = _extract_3d(ds.variables["T"], time_indices, i_lat, i_lon)
        out["T_K"]    = T_K
        out["sigma_T"] = _compute_sigma("T", T_K, P_hPa)
    else:
        warnings.warn(
            "T (temperature) not in file. Using US Standard Atmosphere "
            "approximation for altitude and number density calculations. "
            "Download T for accurate results.",
            UserWarning,
        )
        # US Std Atm rough approximation: T = 288.15 - 6.5*z (troposphere)
        # We don't know z yet, so use isothermal 255 K as fallback
        T_K = np.full(n_lev, 255.0)

    # ── Altitude and number density ───────────────────────────────────────────
    # Strip levels where T has fill values — these corrupt altitude integration
    valid_mask = np.isfinite(T_K) & (T_K > 100.0) & (T_K < 400.0)
    if not np.all(valid_mask):
        n_invalid = np.sum(~valid_mask)
        print(f"  Masking {n_invalid} fill-value levels from T")
        T_K = T_K[valid_mask]
        P_hPa_use = P_hPa[valid_mask]
        P_Pa_use  = P_Pa[valid_mask]
    else:
        P_hPa_use = P_hPa
        P_Pa_use  = P_Pa

    z_km    = _pressure_to_altitude(P_hPa_use, T_K)
    n_total = _number_density(P_Pa_use, T_K)

    # Update level arrays to valid subset
    out["p_levels_hPa"] = P_hPa_use
    out["P_Pa"]         = P_Pa_use
    out["T_K"]          = T_K
    out["sigma_T"]      = _compute_sigma("T", T_K, P_hPa_use)
    out["z_km"]    = z_km
    out["n_total"] = n_total

    # ── Wind components ───────────────────────────────────────────────────────
    for var_name in ("U", "V"):
        if var_name in ds_vars:
            vals = _extract_3d(ds.variables[var_name],
                               time_indices, i_lat, i_lon)
            key  = _VAR_CATALOGUE[var_name]["out_key"]
            out[key] = vals
            sigma = _compute_sigma(var_name, vals, P_hPa)
            if sigma is not None:
                out[f"sigma_{key}"] = sigma

    # ── Ertel PV ──────────────────────────────────────────────────────────────
    if "EPV" in ds_vars:
        out["EPV"] = _extract_3d(ds.variables["EPV"],
                                 time_indices, i_lat, i_lon)

    # ── Cloud fractions ───────────────────────────────────────────────────────
    for var_name in ("CLOUD", "QL"):
        if var_name in ds_vars:
            vals = _extract_3d(ds.variables[var_name],
                               time_indices, i_lat, i_lon)
            out[_VAR_CATALOGUE[var_name]["out_key"]] = vals
            sigma = _compute_sigma(var_name, vals, P_hPa)
            if sigma is not None:
                out[f"sigma_{_VAR_CATALOGUE[var_name]['out_key']}"] = sigma

    # ── Specific humidity → H2O VMR ───────────────────────────────────────────
    if "QV" in ds_vars:
        QV      = _extract_3d(ds.variables["QV"],
                              time_indices, i_lat, i_lon)[valid_mask]
        vmr_H2O = _mmr_to_vmr(QV, _M_H2O)
        vmr_H2O = np.clip(vmr_H2O, 0.0, 1.0)
        col_H2O = _column_density(vmr_H2O, n_total, z_km)
        out["vmr_H2O"]      = vmr_H2O
        out["col_H2O"]      = col_H2O
        out["sigma_vmr_H2O"]= _compute_sigma("QV", vmr_H2O, P_hPa)
        print(f"  H2O column: {col_H2O:.3e} molecules/m^2")

    # ── Ozone mass mixing ratio → O3 VMR ──────────────────────────────────────
    if "O3" in ds_vars:
        O3_mmr  = _extract_3d(ds.variables["O3"],
                              time_indices, i_lat, i_lon)[valid_mask]
        vmr_O3  = _mmr_to_vmr(O3_mmr, _M_O3)
        vmr_O3  = np.clip(vmr_O3, 0.0, 1.0)
        col_O3  = _column_density(vmr_O3, n_total, z_km)
        o3_du   = col_O3 / 2.6867e20
        out["vmr_O3"]      = vmr_O3
        out["col_O3"]      = col_O3
        out["sigma_vmr_O3"]= _compute_sigma("O3", vmr_O3, P_hPa)
        print(f"  O3 column : {o3_du:.1f} DU  "
              f"(expected ~250-350 DU mid-latitude)")

    # ── Relative humidity ─────────────────────────────────────────────────────
    if "RH" in ds_vars:
        RH = _extract_3d(ds.variables["RH"], time_indices, i_lat, i_lon)
        out["RH"]       = RH
        out["sigma_RH"] = _compute_sigma("RH", RH, P_hPa)

    # ── Vertical pressure velocity ────────────────────────────────────────────
    if "OMEGA" in ds_vars:
        omega = _extract_3d(ds.variables["OMEGA"], time_indices, i_lat, i_lon)
        out["OMEGA"]       = omega
        out["sigma_OMEGA"] = _compute_sigma("OMEGA", omega, P_hPa)

    # ── Surface fields (no lev dimension) ────────────────────────────────────
    for var_name, key in [("SLP", "SLP_Pa"), ("PS", "PS_Pa"),
                           ("PHIS", "PHIS_m2s2")]:
        if var_name in ds_vars:
            out[key] = _extract_sfc(ds.variables[var_name],
                                    time_indices, i_lat, i_lon)

    return out


# ══════════════════════════════════════════════════════════════════════════════
# Writer
# ══════════════════════════════════════════════════════════════════════════════

def write_npz(output_path, column):
    """Write column dict to LEOS .npz — all fields, strings as object arrays."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    save_dict = {}
    for k, v in column.items():
        if isinstance(v, str):
            save_dict[k] = np.array([v], dtype=object)
        elif isinstance(v, (int, float)):
            save_dict[k] = np.array(v, dtype=np.float64)
        else:
            save_dict[k] = np.asarray(v, dtype=np.float64)

    np.savez(output_path, **save_dict)

    size_kb = os.path.getsize(output_path) / 1e3
    print(f"\n  Written : {output_path} ({size_kb:.1f} KB)")
    print(f"  Levels  : {len(column['z_km'])}")
    print(f"  z range : {column['z_km'].min():.1f}–"
          f"{column['z_km'].max():.1f} km")
    if "T_K" in column:
        print(f"  T range : {column['T_K'].min():.1f}–"
              f"{column['T_K'].max():.1f} K")
    print(f"  Fields  : {', '.join(k for k in column if not k.startswith('_'))}")


# ══════════════════════════════════════════════════════════════════════════════
# Default output path
# ══════════════════════════════════════════════════════════════════════════════

def _default_output_path(nc_path, lat, lon):
    basename = os.path.basename(nc_path)
    date_str = "unknown"
    for part in basename.replace(".", "_").split("_"):
        if len(part) == 8 and part.isdigit():
            try:
                datetime.strptime(part, "%Y%m%d")
                date_str = f"{part[:4]}-{part[4:6]}-{part[6:8]}"
                break
            except ValueError:
                continue
    lat_str = f"{abs(lat):.2f}{'N' if lat >= 0 else 'S'}"
    lon_str = f"{abs(lon):.2f}{'E' if lon >= 0 else 'W'}"
    return os.path.join(_DEFAULT_OUT_DIR,
                        f"merra2_{date_str}_{lat_str}_{lon_str}.npz")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("nc_file",
                        help="Path to MERRA-2 netCDF4 file (.nc4 or .nc).")
    parser.add_argument("--lat",  type=float,
                        help="Target latitude [deg]. Required unless "
                             "--list-times or --list-vars.")
    parser.add_argument("--lon",  type=float,
                        help="Target longitude [deg]. Required unless "
                             "--list-times or --list-vars.")

    time_group = parser.add_mutually_exclusive_group()
    time_group.add_argument("--time", default=None,
                             metavar="YYYY-MM-DDTHH:MM",
                             help="Select nearest time step to this datetime.")
    time_group.add_argument("--time-index", type=int, default=None,
                             metavar="N",
                             help="Select time step by 0-based index.")
    time_group.add_argument("--time-average", action="store_true",
                             help="Average all time steps.")

    parser.add_argument("--list-times", action="store_true",
                        help="Print available time steps and exit.")
    parser.add_argument("--list-vars",  action="store_true",
                        help="Print variables in the file and exit.")
    parser.add_argument("--output", default=None,
                        help="Output .npz path. "
                             "Default: leos/data/merra2_DATE_LAT_LON.npz")

    args = parser.parse_args()

    if not os.path.exists(args.nc_file):
        print(f"ERROR: File not found: {args.nc_file}", file=sys.stderr)
        sys.exit(1)

    print(f"\nLEOS MERRA-2 converter")
    print(f"  input : {args.nc_file}")

    try:
        ds = nc.Dataset(args.nc_file, "r")
    except OSError as e:
        sys.exit(
            f"ERROR: Could not open '{args.nc_file}' as NetCDF ({e}).\n"
            f"If you downloaded from GES DISC's OPeNDAP or Subsetter, "
            f"make sure the output format was set to 'netCDF', not 'ASCII'/'CSV'."
        )

    # ── List modes ────────────────────────────────────────────────────────────
    if args.list_times:
        times = _decode_times(ds.variables["time"])
        print(f"\n  {len(times)} time step(s):")
        for i, t in enumerate(times):
            print(f"    [{i}]  {t.isoformat()}")
        ds.close()
        return

    if args.list_vars:
        print(f"\n  Variables in file:")
        for k in _VAR_CATALOGUE:
            status = "PRESENT" if k in ds.variables else "absent"
            meta   = _VAR_CATALOGUE[k]
            print(f"    {k:6s}  {status:7s}  {meta['long']}  [{meta['units']}]")
        print(f"\n  Coordinate variables:")
        for k in ("time", "lat", "lon", "lev"):
            if k in ds.variables:
                v = ds.variables[k]
                print(f"    {k:6s}  shape={v.shape}  "
                      f"units={getattr(v,'units','?')}")
        ds.close()
        return

    # ── lat/lon required for extraction ──────────────────────────────────────
    if args.lat is None or args.lon is None:
        parser.error("--lat and --lon are required for extraction. "
                     "Use --list-times or --list-vars to inspect the file.")

    # ── Time index ────────────────────────────────────────────────────────────
    time_idx = None
    if args.time is not None:
        times    = _decode_times(ds.variables["time"])
        time_idx = _nearest_time_index(times, args.time)
        print(f"  Requested: {args.time}  → nearest [{time_idx}]: "
              f"{times[time_idx].isoformat()}")
    elif args.time_index is not None:
        time_idx = args.time_index

    # ── Extract ───────────────────────────────────────────────────────────────
    column = extract_column(
        ds,
        lat_target   = args.lat,
        lon_target   = args.lon,
        time_idx     = time_idx,
        time_average = args.time_average,
    )
    ds.close()

    # ── Write ─────────────────────────────────────────────────────────────────
    output_path = args.output or _default_output_path(
        args.nc_file, args.lat, args.lon
    )
    print(f"  output: {output_path}")
    write_npz(output_path, column)

    print("\nLoad in LEOS with:")
    print("  from leos.atmosphere_earth import AtmosphericColumn")
    print(f"  col = AtmosphericColumn.from_npz('{output_path}')")


if __name__ == "__main__":
    main()
