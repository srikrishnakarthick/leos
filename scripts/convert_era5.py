#!/usr/bin/env python3
"""
scripts/convert_era5.py
-------------------------
Convert an ERA5 reanalysis-era5-pressure-levels netCDF file (downloaded
via the cdsapi) into the LEOS atmospheric column .npz format used by
atmosphere_earth.py — same output schema as scripts/convert_merra2.py,
so AtmosphericColumn.from_npz() works identically regardless of source.

WHERE TO GET ERA5 DATA
-----------------------
1. Register (free): https://cds.climate.copernicus.eu
2. After logging in, go to your profile page and copy your UID and API key.
3. pip install cdsapi
4. Create ~/.cdsapirc with:

       url: https://cds.climate.copernicus.eu/api/v2
       key: YOUR-UID:YOUR-API-KEY

5. IMPORTANT — before your first request will work, you must accept the
   dataset's Terms of Use / licence. Go to:

       https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=download

   and accept ALL required licences shown there (there may be more than
   one). Requests will fail with a permissions error until this is done
   — this is a one-time step per CDS account.

6. Request the data (adjust year/month/day/time/area as needed):

    import cdsapi
    c = cdsapi.Client()
    c.retrieve(
        'reanalysis-era5-pressure-levels',
        {
            'product_type': 'reanalysis',
            'variable': [
                'temperature',
                'specific_humidity',
                'ozone_mass_mixing_ratio',
            ],
            'pressure_level': [
                '1', '2', '3', '5', '7', '10', '20', '30', '50',
                '70', '100', '125', '150', '175', '200', '225', '250',
                '300', '350', '400', '450', '500', '550', '600', '650',
                '700', '750', '775', '800', '825', '850', '875', '900',
                '925', '950', '975', '1000',
            ],
            'year': '2024', 'month': '01', 'day': '15', 'time': '12:00',
            'area': [19, 77, 18, 78],  # North, West, South, East
            'format': 'netcdf',
        },
        'era5_sample.nc'
    )

EXPECTED INPUT FORMAT
-----------------------
A netCDF file (opened via xarray) with:

  dims  : (valid_time, pressure_level, latitude, longitude)
  coords: pressure_level in hPa, descending (e.g. 1000 ... 1)
          latitude, longitude in degrees
  data variables (any subset of):
    t   temperature              [K]
    q   specific_humidity        [kg/kg]
    o3  ozone_mass_mixing_ratio  [kg/kg]

Only t, q, o3 are handled (the variables available in
reanalysis-era5-pressure-levels relevant to LEOS). If your request
included other variables (u, v, z, etc.) they are ignored.

OUTPUT SCHEMA (identical to convert_merra2.py)
-------------------------------------------------
  z_km, T_K, P_Pa, p_levels_hPa, n_total   — altitude/temperature/pressure/
                                              number density profiles
  vmr_H2O, col_H2O, sigma_vmr_H2O          — from q (if present)
  vmr_O3,  col_O3,  sigma_vmr_O3           — from o3 (if present)
  sigma_T                                   — temperature uncertainty
  lat, lon, time_iso, source='era5', vars_present

USAGE
-----
    # Nearest grid point + nearest time
    python scripts/convert_era5.py era5_sample.nc --lat 18.4 --lon 77.5 \\
        --time 2024-01-15T12:00

    # Average over all times in the file
    python scripts/convert_era5.py era5_sample.nc --lat 18.4 --lon 77.5 \\
        --time-average

    # List available times / variables
    python scripts/convert_era5.py era5_sample.nc --list-times
    python scripts/convert_era5.py era5_sample.nc --list-vars
"""

import argparse
import os
import sys
import warnings
import numpy as np
from datetime import datetime

try:
    import xarray as xr
except ImportError:
    sys.exit(
        "ERROR: xarray is required to read ERA5 netCDF files.\n"
        "Install with: pip install xarray netCDF4 --break-system-packages"
    )

# ── LEOS output directory ─────────────────────────────────────────────────────
_DEFAULT_OUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "leos", "data"
)

# ── Physical constants ────────────────────────────────────────────────────────
_kB    = 1.380649e-23   # J/K
_g0    = 9.807          # m/s^2
_Rd    = 287.058        # J/(kg K) dry air
_Mair  = 28.966e-3       # kg/mol
_M_H2O = 18.015e-3       # kg/mol
_M_O3  = 47.997e-3       # kg/mol

# ── Variable name mapping: ERA5 short name -> role ────────────────────────────
_VAR_MAP = {
    "t":  "T_K",      # temperature [K]
    "q":  "vmr_H2O",  # specific humidity [kg/kg] -> converted to VMR
    "o3": "vmr_O3",   # ozone mass mixing ratio [kg/kg] -> converted to VMR
}

# ── Uncertainty estimates (Hersbach et al. 2020, ERA5 validation literature) ──
_SIGMA = {
    "t": {"type": "level", "values": {
            (100, 1000): 0.5,   # K, troposphere — ERA5 generally well-constrained
            (10,   100): 1.0,   # K, lower stratosphere
            (0.0,   10): 2.0,   # K, upper stratosphere
          }},
    "q":  {"type": "frac", "value": 0.10},   # 10% on specific humidity
    "o3": {"type": "frac", "value": 0.15},   # 15% on ozone MMR
}


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _nearest_index(arr, value):
    return int(np.argmin(np.abs(np.asarray(arr) - value)))


def _pressure_to_altitude(P_hPa, T_K):
    """
    Geometric altitude [km] from pressure [hPa] and temperature [K]
    via the hypsometric equation, integrated layer by layer from the
    surface (highest pressure) upward. Same approach as convert_merra2.py.
    """
    P_Pa = np.asarray(P_hPa, dtype=float) * 100.0
    T    = np.asarray(T_K,    dtype=float)

    order  = np.argsort(P_Pa)[::-1]   # surface (high P) -> top (low P)
    P_sort = P_Pa[order]
    T_sort = T[order]

    z = np.zeros(len(P_sort))
    for i in range(1, len(P_sort)):
        T_mean = 0.5 * (T_sort[i - 1] + T_sort[i])
        dz     = -(_Rd * T_mean / _g0) * np.log(P_sort[i] / P_sort[i - 1])
        z[i]   = z[i - 1] + dz

    z_km = z / 1000.0
    result = np.empty_like(z_km)
    result[order] = z_km
    return result


def _number_density(P_Pa, T_K):
    """Total number density [molecules/m^3] from ideal gas law."""
    return np.asarray(P_Pa) / (_kB * np.asarray(T_K))


def _mmr_to_vmr(mmr, mol_mass_species):
    """Mass mixing ratio [kg/kg] -> volume mixing ratio."""
    return np.asarray(mmr) * (_Mair / mol_mass_species)


def _column_density(vmr, n_total, z_km):
    """Vertical column [molecules/m^2] by trapezoidal integration over z."""
    z_m = np.asarray(z_km) * 1e3
    return float(np.trapezoid(np.asarray(vmr) * np.asarray(n_total), z_m))


def _compute_sigma(var_name, values, P_hPa):
    """Per-level uncertainty for a given ERA5 variable."""
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

def extract_column(ds, lat_target, lon_target,
                    time_idx=None, time_average=False):
    """
    Extract an atmospheric column from an ERA5 pressure-level dataset.

    Parameters
    ----------
    ds : xarray.Dataset
    lat_target, lon_target : float — degrees
    time_idx : int or None
    time_average : bool

    Returns
    -------
    dict — output fields (see module docstring OUTPUT SCHEMA)
    """
    lats = ds["latitude"].values
    lons = ds["longitude"].values
    levs = ds["pressure_level"].values   # hPa

    i_lat = _nearest_index(lats, lat_target)
    i_lon = _nearest_index(lons, lon_target)
    actual_lat = float(lats[i_lat])
    actual_lon = float(lons[i_lon])

    if abs(actual_lat - lat_target) > 1.0 or abs(actual_lon - lon_target) > 1.0:
        warnings.warn(
            f"Nearest grid point (lat={actual_lat:.3f}, lon={actual_lon:.3f}) "
            f"is >1 deg from requested ({lat_target}, {lon_target}). "
            f"Request a smaller 'area' box for a closer match.",
            UserWarning,
        )
    print(f"  Grid point: lat={actual_lat:.3f}  lon={actual_lon:.3f}")

    # ── Time selection ────────────────────────────────────────────────────────
    times = ds["valid_time"].values
    n_times = len(times)

    if time_average:
        time_sel = slice(None)
        time_label = (
            f"{str(times[0])[:16]}/{str(times[-1])[:16]}_avg"
        )
        print(f"  Averaging {n_times} time step(s)")
    else:
        idx = time_idx if time_idx is not None else 0
        time_sel = idx
        time_label = str(times[idx])[:19]
        print(f"  Time step [{idx}]: {time_label}")

    P_hPa = np.asarray(levs, dtype=np.float64)

    found_vars = [v for v in _VAR_MAP if v in ds.data_vars]
    missing    = [v for v in _VAR_MAP if v not in ds.data_vars]
    if missing:
        print(f"  Variables not in file (skipped): {', '.join(missing)}")
    print(f"  Variables extracting: {', '.join(found_vars)}")

    if "t" not in found_vars:
        sys.exit(
            "ERROR: 'temperature' (t) not found in this file. "
            "It is required to compute altitude and number density. "
            "Re-request the data with 'temperature' included."
        )

    def _get(varname):
        da = ds[varname].isel(latitude=i_lat, longitude=i_lon)
        if time_average:
            da = da.mean(dim="valid_time")
        else:
            da = da.isel(valid_time=time_sel)
        return da.values.astype(np.float64)   # shape (pressure_level,)

    T_K_raw = _get("t")

    # ── Mask any NaN / fill levels ────────────────────────────────────────────
    valid_mask = np.isfinite(T_K_raw) & (T_K_raw > 100.0) & (T_K_raw < 400.0)
    if not np.all(valid_mask):
        n_invalid = int(np.sum(~valid_mask))
        print(f"  Masking {n_invalid} invalid level(s) from T")

    P_hPa_use = P_hPa[valid_mask]
    T_K       = T_K_raw[valid_mask]

    z_km    = _pressure_to_altitude(P_hPa_use, T_K)
    P_Pa    = P_hPa_use * 100.0
    n_total = _number_density(P_Pa, T_K)

    out = {
        "p_levels_hPa": P_hPa_use,
        "P_Pa"        : P_Pa,
        "T_K"         : T_K,
        "sigma_T"     : _compute_sigma("t", T_K, P_hPa_use),
        "z_km"        : z_km,
        "n_total"     : n_total,
        "lat"         : actual_lat,
        "lon"         : actual_lon,
        "time_iso"    : time_label,
        "source"      : "era5",
        "product"     : "reanalysis-era5-pressure-levels",
        "vars_present": ",".join(found_vars),
    }

    # ── Specific humidity -> H2O VMR + column ─────────────────────────────────
    if "q" in found_vars:
        q = _get("q")[valid_mask]
        vmr_H2O = np.clip(_mmr_to_vmr(q, _M_H2O), 0.0, 1.0)
        col_H2O = _column_density(vmr_H2O, n_total, z_km)
        out["vmr_H2O"]       = vmr_H2O
        out["col_H2O"]       = col_H2O
        out["sigma_vmr_H2O"] = _compute_sigma("q", vmr_H2O, P_hPa_use)
        print(f"  H2O column: {col_H2O:.3e} molecules/m^2")

    # ── Ozone mass mixing ratio -> O3 VMR + column ────────────────────────────
    if "o3" in found_vars:
        o3 = _get("o3")[valid_mask]
        vmr_O3 = np.clip(_mmr_to_vmr(o3, _M_O3), 0.0, 1.0)
        col_O3 = _column_density(vmr_O3, n_total, z_km)
        o3_du  = col_O3 / 2.6867e20
        out["vmr_O3"]       = vmr_O3
        out["col_O3"]       = col_O3
        out["sigma_vmr_O3"] = _compute_sigma("o3", vmr_O3, P_hPa_use)
        print(f"  O3 column : {o3_du:.1f} DU  "
              f"(expected ~250-350 DU mid-latitude)")

    return out


# ══════════════════════════════════════════════════════════════════════════════
# Writer
# ══════════════════════════════════════════════════════════════════════════════

def write_npz(output_path, column):
    """Write column dict to LEOS .npz — same convention as convert_merra2.py."""
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
    print(f"  z range : {column['z_km'].min():.1f}-"
          f"{column['z_km'].max():.1f} km")
    print(f"  T range : {column['T_K'].min():.1f}-"
          f"{column['T_K'].max():.1f} K")
    print(f"  Fields  : {', '.join(k for k in column if not k.startswith('_'))}")


def _default_output_path(nc_path, lat, lon, time_label):
    date_str = "unknown"
    for token in str(time_label).replace("T", "-").split("-"):
        if len(token) == 8 and token.isdigit():
            date_str = f"{token[:4]}-{token[4:6]}-{token[6:8]}"
            break
        if len(token) == 4 and token.isdigit() and token.startswith(("19", "20")):
            date_str = token
    lat_str = f"{abs(lat):.2f}{'N' if lat >= 0 else 'S'}"
    lon_str = f"{abs(lon):.2f}{'E' if lon >= 0 else 'W'}"
    return os.path.join(_DEFAULT_OUT_DIR, f"era5_{date_str}_{lat_str}_{lon_str}.npz")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("nc_file", help="Path to ERA5 netCDF file (.nc).")
    parser.add_argument("--lat", type=float, help="Target latitude [deg].")
    parser.add_argument("--lon", type=float, help="Target longitude [deg].")

    time_group = parser.add_mutually_exclusive_group()
    time_group.add_argument("--time", default=None,
                             metavar="YYYY-MM-DDTHH:MM",
                             help="Select nearest time step to this datetime.")
    time_group.add_argument("--time-index", type=int, default=None,
                             metavar="N", help="Select time step by 0-based index.")
    time_group.add_argument("--time-average", action="store_true",
                             help="Average all time steps.")

    parser.add_argument("--list-times", action="store_true",
                         help="Print available time steps and exit.")
    parser.add_argument("--list-vars", action="store_true",
                         help="Print variables present in the file and exit.")
    parser.add_argument("--output", default=None,
                         help="Output .npz path. "
                              "Default: leos/data/era5_DATE_LAT_LON.npz")

    args = parser.parse_args()

    if not os.path.exists(args.nc_file):
        sys.exit(f"ERROR: File not found: {args.nc_file}")

    print("\nLEOS ERA5 converter")
    print(f"  input : {args.nc_file}")

    try:
        ds = xr.open_dataset(args.nc_file)
    except Exception as e:
        sys.exit(
            f"ERROR: Could not open '{args.nc_file}' as netCDF ({e}).\n"
            f"Ensure the CDS download request used format: 'netcdf', and "
            f"that you have accepted the dataset's licence at\n"
            f"https://cds.climate.copernicus.eu/datasets/"
            f"reanalysis-era5-pressure-levels?tab=download"
        )

    # ── List modes ────────────────────────────────────────────────────────────
    if args.list_times:
        times = ds["valid_time"].values
        print(f"\n  {len(times)} time step(s):")
        for i, t in enumerate(times):
            print(f"    [{i}]  {str(t)[:19]}")
        return

    if args.list_vars:
        print("\n  Variables in file:")
        for v in _VAR_MAP:
            status = "PRESENT" if v in ds.data_vars else "absent"
            print(f"    {v:4s}  {status}")
        print("\n  Coordinates:")
        for c in ("valid_time", "latitude", "longitude", "pressure_level"):
            if c in ds.coords:
                print(f"    {c:15s} shape={ds[c].shape}")
        return

    if args.lat is None or args.lon is None:
        parser.error(
            "--lat and --lon are required for extraction. "
            "Use --list-times or --list-vars to inspect the file."
        )

    # ── Time index ────────────────────────────────────────────────────────────
    time_idx = None
    if args.time is not None:
        times = ds["valid_time"].values
        target = np.datetime64(args.time)
        time_idx = int(np.argmin(np.abs(times - target)))
        print(f"  Requested: {args.time} -> nearest [{time_idx}]: "
              f"{str(times[time_idx])[:19]}")
    elif args.time_index is not None:
        time_idx = args.time_index

    column = extract_column(
        ds,
        lat_target=args.lat,
        lon_target=args.lon,
        time_idx=time_idx,
        time_average=args.time_average,
    )

    output_path = args.output or _default_output_path(
        args.nc_file, args.lat, args.lon, column["time_iso"]
    )
    print(f"  output: {output_path}")
    write_npz(output_path, column)

    print("\nLoad in LEOS with:")
    print("  from leos.atmosphere_earth import AtmosphericColumn")
    print(f"  col = AtmosphericColumn.from_npz('{output_path}')")


if __name__ == "__main__":
    main()
