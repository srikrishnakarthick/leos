#!/usr/bin/env python3
"""
scripts/convert_lola.py
------------------------
Convert a LOLA lunar elevation GeoTIFF or NetCDF file into a LEOS
surface conditions .npz for use with leos/atmosphere_moon.py.

Computes for a given lat/lon point:
  - Surface elevation [m above reference radius 1737.4 km]
  - Local terrain slope [deg]
  - Local terrain aspect [deg, azimuth of downslope direction]
  - Horizon elevation angles at N azimuths [deg]
  - Illumination fraction from horizon mask

INPUT FORMATS
-------------

PRIMARY — GeoTIFF (recommended)
  Pixel-registered, MOON_PA DE421, readable with rasterio.
  Download: https://pgda.gsfc.nasa.gov/products/95
  Recommended file: LOLA 64ppd Global Shape Model GeoTIFF (~683 MB)

  pip install rasterio

FALLBACK — NetCDF/GMT
  Gridline-registered, older PDS products, readable with xarray.
  Download: https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/
  Products: ldam_4_float (4ppd), ldam_8_float (8ppd)

  pip install xarray netCDF4

NOT SUPPORTED — PDS3 .IMG binary
  Convert to GeoTIFF first:
    gdal_translate -of GTiff LDEM_4.IMG LDEM_4.tif

OUTPUT SCHEMA (leos/data/lola_LAT_LON.npz)
-------------------------------------------
  elevation_m           float64        elevation above 1737.4 km ref [m]
  slope_deg             float64        local slope [deg]
  aspect_deg            float64        downslope azimuth [deg]
  horizon_elevation_deg float64[n_az]  horizon angles at n_az azimuths
  illumination_fraction float64        fraction illuminated by sun
                                       (requires --sun-az and --sun-el)
  dem_patch_m           float64[n,n]   DEM patch around point [m]
  dem_patch_lat         float64[n]     latitude grid of patch
  dem_patch_lon         float64[n]     longitude grid of patch
  lat, lon              float64        target point
  elevation_ref_km      float64        reference radius [km] = 1737.4
  source                str            'lola_geotiff' or 'lola_netcdf'
  n_azimuths            int            number of horizon azimuths
  resolution_ppd        float64        pixels per degree of input DEM
  file_format           str            'geotiff' or 'netcdf'

USAGE
-----
  # Basic — elevation, slope, aspect only
  python scripts/convert_lola.py LDEM_64.tif --lat -89.5 --lon 0.0

  # With horizon mask (radius 5 km around point)
  python scripts/convert_lola.py LDEM_64.tif --lat -89.5 --lon 0.0 \\
      --horizon --radius-km 5.0 --n-azimuths 360

  # With illumination fraction (needs Sun position)
  python scripts/convert_lola.py LDEM_64.tif --lat -89.5 --lon 0.0 \\
      --horizon --radius-km 5.0 \\
      --sun-az 45.0 --sun-el 2.5

  # NetCDF fallback
  python scripts/convert_lola.py ldam_4.nc --lat 0.0 --lon 0.0 \\
      --format netcdf

  # Custom output path
  python scripts/convert_lola.py LDEM_64.tif --lat -89.5 --lon 0.0 \\
      --output my_lola_south_pole.npz

  # Inspect file metadata only
  python scripts/convert_lola.py LDEM_64.tif --info
"""

import argparse
import os
import sys
import warnings
import numpy as np

# ── LEOS output directory ─────────────────────────────────────────────────────
_DEFAULT_OUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "leos", "data"
)

# ── Moon reference radius ─────────────────────────────────────────────────────
_R_MOON_KM  = 1737.4
_R_MOON_M   = _R_MOON_KM * 1e3

# ══════════════════════════════════════════════════════════════════════════════
# Format detection
# ══════════════════════════════════════════════════════════════════════════════

def _detect_format(path: str, force: str = None) -> str:
    """
    Detect input file format.
    Returns 'geotiff' or 'netcdf'.
    Raises SystemExit for unsupported formats.
    """
    if force:
        return force.lower().strip()

    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        return "geotiff"
    if ext in (".nc", ".nc4", ".h5", ".hdf5", ".cdf"):
        return "netcdf"
    if ext in (".img", ".raw", ".dat"):
        sys.exit(
            f"ERROR: PDS3 .IMG binary format is not supported.\n"
            f"Convert to GeoTIFF first with GDAL:\n"
            f"  gdal_translate -of GTiff {path} output.tif\n"
            f"Then rerun with the .tif file."
        )
    # Default to GeoTIFF for unknown extensions
    warnings.warn(
        f"Unknown extension '{ext}' — assuming GeoTIFF. "
        f"Use --format netcdf if this is a NetCDF file.",
        UserWarning,
    )
    return "geotiff"


# ══════════════════════════════════════════════════════════════════════════════
# GeoTIFF reader (PRIMARY)
# ══════════════════════════════════════════════════════════════════════════════

def _read_geotiff(path: str):
    """
    Read LOLA GeoTIFF. Returns (data_array, transform, crs, meta).
    Requires rasterio.
    """
    try:
        import rasterio
    except ImportError:
        sys.exit(
            "ERROR: rasterio is required for GeoTIFF files.\n"
            "Install with: pip install rasterio"
        )

    with rasterio.open(path) as src:
        meta = {
            "driver"   : src.driver,
            "dtype"    : src.dtypes[0],
            "width"    : src.width,
            "height"   : src.height,
            "crs"      : str(src.crs),
            "bounds"   : src.bounds,
            "res"      : src.res,       # (pixel_width_deg, pixel_height_deg)
            "nodata"   : src.nodata,
            "count"    : src.count,
        }
        transform = src.transform
        data      = src.read(1).astype(np.float64)
        crs       = src.crs

        # Replace nodata with NaN
        if meta["nodata"] is not None:
            data[data == meta["nodata"]] = np.nan

    return data, transform, crs, meta


def _geotiff_extract_patch(path: str, lat: float, lon: float,
                            radius_km: float = 2.0):
    """
    Extract a DEM patch of radius_km around (lat, lon) from a GeoTIFF.

    Returns
    -------
    patch : float64 2D array   elevation [m]
    lat_grid : float64 1D      latitude of each row [deg]
    lon_grid : float64 1D      longitude of each col [deg]
    center_elev : float        elevation at (lat, lon) [m]
    res_ppd : float            resolution in pixels per degree
    """
    try:
        import rasterio
        from rasterio.windows import from_bounds
    except ImportError:
        sys.exit("ERROR: rasterio not installed. Run: pip install rasterio")

    with rasterio.open(path) as src:
        res_deg = abs(src.res[0])           # degrees per pixel
        res_ppd = 1.0 / res_deg

        # Radius in degrees (approximate, valid for small areas)
        radius_deg = radius_km / (np.pi * _R_MOON_KM / 180.0)
        pad        = max(radius_deg, 2 * res_deg)   # at least 2 pixels

        win_bounds = (
            lon - pad,   # left
            lat - pad,   # bottom
            lon + pad,   # right
            lat + pad,   # top
        )

        # Clamp to file bounds
        file_bounds = src.bounds
        win_bounds = (
            max(win_bounds[0], file_bounds.left),
            max(win_bounds[1], file_bounds.bottom),
            min(win_bounds[2], file_bounds.right),
            min(win_bounds[3], file_bounds.top),
        )

        window = from_bounds(*win_bounds, transform=src.transform)
        patch  = src.read(1, window=window).astype(np.float64)

        if src.nodata is not None:
            patch[patch == src.nodata] = np.nan

        # Build coordinate grids for the window
        win_transform = src.window_transform(window)
        nrows, ncols  = patch.shape
        lon_grid = np.array([
            win_transform * (c + 0.5, 0.5) for c in range(ncols)
        ])[:, 0]
        lat_grid = np.array([
            win_transform * (0.5, r + 0.5) for r in range(nrows)
        ])[:, 1]

        # Extract center elevation
        row, col = src.index(lon, lat)
        try:
            center_elev = float(src.read(1)[row, col])
            if src.nodata is not None and center_elev == src.nodata:
                center_elev = np.nan
        except (IndexError, Exception):
            center_elev = float(np.nanmean(patch))

    return patch, lat_grid, lon_grid, center_elev, res_ppd


# ══════════════════════════════════════════════════════════════════════════════
# NetCDF reader (FALLBACK)
# ══════════════════════════════════════════════════════════════════════════════

def _read_netcdf_info(path: str):
    """Print NetCDF file metadata."""
    try:
        import xarray as xr
    except ImportError:
        sys.exit(
            "ERROR: xarray is required for NetCDF files.\n"
            "Install with: pip install xarray netCDF4\n"
            "NOTE: GeoTIFF (PRIMARY format) is recommended over NetCDF.\n"
            "Download GeoTIFF from: https://pgda.gsfc.nasa.gov/products/95"
        )
    ds = xr.open_dataset(path)
    print(ds)
    print("\nVariables:", list(ds.data_vars))
    print("Coordinates:", list(ds.coords))
    ds.close()


def _netcdf_extract_patch(path: str, lat: float, lon: float,
                           radius_km: float = 2.0):
    """
    Extract DEM patch from NetCDF/GMT LOLA file.

    NOTE: NetCDF LOLA products are gridline-registered (pixel edges at
    grid coordinates, not centres). A half-pixel correction is applied
    automatically here for consistency with the GeoTIFF (pixel-registered)
    output.

    Returns same tuple as _geotiff_extract_patch.
    """
    try:
        import xarray as xr
    except ImportError:
        sys.exit(
            "ERROR: xarray is required for NetCDF files.\n"
            "Install with: pip install xarray netCDF4\n"
            "RECOMMENDATION: Use GeoTIFF format instead "
            "(https://pgda.gsfc.nasa.gov/products/95) — it is "
            "pixel-registered and does not require this correction."
        )

    ds = xr.open_dataset(path)

    # Find lat/lon/elevation variable names flexibly
    lat_var = next((v for v in ds.coords if v.lower() in ("lat", "latitude", "y")), None)
    lon_var = next((v for v in ds.coords if v.lower() in ("lon", "longitude", "x")), None)
    elev_var = next((v for v in ds.data_vars
                     if any(k in v.lower() for k in ("elev", "dem", "z", "topo", "height"))),
                    list(ds.data_vars)[0] if ds.data_vars else None)

    if lat_var is None or lon_var is None or elev_var is None:
        ds.close()
        sys.exit(
            f"ERROR: Could not identify lat/lon/elevation variables in {path}.\n"
            f"Variables: {list(ds.data_vars)}\n"
            f"Coords: {list(ds.coords)}"
        )

    lats = ds[lat_var].values
    lons = ds[lon_var].values
    res_deg = abs(float(lats[1] - lats[0]))
    res_ppd = 1.0 / res_deg

    # Half-pixel correction for gridline-registered data
    # (gridline: coordinates refer to pixel edges → shift by half pixel
    #  to get pixel centres, consistent with GeoTIFF pixel-registered output)
    half = res_deg / 2.0
    lats_centre = lats + half
    lons_centre = lons + half

    radius_deg = radius_km / (np.pi * _R_MOON_KM / 180.0)
    pad        = max(radius_deg, 2 * res_deg)

    lat_mask = (lats_centre >= lat - pad) & (lats_centre <= lat + pad)
    lon_mask = (lons_centre >= lon - pad) & (lons_centre <= lon + pad)

    patch_da = ds[elev_var].isel(
        **{lat_var: lat_mask, lon_var: lon_mask}
    )
    patch = patch_da.values.astype(np.float64)

    # Handle NaN/fill values
    fill = ds[elev_var].attrs.get("_FillValue",
           ds[elev_var].attrs.get("missing_value", None))
    if fill is not None:
        patch[patch == fill] = np.nan

    lat_grid = lats_centre[lat_mask]
    lon_grid = lons_centre[lon_mask]

    # Center elevation
    i_lat = int(np.argmin(np.abs(lat_grid - lat)))
    i_lon = int(np.argmin(np.abs(lon_grid - lon)))
    center_elev = float(patch[i_lat, i_lon])

    ds.close()
    return patch, lat_grid, lon_grid, center_elev, res_ppd


# ══════════════════════════════════════════════════════════════════════════════
# Terrain analysis
# ══════════════════════════════════════════════════════════════════════════════

def _compute_slope_aspect(patch: np.ndarray,
                           res_m: float) -> tuple:
    """
    Compute slope and aspect from a DEM patch using finite differences.

    Parameters
    ----------
    patch  : 2D float array   elevation [m], shape (nrows, ncols)
    res_m  : float            pixel size [m]

    Returns
    -------
    slope_deg  : float   slope at centre pixel [deg from horizontal]
    aspect_deg : float   downslope azimuth [deg clockwise from North]
    """
    nrows, ncols = patch.shape
    cr, cc = nrows // 2, ncols // 2   # centre pixel

    # 3×3 neighbourhood
    if nrows < 3 or ncols < 3:
        return 0.0, 0.0

    z = patch[cr-1:cr+2, cc-1:cc+2]
    if np.any(np.isnan(z)):
        return 0.0, 0.0

    # Horn (1981) finite difference
    dz_dx = ((z[0,2] + 2*z[1,2] + z[2,2]) -
              (z[0,0] + 2*z[1,0] + z[2,0])) / (8 * res_m)
    dz_dy = ((z[0,0] + 2*z[0,1] + z[0,2]) -
              (z[2,0] + 2*z[2,1] + z[2,2])) / (8 * res_m)

    slope_rad  = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    aspect_rad = np.arctan2(-dz_dy, dz_dx)

    slope_deg  = np.rad2deg(slope_rad)
    aspect_deg = (90.0 - np.rad2deg(aspect_rad)) % 360.0

    return float(slope_deg), float(aspect_deg)


def _compute_horizon(patch: np.ndarray,
                     lat_grid: np.ndarray,
                     lon_grid: np.ndarray,
                     center_lat: float,
                     center_lon: float,
                     center_elev: float,
                     n_azimuths: int = 360) -> np.ndarray:
    """
    Compute horizon elevation angle at n_azimuths equally-spaced azimuths.

    For each azimuth, scans outward from the centre pixel and finds the
    maximum elevation angle to any DEM point in that direction.

    Parameters
    ----------
    patch       : 2D float array   elevation [m]
    lat_grid    : 1D float         latitude of each row [deg]
    lon_grid    : 1D float         longitude of each col [deg]
    center_lat/lon/elev : float    observer position
    n_azimuths  : int              number of azimuth directions

    Returns
    -------
    horizon_el : float64[n_azimuths]   horizon elevation [deg]
    """
    azimuths   = np.linspace(0, 360, n_azimuths, endpoint=False)
    horizon_el = np.zeros(n_azimuths)

    # Pixel size in meters (approximate, valid for small patches)
    if len(lat_grid) < 2 or len(lon_grid) < 2:
        return horizon_el

    dlat_m = abs(lat_grid[1] - lat_grid[0]) * np.pi * _R_MOON_M / 180.0
    dlon_m = abs(lon_grid[1] - lon_grid[0]) * np.pi * _R_MOON_M / 180.0 \
             * np.cos(np.deg2rad(center_lat))
    res_m  = 0.5 * (dlat_m + dlon_m)

    nrows, ncols = patch.shape
    cr = np.argmin(np.abs(lat_grid - center_lat))
    cc = np.argmin(np.abs(lon_grid - center_lon))

    for i, az_deg in enumerate(azimuths):
        az_rad = np.deg2rad(az_deg)
        drow   = -np.cos(az_rad)   # North = row decreasing
        dcol   =  np.sin(az_rad)   # East  = col increasing

        max_el = 0.0
        step   = 1
        while True:
            r = int(round(cr + drow * step))
            c = int(round(cc + dcol * step))
            if r < 0 or r >= nrows or c < 0 or c >= ncols:
                break
            z = patch[r, c]
            if not np.isnan(z):
                dist_m = step * res_m
                el_rad = np.arctan2(z - center_elev, dist_m)
                el_deg = np.rad2deg(el_rad)
                if el_deg > max_el:
                    max_el = el_deg
            step += 1

        horizon_el[i] = max(0.0, max_el)

    return horizon_el


def _illumination_from_horizon(horizon_el: np.ndarray,
                                sun_az_deg: float,
                                sun_el_deg: float) -> float:
    """
    Compute illumination fraction from horizon mask and Sun position.

    Returns 1.0 if Sun is above the local horizon at its azimuth,
    0.0 if below (in shadow). Interpolates between azimuth samples.

    Parameters
    ----------
    horizon_el  : float64[n]   horizon elevation [deg] at n azimuths
    sun_az_deg  : float        Sun azimuth [deg clockwise from North]
    sun_el_deg  : float        Sun elevation angle [deg above horizontal]
    """
    if sun_el_deg <= 0.0:
        return 0.0   # Sun below geometric horizon

    n = len(horizon_el)
    azimuths = np.linspace(0, 360, n, endpoint=False)

    # Interpolate horizon elevation at Sun azimuth
    sun_az = sun_az_deg % 360.0
    horizon_at_sun = float(np.interp(sun_az, azimuths,
                                     horizon_el, period=360.0))

    if sun_el_deg > horizon_at_sun:
        return 1.0   # illuminated
    else:
        return 0.0   # in shadow (terrain occults Sun)


# ══════════════════════════════════════════════════════════════════════════════
# Writer
# ══════════════════════════════════════════════════════════════════════════════

def write_npz(output_path: str, result: dict):
    """Write result dict to LEOS .npz schema."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    save_dict = {}
    for k, v in result.items():
        if isinstance(v, str):
            save_dict[k] = np.array([v], dtype=object)
        elif isinstance(v, (int, float, np.floating, np.integer)):
            save_dict[k] = np.array(v, dtype=np.float64)
        elif v is None:
            continue
        else:
            save_dict[k] = np.asarray(v, dtype=np.float64)

    np.savez(output_path, **save_dict)
    size_kb = os.path.getsize(output_path) / 1e3
    print(f"\n  Written : {output_path} ({size_kb:.1f} KB)")
    fields = [k for k in save_dict if not k.startswith("_")]
    print(f"  Fields  : {', '.join(fields)}")


def _default_path(lat: float, lon: float) -> str:
    lat_str = f"{abs(lat):.2f}{'N' if lat >= 0 else 'S'}"
    lon_str = f"{abs(lon):.2f}{'E' if lon >= 0 else 'W'}"
    return os.path.join(_DEFAULT_OUT_DIR, f"lola_{lat_str}_{lon_str}.npz")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("dem_file",
                        help="Path to LOLA DEM file (.tif GeoTIFF or .nc NetCDF).")
    parser.add_argument("--lat",  type=float,
                        help="Target latitude [deg N].")
    parser.add_argument("--lon",  type=float,
                        help="Target longitude [deg E].")
    parser.add_argument("--radius-km", type=float, default=2.0,
                        help="Radius of DEM patch to extract [km]. Default 2.")
    parser.add_argument("--horizon", action="store_true",
                        help="Compute horizon elevation mask.")
    parser.add_argument("--n-azimuths", type=int, default=360,
                        help="Number of azimuth directions for horizon. Default 360.")
    parser.add_argument("--sun-az", type=float, default=None,
                        help="Sun azimuth [deg clockwise from North] for "
                             "illumination fraction.")
    parser.add_argument("--sun-el", type=float, default=None,
                        help="Sun elevation angle [deg] for illumination fraction.")
    parser.add_argument("--format", default=None,
                        choices=["geotiff", "netcdf"],
                        help="Force input format. Auto-detected from extension "
                             "if not specified.")
    parser.add_argument("--info", action="store_true",
                        help="Print file metadata and exit.")
    parser.add_argument("--output", default=None,
                        help="Output .npz path.")

    args = parser.parse_args()

    if not os.path.exists(args.dem_file):
        sys.exit(f"ERROR: File not found: {args.dem_file}")

    fmt = _detect_format(args.dem_file, args.format)

    print(f"\nLEOS LOLA converter")
    print(f"  input  : {args.dem_file}")
    print(f"  format : {fmt} ({'PRIMARY — recommended' if fmt=='geotiff' else 'FALLBACK — GeoTIFF preferred'})")

    # ── Info mode ─────────────────────────────────────────────────────────────
    if args.info:
        if fmt == "geotiff":
            try:
                import rasterio
                with rasterio.open(args.dem_file) as src:
                    print(f"\n  Driver   : {src.driver}")
                    print(f"  Size     : {src.width} x {src.height} pixels")
                    print(f"  CRS      : {src.crs}")
                    print(f"  Bounds   : {src.bounds}")
                    print(f"  Res      : {src.res} deg/pixel")
                    print(f"  PPD      : {1/src.res[0]:.1f}")
                    print(f"  Dtype    : {src.dtypes[0]}")
                    print(f"  NoData   : {src.nodata}")
                    print(f"  Bands    : {src.count}")
            except ImportError:
                sys.exit("ERROR: rasterio not installed. Run: pip install rasterio")
        else:
            _read_netcdf_info(args.dem_file)
        return

    # ── Extraction requires lat/lon ───────────────────────────────────────────
    if args.lat is None or args.lon is None:
        parser.error("--lat and --lon are required for extraction. "
                     "Use --info to inspect the file.")

    print(f"  lat    : {args.lat}")
    print(f"  lon    : {args.lon}")
    print(f"  radius : {args.radius_km} km")

    # ── Extract DEM patch ─────────────────────────────────────────────────────
    if fmt == "geotiff":
        patch, lat_grid, lon_grid, center_elev, res_ppd = \
            _geotiff_extract_patch(args.dem_file, args.lat, args.lon,
                                   args.radius_km)
    else:
        print("\n  NOTE: Using NetCDF fallback format. GeoTIFF is recommended.")
        print("  Download GeoTIFF from: https://pgda.gsfc.nasa.gov/products/95")
        patch, lat_grid, lon_grid, center_elev, res_ppd = \
            _netcdf_extract_patch(args.dem_file, args.lat, args.lon,
                                  args.radius_km)

    print(f"\n  DEM patch: {patch.shape[0]}×{patch.shape[1]} pixels")
    print(f"  Resolution: {res_ppd:.1f} ppd")
    print(f"  Elevation at point: {center_elev:.1f} m")

    # ── Slope and aspect ──────────────────────────────────────────────────────
    # Pixel size in metres at this latitude
    res_deg = 1.0 / res_ppd
    res_m   = res_deg * np.pi * _R_MOON_M / 180.0 \
              * np.cos(np.deg2rad(args.lat))
    res_m   = max(res_m, 1.0)   # guard against zero at poles

    slope_deg, aspect_deg = _compute_slope_aspect(patch, res_m)
    print(f"  Slope : {slope_deg:.2f} deg")
    print(f"  Aspect: {aspect_deg:.1f} deg")

    # ── Horizon mask ──────────────────────────────────────────────────────────
    horizon_el        = None
    illumination_frac = None

    if args.horizon:
        print(f"\n  Computing horizon mask ({args.n_azimuths} azimuths)...")
        horizon_el = _compute_horizon(
            patch, lat_grid, lon_grid,
            args.lat, args.lon, center_elev,
            n_azimuths=args.n_azimuths,
        )
        print(f"  Horizon: min={horizon_el.min():.1f}° "
              f"max={horizon_el.max():.1f}° "
              f"mean={horizon_el.mean():.1f}°")

        if args.sun_az is not None and args.sun_el is not None:
            illumination_frac = _illumination_from_horizon(
                horizon_el, args.sun_az, args.sun_el
            )
            print(f"  Sun az={args.sun_az}° el={args.sun_el}° → "
                  f"illumination={illumination_frac:.0f}")
    else:
        if args.sun_az is not None or args.sun_el is not None:
            print("  NOTE: --sun-az/--sun-el ignored without --horizon.")

    # ── Assemble result ───────────────────────────────────────────────────────
    result = {
        "lat"                   : args.lat,
        "lon"                   : args.lon,
        "elevation_m"           : center_elev,
        "slope_deg"             : slope_deg,
        "aspect_deg"            : aspect_deg,
        "elevation_ref_km"      : _R_MOON_KM,
        "resolution_ppd"        : res_ppd,
        "dem_patch_m"           : patch,
        "dem_patch_lat"         : lat_grid,
        "dem_patch_lon"         : lon_grid,
        "source"                : f"lola_{fmt}",
        "file_format"           : fmt,
        "n_azimuths"            : args.n_azimuths if args.horizon else 0,
    }

    if horizon_el is not None:
        result["horizon_elevation_deg"] = horizon_el
    if illumination_frac is not None:
        result["illumination_fraction"] = illumination_frac
    if args.sun_az is not None:
        result["sun_az_deg"] = args.sun_az
    if args.sun_el is not None:
        result["sun_el_deg"] = args.sun_el

    # ── Write ─────────────────────────────────────────────────────────────────
    output_path = args.output or _default_path(args.lat, args.lon)
    print(f"\n  output: {output_path}")
    write_npz(output_path, result)

    print(f"\nLoad in LEOS with:")
    print(f"  from leos.atmosphere_moon import MoonSurfaceConditions")
    print(f"  cond = MoonSurfaceConditions.from_npz('{output_path}')")


if __name__ == "__main__":
    main()
