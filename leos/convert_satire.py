#!/usr/bin/env python3
"""
scripts/convert_satire.py
--------------------------
Convert SATIRE family reconstruction files to the LEOS .npz format
expected by leos/solar_spectrum.py (_get_satire and related fetchers).

Supported variants
------------------
  satire-s   SATIRE-S daily reconstruction (1974–present)
              IDL binary .sav or structured ASCII with julian/wl_lb/wl_ub/ssi fields.
  satire-t   SATIRE-T sunspot-based reconstruction (1611–2025)
              SSI_ISN.txt or SSI_GSN.txt + optional TSI file.
  satire-m   SATIRE-M millennial reconstruction (6755 BC–1885 AD)
              SATIRE-M_wu18_ssi.txt format.
  pmip4      PMIP4 composite (Holocene–2015)
              Same IDL 4-array layout as SATIRE-M but longer header.

Output
------
All variants write to leos/data/satire_{variant}.npz with schema:
  wavelengths   [nm]          1D float64, LEOS master grid (115–2400 nm, 1 nm)
  flux          [W/m²/nm]     1D float64, time-averaged SSI on master grid
  uncertainty   [W/m²/nm]     1D float64, combined σ(λ)
  source_wl     [nm]          1D float64, original wavelength centres
  source_flux   [W/m²/nm]     2D float64, full SSI matrix [n_wl, n_time] (if --keep-time-series)
  source_time   [JD or year]  1D float64, time axis
  tsi           [W/m²]        1D float64, TSI series (if available)

Usage examples
--------------
  # SATIRE-S (ASCII structured text)
  python scripts/convert_satire.py satire-s path/to/satire_s_ssi.txt

  # SATIRE-T using ISN sunspot number, extract single date
  python scripts/convert_satire.py satire-t path/to/SSI_ISN.txt --date 2024-01-15

  # SATIRE-T keep full time series
  python scripts/convert_satire.py satire-t path/to/SSI_GSN.txt --keep-time-series

  # SATIRE-M millennial
  python scripts/convert_satire.py satire-m path/to/SATIRE-M_wu18_ssi.txt

  # PMIP4 composite
  python scripts/convert_satire.py pmip4 path/to/pmip4_ssi.txt --keep-time-series

  # Override output path
  python scripts/convert_satire.py satire-t path/to/SSI_ISN.txt --output leos/data/my_satire.npz

Download locations
------------------
  SATIRE-S : https://www2.mps.mpg.de/projects/sun-climate/data.html
  SATIRE-T : https://www2.mps.mpg.de/projects/sun-climate/data.html
  SATIRE-M : https://www2.mps.mpg.de/projects/sun-climate/data.html
  PMIP4    : https://www2.mps.mpg.de/projects/sun-climate/data/PMIP6/
"""

import argparse
import os
import sys
import numpy as np
from datetime import datetime, timedelta

# ── LEOS master wavelength grid ───────────────────────────────────────────────
# 1 nm spacing, 115–2400 nm — consistent with solar_spectrum.py pipeline
MASTER_WL = np.arange(115.0, 2401.0, 1.0)   # 2286 points

# ── Default output directory ──────────────────────────────────────────────────
_DEFAULT_OUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "leos", "data"
)


# ══════════════════════════════════════════════════════════════════════════════
# Master grid interpolation
# ══════════════════════════════════════════════════════════════════════════════

def _to_master_grid(source_wl, flux_1d):
    """
    Interpolate a single flux spectrum onto the LEOS master grid.
    Extrapolated regions (outside source coverage) are set to 0.

    Parameters
    ----------
    source_wl : 1D array, wavelengths in nm
    flux_1d   : 1D array, flux in W/m²/nm

    Returns
    -------
    1D array on MASTER_WL
    """
    interp = np.interp(MASTER_WL, source_wl, flux_1d,
                       left=0.0, right=0.0)
    return interp


def _sigma_from_bounds(flux, ll, ul):
    """
    Convert asymmetric lower/upper bounds to 1-sigma uncertainty.
    σ = (ul - ll) / 2  (treat bounds as ±1σ around central value).
    """
    return np.abs(ul - ll) / 2.0


def _calibration_sigma(wl_nm, flux):
    """
    Fallback: assign calibration uncertainty when no explicit bounds given.
    Uses SATIRE literature estimates:
      UV < 300 nm  : 5%
      300–400 nm   : 3%
      400–700 nm   : 2%
      > 700 nm     : 2%
    """
    sigma = np.zeros_like(flux)
    bands = [
        (115,  200, 8.0),
        (200,  300, 5.0),
        (300,  400, 3.0),
        (400,  700, 2.0),
        (700, 1e6,  2.0),
    ]
    for lo, hi, pct in bands:
        mask = (wl_nm >= lo) & (wl_nm < hi)
        sigma[mask] = flux[mask] * (pct / 100.0)
    return sigma


# ══════════════════════════════════════════════════════════════════════════════
# Julian date utilities
# ══════════════════════════════════════════════════════════════════════════════

def _jd_to_datetime(jd):
    """Convert Julian Day Number to Python datetime (approximate)."""
    # JD 2440587.5 = 1970-01-01
    unix = (jd - 2440587.5) * 86400.0
    return datetime(1970, 1, 1) + timedelta(seconds=unix)


def _decimal_year_to_jd(year_float):
    """Convert decimal year (e.g. 2024.5) to Julian Day."""
    year_int = int(year_float)
    frac = year_float - year_int
    # Days in this year
    days_in_year = 366 if (year_int % 4 == 0 and
                           (year_int % 100 != 0 or year_int % 400 == 0)) else 365
    day_of_year = frac * days_in_year
    # JD of Jan 1.0 of year_int
    # Using formula: JD(Jan 1, Y) ≈ 365.25*(Y-4716) + ... simplified:
    dt = datetime(max(year_int, 1), 1, 1) + timedelta(days=day_of_year)
    return (dt - datetime(1970, 1, 1)).total_seconds() / 86400.0 + 2440587.5


def _date_str_to_jd(date_str):
    """Convert 'YYYY-MM-DD' string to Julian Day."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return (dt - datetime(1970, 1, 1)).total_seconds() / 86400.0 + 2440587.5


# ══════════════════════════════════════════════════════════════════════════════
# SATIRE-S parser
# ══════════════════════════════════════════════════════════════════════════════

def parse_satire_s(filepath):
    """
    Parse SATIRE-S ASCII structured file.

    Expected layout (from MPS header):
      julian    — 1D array of Julian days
      tsi       — 1D array of TSI values
      tsi_ll    — lower uncertainty
      tsi_ul    — upper uncertainty
      wl_lb     — wavelength bin lower bounds (1D)
      wl_ub     — wavelength bin upper bounds (1D)
      ssi       — 2D array [n_wl, n_time] W/m²/nm
      ssi_ll    — lower uncertainty bound
      ssi_ul    — upper uncertainty bound

    The MPS ASCII export writes these arrays sequentially.
    Each array is preceded by a comment line starting with ';' or '#'.
    """
    print(f"  Parsing SATIRE-S: {filepath}")

    with open(filepath, "r") as f:
        lines = [l.rstrip() for l in f]

    # Strip comment/header lines, collect numeric blocks
    blocks = []
    current = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("#", ";", "!")):
            if current:
                blocks.append(current)
                current = []
        elif stripped == "":
            if current:
                blocks.append(current)
                current = []
        else:
            current.append(stripped)
    if current:
        blocks.append(current)

    def _parse_block(b):
        return np.array([float(x) for line in b for x in line.split()])

    if len(blocks) < 6:
        raise ValueError(
            f"Expected at least 6 data blocks in SATIRE-S file, found {len(blocks)}. "
            f"Check file format matches MPS v20250322 layout."
        )

    julian  = _parse_block(blocks[0])
    tsi     = _parse_block(blocks[1])
    tsi_ll  = _parse_block(blocks[2])
    tsi_ul  = _parse_block(blocks[3])
    wl_lb   = _parse_block(blocks[4])
    wl_ub   = _parse_block(blocks[5])
    wl_ctr  = 0.5 * (wl_lb + wl_ub)

    n_wl   = len(wl_ctr)
    n_time = len(julian)

    # SSI matrix: blocks 6, 7, 8 = ssi, ssi_ll, ssi_ul
    if len(blocks) >= 9:
        ssi    = _parse_block(blocks[6]).reshape(n_wl, n_time)
        ssi_ll = _parse_block(blocks[7]).reshape(n_wl, n_time)
        ssi_ul = _parse_block(blocks[8]).reshape(n_wl, n_time)
        has_bounds = True
    else:
        ssi    = _parse_block(blocks[6]).reshape(n_wl, n_time)
        ssi_ll = None
        ssi_ul = None
        has_bounds = False

    print(f"    wavelengths: {n_wl}, time steps: {n_time}")
    print(f"    wl range: {wl_ctr[0]:.1f}–{wl_ctr[-1]:.1f} nm")
    print(f"    time range: JD {julian[0]:.1f}–{julian[-1]:.1f}")

    return {
        "julian"    : julian,
        "tsi"       : tsi,
        "tsi_sigma" : _sigma_from_bounds(tsi, tsi_ll, tsi_ul) if has_bounds else None,
        "wl"        : wl_ctr,
        "ssi"       : ssi,        # [n_wl, n_time]
        "ssi_sigma" : _sigma_from_bounds(ssi, ssi_ll, ssi_ul) if has_bounds else None,
        "has_bounds": has_bounds,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SATIRE-T parser
# ══════════════════════════════════════════════════════════════════════════════

def parse_satire_t(filepath):
    """
    Parse SATIRE-T SSI file (SSI_ISN.txt or SSI_GSN.txt).

    File structure (from Temaj et al. 2026 readme):
      Line 1 : header1
      Line 2 : header2
      Line 3 : blank
      Line 4 : wavelength array (space-separated, nm)
      Line 5 : JD array (space-separated)
      Lines 6+: SSI matrix [n_wl rows × n_time cols]
    """
    print(f"  Parsing SATIRE-T: {filepath}")

    with open(filepath, "r") as f:
        header1 = f.readline()
        header2 = f.readline()
        f.readline()   # blank line

        wavelength = np.fromstring(f.readline(), sep=" ")
        jd         = np.fromstring(f.readline(), sep=" ")
        ssi        = np.loadtxt(f)   # [n_wl, n_time]

    n_wl, n_time = ssi.shape
    print(f"    wavelengths: {n_wl}, time steps: {n_time}")
    print(f"    wl range: {wavelength[0]:.1f}–{wavelength[-1]:.1f} nm")
    print(f"    JD range: {jd[0]:.1f}–{jd[-1]:.1f}")
    print(f"    header: {header1.strip()}")

    return {
        "julian" : jd,
        "wl"     : wavelength,
        "ssi"    : ssi,        # [n_wl, n_time]
        "tsi"    : None,       # load separately with --tsi-file
    }


def parse_satire_t_tsi(tsi_filepath):
    """Parse TSI_ISN.txt or TSI_GSN.txt."""
    print(f"  Parsing SATIRE-T TSI: {tsi_filepath}")
    jd, time_yr, tsi = np.loadtxt(tsi_filepath, usecols=(0, 1, 2), unpack=True)
    return jd, tsi


# ══════════════════════════════════════════════════════════════════════════════
# SATIRE-M parser
# ══════════════════════════════════════════════════════════════════════════════

def parse_satire_m(filepath, header_lines=9):
    """
    Parse SATIRE-M_wu18_ssi.txt.

    File structure (IDL layout):
      Lines 1–9  : header (skip)
      Next block : wavelength array [1070 values]
      Next block : wavelength bin widths [1070 values]
      Next block : date array [865 values], calendar year
      Next block : SSI matrix [1070 × 865]
    """
    print(f"  Parsing SATIRE-M: {filepath}")

    with open(filepath, "r") as f:
        # Skip header
        for _ in range(header_lines):
            f.readline()

        # Read remaining content as one stream of floats
        content = f.read()

    values = np.fromstring(content, sep=" ")

    n_wl = 1070

    if len(values) < n_wl * 2 + 2:
        raise ValueError(
            f"File too short — expected at least {n_wl*2+2} values, "
            f"got {len(values)}. Check header_lines (currently {header_lines})."
        )

    idx = 0
    wl  = values[idx : idx + n_wl]; idx += n_wl
    dwl = values[idx : idx + n_wl]; idx += n_wl

    # Infer n_time: remaining values = n_time + n_wl*n_time = n_time*(n_wl+1)
    remaining = len(values) - idx
    n_time = remaining // (n_wl + 1)

    if n_time < 1:
        raise ValueError(
            f"Could not infer n_time from {remaining} remaining values "
            f"with n_wl={n_wl}. Check header_lines (currently {header_lines})."
        )

    date    = values[idx : idx + n_time];          idx += n_time
    ssi_raw = values[idx : idx + n_wl * n_time]
    ssi     = ssi_raw.reshape(n_wl, n_time)

    # TSI from bin widths
    tsi = np.array([np.sum(dwl * ssi[:, i]) for i in range(n_time)])

    print(f"    wavelengths: {n_wl}, time steps: {n_time}")
    print(f"    wl range: {wl[0]:.1f}–{wl[-1]:.1f} nm")
    print(f"    date range: {date[0]:.1f}–{date[-1]:.1f} (calendar year)")

    return {
        "date" : date,    # calendar year (float)
        "julian": None,   # not provided, derived from date
        "wl"   : wl,
        "dwl"  : dwl,
        "ssi"  : ssi,     # [n_wl, n_time]
        "tsi"  : tsi,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PMIP4 composite parser
# ══════════════════════════════════════════════════════════════════════════════

def parse_pmip4(filepath, header_lines=12):
    """
    Parse PMIP4 composite SSI file.

    File structure:
      Lines 1–12 : header (skip; 14C has 12, 10Be may vary)
      Block 1    : wavelength centres [1070 values, nm]
      Block 2    : wavelength bin widths [1070 values, nm]
      Block 3    : time array [N values, decimal year]
      Block 4    : SSI matrix [1070 × N, W/m²/nm]

    N = 69235 for 14C reconstruction
    N = 61595 for 10Be reconstruction
    """
    print(f"  Parsing PMIP4: {filepath}")

    with open(filepath, "r") as f:
        for _ in range(header_lines):
            f.readline()
        content = f.read()

    values = np.fromstring(content, sep=" ")

    n_wl = 1070

    idx = 0
    wl  = values[idx : idx + n_wl]; idx += n_wl
    dwl = values[idx : idx + n_wl]; idx += n_wl

    remaining = len(values) - idx
    n_time = remaining // (n_wl + 1)
    print(f"    inferred n_time = {n_time}")

    time    = values[idx : idx + n_time];          idx += n_time
    ssi_raw = values[idx : idx + n_wl * n_time]
    ssi     = ssi_raw.reshape(n_wl, n_time)

    tsi = np.array([np.sum(dwl * ssi[:, i]) for i in range(n_time)])

    print(f"    wavelengths: {n_wl}, time steps: {n_time}")
    print(f"    wl range: {wl[0]:.1f}–{wl[-1]:.1f} nm")
    print(f"    time range: {time[0]:.2f}–{time[-1]:.2f} (decimal year)")

    return {
        "date"  : time,   # decimal year
        "julian": None,
        "wl"    : wl,
        "dwl"   : dwl,
        "ssi"   : ssi,
        "tsi"   : tsi,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Time selection
# ══════════════════════════════════════════════════════════════════════════════

def _select_time_index(data, date_str, variant):
    """
    Find the index in the time array closest to the requested date.
    Returns int index.
    """
    if date_str is None:
        return None   # caller will time-average

    target_jd = _date_str_to_jd(date_str)

    if "julian" in data and data["julian"] is not None:
        time_jd = data["julian"]
    elif "date" in data and data["date"] is not None:
        # Calendar year or decimal year — convert to JD approximately
        if variant in ("satire-m", "pmip4"):
            time_jd = np.array([_decimal_year_to_jd(y) for y in data["date"]])
        else:
            time_jd = data["date"]
    else:
        raise ValueError("No time array found in parsed data.")

    idx = int(np.argmin(np.abs(time_jd - target_jd)))
    closest = _jd_to_datetime(time_jd[idx]).strftime("%Y-%m-%d") \
              if variant not in ("satire-m", "pmip4") else f"year≈{data['date'][idx]:.1f}"
    print(f"    Selected time index {idx} (closest to {date_str}: {closest})")
    return idx


# ══════════════════════════════════════════════════════════════════════════════
# Output writer
# ══════════════════════════════════════════════════════════════════════════════

def write_npz(output_path, data, variant, date_str, keep_time_series):
    """
    Interpolate onto LEOS master grid and write .npz.
    """
    wl  = data["wl"]
    ssi = data["ssi"]   # [n_wl, n_time]

    time_idx = _select_time_index(data, date_str, variant)

    if time_idx is not None:
        # Single spectrum
        flux_1d = ssi[:, time_idx]
        label   = f"single date index {time_idx}"
    else:
        # Time average
        flux_1d = np.mean(ssi, axis=1)
        label   = "time average"

    print(f"    Computing {label} spectrum...")

    # Interpolate to master grid
    flux_master = _to_master_grid(wl, flux_1d)

    # Uncertainty
    if data.get("ssi_sigma") is not None:
        if time_idx is not None:
            sig_1d = data["ssi_sigma"][:, time_idx]
        else:
            sig_1d = np.mean(data["ssi_sigma"], axis=1)
        sigma_master = _to_master_grid(wl, sig_1d)
    else:
        sigma_master = _calibration_sigma(MASTER_WL, flux_master)

    # TSI series
    tsi = data.get("tsi")

    # Time array
    time_arr = data.get("julian") if data.get("julian") is not None else data.get("date")

    # Build save dict
    save_dict = {
        "wavelengths" : MASTER_WL,
        "flux"        : flux_master,
        "uncertainty" : sigma_master,
        "source_wl"   : wl,
        "variant"     : np.array([variant], dtype=object),
    }

    if tsi is not None:
        save_dict["tsi"] = tsi

    if time_arr is not None:
        save_dict["source_time"] = time_arr

    if keep_time_series:
        print(f"    Keeping full SSI time series [n_wl={ssi.shape[0]}, n_time={ssi.shape[1]}]...")
        save_dict["source_flux"] = ssi

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(output_path, **save_dict)

    size_mb = os.path.getsize(output_path) / 1e6
    print(f"  Written: {output_path} ({size_mb:.1f} MB)")
    print(f"  Master grid flux range: {flux_master.min():.4f}–{flux_master.max():.4f} W/m²/nm")
    tsi_check = np.trapezoid(flux_master, MASTER_WL)
    print(f"  Integrated flux (115–2400 nm): {tsi_check:.1f} W/m²")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "variant",
        choices=["satire-s", "satire-t", "satire-m", "pmip4"],
        help="Which SATIRE variant to convert.",
    )
    parser.add_argument(
        "ssi_file",
        help="Path to the SSI data file.",
    )
    parser.add_argument(
        "--tsi-file",
        default=None,
        help="(satire-t only) Path to TSI_ISN.txt or TSI_GSN.txt.",
    )
    parser.add_argument(
        "--date",
        default=None,
        metavar="YYYY-MM-DD",
        help="Extract spectrum for this date. Default: time average.",
    )
    parser.add_argument(
        "--keep-time-series",
        action="store_true",
        help="Store full SSI[n_wl, n_time] matrix in output (large files).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output .npz path. Default: leos/data/satire_{variant}.npz",
    )
    parser.add_argument(
        "--header-lines",
        type=int,
        default=None,
        help="Number of header lines to skip (satire-m: default 9, pmip4: default 12).",
    )

    args = parser.parse_args()

    # ── Check input file exists ───────────────────────────────────────────────
    if not os.path.exists(args.ssi_file):
        print(f"ERROR: SSI file not found: {args.ssi_file}", file=sys.stderr)
        sys.exit(1)

    # ── Default output path ───────────────────────────────────────────────────
    if args.output is None:
        safe_variant = args.variant.replace("-", "_")
        args.output = os.path.join(
            _DEFAULT_OUT_DIR, f"satire_{safe_variant}.npz"
        )

    print(f"\nLEOS SATIRE converter")
    print(f"  variant  : {args.variant}")
    print(f"  input    : {args.ssi_file}")
    print(f"  output   : {args.output}")
    print(f"  date     : {args.date or 'time average'}")
    print(f"  keep ts  : {args.keep_time_series}")
    print()

    # ── Parse ─────────────────────────────────────────────────────────────────
    if args.variant == "satire-s":
        data = parse_satire_s(args.ssi_file)

    elif args.variant == "satire-t":
        data = parse_satire_t(args.ssi_file)
        if args.tsi_file:
            jd_tsi, tsi = parse_satire_t_tsi(args.tsi_file)
            data["tsi"] = tsi

    elif args.variant == "satire-m":
        hl = args.header_lines if args.header_lines is not None else 9
        data = parse_satire_m(args.ssi_file, header_lines=hl)

    elif args.variant == "pmip4":
        hl = args.header_lines if args.header_lines is not None else 12
        data = parse_pmip4(args.ssi_file, header_lines=hl)

    # ── Write ─────────────────────────────────────────────────────────────────
    write_npz(
        output_path     = args.output,
        data            = data,
        variant         = args.variant,
        date_str        = args.date,
        keep_time_series= args.keep_time_series,
    )

    print("\nDone. Load in LEOS with:")
    print(f"  from leos.solar_spectrum import get_solar_spectrum, SpectralSource")
    print(f"  s = get_solar_spectrum(SpectralSource.SATIRE_S)")


if __name__ == "__main__":
    main()
