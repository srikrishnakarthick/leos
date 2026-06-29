"""
leos/kernels/missions/fetch_maven_kernels.py

MAVEN kernel resolution: static kernels (FK + structure SPK), ancillary
SPK, the latest SCLK, the rolling reconstructed-orbit SPK, and (when a
time window is given) the weekly attitude CK files that actually cover
that window.

This is the one mission module with real dynamic logic -- the others are
plain static (filename, base_url) lists.
"""

import re
import requests

from ... import _kernel_common as _kc
from ...fetch_generic_kernels import select_common_kernels

# ── MAVEN kernel base URL ─────────────────────────────────────────────────────
_MAVEN_BASE = "https://naif.jpl.nasa.gov/pub/naif/MAVEN/kernels/"

# Static kernels: always load these (latest version of each) ──────────────────
MAVEN_STATIC_KERNELS = [
    # FK: latest frames kernel
    ("maven_v12.tf",              _MAVEN_BASE + "fk/"),
    # LSK: handled by select_common_kernels (naif0012.tls) -- skip
    # PCK: handled by select_common_kernels (pck00011.tpc) -- skip
    # Structure SPK
    ("maven_struct_v12.bsp",      _MAVEN_BASE + "spk/"),
]

# Rolling reconstructed SPK (always up to date, covers full mission) ──────────
MAVEN_RECONSTRUCTED_SPK = ("maven_orb_rec.bsp", _MAVEN_BASE + "spk/")

# Predicted SPK (future coverage) ─────────────────────────────────────────────
MAVEN_PREDICTED_SPK = ("maven_orb.bsp", _MAVEN_BASE + "spk/")

# Background Mars + de421 needed by mission kernels (NAIF recommendation) ─────
MAVEN_ANCILLARY_SPK = [
    ("de421.bsp",    _MAVEN_BASE + "spk/"),
    ("mar097s.bsp",  _MAVEN_BASE + "spk/"),
]


def resolve_maven_sclk():
    """
    Fetch the MAVEN sclk/ directory listing and return the filename of the
    highest-numbered MVN_SCLKSCET .tsc file.  Falls back to a known-good
    file if the listing fetch fails.
    """
    _FALLBACK = "MVN_SCLKSCET.00133.tsc"
    try:
        resp = requests.get(_MAVEN_BASE + "sclk/", timeout=10)
        resp.raise_for_status()
        matches = re.findall(r"MVN_SCLKSCET\.(\d{5})\.tsc", resp.text)
        if not matches:
            return _FALLBACK
        latest = f"MVN_SCLKSCET.{max(matches)}.tsc"
        return latest
    except Exception as e:
        print(f"Warning: could not fetch MAVEN SCLK listing ({e}); "
              f"falling back to {_FALLBACK}")
        return _FALLBACK


_MAVEN_CK_DATE_RE = re.compile(
    r"mvn_(sc|app)_rel_(\d{6})_(\d{6})_v\d+\.bc"
)


def _parse_maven_ck_date(token):
    """Parse a YYMMDD string (e.g. '141013') into an astropy Time."""
    try:
        return _kc._to_time_or_none(f"20{token[:2]}-{token[2:4]}-{token[4:6]}")
    except Exception:
        return None


def resolve_maven_ck(time=None, time_range=None, structure="sc"):
    """
    Return the list of MAVEN attitude CK filenames (mvn_sc_rel_* or
    mvn_app_rel_*) whose coverage overlaps the requested time/time_range.

    structure: 'sc' for spacecraft, 'app' for articulated payload platform.

    Fetches the ck/ directory listing, filters by structure and coverage,
    and returns filenames sorted oldest-first (so SPICE loads them in the
    right priority order).

    Falls back to an empty list (attitude-free) if the listing fetch fails.
    """
    req_lo, req_hi = _kc._normalize_window(time, time_range)

    try:
        resp = requests.get(_MAVEN_BASE + "ck/", timeout=15)
        resp.raise_for_status()
        listing = resp.text
    except Exception as e:
        print(f"Warning: could not fetch MAVEN CK listing ({e}); "
              f"CK files will not be included.")
        return []

    candidates = []
    seen_files = set()

    for m in _MAVEN_CK_DATE_RE.finditer(listing):
        str_type, start_str, end_str = m.group(1), m.group(2), m.group(3)
        if str_type != structure:
            continue
        fname = m.group(0)
        
        # Guard 1: Drop duplicate regex evaluations from the HTML line string
        if fname in seen_files:
            continue
        seen_files.add(fname)

        # Guard 2: Exclude archived files by checking the local HTML line prefix context
        match_start = m.start()
        context = listing[max(0, match_start-50):match_start]
        if "archived/" in context:
            continue

        cov_start = _parse_maven_ck_date(start_str)
        cov_end = _parse_maven_ck_date(end_str)
        if cov_start is None or cov_end is None:
            continue
        if _kc._window_contains(req_lo, req_hi, cov_start, cov_end):
            candidates.append((cov_start, fname))

    # sort by coverage start date, oldest first
    candidates.sort(key=lambda x: x[0].jd)
    return [fname for _, fname in candidates]


def get_kernel_urls(time=None, time_range=None, include_ck=True):
    """
    Resolve all MAVEN kernel URLs for a given time or time_range.

    Returns dict[filename -> URL].
    """
    urls = {}

    # common kernels
    for fname, subdir in select_common_kernels(time=time, time_range=time_range):
        urls[fname] = _kc._NAIF_BASE + _kc._NAIF_SUBDIRS[subdir] + fname

    # MAVEN static: FK + structure SPK
    for fname, base_url in MAVEN_STATIC_KERNELS:
        urls[fname] = base_url + fname

    # ancillary Mars/planetary SPK
    for fname, base_url in MAVEN_ANCILLARY_SPK:
        urls[fname] = base_url + fname

    # latest SCLK
    sclk_name = resolve_maven_sclk()
    urls[sclk_name] = _MAVEN_BASE + "sclk/" + sclk_name

    # reconstructed orbit SPK (rolling file, always current)
    orb_fname, orb_url = MAVEN_RECONSTRUCTED_SPK
    urls[orb_fname] = orb_url + orb_fname

    # CK files
    if include_ck and (time is not None or time_range is not None):
        for ck_fname in resolve_maven_ck(time=time, time_range=time_range, structure="sc"):
            urls[ck_fname] = _MAVEN_BASE + "ck/" + ck_fname

    return urls


# Backward-compatible alias for the pre-refactor name.
get_maven_kernel_urls = get_kernel_urls
