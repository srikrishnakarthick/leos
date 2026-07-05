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

_MAVEN_BASE = "https://naif.jpl.nasa.gov/pub/naif/MAVEN/kernels/"

# ── Dynamic resolvers replacing hardcoded MAVEN_STATIC_KERNELS ─────────────
_MAVEN_FK_RE = re.compile(r"^maven_v(\d+)\.tf$", re.IGNORECASE)
_MAVEN_STRUCT_SPK_RE = re.compile(r"^maven_struct_v(\d+)\.bsp$", re.IGNORECASE)


def resolve_latest_maven_fk():
    """Highest-numbered maven_vNN.tf in MAVEN's fk/ directory."""
    listing = _kc._fetch_directory_listing(_MAVEN_BASE + "fk/")
    versions = [
        (int(m.group(1)), fname)
        for fname in listing
        if (m := _MAVEN_FK_RE.match(fname))
    ]
    if not versions:
        print("Warning: could not determine latest MAVEN FK from listing; "
              "falling back to maven_v12.tf.")
        return "maven_v12.tf"
    return max(versions)[1]


def resolve_latest_maven_struct_spk():
    """Highest-numbered maven_struct_vNN.bsp in MAVEN's spk/ directory."""
    listing = _kc._fetch_directory_listing(_MAVEN_BASE + "spk/")
    versions = [
        (int(m.group(1)), fname)
        for fname in listing
        if (m := _MAVEN_STRUCT_SPK_RE.match(fname))
    ]
    if not versions:
        print("Warning: could not determine latest MAVEN structure SPK from "
              "listing; falling back to maven_struct_v12.bsp.")
        return "maven_struct_v12.bsp"
    return max(versions)[1]


# MAVEN_STATIC_KERNELS list is now DELETED -- replaced by the two resolvers above.

MAVEN_RECONSTRUCTED_SPK = ("maven_orb_rec.bsp", _MAVEN_BASE + "spk/")
MAVEN_PREDICTED_SPK = ("maven_orb.bsp", _MAVEN_BASE + "spk/")
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

    # Line-based scan instead of a fixed character lookbehind: robust to
    # NAIF changing row spacing/format, since "archived/" only ever
    # appears as part of the href path on the SAME listing line as the
    # filename it applies to.
    for line in listing.splitlines():
        m = _MAVEN_CK_DATE_RE.search(line)
        if not m:
            continue
        str_type, start_str, end_str = m.group(1), m.group(2), m.group(3)
        if str_type != structure:
            continue
        fname = m.group(0)

        if fname in seen_files:
            continue
        seen_files.add(fname)

        if "archived/" in line:
            continue

        cov_start = _parse_maven_ck_date(start_str)
        cov_end = _parse_maven_ck_date(end_str)
        if cov_start is None or cov_end is None:
            continue
        if _kc._window_contains(req_lo, req_hi, cov_start, cov_end):
            candidates.append((cov_start, fname))

    candidates.sort(key=lambda x: x[0].jd)
    return [fname for _, fname in candidates]


def get_kernel_urls(time=None, time_range=None, include_ck=True):
    urls = {}

    fk_name = resolve_latest_maven_fk()
    urls[fk_name] = _MAVEN_BASE + "fk/" + fk_name

    struct_name = resolve_latest_maven_struct_spk()
    urls[struct_name] = _MAVEN_BASE + "spk/" + struct_name

    for fname, base_url in MAVEN_ANCILLARY_SPK:
        urls[fname] = base_url + fname

    sclk_name = resolve_maven_sclk()
    urls[sclk_name] = _MAVEN_BASE + "sclk/" + sclk_name

    orb_fname, orb_url = MAVEN_RECONSTRUCTED_SPK
    urls[orb_fname] = orb_url + orb_fname

    if include_ck and (time is not None or time_range is not None):
        for ck_fname in resolve_maven_ck(time=time, time_range=time_range, structure="sc"):
            urls[ck_fname] = _MAVEN_BASE + "ck/" + ck_fname

    return urls

# Backward-compatible alias for the pre-refactor name.
get_maven_kernel_urls = get_kernel_urls
