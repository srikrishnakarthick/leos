"""
leos/kernels/_kernel_common.py

Shared, low-level infrastructure used by fetch_generic_kernels.py and by
the per-mission resolvers under kernels/missions/.

This module exists so that fetch_generic_kernels.py and kernels/missions/*
can both build NAIF URLs, filter candidate kernels by time window, and log
citations WITHOUT importing from each other or from fetch_kernels.py --
that would either create a circular import (fetch_kernels.py imports the
generic + mission resolvers) or force every mission module to duplicate
this code.

Nothing here is part of the public `leos.kernels` API on its own; it's
imported by fetch_generic_kernels.py, the mission modules, and
fetch_kernels.py.
"""

import os
import hashlib
import requests
from astropy.time import Time

# ── Directory Architecture ───────────────────────────────────────────────────
KERNEL_ROOT = os.path.join(os.path.dirname(__file__), "data")
_DEFAULT_KERNEL_ROOT = KERNEL_ROOT
_CMT_CACHE_DIR = os.path.join(KERNEL_ROOT, "_cmt_cache")


# ── NAIF Subdirectory Map ────────────────────────────────────────────────────
_NAIF_BASE = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/"
_NAIF_SUBDIRS = {
    "lsk": "lsk/",
    "pck": "pck/",
    "fk_planets": "fk/planets/",
    "fk_satellites": "fk/satellites/",
    "fk_stations": "fk/stations/",
    "spk_planets": "spk/planets/",
    "spk_satellites": "spk/satellites/",
    "spk_asteroids": "spk/asteroids/",
    "spk_comets": "spk/comets/",
    "spk_lagrange_point": "spk/lagrange_point/",
    "spk_stations": "spk/stations/",
    "spk_tno": "spk/tno/",
}
_CHECKSUM_MANIFEST_URL = {
    subdir: _NAIF_BASE + path + "aa_checksums.txt"
    for subdir, path in _NAIF_SUBDIRS.items()
}

# ── Subdirs that actually publish aa_checksums.txt ───────────────────────────
_SUBDIRS_WITH_CHECKSUMS = {
    "spk_satellites",
    "spk_planets",
    "spk_lagrange_point",
    "spk_asteroids",
}


# ── Time Window Helpers ──────────────────────────────────────────────────────

def _to_time_or_none(value):
    if value is None:
        return None
    if isinstance(value, Time):
        return value
    return Time(value)


def _normalize_window(time=None, time_range=None):
    """Returns (lo, hi) as astropy Time or None, or (None, None) if no filtering requested."""
    if time is None and time_range is None:
        return None, None
    if time_range is not None:
        lo, hi = time_range
        return _to_time_or_none(lo), _to_time_or_none(hi)
    t = _to_time_or_none(time)
    return t, t


def _window_contains(req_lo, req_hi, cov_start, cov_end):
    cov_start_t = _to_time_or_none(cov_start)
    cov_end_t = _to_time_or_none(cov_end)
    if cov_start_t is not None and req_lo is not None and req_lo < cov_start_t:
        return False
    if cov_end_t is not None and req_hi is not None and req_hi > cov_end_t:
        return False
    return True


# ── Version-extraction helper (used by _select_time_filtered_kernels) ────────

def _coverage_width_days(cov_start, cov_end):
    """
    Return the coverage span in days as a float, or infinity when either
    bound is None (unbounded kernels are treated as maximally wide so
    bounded/tighter files are preferred when available).
    """
    s = _to_time_or_none(cov_start)
    e = _to_time_or_none(cov_end)
    if s is None or e is None:
        return float("inf")
    return (e - s).jd  # astropy TimeDelta → Julian days


def _select_time_filtered_kernels(entries, time=None, time_range=None, context_label=""):
    """
    entries: iterable of (filename, subdir, cov_start, cov_end).

    Behaviour
    ---------
    * Entries with cov_start=cov_end=None are always included (no time
      preference possible, e.g. leap-second or PCK kernels).
    * Entries that carry a real time window are first filtered to those that
      *contain* the requested time/time_range.
    * Among the surviving bounded entries the function keeps only the
      **single best** candidate, chosen by:
        - Tightest coverage width (the file with the smallest date span wins).
      This prevents e.g. mar099.bsp (1600–2600) from being downloaded when
      mar099s.bsp (1995–2050) already satisfies a modern request.
    * Raises if at least one bounded entry exists but none of them match a
      specified request – unbounded entries matching doesn't count.
    """
    req_lo, req_hi = _normalize_window(time, time_range)

    unbounded = []
    bounded_candidates = []   # entries that carry a real window

    for fname, subdir, cov_start, cov_end in entries:
        if cov_start is None and cov_end is None:
            unbounded.append((fname, subdir))
        else:
            bounded_candidates.append((fname, subdir, cov_start, cov_end))

    if not bounded_candidates:
        # Nothing time-gated; return everything as-is.
        return unbounded

    # ── Filter by time coverage ──────────────────────────────────────────────
    if req_lo is None and req_hi is None:
        # No time preference supplied → include *all* bounded entries
        # (caller did not ask us to narrow anything down).
        matching = [(f, s) for f, s, cs, ce in bounded_candidates]
    else:
        matching = [
            (f, s)
            for f, s, cs, ce in bounded_candidates
            if _window_contains(req_lo, req_hi, cs, ce)
        ]

    if not matching and (req_lo is not None or req_hi is not None):
        raise ValueError(
            f"No registered kernel{' for ' + context_label if context_label else ''} "
            f"covers the requested time window ({req_lo}, {req_hi}). "
            f"Check the kernel registry or widen the request."
        )

    # ── Keep only the best candidate ─────────────────────────────────────────
    # Build a lookup so we can retrieve cov_start/end for ranking.
    cov_map = {f: (cs, ce) for f, _, cs, ce in bounded_candidates}

    def rank(fname_subdir):
        fname = fname_subdir[0]
        cs, ce = cov_map[fname]
        return _coverage_width_days(cs, ce)

    best = min(matching, key=rank)
    return unbounded + [best]


# ── Citation Tracking ────────────────────────────────────────────────────────
CITATION_LOG = []  # list of dicts: {filename, url, context}

_SPICE_CITATION = (
    "Acton, C.H. (1996). Ancillary Data Services of NASA's Navigation and "
    "Ancillary Information Facility. Planetary and Space Science, 44(1), 65-70. "
    "SPICE Toolkit: https://naif.jpl.nasa.gov/naif/"
)
_SPICEYPY_CITATION = (
    "Annex, A.M., Pearson, B., Seignovert, B., et al. (2020). SpiceyPy: a "
    "Pythonic Wrapper for the SPICE Toolkit. Journal of Open Source Software, "
    "5(46), 2050. https://doi.org/10.21105/joss.02050"
)


def get_citations():
    """Returns the running list of kernel + toolkit citations accumulated this session."""
    return {
        "kernels": list(CITATION_LOG),
        "toolkit": [_SPICE_CITATION, _SPICEYPY_CITATION],
    }


def reset_citations():
    """Clears the citation log. Call at the start of a fresh analysis run if desired."""
    CITATION_LOG.clear()


def _log_citation(filename, url, context):
    CITATION_LOG.append({"filename": filename, "url": url, "context": context})


# ── Checksum Utilities ───────────────────────────────────────────────────────

def fetch_remote_md5s(subdir="spk_satellites"):
    """
    Pass the subdir key matching the kernel category you're about to
    download (e.g. "spk_satellites" for moon kernels, "spk_planets" for
    de442.bsp, "pck"/"lsk" for those).
    """
    checksum_url = _CHECKSUM_MANIFEST_URL[subdir]
    md5_dict = {}
    try:
        response = requests.get(checksum_url, timeout=10)
        response.raise_for_status()
        for line in response.text.splitlines():
            parts = line.split()
            if len(parts) >= 2:
                hash_val, filename = parts[0], parts[1]
                md5_dict[filename.lower()] = hash_val.lower()
    except Exception as e:
        print(f"Warning: Could not fetch remote aa_checksums.txt for "
              f"'{subdir}' ({e}).")
    return md5_dict


def calculate_local_md5(filepath):
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


# ── Filename → Subdirectory Inference ────────────────────────────────────────

def _infer_subdir(filename):
    fname = filename.lower()
    if fname.endswith(".tls"):
        return "lsk"
    if fname.endswith(".tpc") or fname.endswith(".bpc"):
        return "pck"
    if fname.endswith(".tf"):
        raise ValueError(
            f"Cannot infer NAIF subdirectory for frame kernel '{filename}' "
            f"(could be fk/planets, fk/satellites, or fk/stations). "
            f"Pass an explicit URL via extra_urls instead."
        )
    if fname.endswith(".bsp"):
        if fname.startswith("de"):
            return "spk_planets"
        if fname.startswith(("l1_", "l2_", "l4_", "l5_")):
            return "spk_lagrange_point"
        if fname.startswith("codes_"):
            return "spk_asteroids"
        if fname.startswith(("c_g_", "ison", "c2013", "siding_spring")):
            return "spk_comets"
        if fname.startswith("tnosat"):
            return "spk_tno"
        if fname.startswith(("dss_", "earthstns", "ndosl")):
            return "spk_stations"
        return "spk_satellites"
    raise ValueError(
        f"Cannot infer NAIF subdirectory for '{filename}'. "
        f"Pass an explicit URL via extra_urls instead."
    )


def _subdir_for(filename):
    """Best-effort guess of which checksum manifest a filename belongs to,
    used by fetch_kernels() when downloading a heterogeneous queue."""
    try:
        return _infer_subdir(filename)
    except ValueError:
        return "spk_satellites"
