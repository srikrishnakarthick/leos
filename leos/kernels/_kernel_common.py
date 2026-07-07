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
import re
from astropy.time import Time

# ── Session-Scoped Cache (in-memory only; never touches disk, never
#    persists past process exit) ──────────────────────────────────────────
# Directory listings and coverage summaries change rarely enough that
# hitting NAIF once per URL per run is plenty fresh, while still avoiding
# a dozen+ redundant round-trips within a single script (e.g. resolving
# LSK/PCK/DE on every fetch_kernels() call, or scanning giant-planet moon
# candidates repeatedly for the same body lookup).
_SESSION_CACHE = {}


def reset_session_cache():
    """
    Clears the in-memory listing/comment cache. Call this if you want a
    long-running process (a notebook, a service) to re-check NAIF for
    updates without restarting -- normal short-lived scripts don't need
    to call this at all, since the cache dies with the process anyway.
    """
    _SESSION_CACHE.clear()

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

def _window_overlaps(req_lo, req_hi, cov_start, cov_end):
    """
    Unlike _window_contains (does coverage fully contain the request?),
    this answers "does coverage overlap the request at all?" -- the
    correct test for multi-file sets like weekly CKs, where a multi-week
    request should return every CK that overlaps any part of it rather
    than requiring a single file to span the whole thing.
    """
    cov_start_t = _to_time_or_none(cov_start)
    cov_end_t = _to_time_or_none(cov_end)
    if cov_end_t is not None and req_lo is not None and cov_end_t < req_lo:
        return False
    if cov_start_t is not None and req_hi is not None and cov_start_t > req_hi:
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

# ── Directory-Listing Helpers (for dynamic "latest" resolution) ─────────────

_LSK_URL = _NAIF_BASE + _NAIF_SUBDIRS["lsk"]
_PCK_URL = _NAIF_BASE + _NAIF_SUBDIRS["pck"]
_SPK_PLANETS_URL = _NAIF_BASE + _NAIF_SUBDIRS["spk_planets"]

_SIZE_RE = re.compile(r"([\d.]+)\s*([KMGT]?)\s*$")


def _parse_size_token(token):
    """'114M' -> ~114*1024**2 bytes. Returns None if unparseable."""
    m = _SIZE_RE.search(token.strip())
    if not m:
        return None
    num, unit = m.groups()
    mult = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}.get(unit, 1)
    try:
        return float(num) * mult
    except ValueError:
        return None


_LISTING_HREF_ROW_RE = re.compile(
    r'href="([^"?/][^"]*)"[^<]*</a>\s*'
    r'[\d\-]{10}\s+[\d:]{4,5}\s+(\S+)',
)


def _fetch_directory_listing(url, timeout=15):
    """
    Fetch a NAIF Apache-autoindex directory listing and return
    {filename: size_bytes_or_None}. Cached in-memory for the life of the
    process (see _SESSION_CACHE) -- repeated calls in one run reuse the
    first successful fetch; a fresh process always hits NAIF again.
    Returns {} on any failure so callers treat that as "listing
    unavailable" rather than crashing.
    """
    cache_key = ("listing", url)
    if cache_key in _SESSION_CACHE:
        return _SESSION_CACHE[cache_key]

    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        text = resp.text
    except Exception as e:
        print(f"Warning: could not fetch directory listing {url} ({e}).")
        return {}  # failures are NOT cached -- retry on next call

    entries = {}
    for fname, size_tok in _LISTING_HREF_ROW_RE.findall(text):
        if fname.endswith("/"):
            continue
        entries[fname] = _parse_size_token(size_tok)

    if not entries:
        for line in text.splitlines():
            parts = line.split()
            if len(parts) >= 4 and "href" not in line:
                fname = parts[0].rstrip("/")
                if fname.lower() in ("parent_directory", "[parentdir]"):
                    continue
                entries[fname] = _parse_size_token(parts[-1])

    _SESSION_CACHE[cache_key] = entries
    return entries


def _fetch_spk_coverage_summary(subdir_url, filename="aa_summaries.txt"):
    """
    Tolerantly parse NAIF's aa_summaries.txt into
    {filename: (Time_or_None, Time_or_None)}. Cached in-memory per URL
    for the life of the process, same policy as _fetch_directory_listing.
    """
    cache_key = ("summary", subdir_url, filename)
    if cache_key in _SESSION_CACHE:
        return _SESSION_CACHE[cache_key]

    try:
        resp = requests.get(subdir_url + filename, timeout=15)
        resp.raise_for_status()
        text = resp.text
    except Exception as e:
        print(f"Warning: could not fetch {filename} from {subdir_url} ({e}).")
        return {}  # failures are NOT cached

    coverage = {}
    for fname, start_tok, end_tok in _SUMMARY_BLOCK_RE.findall(text):
        start = _best_effort_parse_date(start_tok)
        end = _best_effort_parse_date(end_tok)
        if start or end:
            coverage[fname] = (start, end)

    _SESSION_CACHE[cache_key] = coverage
    return coverage
# ── LSK ───────────────────────────────────────────────────────────────────

_LSK_VERSION_RE = re.compile(r"^naif(\d+)\.tls$", re.IGNORECASE)


def resolve_latest_lsk():
    """
    Highest-numbered naifNNNN.tls in NAIF's lsk/ directory. Deliberately
    excludes 'latest_leapseconds.tls' (an alias, not a stable filename to
    pin to) and '.tls.pc' variants. Falls back to naif0012.tls if the
    listing fetch fails or nothing matches.
    """
    listing = _fetch_directory_listing(_LSK_URL)
    versions = [
        (int(m.group(1)), fname)
        for fname in listing
        if (m := _LSK_VERSION_RE.match(fname))
    ]
    if not versions:
        print("Warning: could not determine latest LSK from listing; "
              "falling back to naif0012.tls.")
        return "naif0012.tls"
    return max(versions)[1]


# ── PCK ───────────────────────────────────────────────────────────────────

_PCK_VERSION_RE = re.compile(r"^pck(\d+)\.tpc$", re.IGNORECASE)


def resolve_latest_pck():
    """
    Highest-numbered generic orientation/radii PCK (pckNNNNN.tpc) in
    NAIF's pck/ directory. Excludes suffixed re-releases of the same
    version (e.g. 'pck00011_n0066.tpc') and unrelated files sharing the
    directory (Gravity.tpc, gm_de440.tpc, mars_iau2000_v1.tpc,
    moon_pa_*.bpc, earth_*.bpc, etc.) via the strict '^pckNNNNN.tpc$'
    match. Falls back to pck00011.tpc if the listing fetch fails.
    """
    listing = _fetch_directory_listing(_PCK_URL)
    versions = [
        (int(m.group(1)), fname)
        for fname in listing
        if (m := _PCK_VERSION_RE.match(fname))
    ]
    if not versions:
        print("Warning: could not determine latest PCK from listing; "
              "falling back to pck00011.tpc.")
        return "pck00011.tpc"
    return max(versions)[1]


# ── Planetary SPK (DE-series) ────────────────────────────────────────────

_DE_VERSION_RE = re.compile(r"^de(\d+)(s)?\.bsp$", re.IGNORECASE)

_SUMMARY_BLOCK_RE = re.compile(
    r'([a-zA-Z0-9_.\-]+\.bsp)\s.*?'
    r'(?:Start|Begin)[^\n]*?:\s*([\d\-A-Za-z:. ]+?)\s*\n.*?'
    r'(?:Stop|End)[^\n]*?:\s*([\d\-A-Za-z:. ]+?)\s*\n',
    re.IGNORECASE | re.DOTALL,
)


def _best_effort_parse_date(token):
    """Try astropy's Time parser on a raw token; returns None (never
    raises) on failure, since NAIF summary date formatting isn't
    guaranteed stable across releases."""
    try:
        return Time(token.strip())
    except Exception:
        return None


_DE_VERSION_RE = re.compile(r"^de(\d+)(s)?\.bsp$", re.IGNORECASE)
_DE_PART_RE = re.compile(r"^de(\d+)_part-(\d+)\.bsp$", re.IGNORECASE)


def _classify_de_version(fnames):
    """
    Given all filenames on NAIF's spk/planets/ listing belonging to one DE
    version number, classify them as:
      ("single", fname)
      ("short_full", short_fname, full_fname)   -- either may be None
      ("parts", [fname, fname, ...])            -- sorted by part number
    """
    parts = sorted(
        (int(_DE_PART_RE.match(f).group(2)), f)
        for f in fnames if _DE_PART_RE.match(f)
    )
    if parts:
        return ("parts", [f for _, f in parts])

    short = next((f for f in fnames if f.lower().endswith("s.bsp")), None)
    full = next((f for f in fnames if f.lower().endswith(".bsp") and f != short), None)
    if short and full:
        return ("short_full", short, full)
    return ("single", full or short)


def resolve_best_planetary_spk(time=None, time_range=None):
    """
    Returns a LIST of one or more filenames from NAIF's spk/planets/
    needed to cover the request, for the highest-available DE version.

    Behaviour per DE-version file layout:
      - single file (e.g. de440.bsp): return [that file].
- short+full (e.g. de442s.bsp/de442.bsp): return the short file if
        its coverage contains the request, else the full file. With no
        time window given, prefer the short file (smaller download).
      - multi-part (e.g. de441_part-1.bsp/part-2.bsp): return every part
        whose coverage overlaps the requested window. With no time
        window given, return only the most recent part (smallest
        reasonable default rather than downloading the whole multi-GB
        set with no stated need).

    Falls back to ["de442.bsp"] if the listing fetch fails entirely.
    """
    listing = _fetch_directory_listing(_SPK_PLANETS_URL)
    by_version = {}
    for fname in listing:
        m = _DE_VERSION_RE.match(fname) or _DE_PART_RE.match(fname)
        if not m:
continue
        num = int(m.group(1))
        by_version.setdefault(num, []).append(fname)

    if not by_version:
        print("Warning: could not determine latest planetary SPK from "
              "listing; falling back to de442.bsp.")
        return ["de442.bsp"]

    best_version = max(by_version)
    kind, *rest = _classify_de_version(by_version[best_version])
    req_lo, req_hi = _normalize_window(time, time_range)
    coverage = _fetch_spk_coverage_summary(_SPK_PLANETS_URL)

    if kind == "single":
        fname = rest[0]
        _warn_if_uncovered(fname, time, time_range, coverage)
       return [fname]

    if kind == "short_full":
        short_name, full_name = rest
        if req_lo is None and req_hi is None:
            return [short_name] if short_name else [full_name]
        if short_name:
            short_cov = coverage.get(short_name)
            if short_cov and _window_contains(req_lo, req_hi, *short_cov):
                return [short_name]
        _warn_if_uncovered(full_name, time, time_range, coverage)
        return [full_name]

    # kind == "parts"
    part_files = rest[0]
    if req_lo is None and req_hi is None:
        # No window given -- default to the most recent part only, since
        # "shortest" for a multi-part deep-time ephemeris means "the part
        # that covers the modern era", not "fewest bytes across all parts".
        chosen = part_files[-1]
        print(
            f"No time window given for de{best_version} (multi-part "
            f"ephemeris); defaulting to '{chosen}' (most recent part) "
            f"rather than downloading all {len(part_files)} parts. Pass "
            f"time= or time_range= to select parts by actual coverage."
        )
        return [chosen]

    overlapping = [
        f for f in part_files
        if (cov := coverage.get(f)) and _window_overlaps(req_lo, req_hi, *cov)
    ]
    if not overlapping:
        print(
            f"Warning: could not verify which de{best_version} part(s) "
            f"cover the requested window ({req_lo}, {req_hi}) -- "
            f"aa_summaries.txt didn't yield parseable entries. Falling "
            f"back to all {len(part_files)} parts to be safe."
        )
        return list(part_files)
    return overlapping

# ── Mars Satellite SPK (marNNN-series) ───────────────────────────────────

_MAR_VERSION_RE = re.compile(r"^mar(\d+)(s)?\.bsp$", re.IGNORECASE)
_SPK_SATELLITES_URL = _NAIF_BASE + _NAIF_SUBDIRS["spk_satellites"]


def resolve_best_mars_spk(time=None, time_range=None):
    """
    Same short/full preference logic as resolve_best_planetary_spk, but for
    NAIF's marNNN(s).bsp series (spk/satellites/, not spk/planets/):
    marNNNs.bsp = tighter modern-coverage file, marNNN.bsp = long-coverage
    fallback. Falls back to mar099s.bsp / mar099.bsp if the listing fetch
    fails or nothing matches -- the same pair currently hardcoded in
    BODY_KERNELS.
    """
    listing = _fetch_directory_listing(_SPK_SATELLITES_URL)
    versions = {}
    for fname in listing:
        m = _MAR_VERSION_RE.match(fname)
        if not m:
            continue
        num = int(m.group(1))
        kind = "short" if m.group(2) else "full"
        versions.setdefault(num, {})[kind] = fname

    if not versions:
        print("Warning: could not determine latest Mars satellite SPK from "
              "listing; falling back to mar099s.bsp/mar099.bsp.")
        return [
            ("mar099s.bsp", "spk_satellites", "1995-01-01", "2050-01-01"),
            ("mar099.bsp", "spk_satellites", "1600-01-01", "2600-01-02"),
        ]

    candidates = versions[max(versions)]
    full_name, short_name = candidates.get("full"), candidates.get("short")

    coverage = _fetch_spk_coverage_summary(_SPK_SATELLITES_URL)

    def _cov_or_none(fname):
        return coverage.get(fname)

    if not short_name:
        cov = _cov_or_none(full_name)
        return [(full_name, "spk_satellites", cov[0] if cov else None, cov[1] if cov else None)]
    if not full_name:
        cov = _cov_or_none(short_name)
        return [(short_name, "spk_satellites", cov[0] if cov else None, cov[1] if cov else None)]

    req_lo, req_hi = _normalize_window(time, time_range)
    short_cov = _cov_or_none(short_name)
    full_cov = _cov_or_none(full_name)

    entries = []
    if short_cov:
        entries.append((short_name, "spk_satellites", short_cov[0], short_cov[1]))
    else:
        entries.append((short_name, "spk_satellites", None, None))
    if full_cov:
        entries.append((full_name, "spk_satellites", full_cov[0], full_cov[1]))
    else:
        entries.append((full_name, "spk_satellites", None, None))

    return entries


# ── Lagrange Point / Planetary DE-version consistency ───────────────────

_LAGRANGE_VERSION_RE = re.compile(r"^L([12345])_de(\d+)\.bsp$", re.IGNORECASE)
_ANY_DE_FILENAME_RE = re.compile(r"^de(\d+)(s)?(?:_part-(\d))?\.bsp$", re.IGNORECASE)
_LAGRANGE_POINT_URL = _NAIF_BASE + _NAIF_SUBDIRS["spk_lagrange_point"]


def _de_version_from_filename(fname):
    """Extract the DE version number from any de-series filename
    (de442.bsp, de442s.bsp, de441_part-1.bsp, ...). Returns None if the
    filename doesn't match the expected pattern."""
    m = _ANY_DE_FILENAME_RE.match(fname)
    return int(m.group(1)) if m else None


def resolve_matching_lagrange_kernel(point, hardcoded_entry, planetary_de_version):
    """
    Resolve the Lagrange point file for `point` (e.g. "L1") whose DE
    version best matches `planetary_de_version` -- the DE version
    resolve_best_planetary_spk() picked for this session.

    hardcoded_entry: the existing (filename, subdir, cov_start, cov_end)
    tuple from LAGRANGE_KERNELS, used as the fallback if no matching
    version is available on NAIF.

    Returns (filename, subdir, cov_start, cov_end), and always prints a
    message when the returned file's DE version differs from
    planetary_de_version, so a session-wide mismatch is never silent.

    Plan B (no exact-version match found on NAIF, in either direction):
    fall back to hardcoded_entry and warn. This never raises -- a
    Lagrange/DE mismatch is a precision concern, not a hard failure, and
    the caller should still get a usable kernel.
    """
    hc_fname, hc_subdir, hc_start, hc_end = hardcoded_entry
    m = _LAGRANGE_VERSION_RE.match(hc_fname)
    if not m:
        # Unexpected filename shape; can't reason about versions at all.
        return hardcoded_entry
    hc_version = int(m.group(2))

    if hc_version == planetary_de_version:
        return hardcoded_entry  # No issue -- versions already match.

    # Versions differ -- see if NAIF has a Lagrange file for the DE
    # version the planetary resolver actually picked.
    listing = _fetch_directory_listing(_LAGRANGE_POINT_URL)
    target_fname = f"{point}_de{planetary_de_version}.bsp"

    if target_fname in listing:
        coverage = _fetch_spk_coverage_summary(_LAGRANGE_POINT_URL)
        cov = coverage.get(target_fname, (None, None))
        direction = "older" if hc_version < planetary_de_version else "newer"
        print(
            f"Note: switching {point} Lagrange kernel from '{hc_fname}' "
            f"(de{hc_version}, {direction} than the resolved planetary "
            f"SPK) to '{target_fname}' to match the session's planetary "
            f"DE version (de{planetary_de_version})."
        )
        return (target_fname, "spk_lagrange_point", cov[0], cov[1])

    # Plan B: no matching version published for this Lagrange point.
    # Fall back to the hardcoded file and make the mismatch loud.
    direction = (
        "older than" if hc_version < planetary_de_version else "newer than"
    )
    print(
        f"Warning: '{target_fname}' is not available on NAIF's "
        f"lagrange_point/ directory. Falling back to the hardcoded "
        f"'{hc_fname}', which is {direction} the resolved planetary SPK "
        f"(de{planetary_de_version}). Positions for {point} will be "
        f"computed from a different DE solution than the rest of this "
        f"session's planetary ephemeris -- check "
        f"{_LAGRANGE_POINT_URL}aa_summaries.txt if precision at this "
        f"level matters for your use case."
    )
    return hardcoded_entry


def _warn_if_uncovered(fname, time, time_range, coverage=None):
    req_lo, req_hi = _normalize_window(time, time_range)
    if req_lo is None and req_hi is None:
        return
    if coverage is None:
        coverage = _fetch_spk_coverage_summary(_SPK_PLANETS_URL)
    cov = coverage.get(fname)
    if cov is None:
        print(f"Warning: could not verify '{fname}' covers the requested "
              f"time window ({req_lo}, {req_hi}) -- aa_summaries.txt "
              f"didn't yield a parseable entry for it. Proceeding, but "
              f"double-check coverage manually if precision matters.")
        return
    if not _window_contains(req_lo, req_hi, *cov):
        print(f"Warning: '{fname}' coverage ({cov[0]}, {cov[1]}) may not "
              f"fully contain the requested window ({req_lo}, {req_hi}). "
              f"This is NAIF's highest-version planetary SPK available, "
              f"but your request may fall outside its validated range.")
