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
DATA_DIRS = {
    "generic": os.path.join(KERNEL_ROOT, "generic"),
    "mission": os.path.join(KERNEL_ROOT, "mission"),
}

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
# ── Generic Multi-File Kernel Resolution ─────────────────────────────────
# Any kernel family (LSK, PCK, DE-series, Mars SPK, moon SPK, Lagrange,
# ...) may have more than one file representing "the same logical
# kernel version" -- either because NAIF ships a short/full pair
# (de442s.bsp + de442.bsp), or a numbered-part split for size reasons
# (de441_part-1.bsp + de441_part-2.bsp), or in principle both at once.
# This is the one shared classifier + selector every per-family resolver
# calls, replacing bespoke short/full and part-detection logic that used
# to live separately in the DE-series and Mars-SPK resolvers.

_GENERIC_PART_RE = re.compile(r"^(.*?)[_\-]part-?(\d+)(\..+)$", re.IGNORECASE)
_GENERIC_SHORT_RE = re.compile(r"^(.*)s(\.[a-zA-Z0-9]+)$", re.IGNORECASE)


def _classify_version_group(fnames, listing_sizes=None):
    """
    Given all filenames belonging to ONE logical kernel version (already
    grouped by version number by the caller -- e.g. all files for de442,
    or all files for naif0012), classify the relationship between them.

    Returns a list of "candidate sets", where each candidate set is a
    list of filenames that must be used TOGETHER to get one flavor of
    complete coverage. Typically there are 1-2 candidate sets:
      - one file, no variants:            [[fname]]
      - short/full variants (no parts):   [[short], [full]]
      - numbered parts (no short/full):   [[part1, part2, ...]]
      - parts x short/full (general case, not seen yet but handled):
                                           [[s_part1, s_part2, ...],
                                            [f_part1, f_part2, ...]]

    Candidate sets are ordered smallest-total-size first when sizes are
    known (listing_sizes: {fname: bytes_or_None}), so callers that want
    "smallest among latest" can just take candidate_sets[0].
    """
    parts_by_variant = {}   # variant_key -> {part_num: fname}
    non_part_files = []

    for fname in fnames:
        m = _GENERIC_PART_RE.match(fname)
        if m:
            base, part_num, ext = m.group(1), int(m.group(2)), m.group(3)
            sm = _GENERIC_SHORT_RE.match(base + ext)
            variant_key = "short" if sm else "full"
            parts_by_variant.setdefault(variant_key, {})[part_num] = fname
        else:
            non_part_files.append(fname)

    candidate_sets = []

    if parts_by_variant:
        for variant_key in ("short", "full"):
            if variant_key in parts_by_variant:
                ordered = [
                    parts_by_variant[variant_key][n]
                    for n in sorted(parts_by_variant[variant_key])
                ]
                candidate_sets.append(ordered)
    else:
        short_files = [f for f in non_part_files if _GENERIC_SHORT_RE.match(f)]
        full_files = [f for f in non_part_files if f not in short_files]
        if short_files:
            candidate_sets.append(short_files[:1])
        if full_files:
            candidate_sets.append(full_files[:1])

    if not candidate_sets:
        return []

    if listing_sizes:
        def total_size(cset):
            sizes = [listing_sizes.get(f) for f in cset]
            return sum(s for s in sizes if s is not None) if all(s is not None for s in sizes) else float("inf")
        candidate_sets.sort(key=total_size)

    return candidate_sets


def resolve_versioned_kernel(version_groups, time=None, time_range=None,
                               coverage_url=None, listing_sizes=None,
                               fallback=None, label=""):
    """
    Shared entry point for every "pick the latest version, then pick the
    right file(s) for the request" resolver in this module.

    Parameters
    ----------
    version_groups : dict[version_number, list[filename]]
        All filenames for this kernel family, grouped by version number
        (grouping logic is family-specific and stays in the caller,
        since "what counts as a version" differs per family).
    coverage_url : str, optional
        Base URL to fetch aa_summaries.txt from, for time-window
        filtering. If None, time-window filtering is skipped (used for
        kernel types like LSK/PCK that have no per-file time coverage
        concept at all).
    listing_sizes : dict[filename, bytes_or_None], optional
        Byte sizes from _fetch_directory_listing, used to pick the
        smallest candidate set when no time window is given.
    fallback : list[str], optional
        Returned if version_groups is empty (listing fetch failed).
    label : str
        Used in print() messages only.

    Returns
    -------
    list[str]
        One or more filenames needed to satisfy the request.
    """
    if not version_groups:
        print(f"Warning: could not determine latest {label} from listing; "
              f"falling back to {fallback}.")
        return list(fallback) if fallback else []

    best_version = max(version_groups)
    candidate_sets = _classify_version_group(
        version_groups[best_version], listing_sizes=listing_sizes
    )
    if not candidate_sets:
        print(f"Warning: could not classify {label} version {best_version}'s "
              f"files; falling back to {fallback}.")
        return list(fallback) if fallback else []

    req_lo, req_hi = _normalize_window(time, time_range)

    if req_lo is None and req_hi is None:
        # No window given: pick the single smallest file across ALL
        # candidates, not the smallest candidate SET -- a multi-part set
        # like [de441_part-1.bsp, de441_part-2.bsp] would otherwise
        # return both parts (multi-GB) when the person asked for nothing
        # in particular. Falls back to candidate_sets[0] (smallest SET)
        # only if no size data is available to compare individual files.
        all_files = [f for cset in candidate_sets for f in cset]
        if listing_sizes and all(listing_sizes.get(f) is not None for f in all_files):
            return [min(all_files, key=lambda f: listing_sizes[f])]
        print(
            f"Note: no time window given for {label}; file sizes "
            f"unavailable from the directory listing, so returning the "
            f"smallest known candidate SET ({candidate_sets[0]}) rather "
            f"than the single smallest file. Pass time= or time_range= "
            f"for precise part selection."
        )
        return candidate_sets[0]

    if coverage_url is None:
        # This kernel type has no time-coverage concept (LSK/PCK); just
        # return the smallest candidate set regardless of window.
        return candidate_sets[0]

    coverage = _fetch_spk_coverage_summary(coverage_url)

    # Prefer whichever candidate set fully satisfies the window using
    # the fewest/smallest files; fall back to whichever OVERLAPS it.
    for cset in candidate_sets:
        covs = [coverage.get(f) for f in cset]
        if all(covs) and all(
            _window_contains(req_lo, req_hi, *cov) for cov in covs
        ):
            return cset

    # Nothing fully contains the window -- for multi-part sets, return
    # every part that overlaps at all (a request spanning a part
    # boundary needs both neighbors).
    for cset in candidate_sets:
        overlapping = [
            f for f in cset
            if (cov := coverage.get(f)) and _window_overlaps(req_lo, req_hi, *cov)
        ]
        if overlapping:
            return overlapping

    print(f"Warning: could not verify any {label} candidate covers the "
          f"requested window ({req_lo}, {req_hi}); falling back to the "
          f"largest/most complete candidate set to be safe.")
    return candidate_sets[-1]

# ── LSK ───────────────────────────────────────────────────────────────────

_LSK_VERSION_RE = re.compile(r"^naif(\d+)\.tls$", re.IGNORECASE)


def resolve_latest_lsk(time=None, time_range=None):
    """
    Highest-numbered naifNNNN.tls in NAIF's lsk/ directory, routed
    through the shared resolve_versioned_kernel() so it's handled
    identically to every other kernel family if NAIF ever splits it.
    Returns a list (one filename in the normal case).
    """
    listing = _fetch_directory_listing(_LSK_URL)
    by_version = {}
    for fname in listing:
        m = _LSK_VERSION_RE.match(fname)
        if m:
            by_version.setdefault(int(m.group(1)), []).append(fname)

    return resolve_versioned_kernel(
        by_version, time=time, time_range=time_range,
        coverage_url=None,  # LSK has no per-file time coverage concept
        listing_sizes=listing,
        fallback=["naif0012.tls"], label="LSK",
    )

# ── PCK ───────────────────────────────────────────────────────────────────

_PCK_VERSION_RE = re.compile(r"^pck(\d+)\.tpc$", re.IGNORECASE)


def resolve_latest_pck(time=None, time_range=None):
    """
    Highest-numbered pckNNNNN.tpc in NAIF's pck/ directory, routed
    through resolve_versioned_kernel(). Returns a list.
    """
    listing = _fetch_directory_listing(_PCK_URL)
    by_version = {}
    for fname in listing:
        m = _PCK_VERSION_RE.match(fname)
        if m:
            by_version.setdefault(int(m.group(1)), []).append(fname)

    return resolve_versioned_kernel(
        by_version, time=time, time_range=time_range,
        coverage_url=None,
        listing_sizes=listing,
        fallback=["pck00011.tpc"], label="PCK",
    )

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


_DE_ANY_RE = re.compile(r"^de(\d+)(s)?(?:[_\-]part-?\d+)?\.bsp$", re.IGNORECASE)


def resolve_best_planetary_spk(time=None, time_range=None):
    """
    Highest-available DE version on NAIF's spk/planets/, routed through
    resolve_versioned_kernel(). Handles single-file (de440.bsp),
    short+full (de442s/de442), and multi-part (de441_part-1/part-2)
    layouts uniformly. Returns a list of one or more filenames.
    """
    listing = _fetch_directory_listing(_SPK_PLANETS_URL)
    by_version = {}
    for fname in listing:
        m = _DE_ANY_RE.match(fname)
        if m:
            by_version.setdefault(int(m.group(1)), []).append(fname)

    return resolve_versioned_kernel(
        by_version, time=time, time_range=time_range,
        coverage_url=_SPK_PLANETS_URL,
        listing_sizes=listing,
        fallback=["de442.bsp"], label="planetary SPK",
    )

# ── Mars Satellite SPK (marNNN-series) ───────────────────────────────────
_SPK_SATELLITES_URL = _NAIF_BASE + _NAIF_SUBDIRS["spk_satellites"]
_MAR_ANY_RE = re.compile(r"^mar(\d+)(s)?(?:[_\-]part-?\d+)?\.bsp$", re.IGNORECASE)


def resolve_best_mars_spk(time=None, time_range=None):

    """
    Highest-available marNNN version on NAIF's spk/satellites/, routed
    through resolve_versioned_kernel(). Returns a list of filenames
    (plain strings -- callers needing the (fname, subdir, cov_start,
    cov_end) tuple shape used elsewhere in BODY_KERNELS should look up
    coverage separately via _fetch_spk_coverage_summary if needed).
    """
    listing = _fetch_directory_listing(_SPK_SATELLITES_URL)
    by_version = {}
    for fname in listing:
        m = _MAR_ANY_RE.match(fname)
        if m:
            by_version.setdefault(int(m.group(1)), []).append(fname)

    return resolve_versioned_kernel(
        by_version, time=time, time_range=time_range,
        coverage_url=_SPK_SATELLITES_URL,
        listing_sizes=listing,
        fallback=["mar099s.bsp", "mar099.bsp"], label="Mars satellite SPK",
    )
# ── Lagrange Point / Planetary DE-version consistency ───────────────────

_LAGRANGE_VERSION_RE = re.compile(r"^L([12345])_de(\d+)\.bsp$", re.IGNORECASE)
_ANY_DE_FILENAME_RE = re.compile(r"^de(\d+)(s)?(?:[_\-]part-?\d+)?\.bsp$", re.IGNORECASE)
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
        cov = coverage.get(target_fname)
        if cov is not None:
            direction = "older" if hc_version < planetary_de_version else "newer"
            print(
                f"Note: switching {point} Lagrange kernel from '{hc_fname}' "
                f"(de{hc_version}, {direction} than the resolved planetary "
                f"SPK) to '{target_fname}' to match the session's planetary "
                f"DE version (de{planetary_de_version})."
            )
            return (target_fname, "spk_lagrange_point", cov[0], cov[1])
        print(
            f"Warning: '{target_fname}' exists on NAIF but its coverage "
            f"window could not be parsed from aa_summaries.txt. Falling "
            f"back to the hardcoded '{hc_fname}' rather than trust an "
            f"unbounded window for an unverified file."
        )

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
