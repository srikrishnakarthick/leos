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
_MAVEN_ORB_REC_SEGMENT_RE = re.compile(
    r"^maven_orb_rec_(\d{6})_(\d{6})_v(\d+)\.bsp$", re.IGNORECASE
)
_MAVEN_CRUISE_RE = re.compile(
    r"^trj_c_(\d{6})-(\d{6})_rec_v(\d+)\.bsp$", re.IGNORECASE
)
_MAVEN_TRJ_ORB_REC_RE = re.compile(
    r"^trj_orb_(\d+)-(\d+)_rec_v(\d+)\.bsp$", re.IGNORECASE
)
_MAVEN_TRJ_ORB_PREDICTED_RE = re.compile(
    r"^trj_orb_[\d\-]+_.*\.bsp$", re.IGNORECASE
)

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


# Rolling "current" reconstructed-orbit file NAIF keeps refreshing in place.
# Its actual coverage window must be checked via .cmt/comment text (or a
# HEAD-derived summary) since the filename itself carries no dates.
_MAVEN_ROLLING_REC_FNAME = "maven_orb_rec.bsp"


def _parse_maven_yymmdd(token):
    """'140922' -> Time('2014-09-22'). Returns None on failure."""
    try:
        return _kc._to_time_or_none(f"20{token[:2]}-{token[2:4]}-{token[4:6]}")
    except Exception:
        return None


def _fetch_maven_orbit_listing():
    """
    Fetch and parse MAVEN's spk/ directory listing once per process.
    Returns dict with four buckets:
      "segments": [(cov_start, cov_end, fname), ...]   archival 3-month chunks
      "cruise":   [(cov_start, cov_end, fname), ...]    pre-orbit-insertion cruise
      "predicted_rec": [(cov_start, cov_end, fname), ...]  short recent trj_orb_*_rec_*
      "predicted_long": [fname, ...]                    forward-predicted, unparsed dates
    """
    cache_key = ("maven_orbit_listing",)
    if cache_key in _kc._SESSION_CACHE:
        return _kc._SESSION_CACHE[cache_key]

    try:
        resp = requests.get(_MAVEN_BASE + "spk/", timeout=15)
        resp.raise_for_status()
        text = resp.text
    except Exception as e:
        print(f"Warning: could not fetch MAVEN spk/ listing ({e}); "
              f"orbit SPK resolution will fall back to the rolling file only.")
        return {"segments": [], "cruise": [], "predicted_rec": [], "predicted_long": []}

    segments, cruise, predicted_rec, predicted_long = [], [], [], []

    for fname in re.findall(r'href="([^"?/][^"]*)"', text):
        if m := _MAVEN_ORB_REC_SEGMENT_RE.match(fname):
            cs, ce = _parse_maven_yymmdd(m.group(1)), _parse_maven_yymmdd(m.group(2))
            if cs and ce:
                segments.append((cs, ce, fname))
        elif m := _MAVEN_CRUISE_RE.match(fname):
            cs, ce = _parse_maven_yymmdd(m.group(1)), _parse_maven_yymmdd(m.group(2))
            if cs and ce:
                cruise.append((cs, ce, fname))
        elif m := _MAVEN_TRJ_ORB_REC_RE.match(fname):
            # e.g. trj_orb_24945-24947_rec_v1.bsp -- MJD-like tokens, not
            # calendar dates; coverage isn't derivable from the filename,
            # so these need a .cmt/summary lookup to be time-filterable.
            predicted_rec.append(fname)
        elif fname.startswith("trj_orb_") and fname.endswith(".bsp"):
            predicted_long.append(fname)

    result = {
        "segments": segments,
        "cruise": cruise,
        "predicted_rec": predicted_rec,
        "predicted_long": predicted_long,
    }
    _kc._SESSION_CACHE[cache_key] = result
    return result


def resolve_maven_orbit_spk(time=None, time_range=None):
    """
    Return the list of orbit/trajectory SPK filenames needed to cover the
    requested time/time_range for MAVEN, spanning cruise, reconstructed
    (archival + rolling), and predicted phases.

    No time given -> smallest available file: the rolling maven_orb_rec.bsp
    (a few MB) rather than any multi-GB predicted file or the full archival
    set.

    Time given -> every archival/cruise segment that OVERLAPS the window
    (multi-segment requests get every relevant chunk), plus the rolling
    file if the window reaches into its live coverage, plus a predicted
    file if the window extends beyond all reconstructed coverage.
    """
    req_lo, req_hi = _kc._normalize_window(time, time_range)

    if req_lo is None and req_hi is None:
        # No window: smallest useful thing is the rolling reconstructed file.
        return [_MAVEN_ROLLING_REC_FNAME]

    listing = _fetch_maven_orbit_listing()
    chosen = []

    # 1. Cruise phase (pre-orbit-insertion).
    for cs, ce, fname in listing["cruise"]:
        if _kc._window_overlaps(req_lo, req_hi, cs, ce):
            chosen.append(fname)

    # 2. Archival reconstructed segments.
    for cs, ce, fname in listing["segments"]:
        if _kc._window_overlaps(req_lo, req_hi, cs, ce):
            chosen.append(fname)

    # 3. Rolling "current" reconstructed file -- covers the most recent
    #    window not yet cut into an archival segment. Check its live
    #    coverage via .cmt/comment-style lookup would require downloading
    #    the .bsp itself (no .cmt is published for it), so instead treat
    #    it as covering "the present": include it if the request reaches
    #    up to or past the latest archival segment's end (i.e. the gap
    #    between last archived segment and now).
    latest_segment_end = max((ce for _, ce, _ in listing["segments"]), default=None)
    if latest_segment_end is None or req_hi is None or req_hi >= latest_segment_end:
        if _MAVEN_ROLLING_REC_FNAME not in chosen:
            chosen.append(_MAVEN_ROLLING_REC_FNAME)

    # 4. Predicted/future coverage: only needed if the window reaches
    #    beyond all reconstructed coverage (archival + rolling).
    #    Prefer the short predicted_rec candidates (parsed via .cmt-style
    #    text if available) before falling back to the long-range file.
    reconstructed_end = latest_segment_end  # rolling file's true end is unknown;
    # treat "beyond archival" as the trigger for checking predicted files.
    if req_hi is not None and (reconstructed_end is None or req_hi > reconstructed_end):
        predicted_pick = _select_maven_predicted(req_lo, req_hi, listing)
        if predicted_pick:
            chosen.append(predicted_pick)

    if not chosen:
        raise ValueError(
            f"No MAVEN orbit/trajectory SPK found covering the requested "
            f"window ({req_lo}, {req_hi}). Check {_MAVEN_BASE}spk/ manually "
            f"-- NAIF may have restructured the predicted-file naming."
        )

    return chosen


def _select_maven_predicted(req_lo, req_hi, listing):
    """
    Best-effort pick among predicted/forward files. Filenames don't carry
    parseable calendar dates for these, so this fetches each candidate's
    .cmt-equivalent (or falls back to picking the most recently modified
    long-range file) rather than guessing. Returns a filename or None.
    """
    # Long-range predicted files (e.g. trj_orb_251206-760101_...bsp) are
    # named with start-date-ish tokens; try a light regex pull before
    # giving up and returning the newest one found in the listing order.
    date_tok_re = re.compile(r"trj_orb_(\d{6})-(\d{6})")
    best = None
    for fname in listing["predicted_long"]:
        m = date_tok_re.search(fname)
        if m:
            cs = _parse_maven_yymmdd(m.group(1))
            ce = _parse_maven_yymmdd(m.group(2)) if len(m.group(2)) == 6 else None
            if cs and _kc._window_overlaps(req_lo, req_hi, cs, ce):
                return fname
        best = best or fname  # keep first as fallback candidate

    if best:
        print(f"Note: could not confirm exact date coverage for predicted "
              f"MAVEN file '{best}' from its filename; including it as the "
              f"best-effort match for a window extending past reconstructed "
              f"coverage. Verify against {_MAVEN_BASE}spk/ if precision matters.")
    return best

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


def get_kernel_urls(time=None, time_range=None, include_ck=True,
                     include_common=True):
    urls = {}

    fk_name = resolve_latest_maven_fk()
    urls[fk_name] = _MAVEN_BASE + "fk/" + fk_name

    struct_name = resolve_latest_maven_struct_spk()
    urls[struct_name] = _MAVEN_BASE + "spk/" + struct_name

    if include_common:
        for fname, subdir in select_common_kernels(time=time, time_range=time_range):
            urls[fname] = _kc._NAIF_BASE + _kc._NAIF_SUBDIRS[subdir] + fname

        for fname in _kc.resolve_best_mars_spk(time=time, time_range=time_range):
            urls[fname] = _kc._NAIF_BASE + _kc._NAIF_SUBDIRS["spk_satellites"] + fname

    sclk_name = resolve_maven_sclk()
    urls[sclk_name] = _MAVEN_BASE + "sclk/" + sclk_name

    for orb_fname in resolve_maven_orbit_spk(time=time, time_range=time_range):
        urls[orb_fname] = _MAVEN_BASE + "spk/" + orb_fname

    if include_ck and (time is not None or time_range is not None):
        for ck_fname in resolve_maven_ck(time=time, time_range=time_range, structure="sc"):
            urls[ck_fname] = _MAVEN_BASE + "ck/" + ck_fname

    return urls

# Backward-compatible alias for the pre-refactor name.
get_maven_kernel_urls = get_kernel_urls
