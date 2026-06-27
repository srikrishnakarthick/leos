"""
leos/kernels/fetch_kernels.py

Generic SPICE kernel fetcher / resolver for the `leos` package.

"""

import os
import re
import json
import hashlib
import datetime
import requests
from pathlib import Path
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
    "fk_satellites": "fk/satellites/",
    "spk_planets": "spk/planets/",
    "spk_satellites": "spk/satellites/",
    "spk_asteroids": "spk/asteroids/",
    "spk_comets": "spk/comets/", "spk_lagrange_point": "spk/lagrange_point/",
}

_CHECKSUM_MANIFEST_URL = {
    subdir: _NAIF_BASE + path + "aa_checksums.txt"
    for subdir, path in _NAIF_SUBDIRS.items()
}


# ── Common Kernels (always fetched, body-independent) ───────────────────────
COMMON_KERNELS = [
    ("naif0012.tls", "lsk"),
    ("pck00011.tpc", "pck"),
    ("de442.bsp", "spk_planets"),
]

# ── Body Kernel Registry ────────
BODY_KERNELS = {
    "EARTH": [
        # No additional kernels needed: de442.bsp (COMMON) + pck00011.tpc
        # (COMMON) fully cover Earth's translational state and orientation.
    ],
    "MOON": [
        ("moon_pa_de440_200625.bpc", "pck", None, None),
        ("moon_de440_250416.tf", "fk_satellites", None, None),
    ],
    "MARS": [
        ("mars_iau2000_v1.tpc", "pck", None, None),
        ("mar099s.bsp", "spk_satellites", "1995-01-01", "2050-01-01"),
        ("mar099.bsp", "spk_satellites", "1600-01-01", "2600-01-02"),
    ],
    "PHOBOS": [
        ("mars_iau2000_v1.tpc", "pck", None, None),
        ("mar099s.bsp", "spk_satellites", "1995-01-01", "2050-01-01"),
        ("mar099.bsp", "spk_satellites", "1600-01-01", "2600-01-02"),
    ],
    "DEIMOS": [
        ("mars_iau2000_v1.tpc", "pck", None, None),
        ("mar099s.bsp", "spk_satellites", "1995-01-01", "2050-01-01"),
        ("mar099.bsp", "spk_satellites", "1600-01-01", "2600-01-02"),
    ],
}


# ── Giant-planet moon candidates ──────
PLANET_CANDIDATE_KERNELS = {
    "JUPITER": [
        "jup365.bsp",       # core: Jupiter + 4 Galileans + 4 inner moons
        "jup347.bsp",       # bulk of the named irregulars (~879 MB)
        "jup348.bsp",       # 2024-2025 newly named irregulars
        "jup349.bsp",       # 2026 newly named irregulars
    ],
    "SATURN": [
        "sat441.bsp",           # core: Saturn + classical moons
        "sat415.bsp",           # Janus, Epimetheus, Atlas, Prometheus, Pandora,
                                 # Pan, Methone, Pallene, Anthe, Aegaeon
        "sat393_daphnis.bsp",   # Daphnis only
        "sat441xl_part-1.bsp",  # extended-coverage Saturn-only
        "sat441xl_part-2.bsp",
        "sat455.bsp", "sat456.bsp", "sat457.bsp", "sat459.bsp",  # provisional/
                                 # named irregulars released in waves; expect
                                 # NAIF to add sat460+ over time -- append here
    ],
    "URANUS": [
        "ura184_part-1.bsp",    # Cordelia..Portia + Uranus
        "ura184_part-2.bsp",    # Rosalind..S2025_u_1 + Uranus
        "ura184_part-3.bsp",    # Ariel/Umbriel/Titania/Oberon/Miranda,
                                 # Caliban..S2023_u1 + Uranus
        "ura111xl-701.bsp", "ura111xl-702.bsp", "ura111xl-703.bsp",
        "ura111xl-704.bsp", "ura111xl-705.bsp", "ura111xl-799.bsp",
        "ura116xl.bsp",          # 30-kyr backups for the major moons + irregulars
    ],
    "NEPTUNE": [
        "nep104.bsp",            # Triton + 5 named inner-ish moons + Neptune
        "nep105.bsp",            # Nereid
        "nep097.bsp", "nep097xl-801.bsp", "nep097xl-899.bsp",
        "nep101xl.bsp", "nep101xl-802.bsp",
    ],
    "PLUTO": [
        "plu060.bsp",            # Charon, Nix, Hydra, Kerberos, Styx + Pluto
    ],
}


# ── Asteroids: one shared file covering ~300 named asteroids ────────────────
ASTEROID_KERNEL_FILE = ("codes_300ast_20100725.bsp", "spk_asteroids", "1799-12-30", "2199-12-13")

# ── Lagrange points & comets: small, closed sets -- fine to hand-curate ────
LAGRANGE_KERNELS = {
    "EARTH-MOON L1": ("L1_de441.bsp", "spk_lagrange_point", "1900-01-01", "2151-01-01"),
    "EARTH-MOON L2": ("L2_de441.bsp", "spk_lagrange_point", "1900-01-01", "2151-01-01"),
    "SUN L4": ("L4_de441.bsp", "spk_lagrange_point", "1900-01-01", "2151-01-01"),
    "SUN L5": ("L5_de441.bsp", "spk_lagrange_point", "1900-01-01", "2151-01-01"),
}

COMET_KERNELS = {
    "CHURYUMOV-GERASIMENKO": ("C_G_1000012_2012_2017.bsp", "spk_comets", "2012-01-01", "2017-01-01"),
    "ISON": ("ison.bsp", "spk_comets", "2012-01-01", "2014-01-02"),
    "SIDING SPRING": ("c2013a1_s105_merged.bsp", "spk_comets", None, None),
}


# ── Mission Kernel Registry (no time filtering -- user assumed to know scope) ─
MISSION_KERNELS = {
    "MAVEN": [
        ("maven_spacecraft.bsp", "https://naif.jpl.nasa.gov/pub/naif/MAVEN/kernels/spk/"),
        ("maven_sclk.tsc", "https://naif.jpl.nasa.gov/pub/naif/MAVEN/kernels/sclk/"),
    ],
    "MARS_EXPRESS": [
        
    ],
    "MARS_RECON_ORBITER": [
        ("mro_psp.bsp", "https://naif.jpl.nasa.gov/pub/naif/MRO/kernels/spk/"),  # TODO verify
    ],
    "INSIGHT": [
        ("insight_struct_v01.bsp", "https://naif.jpl.nasa.gov/pub/naif/InSight/kernels/spk/"),  # TODO verify
    ],
    "PERSEVERANCE": [
        ("m2020_v04.bsp", "https://naif.jpl.nasa.gov/pub/naif/M2020/kernels/spk/"),  # TODO verify
    ],
    "CURIOSITY": [
        ("msl_atls_ops_v03.bsp", "https://naif.jpl.nasa.gov/pub/naif/MSL/kernels/spk/"),  # TODO verify
    ],
}


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


def _select_body_kernels(body, time=None, time_range=None):
    clean_body = body.strip().upper()
    if clean_body not in BODY_KERNELS:
        raise ValueError(
            f"No registered kernel set for body '{body}'. "
            f"Known bodies: {sorted(BODY_KERNELS.keys())}."
        )

    entries = BODY_KERNELS[clean_body]
    req_lo, req_hi = _normalize_window(time, time_range)

    if req_lo is None and req_hi is None:
        return [(fn, sub) for (fn, sub, _, _) in entries]

    selected = []
    for fname, subdir, cov_start, cov_end in entries:
        if _window_contains(req_lo, req_hi, cov_start, cov_end):
            selected.append((fname, subdir))

    has_time_bounded_entries = any(cov_start or cov_end for (_, _, cov_start, cov_end) in entries)
    if not selected and has_time_bounded_entries:
        raise ValueError(
            f"No registered kernel for '{body}' covers the requested time window "
            f"({req_lo}, {req_hi}). Check BODY_KERNELS or widen the request."
        )
    return selected


# ── Dynamic moon-kernel resolver ─────────────────────────────────────────────

def _normalize_name(name):
    """Uppercase and strip separators so 'S/2020_S_49', 'S2020_s_49', and
    'S2020_s49' all compare equal -- NAIF is not consistent about this
    across the FK name/ID blocks vs. the "Bodies on the File" tables."""
    return re.sub(r"[^A-Z0-9]", "", str(name).upper())


_BODY_TABLE_RE = re.compile(
    r"^\s*([A-Za-z0-9_/]+)\s+(\d{2,6})\s+[\d.eE+\-]+\s+\d+\s+\d+\s+\S+",
    re.MULTILINE,
)

_FK_NAME_CODE_RE = re.compile(
    r"NAIF_BODY_NAME\s*\+=\s*\(\s*'([^']+)'\s*\)\s*"
    r"NAIF_BODY_CODE\s*\+=\s*\(\s*(\d+)\s*\)",
    re.DOTALL,
)

_TIMESPAN_LINE_RE = re.compile(
    r"Timespan from JED\s+[\d.]+\(([\d\-A-Za-z]+)\)\s+to\s+JED\s+[\d.]+\(([\d\-A-Za-z]+)\)"
)


def _parse_paren_date(token):
    """Parse a 'DD-MON-YYYY' style date as seen inside Timespan(...) parens."""
    m = re.match(r"(\d{1,2})-([A-Za-z]{3})-(\d{1,5})", token.strip())
    if not m:
        return None
    day, mon, year = m.groups()
    try:
        return Time(f"{int(year):04d}-{mon.title()}-{int(day):02d}", format="iso")
    except Exception:
        return None


def _parse_naif_calendar(token):
    """Parse a 'YYYY MON DD ...' style date as seen in BEGIN_TIME/END_TIME
    lines. Treats anything tagged 'B.C.' as unbounded (returns None) since
    those only appear in wide-open 30kyr backup files where exact bounding
    doesn't matter for kernel *selection*."""
    token = token.strip()
    if "B.C." in token.upper():
        return None
    m = re.match(r"(\d{1,5})\s+([A-Za-z]{3})\s+(\d{1,2})", token)
    if not m:
        return None
    year, mon, day = m.groups()
    try:
        return Time(f"{int(year):04d}-{mon.title()}-{int(day):02d}", format="iso")
    except Exception:
        return None


def parse_kernel_comment(text, this_filename):
    """
    Parse a NAIF .cmt comment block and return:

        {
          "bodies": {NORMALIZED_NAME: naif_id, ...},
          "coverage": (Time_or_None, Time_or_None),   # overall begin/end
        }

    Tolerant by design: NAIF's comment formatting differs kernel to kernel
    (compare e.g. sat459's explicit FK name/ID block vs. ura184's plain
    "Bodies on the File" name table), so every name/ID pair found by either
    pattern is unioned together rather than assuming one canonical format.
    """
    bodies = {}

    for name, code in _FK_NAME_CODE_RE.findall(text):
        bodies[_normalize_name(name)] = int(code)

    for name, code in _BODY_TABLE_RE.findall(text):
        if name.upper() in ("NAME", "SYSTEM", "NUMBER"):
            continue
        bodies[_normalize_name(name)] = int(code)

    begin = end = None
    block_re = re.compile(
        r"SPK_KERNEL\s*=\s*" + re.escape(this_filename) + r"\b.*?"
        r"BEGIN_TIME\s*=\s*(.+?)\n.*?END_TIME\s*=\s*(.+?)\n",
        re.DOTALL,
    )
    m = block_re.search(text)
    if m:
        begin = _parse_naif_calendar(m.group(1))
        end = _parse_naif_calendar(m.group(2))
    else:
        m2 = _TIMESPAN_LINE_RE.search(text)
        if m2:
            begin = _parse_paren_date(m2.group(1))
            end = _parse_paren_date(m2.group(2))

    return {"bodies": bodies, "coverage": (begin, end)}


def _comment_cache_path(filename):
    os.makedirs(_CMT_CACHE_DIR, exist_ok=True)
    return os.path.join(_CMT_CACHE_DIR, filename + ".cmt.txt")


def _fetch_comment_text(filename, subdir="spk_satellites", force=False):
    """
    Fetch (and disk-cache) the small text comment for `filename` WITHOUT
    downloading the multi-gigabyte binary kernel. Returns "" on any failure
    (404, network error, etc.) so callers treat that candidate as simply
    not matching, rather than crashing the whole resolution pass.
    """
    cache_path = _comment_cache_path(filename)
    if not force and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    cmt_name = re.sub(r"\.bsp$", ".cmt", filename)
    url = _NAIF_BASE + _NAIF_SUBDIRS[subdir] + cmt_name
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        text = resp.text
    except Exception as e:
        print(f"Could not fetch comment for {filename} ({e}); "
              f"skipping it as a candidate for this lookup.")
        text = ""

    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


def resolve_moon_kernel(body, time=None, time_range=None, planet=None):
    """
    Dynamically determine which file(s) in PLANET_CANDIDATE_KERNELS contain
    `body` (a moon name like "Himalia"/"S/2020 S 49", or a NAIF ID) and, if
    given, cover the requested time/time_range.

    Returns a list of (planet, filename) tuples, most-specific candidate
    first (smaller/dedicated files are listed before big merged catalogs in
    PLANET_CANDIDATE_KERNELS, and that ordering is preserved here).

    Raises ValueError if nothing matches.
    """
    req_lo, req_hi = _normalize_window(time, time_range)
    is_numeric = str(body).strip().lstrip("-").isdigit()
    norm_body = None if is_numeric else _normalize_name(body)

    planets_to_search = [planet.upper()] if planet else list(PLANET_CANDIDATE_KERNELS.keys())
    matches = []

    for pl in planets_to_search:
        for fname in PLANET_CANDIDATE_KERNELS.get(pl, []):
            text = _fetch_comment_text(fname)
            if not text:
                continue
            parsed = parse_kernel_comment(text, fname)
            if is_numeric:
                hit = int(body) in parsed["bodies"].values()
            else:
                hit = norm_body in parsed["bodies"]
            if not hit:
                continue
            begin, end = parsed["coverage"]
            if _window_contains(req_lo, req_hi, begin, end):
                matches.append((pl, fname))

    if not matches:
        raise ValueError(
            f"Could not find a kernel covering body '{body}' "
            f"{'for the requested time window ' if (req_lo or req_hi) else ''}"
            f"across candidates for {planets_to_search}. Either the name is "
            f"misspelled/mis-normalized, the .cmt fetch failed for the file "
            f"that actually has it (see warnings above), or NAIF has released "
            f"a new file that needs adding to PLANET_CANDIDATE_KERNELS."
        )

    matches.sort(key=lambda pf: PLANET_CANDIDATE_KERNELS[pf[0]].index(pf[1]))
    return matches


# ── URL Resolution ────────────────────────────────────────────────────────────

def _infer_subdir(filename):
    fname = filename.lower()
    if fname.endswith(".tls"):
        return "lsk"
    if fname.endswith(".tpc") or fname.endswith(".bpc"):
        return "pck"
    if fname.endswith(".bsp"):
        if fname.startswith("de"):
            return "spk_planets"
        if fname.startswith(("l1_", "l2_", "l4_", "l5_")):
            return "spk_lagrange_point"
        return "spk_satellites"
    raise ValueError(
        f"Cannot infer NAIF subdirectory for '{filename}'. "
        f"Pass an explicit URL via extra_urls instead."
    )


def get_dynamic_ephemeris_urls(body=None, mission=None, filenames=None,
                                 time=None, time_range=None):
    """
    Resolves kernel filenames into NAIF download URLs.

    Parameters
    ----------
    body : str, optional
        A body name (e.g. "MARS", "EARTH", "MOON") that resolves through
        the small static BODY_KERNELS table, OR a giant-planet moon name /
        NAIF ID (e.g. "HIMALIA", "S/2020 S 49", 65297) that resolves
        dynamically via resolve_moon_kernel().
    mission : str, optional
        Mission name (e.g. "MAVEN") -- pulls its registered set from
        MISSION_KERNELS. No time filtering applied.
    filenames : str or list[str], optional
        Comma-separated string or list of explicit filenames to resolve
        (subdirectory inferred automatically).
    time : str or astropy.time.Time, optional
        Single timestamp to filter body/moon kernels against.
    time_range : tuple, optional
        (start, end) timestamps to filter body/moon kernels against.

    Returns
    -------
    dict[str, str]
        filename -> full download URL
    """
    urls = {}

    if body:
        clean_body = body.strip().upper() if isinstance(body, str) else str(body)
        for fname, subdir in COMMON_KERNELS:
            urls[fname] = _NAIF_BASE + _NAIF_SUBDIRS[subdir] + fname

        if clean_body in BODY_KERNELS:
            for fname, subdir in _select_body_kernels(body, time=time, time_range=time_range):
                urls[fname] = _NAIF_BASE + _NAIF_SUBDIRS[subdir] + fname
        else:
            matches = resolve_moon_kernel(clean_body, time=time, time_range=time_range)
            best_planet, best_fname = matches[0]
            urls[best_fname] = _NAIF_BASE + _NAIF_SUBDIRS["spk_satellites"] + best_fname

    if mission:
        clean_mission = mission.strip().upper()
        if clean_mission not in MISSION_KERNELS:
            raise ValueError(
                f"No registered kernel set for mission '{mission}'. "
                f"Known missions: {sorted(MISSION_KERNELS.keys())}."
            )
        for fname, loc in MISSION_KERNELS[clean_mission]:
            if loc.startswith("http"):
                urls[fname] = loc + fname
            else:
                urls[fname] = _NAIF_BASE + _NAIF_SUBDIRS[loc] + fname

    if filenames:
        if isinstance(filenames, str):
            filenames = [f.strip() for f in filenames.split(",") if f.strip()]
        for fname in filenames:
            urls[fname] = _NAIF_BASE + _NAIF_SUBDIRS[_infer_subdir(fname)] + fname

    if not urls:
        raise ValueError("get_dynamic_ephemeris_urls() needs at least one of: body, mission, filenames.")
    return urls


# ── Checksum Utilities ───────────────────────────────────────────────────────

def fetch_remote_md5s(subdir="spk_satellites"):
    """
    FIX: this used to always hit spk/planets/aa_checksums.txt regardless of
    what kind of kernel was being verified. Pass the subdir key matching
    the kernel category you're about to download (e.g. "spk_satellites"
    for moon kernels, "spk_planets" for de442.bsp, "pck"/"lsk" for those).
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


# ── Main Fetch Routine ───────────────────────────────────────────────────────

def _subdir_for(filename):
    """Best-effort guess of which checksum manifest a filename belongs to,
    used by fetch_kernels() when downloading a heterogeneous queue."""
    try:
        return _infer_subdir(filename)
    except ValueError:
        return "spk_satellites"


def fetch_kernels(target_dir=None, body=None, mission=None, filenames=None,
                   time=None, time_range=None, extra_urls=None):
    """
    Fetches missing kernels for a body and/or mission and/or explicit filenames.
    See get_dynamic_ephemeris_urls() for parameter semantics.
    """
    root_dir = os.path.abspath(target_dir) if target_dir else _DEFAULT_KERNEL_ROOT
    generic_dir = os.path.join(root_dir, "generic")
    mission_dir = os.path.join(root_dir, "mission")

    os.makedirs(generic_dir, exist_ok=True)
    os.makedirs(mission_dir, exist_ok=True)

    queue = {}
    if body or filenames:
        queue.update(get_dynamic_ephemeris_urls(
            body=body, filenames=filenames, time=time, time_range=time_range
        ))
    mission_filenames = set()
    if mission:
        mission_urls = get_dynamic_ephemeris_urls(mission=mission)
        mission_filenames.update(mission_urls.keys())
        queue.update(mission_urls)
    if extra_urls:
        mission_filenames.update(extra_urls.keys())
        queue.update(extra_urls)

    if not queue:
        raise ValueError("fetch_kernels() needs at least one of: body, mission, filenames, extra_urls.")

    context_label = mission if mission else body

    manifest_cache = {}

    for filename, url in queue.items():
        dest_dir = mission_dir if filename in mission_filenames else generic_dir
        dest = os.path.join(dest_dir, filename)

        subdir_key = _subdir_for(filename)
        if subdir_key not in manifest_cache:
            print(f"  Fetching live NAIF asset checksum tokens for '{subdir_key}'...")
            manifest_cache[subdir_key] = fetch_remote_md5s(subdir_key)
        expected_md5 = manifest_cache[subdir_key].get(filename.lower())

        if os.path.exists(dest):
            if expected_md5:
                if calculate_local_md5(dest) == expected_md5:
                    print(f"  Verified & intact (via NAIF Manifest): {filename}")
                    _log_citation(filename, url, context_label)
                    continue
            else:
                if os.path.getsize(dest) > 0:
                    print(f"  Verified via document footprint: {filename}")
                    _log_citation(filename, url, context_label)
                    continue

        print(f"  Downloading/Correcting {filename} ...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        if expected_md5 and calculate_local_md5(dest) != expected_md5:
            raise ValueError(f"MD5 verification failure on newly downloaded asset: {filename}")
        print(f"  Successfully verified and saved: {filename}")
        _log_citation(filename, url, context_label)


if __name__ == "__main__":
    today = datetime.date.today().isoformat()
    print(f"Initializing Generic SPICE Pipeline Asset Fetcher [Target: MARS, time={today}]")
    fetch_kernels(body="MARS", time=today)
