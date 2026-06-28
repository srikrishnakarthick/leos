"""
leos/kernels/fetch_generic_kernels.py

Resolvers for body-, planet-, asteroid-, comet-, and Lagrange-point kernels
published in NAIF's *generic_kernels* tree, plus the small always-fetched
"common" kernel set (leap seconds, orientation, planetary ephemeris) that
both bodies and missions build on.

This module has no knowledge of any individual spacecraft mission --
mission-specific kernel sets live under kernels/missions/.
"""

import os
import re
import requests
from astropy.time import Time

from ._kernel_common import (
    _NAIF_BASE,
    _NAIF_SUBDIRS,
    _CMT_CACHE_DIR,
    _normalize_window,
    _window_contains,
    _select_time_filtered_kernels,
    _infer_subdir,
)

# ── Common Kernels (always fetched, body-independent) ───────────────────────
COMMON_KERNELS = [
    ("naif0012.tls", "lsk", None, None),
    ("pck00011.tpc", "pck", None, None),
    ("de442.bsp", "spk_planets", "1549-12-31", "2650-01-25"),
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
    "MERCURY": [
    # de442.bsp (COMMON) fully covers Mercury's translational state.
    # No moons. No orientation kernel in the generic set.
    ],
    "VENUS": [
    # de442.bsp (COMMON) fully covers Venus's translational state.
    # No moons. No orientation kernel in the generic set.
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
# Bodies NOT covered by any NAIF generic kernel and requiring an on-demand
# JPL Horizons SPK request (https://ssd.jpl.nasa.gov/horizons/):
#   - Small NEOs and quasi-satellites: 2002 VE68 (Zoozve), 2016 HO3 (Kamo'oalewa), etc.
#   - Any asteroid outside the ~300 most massive named bodies above
#   - Mission-specific small body targets not yet in a dedicated kernel
# For these, use fetch_kernels(extra_urls={...}) with a Horizons-generated .bsp.

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


# ── Selection Helpers ────────────────────────────────────────────────────────

def _select_body_kernels(body, time=None, time_range=None):
    clean_body = body.strip().upper()
    if clean_body not in BODY_KERNELS:
        raise ValueError(
            f"No registered kernel set for body '{body}'. "
            f"Known bodies: {sorted(BODY_KERNELS.keys())}."
        )
    return _select_time_filtered_kernels(
        BODY_KERNELS[clean_body], time=time, time_range=time_range,
        context_label=f"'{body}'",
    )


def select_common_kernels(time=None, time_range=None):
    """
    Public (no leading underscore, unlike the rest of this module's helpers)
    because kernels/missions/* needs it too -- e.g. MAVEN's kernel set
    bundles in the same naif0012.tls/pck00011.tpc/de442.bsp set as any
    plain body request.
    """
    return _select_time_filtered_kernels(
        COMMON_KERNELS, time=time, time_range=time_range,
        context_label="the common kernel set",
    )


def _select_named_static_kernel(entry, time=None, time_range=None, label=""):
    """
    entry: a (filename, subdir, cov_start, cov_end) tuple, e.g. a value
    pulled from LAGRANGE_KERNELS or COMET_KERNELS. Returns
    (filename, subdir) if the request falls within the kernel's
    validity window; raises ValueError otherwise.
    """
    fname, subdir, cov_start, cov_end = entry
    req_lo, req_hi = _normalize_window(time, time_range)
    if not _window_contains(req_lo, req_hi, cov_start, cov_end):
        label_suffix = f" ({label})" if label else ""
        raise ValueError(
            f"'{fname}'{label_suffix} does not cover the requested time "
            f"window ({req_lo}, {req_hi})."
        )
    return fname, subdir


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


def _fetch_asteroid_tf_text(force=False):
    """
    Fetch (and disk-cache) codes_300ast_20100725.tf, which contains
    NAIF_BODY_NAME/CODE blocks for all 300 asteroids.  The .cmt for this
    file has no body listing ('use BRIEF') -- the .tf does.
    """
    tf_name = "codes_300ast_20100725.tf"
    cache_path = _comment_cache_path(tf_name)
    if not force and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    url = _NAIF_BASE + _NAIF_SUBDIRS["spk_asteroids"] + tf_name
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        text = resp.text
    except Exception as e:
        print(f"Could not fetch asteroid TF ({e}); asteroid name lookup unavailable.")
        text = ""

    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


_AST_TF_RE = re.compile(
    r"NAIF_BODY_NAME\s*\+=\s*\(\s*'(?:\d+\s+)?([^']+)'\s*\)\s*"
    r"NAIF_BODY_CODE\s*\+=\s*\(\s*(\d+)\s*\)",
    re.DOTALL,
)


def _parse_asteroid_tf(text):
    """
    Parse codes_300ast_20100725.tf and return {NORMALIZED_NAME: naif_id}.
    Names in the file look like '1 CERES', '2 PALLAS' -- the leading
    minor-planet number is stripped so 'CERES' and '1 CERES' both resolve.
    """
    bodies = {}
    for name, code in _AST_TF_RE.findall(text):
        bodies[_normalize_name(name)] = int(code)
    return bodies


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


def resolve_asteroid_kernel(body, time=None, time_range=None):
    """
    Check whether `body` (an asteroid name like 'CERES', a prefixed name
    like '1 CERES', a NAIF ID like 2000001, or a minor-planet number like 1)
    is one of the ~300 named asteroids on ASTEROID_KERNEL_FILE, and that the
    file's validity window covers the requested time/time_range.

    Body names are resolved via codes_300ast_20100725.tf (NAIF_BODY_NAME/CODE
    blocks).  Numeric IDs skip the network call entirely.

    Returns (filename, subdir) on success.  Raises ValueError otherwise.
    """
    fname, subdir, cov_start, cov_end = ASTEROID_KERNEL_FILE
    req_lo, req_hi = _normalize_window(time, time_range)

    if not _window_contains(req_lo, req_hi, cov_start, cov_end):
        raise ValueError(
            f"'{fname}' does not cover the requested time window "
            f"({req_lo}, {req_hi})."
        )

    is_numeric = str(body).strip().lstrip("-").isdigit()

    if is_numeric:
        raw = int(str(body).strip())
        naif_id = raw if raw > 1_000_000 else raw + 2_000_000
        if 2_000_001 <= naif_id <= 2_000_300:
            return fname, subdir
        raise ValueError(
            f"NAIF ID {naif_id} is outside the range covered by {fname} "
            f"(2000001–2000300)."
        )

    text = _fetch_asteroid_tf_text()
    if not text:
        raise ValueError(
            f"Could not verify '{body}' is on {fname}: .tf fetch failed."
        )

    bodies = _parse_asteroid_tf(text)
    norm = _normalize_name(body)
    norm_stripped = re.sub(r"^\d+", "", norm)   # '1CERES' -> 'CERES'

    if norm not in bodies and norm_stripped not in bodies:
        raise ValueError(
            f"'{body}' not found among the named bodies in "
            f"codes_300ast_20100725.tf. Either the name is wrong, or it "
            f"isn't one of the ~300 named asteroids covered by this file."
        )
    return fname, subdir


# ── Public Entry Point ───────────────────────────────────────────────────────

def get_generic_kernel_urls(body=None, filenames=None, time=None, time_range=None):
    """
    Resolves non-mission kernel filenames into NAIF download URLs.

    Parameters
    ----------
    body : str, optional
        A body name (e.g. "MARS", "EARTH", "MOON") that resolves through
        BODY_KERNELS, a Lagrange point / comet name, OR a giant-planet
        moon name / NAIF ID (e.g. "HIMALIA", "S/2020 S 49", 65297) that
        resolves dynamically via resolve_moon_kernel(), OR a named
        asteroid via resolve_asteroid_kernel().
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
        for fname, subdir in select_common_kernels(time=time, time_range=time_range):
            urls[fname] = _NAIF_BASE + _NAIF_SUBDIRS[subdir] + fname

        if clean_body in BODY_KERNELS:
            for fname, subdir in _select_body_kernels(body, time=time, time_range=time_range):
                urls[fname] = _NAIF_BASE + _NAIF_SUBDIRS[subdir] + fname
        elif clean_body in LAGRANGE_KERNELS:
            fname, subdir = _select_named_static_kernel(
                LAGRANGE_KERNELS[clean_body], time, time_range, clean_body
            )
            urls[fname] = _NAIF_BASE + _NAIF_SUBDIRS[subdir] + fname
        elif clean_body in COMET_KERNELS:
            fname, subdir = _select_named_static_kernel(
                COMET_KERNELS[clean_body], time, time_range, clean_body
            )
            urls[fname] = _NAIF_BASE + _NAIF_SUBDIRS[subdir] + fname
        else:
            moon_err = ast_err = None

            # Asteroid first: one cached .tf fetch vs. scanning dozens of moon
            # .cmt files.  Unknown bodies fail fast here instead of hanging.
            try:
                fname, subdir = resolve_asteroid_kernel(
                    clean_body, time=time, time_range=time_range
                )
                urls[fname] = _NAIF_BASE + _NAIF_SUBDIRS[subdir] + fname
            except ValueError as e:
                ast_err = e
                try:
                    matches = resolve_moon_kernel(
                        clean_body, time=time, time_range=time_range
                    )
                    best_planet, best_fname = matches[0]
                    urls[best_fname] = (
                        _NAIF_BASE + _NAIF_SUBDIRS["spk_satellites"] + best_fname
                    )
                except ValueError as e2:
                    moon_err = e2

            if ast_err is not None and moon_err is not None:
                raise ValueError(
                    f"Could not resolve body '{body}' against any registry "
                    f"(BODY_KERNELS, LAGRANGE_KERNELS, COMET_KERNELS, "
                    f"PLANET_CANDIDATE_KERNELS, or ASTEROID_KERNEL_FILE).\n"
                    f"  Moon-resolution attempt: {moon_err}\n"
                    f"  Asteroid-resolution attempt: {ast_err}"
                )

    if filenames:
        if isinstance(filenames, str):
            filenames = [f.strip() for f in filenames.split(",") if f.strip()]
        for fname in filenames:
            urls[fname] = _NAIF_BASE + _NAIF_SUBDIRS[_infer_subdir(fname)] + fname

    if not urls:
        raise ValueError("get_generic_kernel_urls() needs at least one of: body, filenames.")
    return urls
