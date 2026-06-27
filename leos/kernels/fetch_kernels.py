import os
import hashlib
import requests
from pathlib import Path
from astropy.time import Time

# ── Directory Architecture ───────────────────────────────────────────────────
KERNEL_ROOT = os.path.join(os.path.dirname(__file__), "data")
_DEFAULT_KERNEL_ROOT = KERNEL_ROOT


# ── NAIF Subdirectory Map ────────────────────────────────────────────────────
_NAIF_BASE = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/"
_NAIF_SUBDIRS = {
    "lsk": "lsk/",
    "pck": "pck/",
    "fk_satellites": "fk/satellites/",
    "spk_planets": "spk/planets/",
    "spk_satellites": "spk/satellites/",
    "spk_asteroids": "spk/asteroids/",
}

# ── Common Kernels (always fetched, body-independent) ───────────────────────
# de442.bsp covers Mercury..Pluto barycenters, Sun, Earth, and Moon
# translational state, span 1549-12-31 to 2650-01-25 (verified via brief).
COMMON_KERNELS = [
    ("naif0012.tls", "lsk"),
    ("pck00011.tpc", "pck"),
    ("de442.bsp", "spk_planets"),
]

# ── Body Kernel Registry ─────────────────────────────────────────────────────
# Each entry: (filename, subdir_key, coverage_start, coverage_end)
BODY_KERNELS = {
    "EARTH": [
        # No additional kernels needed: de442.bsp (COMMON) + pck00011.tpc
        # (COMMON) fully cover Earth's translational state and orientation.
    ],
    "MOON": [
        # Lunar orientation (PA frame) — verified coverage ~1550-2650,
        # matching de442's span. Must be paired with its frame kernel.
        ("moon_pa_de440_200625.bpc", "pck", None, None),
        ("moon_de440_250416.tf", "fk_satellites", None, None),
    ],
    "MARS": [
        ("mars_iau2000_v1.tpc", "pck", None, None),
        # Short-span file first (smaller download) — only valid 1995-2050
        ("mar099s.bsp", "spk_satellites", "1995-01-01", "2050-01-01"),
        # Full-span fallback for anything outside that window
        ("mar099.bsp", "spk_satellites", "1600-01-01", "2600-01-02"),
    ],
    "PHOBOS": [
        # Same SPK system as Mars: mar099/mar099s carries Phobos (401),
        # Deimos (402), and Mars barycenter (499) together.
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

# ── Planetary System Groups ──────────────────────────────────────────────────
# Each planet's major/named moons + the planet itself, keyed by file role.
# A body may need MULTIPLE files (e.g. an outer Jovian moon needs jup347
# AND de442 for context, but not jup365).
PLANETARY_SYSTEM_KERNELS["JUPITER"] = {
    "core": [("jup365.bsp", "spk_satellites", "1600-01-10", "2200-01-10")],  # Jupiter + 4 Galileans + 4 inner moons
    "moons": {
        "IO": [], "EUROPA": [], "GANYMEDE": [], "CALLISTO": [],
        "AMALTHEA": [], "THEBE": [], "ADRASTEA": [], "METIS": [],
        # jup347.bsp: NAIF IDs 506,507,510,513,517,519-545,547-550,551-572,55501-55526
        "HIMALIA": [("jup347.bsp", "spk_satellites", "1799-12-21", "2200-01-10")],
        "ELARA": [("jup347.bsp", "spk_satellites", "1799-12-21", "2200-01-10")],
        "PASIPHAE": [("jup347.bsp", "spk_satellites", "1799-12-21", "2200-01-10")],
        "SINOPE": [("jup347.bsp", "spk_satellites", "1799-12-21", "2200-01-10")],
        "LYSITHEA": [("jup347.bsp", "spk_satellites", "1799-12-21", "2200-01-10")],
        "CARME": [("jup347.bsp", "spk_satellites", "1799-12-21", "2200-01-10")],
        "ANANKE": [("jup347.bsp", "spk_satellites", "1799-12-21", "2200-01-10")],
        "LEDA": [("jup347.bsp", "spk_satellites", "1799-12-21", "2200-01-10")],
        # ... (all other named irregulars in jup347's BODIES list follow the same entry)
        # jup348.bsp: NAIF IDs 55527-55530 (newly named 2024-2025 discoveries)
        # jup349.bsp: NAIF IDs 55531-55544 (newly named 2026 discoveries)
        # These are provisional-designation moons unlikely to be queried by name yet.
    },

    "SATURN": {
        "core": [("sat441.bsp", "spk_satellites", "1749-12-30", "2250-01-06")],
        "moons": {
            "MIMAS": [], "ENCELADUS": [], "TETHYS": [], "DIONE": [], "RHEA": [],
            "TITAN": [], "HYPERION": [], "IAPETUS": [], "PHOEBE": [],
            # small named moons needing sat415 in addition to sat441:
            "JANUS":      [("sat415.bsp", "spk_satellites", "1949-12-26", "2050-01-10")],
            "EPIMETHEUS": [("sat415.bsp", "spk_satellites", "1949-12-26", "2050-01-10")],
            "PAN":        [("sat415.bsp", "spk_satellites", "1949-12-26", "2050-01-10")],
            "DAPHNIS":    [("sat393_daphnis.bsp", "spk_satellites", "1949-12-26", "2050-01-10")],
            # irregular moons in sat456/457/459 similarly...
        },
    },
    "URANUS": {
        "core": [],  # Uranus barycenter/planet itself comes from ura184_part-*
        "moons": {
            "ARIEL": [("ura184_part-3.bsp", "spk_satellites", "1600-01-04", "2399-12-17")],
            "MIRANDA": [("ura184_part-3.bsp", "spk_satellites", "1600-01-04", "2399-12-17")],
            "CORDELIA": [("ura184_part-1.bsp", "spk_satellites", "1900-01-01", "2100-01-24")],
            "PUCK": [("ura184_part-2.bsp", "spk_satellites", "1900-01-01", "2100-01-24")],
            # etc — each moon mapped to whichever ura184_part-N actually contains it
        },
    },
    "NEPTUNE": {
        "core": [("nep104.bsp", "spk_satellites", "1600-01-04", "2399-12-25")],  # Neptune + Halimede/Psamathe/Sao/Laomedeia/Neso
        "moons": {
            "TRITON": [],  # in nep104 already
            "NEREID": [("nep105.bsp", "spk_satellites", "1600-01-04", "2400-01-02")],
            "NAIAD": [], "THALASSA": [], "DESPINA": [], "GALATEA": [],
            "LARISSA": [], "PROTEUS": [], "HIPPOCAMP": [],  # all in nep095, superseded by nep104 per body list
            "HALIMEDE": [], "PSAMATHE": [], "SAO": [], "LAOMEDEIA": [], "NESO": [],  # in nep104
        },
    },
    "PLUTO": {
        "core": [("plu060.bsp", "spk_satellites", "1800-01-02", "2199-12-30")],
        "moons": {
            "CHARON": [], "NIX": [], "HYDRA": [], "KERBEROS": [], "STYX": [],
        },
    },
}

# ── Asteroids: shared multi-body file, looked up by name/ID, not per-file ───
ASTEROID_KERNEL_FILE = ("codes_300ast_20100725.bsp", "spk_asteroids", "1799-12-30", "2199-12-13")
ASTEROID_NAMES = {"CERES", "VESTA", "LUTETIA", "KLEOPATRA", "EROS", ...}  # from aa_summaries.txt

# ── Lagrange points: one file per point, no time filtering needed (wide span) ─
LAGRANGE_KERNELS = {
    "EARTH-MOON L1": ("L1_de441.bsp", "spk_lagrange_point", "1900-01-01", "2151-01-01"),
    "EARTH-MOON L2": ("L2_de441.bsp", "spk_lagrange_point", "1900-01-01", "2151-01-01"),
    "SUN L4": ("L4_de441.bsp", "spk_lagrange_point", "1900-01-01", "2151-01-01"),
    "SUN L5": ("L5_de441.bsp", "spk_lagrange_point", "1900-01-01", "2151-01-01"),
}

COMET_KERNELS = {
    "CHURYUMOV-GERASIMENKO": ("C_G_1000012_2012_2017.bsp", "spk_comets", "2012-01-01", "2017-01-01"),
    "ISON": ("ison.bsp", "spk_comets", "2012-01-01", "2014-01-02"),
    # Two files both cover "Siding Spring" with different windows — keep both,
    # priority order: wide-window first as primary, narrow as fallback isn't needed
    # since c2013a1_s105_merged.bsp's window is a superset of siding_spring_8-19-14.bsp.
    "SIDING SPRING": ("c2013a1_s105_merged.bsp", "spk_comets", None, None),  # effectively unbounded (3002 BC–2999 AD)
}

# ── Mission Kernel Registry (no time filtering — user assumed to know scope) ─
# Each entry: (filename, url_or_subdir_key)
# If the second element starts with "http", it's used as a direct full URL
# (filename appended). Otherwise treated as a _NAIF_SUBDIRS key.
#
# VERIFY all URLs/filenames against each mission's actual NAIF kernel directory,
# e.g. https://naif.jpl.nasa.gov/pub/naif/<MISSION>/kernels/
MISSION_KERNELS = {
    "MAVEN": [
        ("maven_spacecraft.bsp", "https://naif.jpl.nasa.gov/pub/naif/MAVEN/kernels/spk/"),
        ("maven_sclk.tsc", "https://naif.jpl.nasa.gov/pub/naif/MAVEN/kernels/sclk/"),
    ],
    "MARS_EXPRESS": [
        ("ORMF_______ .bsp".strip(), "https://naif.jpl.nasa.gov/pub/naif/MEX/kernels/spk/"),  # PLACEHOLDER — verify exact filename
    ],
    "MARS_RECON_ORBITER": [
        ("mro_psp.bsp", "https://naif.jpl.nasa.gov/pub/naif/MRO/kernels/spk/"),  # PLACEHOLDER — verify
    ],
    "INSIGHT": [
        ("insight_struct_v01.bsp", "https://naif.jpl.nasa.gov/pub/naif/InSight/kernels/spk/"),  # PLACEHOLDER — verify
    ],
    "PERSEVERANCE": [
        ("m2020_v04.bsp", "https://naif.jpl.nasa.gov/pub/naif/M2020/kernels/spk/"),  # PLACEHOLDER — verify
    ],
    "CURIOSITY": [
        ("msl_atls_ops_v03.bsp", "https://naif.jpl.nasa.gov/pub/naif/MSL/kernels/spk/"),  # PLACEHOLDER — verify
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

    # If nothing in BODY_KERNELS matched but the body has zero time-bounded
    # entries at all (e.g. EARTH, which has no SPK rows), that's fine — return empty.
    has_time_bounded_entries = any(cov_start or cov_end for (_, _, cov_start, cov_end) in entries)
    if not selected and has_time_bounded_entries:
        raise ValueError(
            f"No registered kernel for '{body}' covers the requested time window "
            f"({req_lo}, {req_hi}). Check BODY_KERNELS or widen the request."
        )
    return selected

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
        Body name (e.g. "MARS") — pulls COMMON_KERNELS + that body's
        registered set from BODY_KERNELS, filtered by time/time_range if given.
    mission : str, optional
        Mission name (e.g. "MAVEN") — pulls its registered set from
        MISSION_KERNELS. No time filtering applied.
    filenames : str or list[str], optional
        Comma-separated string or list of explicit filenames to resolve
        (subdirectory inferred automatically).
    time : str or astropy.time.Time, optional
        Single timestamp to filter body kernels against.
    time_range : tuple, optional
        (start, end) timestamps to filter body kernels against.

    Returns
    -------
    dict[str, str]
        filename -> full download URL
    """
    urls = {}

    if body:
        for fname, subdir in COMMON_KERNELS:
            urls[fname] = _NAIF_BASE + _NAIF_SUBDIRS[subdir] + fname
        for fname, subdir in _select_body_kernels(body, time=time, time_range=time_range):
            urls[fname] = _NAIF_BASE + _NAIF_SUBDIRS[subdir] + fname

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
        raise ValueError("Must supply at least one of: body, mission, filenames.")
    return urls

# ── Checksum Utilities ───────────────────────────────────────────────────────

def fetch_remote_md5s():
    checksum_url = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/aa_checksums.txt"
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
        print(f"  ⚠️ Warning: Could not fetch remote aa_checksums.txt ({e}).")
    return md5_dict

def calculate_local_md5(filepath):
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

# ── Main Fetch Routine ───────────────────────────────────────────────────────

def fetch_kernels(target_dir=None, body=None, mission=None, filenames=None,
                   time=None, time_range=None, extra_urls=None):
    """
    Fetches missing kernels for a body and/or mission and/or explicit filenames.

    Parameters
    ----------
    target_dir : str, optional
        Root directory to download into. Defaults to leos/kernels/data.
    body : str, optional
        Body name (e.g. "MARS").
    mission : str, optional
        Mission name (e.g. "MAVEN").
    filenames : str or list[str], optional
        Explicit comma-separated filenames.
    time : str or astropy.time.Time, optional
        Single timestamp, used to select the correct body SPK.
    time_range : tuple, optional
        (start, end) timestamps, used to select the correct body SPK.
    extra_urls : dict[str, str], optional
        filename -> full URL overrides for kernels not in any registry.
    """
    root_dir = os.path.abspath(target_dir) if target_dir else _DEFAULT_KERNEL_ROOT
    generic_dir = os.path.join(root_dir, "generic")
    mission_dir = os.path.join(root_dir, "mission")

    os.makedirs(generic_dir, exist_ok=True)
    os.makedirs(mission_dir, exist_ok=True)

    print("  Fetching live NAIF asset checksum tokens...")
    nasa_md5s = fetch_remote_md5s()

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

    for filename, url in queue.items():
        dest_dir = mission_dir if filename in mission_filenames else generic_dir
        dest = os.path.join(dest_dir, filename)
        expected_md5 = nasa_md5s.get(filename.lower())

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
    print("Initializing Generic SPICE Pipeline Asset Fetcher [Target: MARS]")
    fetch_kernels(body="MARS", time="2026-06-27")
