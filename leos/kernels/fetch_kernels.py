"""
leos/kernels/fetch_kernels.py

Public entry point for kernel resolution + download. This module dispatches
to:
  - fetch_generic_kernels.py for bodies, moons, asteroids, comets, Lagrange
    points, and explicit filenames
  - kernels/missions/fetch_<mission>_kernels.py for spacecraft mission
    kernel sets

and owns the actual download / checksum-verification / citation-logging
loop, since that part is identical regardless of where the URLs came from.
"""

import os
import requests

from . import _kernel_common as _kc
from . import fetch_generic_kernels
from .missions.Mars import fetch_maven_kernels
from .missions.Mars import fetch_mars_express_kernels
from .missions.Mars import fetch_mro_kernels
from .missions.Mars import fetch_insight_kernels
from .missions.Mars import fetch_perseverance_kernels
from .missions.Mars import fetch_curiosity_kernels

# ── Mission Resolver Registry ────────────────────────────────────────────────
# Every entry maps a mission= string to a get_kernel_urls(time=, time_range=)
# function with the same signature, returning dict[filename -> URL]. To add
# a new mission: drop a kernels/missions/fetch_<name>_kernels.py module with
# that function, import it above, and add one line here.
MISSION_RESOLVERS = {
    "MAVEN": fetch_maven_kernels.get_kernel_urls,
    "MARS_EXPRESS": fetch_mars_express_kernels.get_kernel_urls,
    "MARS_RECON_ORBITER": fetch_mro_kernels.get_kernel_urls,
    "INSIGHT": fetch_insight_kernels.get_kernel_urls,
    "PERSEVERANCE": fetch_perseverance_kernels.get_kernel_urls,
    "CURIOSITY": fetch_curiosity_kernels.get_kernel_urls,
}


# ── URL Resolution ────────────────────────────────────────────────────────────

def get_dynamic_ephemeris_urls(body=None, mission=None, filenames=None,
                                 time=None, time_range=None):
    """
    Resolves kernel filenames into NAIF/mission download URLs. Combines
    generic resolution (body/filenames, via fetch_generic_kernels.py) with
    mission resolution (via kernels/missions/*) -- either or both can be
    supplied in one call.

    See fetch_generic_kernels.get_generic_kernel_urls() and the individual
    mission modules' get_kernel_urls() for parameter semantics.

    Returns
    -------
    dict[str, str]
        filename -> full download URL
    """
    urls = {}

    if body or filenames:
        urls.update(fetch_generic_kernels.get_generic_kernel_urls(
            body=body, filenames=filenames, time=time, time_range=time_range
        ))

    if mission:
        clean_mission = mission.strip().upper()
        if clean_mission not in MISSION_RESOLVERS:
            raise ValueError(
                f"No registered kernel set for mission '{mission}'. "
                f"Known missions: {sorted(MISSION_RESOLVERS.keys())}."
            )
        urls.update(MISSION_RESOLVERS[clean_mission](time=time, time_range=time_range))

    if not urls:
        raise ValueError("get_dynamic_ephemeris_urls() needs at least one of: body, mission, filenames.")
    return urls


# ── Main Fetch Routine ───────────────────────────────────────────────────────

def fetch_kernels(target_dir=None, body=None, mission=None, filenames=None,
                   time=None, time_range=None, extra_urls=None):
    """
    Fetches missing kernels for a body and/or mission and/or explicit filenames.
    See get_dynamic_ephemeris_urls() for parameter semantics.
    """
    root_dir = os.path.abspath(target_dir) if target_dir else _kc._DEFAULT_KERNEL_ROOT
    generic_dir = os.path.join(root_dir, "generic")
    mission_dir = os.path.join(root_dir, "mission")

    os.makedirs(generic_dir, exist_ok=True)
    os.makedirs(mission_dir, exist_ok=True)

    queue = {}
    if body or filenames:
        queue.update(fetch_generic_kernels.get_generic_kernel_urls(
            body=body, filenames=filenames, time=time, time_range=time_range
        ))

    mission_filenames = set()
    if mission:
        # NOTE (refactor fix): the pre-split version called this without
        # time/time_range, which silently disabled MAVEN's time-windowed CK
        # selection whenever you went through fetch_kernels() -- it only
        # worked if you called get_maven_kernel_urls() directly. time/
        # time_range now flow through to mission resolvers too.
        mission_urls = get_dynamic_ephemeris_urls(mission=mission, time=time, time_range=time_range)
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

        subdir_key = _kc._subdir_for(filename)
        if subdir_key in _kc._SUBDIRS_WITH_CHECKSUMS:
            if subdir_key not in manifest_cache:
                print(f"  Fetching live NAIF asset checksum tokens for '{subdir_key}'...")
                manifest_cache[subdir_key] = _kc.fetch_remote_md5s(subdir_key)
            expected_md5 = manifest_cache[subdir_key].get(filename.lower())
        else:
            expected_md5 = None

        if os.path.exists(dest):
            if expected_md5:
                if _kc.calculate_local_md5(dest) == expected_md5:
                    print(f"  Verified & intact (via NAIF Manifest): {filename}")
                    _kc._log_citation(filename, url, context_label)
                    continue
            else:
                if os.path.getsize(dest) > 0:
                    print(f"  Verified via document footprint: {filename}")
                    _kc._log_citation(filename, url, context_label)
                    continue

        print(f"  Downloading/Correcting {filename} ...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        if expected_md5:
            if _kc.calculate_local_md5(dest) != expected_md5:
                raise ValueError(f"MD5 verification failure on newly downloaded asset: {filename}")
            print(f"  Successfully verified and saved: {filename}")
        else:
            print(f"  Warning: no checksum available for '{filename}'; downloaded but unverified.")
        _kc._log_citation(filename, url, context_label)


# ── Citation Tracking (re-exported for backward compatibility) ──────────────
get_citations = _kc.get_citations
reset_citations = _kc.reset_citations


# ── Backward-compatible re-exports ───────────────────────────────────────────
# Anything elsewhere in leos/ that previously did e.g.
#   from leos.kernels.fetch_kernels import resolve_moon_kernel
# keeps working without touching that call site. Grep your codebase for
# `fetch_kernels.` and `from leos.kernels.fetch_kernels import` to confirm
# nothing relies on a name that ISN'T re-exported here.
BODY_KERNELS = fetch_generic_kernels.BODY_KERNELS
PLANET_CANDIDATE_KERNELS = fetch_generic_kernels.PLANET_CANDIDATE_KERNELS
ASTEROID_KERNEL_FILE = fetch_generic_kernels.ASTEROID_KERNEL_FILE
LAGRANGE_KERNELS = fetch_generic_kernels.LAGRANGE_KERNELS
COMET_KERNELS = fetch_generic_kernels.COMET_KERNELS
get_common_kernels = fetch_generic_kernels.select_common_kernels
resolve_moon_kernel = fetch_generic_kernels.resolve_moon_kernel
resolve_asteroid_kernel = fetch_generic_kernels.resolve_asteroid_kernel
parse_kernel_comment = fetch_generic_kernels.parse_kernel_comment
get_maven_kernel_urls = fetch_maven_kernels.get_kernel_urls
resolve_maven_sclk = fetch_maven_kernels.resolve_maven_sclk
resolve_maven_ck = fetch_maven_kernels.resolve_maven_ck
fetch_remote_md5s = _kc.fetch_remote_md5s
calculate_local_md5 = _kc.calculate_local_md5
