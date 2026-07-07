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
        clean_mission = mission.strip().upper()
        if clean_mission not in MISSION_RESOLVERS:
            raise ValueError(
                f"No registered kernel set for mission '{mission}'. "
                f"Known missions: {sorted(MISSION_RESOLVERS.keys())}."
            )
        mission_urls = get_dynamic_ephemeris_urls(mission=mission, time=time, time_range=time_range)
        mission_filenames.update(mission_urls.keys())
        queue.update(mission_urls)

    if extra_urls:
        mission_filenames.update(extra_urls.keys())
        queue.update(extra_urls)

    if not queue:
        raise ValueError("fetch_kernels() needs at least one of: body, mission, filenames, extra_urls.")

    # ── De-dupe against generic/: if a mission asked for a common-kernel-type
    # file (LSK/PCK/planetary or satellite SPK) that happens to ALREADY exist
    # in generic/ -- whether because a previous body= call put it there, or
    # because the mission's own preferred version coincidentally matches --
    # reuse that file instead of downloading a second copy into mission/.
    # This is a pure disk-location decision; it never changes what
    # select_common_kernels()/resolve_best_planetary_spk() consider "latest"
    # for future generic calls, so a mission's older/pinned version can never
    # supersede the generic "best" globally.
    reused_from_generic = set()
    for filename in list(mission_filenames):
        if filename in (body and queue.keys() or []):
            continue  # already a generic-resolved file, not a mission dupe case
        subdir_key = _kc._subdir_for(filename)
        if subdir_key not in ("lsk", "pck", "spk_planets", "spk_satellites"):
            continue  # not a "common-kernel-type" file; leave mission routing alone
        candidate_path = os.path.join(generic_dir, filename)
        if os.path.exists(candidate_path):
            reused_from_generic.add(filename)

    context_label = mission if mission else body
    manifest_cache = {}

    for filename, url in queue.items():
        if filename in reused_from_generic:
            print(f"  Reusing existing generic copy (no duplicate mission "
                  f"download): {filename}")
            _kc._log_citation(filename, url, context_label)
            continue

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
