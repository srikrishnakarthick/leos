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
import shutil
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
        resolver_kwargs = {"time": time, "time_range": time_range}
        if clean_mission == "MAVEN":
            resolver_kwargs["include_common"] = not bool(body)
        urls.update(MISSION_RESOLVERS[clean_mission](**resolver_kwargs))

    if not urls:
        raise ValueError("get_dynamic_ephemeris_urls() needs at least one of: body, mission, filenames.")
    return urls

def _confirm_and_resolve_paths(queue, mission_filenames, generic_dir, mission_dir,
                                reused_from_generic):
    """
    Shows the user every file that would be downloaded (skipping ones
    already verified/reused), its source URL, and its size. Asks Y/N.

    Returns a dict[filename -> local_filepath] that the download loop
    should treat as "already resolved, just verify/copy" -- OR None if
    the user said Y and the normal download loop should proceed as-is.
    """
    to_confirm = [
        (fname, url) for fname, url in queue.items()
        if fname not in reused_from_generic
    ]

    if not to_confirm:
        return None  # nothing to ask about; everything already local

    print(f"\nThe following files will be downloaded to '{generic_dir}' "
          f"(generic) / '{mission_dir}' (mission):\n")
    sized = []
    for fname, url in to_confirm:
        size = _kc.fetch_remote_size(url)
        sized.append((fname, url, size))
        dest_kind = "mission" if fname in mission_filenames else "generic"
        dest_dir = mission_dir if fname in mission_filenames else generic_dir
        dest_path = os.path.join(dest_dir, fname)
        print(f"  [{dest_kind}] {fname}")
        print(f"      URL:  {url}")
        print(f"      Size: {_kc.format_size(size)}")
        print(f"      Path: {dest_path}")

    answer = input("\nProceed with download? [Y/N]: ").strip().upper()

    if answer == "Y":
        return None  # normal download path

    # ── Manual path: N, or anything else, is treated as "can't/won't
    #    download automatically" ──────────────────────────────────────
    print("\nPlease download the following files manually, then provide "
          "their local paths.\n")
    for fname, url, size in sized:
        print(f"  {fname}")
        print(f"      URL:  {url}")
        print(f"      Size: {_kc.format_size(size)}")

    print(f"\nEnter the local paths of the downloaded files, in the exact "
          f"same order as listed above, separated by commas:")
    raw = input("> ").strip()
    provided = [p.strip() for p in raw.split(",")]

    if len(provided) != len(sized):
        raise ValueError(
            f"Expected {len(sized)} path(s), got {len(provided)}. "
            f"Re-run fetch_kernels() and provide exactly one path per "
            f"listed file, in order, comma-separated."
        )

    resolved = {}
    for (fname, url, size), path in zip(sized, provided):
        if not os.path.isfile(path):
            raise ValueError(f"'{path}' (for {fname}) does not exist or is not a file.")
        resolved[fname] = path
    return resolved

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
        resolver_kwargs = {"time": time, "time_range": time_range}
        if clean_mission == "MAVEN":
            resolver_kwargs["include_common"] = not bool(body)
        mission_urls = MISSION_RESOLVERS[clean_mission](**resolver_kwargs)
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
        if body and filename in queue:
            continue  # already a generic-resolved file, not a mission dupe case
        subdir_key = _kc._subdir_for(filename)
        if subdir_key not in ("lsk", "pck", "spk_planets", "spk_satellites"):
            continue  # not a "common-kernel-type" file; leave mission routing alone
        candidate_path = os.path.join(generic_dir, filename)
        if os.path.exists(candidate_path):
            reused_from_generic.add(filename)

    context_label = mission if mission else body
    manifest_cache = {}

    manual_paths = _confirm_and_resolve_paths(
        queue, mission_filenames, generic_dir, mission_dir, reused_from_generic
    )

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

        if manual_paths is not None:
            # ── User supplied a manually-downloaded file instead ──────
            src_path = manual_paths[filename]
            if expected_md5:
                actual_md5 = _kc.calculate_local_md5(src_path)
                if actual_md5 != expected_md5:
                    raise ValueError(
                        f"MD5 verification failed for manually-provided "
                        f"'{filename}' at '{src_path}': "
                        f"expected {expected_md5}, got {actual_md5}. "
                        f"Re-download from {url} and try again."
                    )
                print(f"  Verified (MD5 match) manual file: {filename}")
            else:
                if os.path.getsize(src_path) > 0:
                    print(f"  File size is non-zero, but no checksum was "
                          f"available to verify '{filename}'.")
                else:
                    print(f"  Warning: manually-provided '{filename}' has "
                          f"zero file size.")
            if os.path.abspath(src_path) != os.path.abspath(dest):
                shutil.copyfile(src_path, dest)
            _kc._log_citation(filename, url, context_label)
            continue

        if os.path.exists(dest):
            if expected_md5:
                if _kc.calculate_local_md5(dest) == expected_md5:
                    print(f"  Verified & intact (via NAIF Manifest): {filename}")
                    _kc._log_citation(filename, url, context_label)
                    continue
            else:
                if os.path.getsize(dest) > 0:
                    print(f"  File size is non-zero, but no checksum was "
                          f"available to verify '{filename}'.")
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
            if os.path.getsize(dest) > 0:
                print(f"  File size is non-zero, but no checksum was "
                      f"available to verify '{filename}'.")
            else:
                print(f"  Warning: '{filename}' downloaded but file size "
                      f"is zero.")
        _kc._log_citation(filename, url, context_label)
