"""
LEOS: Kernels Submodule Suite
Manages the retrieval, local buffering, and MD5 validation of NAIF SPICE
ephemeris files and static planetary physical constant kernels.
"""

# ── Shared low-level infrastructure ──────────────────────────────────────
from ._kernel_common import (
    calculate_local_md5, fetch_remote_md5s, fetch_remote_size, format_size,
    KERNEL_ROOT, DATA_DIRS, get_citations, reset_citations,
    reset_session_cache,
    resolve_versioned_kernel, resolve_latest_lsk, resolve_latest_pck,
    resolve_best_planetary_spk, resolve_best_mars_spk,
    resolve_matching_lagrange_kernel,
)

# ── Generic (body/moon/asteroid/comet/Lagrange) kernel resolution ───────
from .fetch_generic_kernels import (
    resolve_moon_kernel, resolve_asteroid_kernel, parse_kernel_comment,
    get_generic_kernel_urls,
    BODY_KERNELS, PLANET_CANDIDATE_KERNELS, ASTEROID_KERNEL_FILE,
    LAGRANGE_KERNELS, COMET_KERNELS,
    select_common_kernels as get_common_kernels,
)

# ── Public dispatch + download entry points ──────────────────────────────
from .fetch_kernels import (
    fetch_kernels,
    get_dynamic_ephemeris_urls,
    MISSION_RESOLVERS,
)

# ── MAVEN mission resolver (the one mission module with real dynamic
#    logic; other missions are plain static lists reached only through
#    MISSION_RESOLVERS / fetch_kernels, not re-exported here) ────────────
from .missions.Mars.fetch_maven_kernels import (
    get_maven_kernel_urls, resolve_maven_sclk, resolve_maven_ck,
    resolve_latest_maven_fk, resolve_latest_maven_struct_spk,
    resolve_maven_orbit_spk,
)

__all__ = [
    "ASTEROID_KERNEL_FILE",
    "BODY_KERNELS",
    "COMET_KERNELS",
    "DATA_DIRS",
    "KERNEL_ROOT",
    "LAGRANGE_KERNELS",
    "MISSION_RESOLVERS",
    "PLANET_CANDIDATE_KERNELS",
    "calculate_local_md5",
    "fetch_kernels",
    "fetch_remote_md5s",
    "fetch_remote_size",
    "format_size",
    "get_citations",
    "get_common_kernels",
    "get_dynamic_ephemeris_urls",
    "get_generic_kernel_urls",
    "get_maven_kernel_urls",
    "parse_kernel_comment",
    "reset_citations",
    "reset_session_cache",
    "resolve_asteroid_kernel",
    "resolve_best_mars_spk",
    "resolve_best_planetary_spk",
    "resolve_latest_lsk",
    "resolve_latest_maven_fk",
    "resolve_latest_maven_struct_spk",
    "resolve_latest_pck",
    "resolve_matching_lagrange_kernel",
    "resolve_maven_ck",
    "resolve_maven_orbit_spk",
    "resolve_maven_sclk",
    "resolve_moon_kernel",
    "resolve_versioned_kernel",
]
