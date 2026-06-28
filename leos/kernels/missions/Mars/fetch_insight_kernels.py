"""
leos/kernels/missions/fetch_insight_kernels.py

InSight kernel URLs.
"""

_INSIGHT_SPK_BASE = "https://naif.jpl.nasa.gov/pub/naif/InSight/kernels/spk/"

INSIGHT_KERNELS = [
    ("insight_struct_v01.bsp", _INSIGHT_SPK_BASE),  # TODO verify
]


def get_kernel_urls(time=None, time_range=None):
    """Returns dict[filename -> URL] for InSight."""
    return {fname: base + fname for fname, base in INSIGHT_KERNELS}
