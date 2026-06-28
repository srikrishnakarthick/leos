"""
leos/kernels/missions/fetch_curiosity_kernels.py

Curiosity (Mars Science Laboratory) kernel URLs.
"""

_MSL_SPK_BASE = "https://naif.jpl.nasa.gov/pub/naif/MSL/kernels/spk/"

CURIOSITY_KERNELS = [
    ("msl_atls_ops_v03.bsp", _MSL_SPK_BASE),  # TODO verify
]


def get_kernel_urls(time=None, time_range=None):
    """Returns dict[filename -> URL] for Curiosity (MSL)."""
    return {fname: base + fname for fname, base in CURIOSITY_KERNELS}
