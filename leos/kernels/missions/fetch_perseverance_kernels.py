"""
leos/kernels/missions/fetch_perseverance_kernels.py

Mars 2020 / Perseverance kernel URLs.
"""

_M2020_SPK_BASE = "https://naif.jpl.nasa.gov/pub/naif/M2020/kernels/spk/"

PERSEVERANCE_KERNELS = [
    ("m2020_v04.bsp", _M2020_SPK_BASE),  # TODO verify
]


def get_kernel_urls(time=None, time_range=None):
    """Returns dict[filename -> URL] for Perseverance (Mars 2020)."""
    return {fname: base + fname for fname, base in PERSEVERANCE_KERNELS}
