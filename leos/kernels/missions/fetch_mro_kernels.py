"""
leos/kernels/missions/fetch_mro_kernels.py

Mars Reconnaissance Orbiter kernel URLs.
"""

_MRO_SPK_BASE = "https://naif.jpl.nasa.gov/pub/naif/MRO/kernels/spk/"

MRO_KERNELS = [
    ("mro_psp.bsp", _MRO_SPK_BASE),  # TODO verify
]


def get_kernel_urls(time=None, time_range=None):
    """
    Returns dict[filename -> URL] for Mars Reconnaissance Orbiter.
    time/time_range are accepted for interface consistency with the other
    mission resolvers (e.g. MAVEN) but aren't currently used to filter
    this mission's kernel set.
    """
    return {fname: base + fname for fname, base in MRO_KERNELS}
