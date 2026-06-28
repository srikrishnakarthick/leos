"""
leos/kernels/missions/fetch_mars_express_kernels.py

Mars Express kernel URLs.

No kernel set has been curated for Mars Express yet -- this mirrors the
empty placeholder that lived in the original MISSION_KERNELS registry.
Add entries here (same (filename, base_url) pattern as fetch_mro_kernels.py)
once you've picked the kernels you need from
https://naif.jpl.nasa.gov/pub/naif/MEX/kernels/.
"""

MARS_EXPRESS_KERNELS = [
    # ("filename.bsp", "https://naif.jpl.nasa.gov/pub/naif/MEX/kernels/spk/"),
]


def get_kernel_urls(time=None, time_range=None):
    """
    Returns dict[filename -> URL] for Mars Express.
    Currently empty -- see module docstring.
    """
    return {fname: base + fname for fname, base in MARS_EXPRESS_KERNELS}
