"""
leos.spice_utils
----------------
Low-level SPICE plumbing. Wraps SpiceyPy calls in clean Python functions
with astropy types. No science logic lives here — just kernel management
and unit-safe SPICE interface functions.

All other LEOS modules that need SPICE import from here.
"""

import os
import spiceypy as spice
from astropy.time import Time
from astropy import units as u

# ── Default kernel paths ──────────────────────────────────────────────────────
_KERNEL_DIR = os.path.join(
    os.path.dirname(__file__), "..", "kernels", "data"
)

_DEFAULT_KERNELS = [
    "naif0012.tls",   # leap seconds
    "pck00011.tpc",   # planetary constants
    "de432s.bsp",     # solar system ephemeris
]


# ── Kernel management ─────────────────────────────────────────────────────────

def load_kernels(kernel_paths=None):
    """
    Load SPICE kernels into the SPICE kernel pool.

    Parameters
    ----------
    kernel_paths : list of str, optional
        Full paths to kernel files. If None, loads the default
        LEOS kernels from kernels/data/.
    """
    if kernel_paths is None:
        kernel_paths = [
            os.path.join(_KERNEL_DIR, k) for k in _DEFAULT_KERNELS
        ]
    for path in kernel_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Kernel not found: {path}\n"
                f"Run `python kernels/fetch_kernels.py` first."
            )
        spice.furnsh(path)


def unload_kernels():
    """Unload all kernels from the SPICE kernel pool."""
    spice.kclear()


def loaded_kernel_count():
    """Return the number of kernels currently loaded."""
    return spice.ktotal("ALL")


# ── Time conversion ───────────────────────────────────────────────────────────

def utc_to_et(time):
    """
    Convert an astropy Time to SPICE Ephemeris Time (ET).

    Parameters
    ----------
    time : astropy Time

    Returns
    -------
    float — ephemeris time in seconds past J2000
    """
    if not isinstance(time, Time):
        raise TypeError("time must be an astropy Time object.")
    return spice.utc2et(time.isot)


def et_to_utc(et, format="ISOC"):
    """
    Convert SPICE Ephemeris Time back to a UTC string.

    Parameters
    ----------
    et : float — ephemeris time in seconds past J2000
    format : str — SPICE time format string (default 'ISOC')

    Returns
    -------
    astropy Time
    """
    utc_str = spice.et2utc(et, format, 3)
    return Time(utc_str, format="isot")


# ── Body utilities ────────────────────────────────────────────────────────────

def body_name_to_id(name):
    """
    Return the NAIF integer ID for a body name.

    Parameters
    ----------
    name : str — e.g. 'Earth', 'Mars', 'Moon'

    Returns
    -------
    int — NAIF body ID
    """
    try:
        code = spice.bodn2c(name.upper())
        return int(code)
    except Exception:
        raise ValueError(f"NAIF ID not found for body: '{name}'")


def body_radii(name):
    """
    Return the triaxial radii of a body in km.

    Parameters
    ----------
    name : str

    Returns
    -------
    astropy Quantity — shape (3,) in km: [a, b, c]
    """
    radii = spice.bodvrd(name.upper(), "RADII", 3)[1]
    return radii * u.km


# ── Position utilities ────────────────────────────────────────────────────────

def sun_position(target_body, et, frame="J2000", observer="SUN"):
    """
    Return the position of the Sun relative to a target body at time et.

    Parameters
    ----------
    target_body : str — e.g. 'EARTH', 'MARS'
    et : float — ephemeris time

    Returns
    -------
    position : astropy Quantity, shape (3,) in km
    light_time : astropy Quantity in seconds
    """
    pos, lt = spice.spkpos(
        "SUN", et, frame, "LT+S", target_body.upper()
    )
    return pos * u.km, lt * u.s
