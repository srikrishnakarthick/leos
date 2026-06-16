import os
from contextlib import contextmanager
import numpy as np
import spiceypy as spice
from astropy.time import Time
from astropy import units as u

_KERNEL_DIR = os.path.join(os.path.dirname(__file__), "..", "kernels", "data")

def discover_kernels():
    """Dynamically scan the data directory to match static frames and any active DE ephemeris."""
    if not os.path.exists(_KERNEL_DIR):
        return []
    
    files = os.listdir(_KERNEL_DIR)
    resolved = []
    
    # Core reference constants frames
    for base in ["naif0012.tls", "pck00011.tpc", "mars_iau2000_v1.tpc"]:
        if base in files:
            resolved.append(os.path.join(_KERNEL_DIR, base))
            
    # Find any active binary planetary ephemeris file matching de*.bsp
    ephem_files = [f for f in files if f.startswith("de") and f.endswith(".bsp")]
    if ephem_files:
        # Prioritize the highest version sorting array string automatically (e.g. de442 over de440)
        ephem_files.sort()
        resolved.append(os.path.join(_KERNEL_DIR, ephem_files[-1]))
        
    return resolved

def load_kernels(kernel_paths=None):
    if kernel_paths is None:
        kernel_paths = discover_kernels()
        if not kernel_paths:
            raise FileNotFoundError(
                f"No usable kernels found in {_KERNEL_DIR}.\nRun `python kernels/fetch_kernels.py` first."
            )
    for path in kernel_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Target kernel file path does not exist: {path}")
        spice.furnsh(path)

def unload_kernels():
    spice.kclear()

@contextmanager
def kernel_sandbox(kernel_paths=None):
    load_kernels(kernel_paths)
    try:
        yield
    finally:
        unload_kernels()

def utc_to_et(time):
    if not isinstance(time, Time):
        raise TypeError("time must be an astropy Time object.")
    if time.isscalar:
        return spice.utc2et(time.isot)
    v_utc2et = np.vectorize(spice.utc2et, otypes=[float])
    return v_utc2et(time.isot)

def et_to_utc(et, format="ISOC"):
    et_arr = np.asarray(et)
    if et_arr.ndim == 0:
        return Time(spice.et2utc(float(et), format, 3), format="isot")
    v_et2utc = np.vectorize(lambda e: spice.et2utc(e, format, 3), otypes=[str])
    return Time(v_et2utc(et_arr), format="isot")

def body_name_to_id(name):
    clean_name = str(name).strip().upper()
    try:
        return int(spice.bodn2c(clean_name))
    except Exception:
        raise ValueError(f"NAIF tracking ID not found for body: '{name}'")

def body_radii(name):
    clean_name = str(name).strip().upper()
    radii = spice.bodvrd(clean_name, "RADII", 3)[1]
    return radii * u.km

def sun_position(target_body, et, frame="J2000"):
    _BARYCENTER = {
        "MERCURY": "1", "VENUS": "2", "EARTH": "3",
        "MARS": "4", "JUPITER": "5", "SATURN": "6",
        "URANUS": "7", "NEPTUNE": "8", "MOON": "301",
    }
    clean_target = str(target_body).strip().upper()
    target = _BARYCENTER.get(clean_target, clean_target)
    et_arr = np.asarray(et)
    
    if et_arr.ndim > 0:
        positions, light_times = [], []
        for e in et_arr.flat:
            pos, lt = spice.spkpos("SUN", float(e), frame, "LT+S", target)
            positions.append(pos)
            light_times.append(lt)
        out_shape = et_arr.shape + (3,)
        return np.array(positions).reshape(out_shape) * u.km, np.array(light_times).reshape(et_arr.shape) * u.s
        
    pos, lt = spice.spkpos("SUN", float(et), frame, "LT+S", target)
    return pos * u.km, lt * u.s
