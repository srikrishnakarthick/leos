import os
from contextlib import contextmanager
import numpy as np
import spiceypy as spice
from astropy.time import Time
from astropy import units as u

_KERNEL_DIR = os.path.join(os.path.dirname(__file__), "..", "kernels", "data")

# ── Updated Absolute Path Mapping ───────────────────────────────────────────
# Looks upward from Coordinates/ to find the sibling kernels/data directory
_KERNEL_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "kernels", "data")
)

def discover_kernels():
    """Dynamically scans generic and mission directories to build a valid loading order."""
    generic_dir = os.path.join(_KERNEL_ROOT, "generic")
    mission_dir = os.path.join(_KERNEL_ROOT, "mission")
    
    resolved = []
    
    # 1. Load Generic base kernels if the directory exists
    if os.path.exists(generic_dir):
        files = os.listdir(generic_dir)
        
        # Chronological priority: LSK (Leapseconds) must be loaded first
        for f in files:
            if f.endswith(".tls"):
                resolved.append(os.path.join(generic_dir, f))
                
        # Structural priority: Text PCK/FK constants
        for f in files:
            if f.endswith(".tpc") or f.endswith(".tf"):
                resolved.append(os.path.join(generic_dir, f))
                
        # Ephemeris planetary tracking paths (SPK)
        spk_files = [f for f in files if f.startswith("de") and f.endswith(".bsp")]
        if spk_files:
            spk_files.sort()
            resolved.append(os.path.join(generic_dir, spk_files[-1]))

    # 2. Automatically sweep up any custom files users dropped into the mission folder
    if os.path.exists(mission_dir):
        for root, _, files in os.walk(mission_dir):
            for f in files:
                # Catch all extensions: CK (.bc/.bck), IK (.ti), SCLK (.tsc), Binary PCK (.bpc), Frames (.tf)
                if f.endswith(('.bc', '.bck', '.ti', '.tsc', '.bpc', '.tf')):
                    resolved.append(os.path.join(root, f))
                    
    return resolved

def load_kernels(kernel_paths=None, extra_paths=None):
    """Loads default planetary assets, plus optional runtime-supplied IK/CK/SCLK paths."""
    if kernel_paths is None:
        kernel_paths = discover_kernels()
        if not kernel_paths and not extra_paths:
            raise FileNotFoundError(
                f"No usable kernels found in {_KERNEL_ROOT}.\n"
                f"Run your asset fetcher script or supply files manually via extra_paths."
            )
            
    # Merge discoverable tracks with explicit runtime strings or lists
    all_paths = list(kernel_paths)
    if extra_paths:
        if isinstance(extra_paths, str):
            all_paths.append(extra_paths)
        else:
            all_paths.extend(extra_paths)
            
    for path in all_paths:
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

def sun_position(target, et, frame="J2000"):
    """Calculates the 3D position vector pointing from a target body toward the Sun.
    
    Accepts explicit planetary body centers (e.g., 'EARTH', 'MARS') or systemic 
    barycenters (e.g., 'EARTH BARYCENTER', 'SATURN BARYCENTER').
    """
    _NAIF_IDS = {
        "MERCURY BARYCENTER": "1",    "SATURN BARYCENTER": "6",    "MERCURY": "199",
        "VENUS BARYCENTER": "2",      "URANUS BARYCENTER": "7",    "VENUS": "299",
        "EARTH BARYCENTER": "3",      "NEPTUNE BARYCENTER": "8",   "MOON": "301",
        "MARS BARYCENTER": "4",       "PLUTO BARYCENTER": "9",     "EARTH": "399",
        "JUPITER BARYCENTER": "5",    "SUN": "10"
    }
    
    # 1. Clean the incoming target string and look up its NAIF ID
    clean_target = str(target).strip().upper()
    target_id = _NAIF_IDS.get(clean_target, clean_target)
    
    et_arr = np.asarray(et, dtype=float)
    
    # 2. Vectorized processing for time arrays
    if et_arr.ndim > 0:
        total_elements = et_arr.size
        positions = np.empty((total_elements, 3), dtype=float)
        light_times = np.empty(total_elements, dtype=float)
        
        for idx, e in enumerate(et_arr.flat):
            pos, lt = spice.spkpos("SUN", e, frame, "LT+S", target_id)
            positions[idx] = pos
            light_times[idx] = lt
            
        out_shape = et_arr.shape + (3,)
        return positions.reshape(out_shape) * u.km, light_times.reshape(et_arr.shape) * u.s
        
    # 3. Scalar processing for a single timestamp
    pos, lt = spice.spkpos("SUN", float(et), frame, "LT+S", target_id)
    return pos * u.km, lt * u.s
def get_spacecraft_attitude(sc_id, instrument_id, et, reference_frame="J2000", tolerance_seconds=1.0):
    """Robust wrapper for CK attitude extraction.
    
    Automatically translates Ephemeris Time (ET) to Spacecraft Clock (SCLK) ticks.
    """
    try:
        # 1. Convert ET to encoded SCLK ticks (Requires an .tsc kernel loaded)
        sclk_ticks = spice.sce2c(sc_id, et)
        
        # 2. Convert look-up tolerance window from seconds to clock ticks
        ticks_tolerance = spice.sctks2(sc_id, tolerance_seconds)
        
        # 3. Query the C-kernel with the converted clock metrics
        matrix, clkout = spice.ckgp(instrument_id, sclk_ticks, ticks_tolerance, reference_frame)
        return matrix
        
    except Exception as e:
        error_str = str(e)
        if "SPICE(KERNELVARNOTFOUND)" in error_str or "SPICE(NOTFOUND)" in error_str:
            raise RuntimeError(
                f"Failed to extract attitude matrix for instrument {instrument_id}.\n"
                f"Reason: Missing required CK (orientation), IK (instrument specs), or SCLK (clock) kernel.\n"
                f"Please ensure all relevant mission kernels are supplied via extra_paths."
            ) from e
        raise e
