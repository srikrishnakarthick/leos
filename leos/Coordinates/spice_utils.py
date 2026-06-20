import os
from contextlib import contextmanager
import numpy as np
import spiceypy as spice
from astropy.time import Time
from astropy import units as u

# ── Updated Absolute Path Mapping ───────────────────────────────────────────
_DEFAULT_KERNEL_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "kernels", "data")
)

def discover_kernels(custom_root=None):
    """
    Dynamically scans generic and mission directories to build a valid loading order.
    Accepts an optional custom_root path string for external workspaces.
    """
    # Fall back to package internal root if no explicit path is passed
    base_root = os.path.abspath(custom_root) if custom_root else _DEFAULT_KERNEL_ROOT
    
    generic_dir = os.path.join(base_root, "generic")
    mission_dir = os.path.join(base_root, "mission")
    
    VALID_EXTENSIONS = ('.tls', '.tpc', '.tf', '.bsp', '.bc', '.bck', '.ti', '.tsc', '.bpc')
    
    generic_buckets = {
        "lsk": [],   
        "meta": [],  
        "spk": []    
    }
    
    if os.path.exists(generic_dir):
        for f in os.listdir(generic_dir):
            full_path = os.path.join(generic_dir, f)
            if not os.path.isfile(full_path):
                continue
                
            if f.endswith(".tls"):
                generic_buckets["lsk"].append(full_path)
            elif f.endswith(".tpc") or f.endswith(".tf"):
                generic_buckets["meta"].append(full_path)
            elif f.endswith(".bsp"):
                generic_buckets["spk"].append(full_path)

    for bucket in generic_buckets.values():
        bucket.sort()

    resolved = []
    resolved.extend(generic_buckets["lsk"])  
    resolved.extend(generic_buckets["meta"]) 
    resolved.extend(generic_buckets["spk"])  

    if os.path.exists(mission_dir):
        mission_files = []
        for root, _, files in os.walk(mission_dir):
            for f in files:
                if f.lower().endswith(VALID_EXTENSIONS):
                    mission_files.append(os.path.join(root, f))
        
        mission_files.sort()
        resolved.extend(mission_files)
                    
    return resolved

def load_kernels(kernel_paths=None, extra_paths=None, custom_dir=None):
    """Loads default planetary assets, plus optional runtime-supplied IK/CK/SCLK paths."""
    if kernel_paths is None:
        # Pass the custom directory down to the scanner pipeline
        kernel_paths = discover_kernels(custom_root=custom_dir)
        if not kernel_paths and not extra_paths:
            target_display_path = custom_dir if custom_dir else _DEFAULT_KERNEL_ROOT
            raise FileNotFoundError(
                f"No usable kernels found in {target_display_path}.\n"
                f"Run your asset fetcher script or supply files manually via extra_paths."
            )
            
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
    """Clears the SPICE kernel pool completely to release system memory handles."""
    spice.kclear()

@contextmanager
def kernel_sandbox(kernel_paths=None):
    """Context manager ensuring execution paths unload kernels after complete cycles."""
    load_kernels(kernel_paths)
    try:
        yield
    finally:
        unload_kernels()

# ── Time and Chronology Utilities ───────────────────────────────────────────

def utc_to_et(time):
    """Converts Astropy Time objects to Ephemeris Time (ET) seconds past J2000."""
    if not isinstance(time, Time):
        raise TypeError("time must be an astropy Time object.")
    if time.isscalar:
        return spice.utc2et(time.isot)
    v_utc2et = np.vectorize(spice.utc2et, otypes=[float])
    return v_utc2et(time.isot)

def et_to_utc(et, format="ISOC"):
    """Converts Ephemeris Time (ET) scalars or arrays back to Astropy Time objects."""
    et_arr = np.asarray(et)
    if et_arr.ndim == 0:
        return Time(spice.et2utc(float(et), format, 3), format="isot")
    v_et2utc = np.vectorize(lambda e: spice.et2utc(e, format, 3), otypes=[str])
    return Time(v_et2utc(et_arr), format="isot")

# ── Central Physical Astrodynamics Properties ────────────────────────────────

def body_name_to_id(name):
    """Maps planet name strings to integer NAIF standard tracking IDs."""
    clean_name = str(name).strip().upper()
    try:
        return int(spice.bodn2c(clean_name))
    except Exception as e:
        # Intercept and convert to RuntimeError to hit our target shields safely
        raise RuntimeError(f"NAIF tracking ID not found for body or instrument: '{name}'") from e

def body_radii(name):
    """Pulls the tri-axial ellipsoid radii [a, b, c] dimensions for a body."""
    clean_name = str(name).strip().upper()
    radii = spice.bodvrd(clean_name, "RADII", 3)[1]
    return radii * u.km

# ── Positions & Distances Engine ─────────────────────────────────────────────

def sun_position(target, et, frame="J2000"):
    """Calculates the 3D position vector pointing from a target body toward the Sun."""
    _NAIF_IDS = {
        "MERCURY BARYCENTER": "1",    "SATURN BARYCENTER": "6",    "MERCURY": "199",
        "VENUS BARYCENTER": "2",      "URANUS BARYCENTER": "7",    "VENUS": "299",
        "EARTH BARYCENTER": "3",      "NEPTUNE BARYCENTER": "8",   "MOON": "301",
        "MARS BARYCENTER": "4",       "PLUTO BARYCENTER": "9",     "EARTH": "399",
        "JUPITER BARYCENTER": "5",    "SUN": "10"
    }
    
    clean_target = str(target).strip().upper()
    target_id = _NAIF_IDS.get(clean_target, clean_target)
    et_arr = np.asarray(et, dtype=float)
    
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
        
    pos, lt = spice.spkpos("SUN", float(et), frame, "LT+S", target_id)
    return pos * u.km, lt * u.s

def get_sub_point(target, observer, et, frame=None, method="NEAR POINT/ELLIPSOID"):
    """Computes the sub-observer point vector on a target planet ellipsoid."""
    if frame is None:
        frame = f"IAU_{target.upper()}"
    et_arr = np.asarray(et, dtype=float)
    
    def _call(e):
        spoint, alt, _ = spice.subpnt(method, target.upper(), e, frame, "LT+S", observer.upper())
        return spoint, alt

    if et_arr.ndim > 0:
        pts = np.empty((et_arr.size, 3))
        alts = np.empty(et_arr.size)
        for i, e in enumerate(et_arr.flat):
            pts[i], alts[i] = _call(e)
        return pts.reshape(et_arr.shape + (3,)) * u.km, alts.reshape(et_arr.shape) * u.km
    
    pt, alt = _call(float(et))
    return pt * u.km, alt * u.km

def get_sub_solar_point(target, et, frame=None, method="NEAR POINT/ELLIPSOID"):
    """Computes the sub-solar point intersection coordinates on a target surface."""
    if frame is None:
        frame = f"IAU_{target.upper()}"
    et_arr = np.asarray(et, dtype=float)
    
    if et_arr.ndim > 0:
        pts = np.empty((et_arr.size, 3))
        for i, e in enumerate(et_arr.flat):
            pts[i], _ = spice.subslr(method, target.upper(), e, frame, "LT+S", "SUN")
        return pts.reshape(et_arr.shape + (3,)) * u.km
    
    pt, _ = spice.subslr(method, target.upper(), float(et), frame, "LT+S", "SUN")
    return pt * u.km

def angular_separation(targ1, targ2, observer, et, shape1="POINT", shape2="POINT"):
    """Computes the true angular separation between two targets from an observer."""
    et_arr = np.asarray(et, dtype=float)
    if et_arr.ndim > 0:
        sep = np.empty(et_arr.size)
        for i, e in enumerate(et_arr.flat):
            sep[i] = spice.trgsep(e, targ1.upper(), shape1, "J2000", targ2.upper(), shape2, "J2000", "LT+S", observer.upper())
        return np.degrees(sep.reshape(et_arr.shape)) * u.deg
    
    sep = spice.trgsep(float(et), targ1.upper(), shape1, "J2000", targ2.upper(), shape2, "J2000", "LT+S", observer.upper())
    return np.degrees(sep) * u.deg

# ── Kinematics & Coordinate Transforms ───────────────────────────────────────

def get_state_vector(target, observer, et, frame="J2000"):
    """Returns 6-element raw state vector [x, y, z, vx, vy, vz] via spkezr."""
    et_arr = np.asarray(et, dtype=float)
    if et_arr.ndim > 0:
        states = np.empty((et_arr.size, 6))
        for i, e in enumerate(et_arr.flat):
            states[i], _ = spice.spkezr(target.upper(), e, frame, "LT+S", observer.upper())
        return states.reshape(et_arr.shape + (6,))
    state, _ = spice.spkezr(target.upper(), float(et), frame, "LT+S", observer.upper())
    return state

def transform_position(vector, from_frame, to_frame, et):
    """Transforms a 3D position vector between coordinate tracks via pxform."""
    et_arr = np.asarray(et, dtype=float)
    vec = np.asarray(vector, dtype=float)
    if et_arr.ndim > 0:
        out = np.empty((et_arr.size, 3), dtype=float)
        for i, e in enumerate(et_arr.flat):
            mat = spice.pxform(from_frame, to_frame, e)
            out[i] = np.dot(mat, vec[i] if vec.ndim > 1 else vec)
        return out.reshape(et_arr.shape + (3,))
    return np.dot(spice.pxform(from_frame, to_frame, float(et)), vec)

def transform_state(state, from_frame, to_frame, et):
    """Transforms a 6-element kinematic state vector between frames via sxform."""
    et_arr = np.asarray(et, dtype=float)
    st = np.asarray(state, dtype=float)
    if et_arr.ndim > 0:
        out = np.empty_like(st)
        for i, e in enumerate(et_arr.flat):
            mat = spice.sxform(from_frame, to_frame, e)
            out[i] = np.dot(mat, st[i] if st.ndim > 1 else st)
        return out
    return np.dot(spice.sxform(from_frame, to_frame, float(et)), st)

def get_spacecraft_attitude(sc_id, instrument_id, et, reference_frame="J2000", tolerance_seconds=1.0):
    """Robust wrapper for CK attitude orientation matrix tracking extraction."""
    try:
        sclk_ticks = spice.sce2c(sc_id, et)
        
        # Calculate tick tolerance using delta ET
        ticks_plus_1 = spice.sce2c(sc_id, et + 1.0)
        ticks_per_second = abs(ticks_plus_1 - sclk_ticks)
        ticks_tolerance = ticks_per_second * tolerance_seconds
        
        matrix, _ = spice.ckgp(instrument_id, sclk_ticks, ticks_tolerance, reference_frame)
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

# ── Illumination & Instrument FOV Intercepts ────────────────────────────────

def get_surface_illumination(target, et, lat, lon, frame=None, method="ELLIPSOID"):
    """Computes Phase, Solar Incidence, and Emission angles at an exact coordinate."""
    clean_target = str(target).strip().upper()
    
    if frame is None:
        frame = f"IAU_{clean_target}"
        
    radii = body_radii(clean_target).to(u.km).value
    re, rp = radii[0], radii[2]
    f = (re - rp) / re
    
    spoint = spice.georec(np.radians(lon), np.radians(lat), 0.0, re, f)
    et_arr = np.asarray(et, dtype=float)
    
    def _call(e):
        _, _, phase, solar_inc, emission = spice.ilumin(method, clean_target, e, frame, "LT+S", "SUN", spoint)
        return np.degrees([phase, solar_inc, emission])

    if et_arr.ndim > 0:
        angles = np.empty((et_arr.size, 3))
        for i, e in enumerate(et_arr.flat):
            angles[i] = _call(e)
        return angles.reshape(et_arr.shape + (3,)) * u.deg
    return _call(float(et)) * u.deg

def get_fov_intercept(inst_name, target, et, inst_frame, method="ELLIPSOID"):
    """Calculates ray-surface terrain footprint intercepts for an instrument boresight."""
    try:
        shape, frame, boresight, nbounds, bounds = spice.getfov(body_name_to_id(inst_name), 3)
    except Exception as e:
        boresight = bounds[0]
    except Exception as e:
        raise RuntimeError(
            f"Failed to extract attitude matrix for instrument {inst_name}.\n"
            f"Reason: Missing required CK (orientation), IK (instrument specs), or SCLK (clock) kernel.\n"
            f"Please ensure all relevant mission kernels are supplied via extra_paths."
        ) from e

    et_arr = np.asarray(et, dtype=float)
    if et_arr.ndim > 0:
        pts = np.empty((et_arr.size, 3))
        found_flags = np.empty(et_arr.size, dtype=bool)
        for i, e in enumerate(et_arr.flat):
            try:
                spoint, _, _, found = spice.sincpt(method, target.upper(), e, frame, "LT+S", inst_name, inst_frame, boresight)
                pts[i] = spoint
                found_flags[i] = found
            except Exception:
                pts[i] = [np.nan, np.nan, np.nan]
                found_flags[i] = False
        return pts.reshape(et_arr.shape + (3,)) * u.km, found_flags.reshape(et_arr.shape)
    spoint, _, _, found = spice.sincpt(method, target.upper(), float(et), frame, "LT+S", inst_name, inst_frame, boresight)
    return spoint * u.km, found
# ── Geometric Events Search Subsystem ────────────────────────────────────────

def check_occultation(target1, shape1, target2, shape2, observer, et, frame1=None, frame2=None):
    """Determines the exact occultation or eclipse state conditions at timestamp frames."""
    if frame1 is None:
        frame1 = f"IAU_{target1.upper()}"
    if frame2 is None:
        frame2 = f"IAU_{target2.upper()}"
        
    et_arr = np.asarray(et, dtype=float)
    if et_arr.ndim > 0:
        codes = np.empty(et_arr.size, dtype=int)
        for i, e in enumerate(et_arr.flat):
            codes[i] = spice.occult(target1.upper(), shape1, frame1, target2.upper(), shape2, frame2, "LT+S", observer.upper(), e)
        return codes.reshape(et_arr.shape)
    return spice.occult(target1.upper(), shape1, frame1, target2.upper(), shape2, frame2, "LT+S", observer.upper(), float(et))
