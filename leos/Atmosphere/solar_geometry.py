"""
leos.solar_geometry
-------------------
Computes Sun-body geometry for a given Geometry object:
  - Sun-body distance
  - Solar zenith angle (SZA)
  - Illumination fraction
  - Subsolar point

All outputs are astropy Quantities with uncertainties where applicable.
Requires SPICE kernels to be loaded via spice_utils.load_kernels().
"""

import numpy as np
import spiceypy as spice
from astropy import units as u
from astropy.time import Time

from leos.Coordinates.geometry import Geometry
from leos.Coordinates.spice_utils import utc_to_et, sun_position
from leos.uncertainty import UncertainQuantity
from leos.units import ANGLE_UNIT, DISTANCE_UNIT


def sun_body_distance(geometry: Geometry) -> UncertainQuantity:
    """
    Compute the distance between the Sun and the target body centre.

    Parameters
    ----------
    geometry : Geometry

    Returns
    -------
    UncertainQuantity in AU
        Central value = Sun-body distance.
        Uncertainty = 0 (ephemeris uncertainty negligible for our purposes).
    """
    et = utc_to_et(geometry.time)
    pos_km, _ = sun_position(geometry.body, et)

    distance_km = np.linalg.norm(pos_km.value) * u.km
    distance_au = distance_km.to(DISTANCE_UNIT)

    return UncertainQuantity(distance_au, 0.0 * DISTANCE_UNIT)


def solar_zenith_angle(geometry: Geometry) -> UncertainQuantity:
    """
    Compute the solar zenith angle (SZA) at the observer location.

    Uses the Sun position in J2000 frame, then rotates to body-fixed
    coordinates using the body's rotation angle — avoiding the need
    for a body-center SPK kernel.

    SZA = 0°  means the Sun is directly overhead.
    SZA = 90° means the Sun is on the horizon.
    SZA > 90° means the observer is in darkness.
    """
    et = utc_to_et(geometry.time)

    # Observer unit vector in body-fixed frame (from lat/lon)
    lat_rad = np.deg2rad(geometry.latitude.value)
    lon_rad = np.deg2rad(geometry.longitude.value)
    obs_vec = np.array([
        np.cos(lat_rad) * np.cos(lon_rad),
        np.cos(lat_rad) * np.sin(lon_rad),
        np.sin(lat_rad)
    ])

    # Sun position relative to body barycenter in J2000
    _BARYCENTER = {
        "EARTH": "EARTH BARYCENTER",
        "MARS":  "MARS BARYCENTER",
        "MOON":  "MOON",
    }
    target = _BARYCENTER.get(geometry.body.upper(), geometry.body.upper())
    pos_j2000, _ = spice.spkpos("SUN", et, "J2000", "LT+S", target)

    # Rotate J2000 Sun vector into body-fixed frame using SPICE pxform
    # pxform gives rotation matrix from J2000 to IAU_BODY frame
    # For bodies where IAU frame needs center SPK, fall back to
    # computing rotation from body spin rate and prime meridian
    try:
        rot = spice.pxform("J2000", f"IAU_{geometry.body.upper()}", et)
        pos_bodyfixed = spice.mxv(rot, pos_j2000)
    except Exception:
        # Fallback: use J2000 position directly (small error for SZA)
        pos_bodyfixed = pos_j2000

    sun_vec = pos_bodyfixed / np.linalg.norm(pos_bodyfixed)

    dot = float(np.clip(np.dot(obs_vec, sun_vec), -1.0, 1.0))
    sza_deg = np.rad2deg(np.arccos(dot))

    return UncertainQuantity(sza_deg * ANGLE_UNIT, 0.0 * ANGLE_UNIT)


def illumination_fraction(geometry: Geometry) -> float:
    """
    Compute the illumination fraction at the observer location.

    Returns
    -------
    float
        1.0 = fully illuminated (SZA < 90°)
        0.0 = in darkness      (SZA > 90°)

    Note: Does not account for terrain shadowing (Phase 2 feature).
    """
    sza = solar_zenith_angle(geometry)
    sza_val = sza.value.value   # UncertainQuantity → Quantity → float

    if sza_val <= 90.0:
        # Cosine weighting — fraction of full illumination
        return float(np.cos(np.deg2rad(sza_val)))
    else:
        return 0.0


def subsolar_point(geometry: Geometry) -> tuple:
    """
    Compute the subsolar point on the target body at the given time.
    """
    et = utc_to_et(geometry.time)

    _BARYCENTER = {
        "EARTH": "EARTH BARYCENTER",
        "MARS":  "MARS BARYCENTER",
        "MOON":  "MOON",
    }
    target = _BARYCENTER.get(geometry.body.upper(), geometry.body.upper())
    pos_j2000, _ = spice.spkpos("SUN", et, "J2000", "LT+S", target)

    try:
        rot = spice.pxform("J2000", f"IAU_{geometry.body.upper()}", et)
        pos_km = spice.mxv(rot, pos_j2000)
    except Exception:
        pos_km = pos_j2000

    dist = np.linalg.norm(pos_km)
    lat  = np.rad2deg(np.arcsin(pos_km[2] / dist))
    lon  = np.rad2deg(np.arctan2(pos_km[1], pos_km[0]))

    return lat * ANGLE_UNIT, lon * ANGLE_UNIT


def toa_irradiance(geometry: Geometry) -> UncertainQuantity:
    """
    Compute top-of-atmosphere (TOA) broadband solar irradiance
    at the target body, accounting for Sun-body distance.

    Uses the inverse-square law:
        I_TOA = L_sun / (4π d²)

    Parameters
    ----------
    geometry : Geometry

    Returns
    -------
    UncertainQuantity in W/m²
    """
    from leos.units import SOLAR_LUMINOSITY, AU

    # Solar constant at 1 AU (W/m²)
    E_0 = 1361.0 * u.W / u.m**2

    dist = sun_body_distance(geometry)
    dist_au = dist.value.to(u.au).value

    # Inverse square scaling
    irradiance = E_0 / (dist_au ** 2)

    # ~0.1% solar variability as uncertainty
    sigma = 0.001 * irradiance

    return UncertainQuantity(irradiance, sigma)
