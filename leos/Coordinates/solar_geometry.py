"""
leos.solar_geometry
-------------------
Computes Sun-body geometry for a given Geometry object:
  - Sun-body distance
  - Solar zenith angle (SZA)
  - Illumination fraction
  - Subsolar point
  - Top-of-Atmosphere (TOA) Irradiance

All outputs are astropy Quantities wrapped in UncertainQuantity structures where applicable.
Requires SPICE kernels to be loaded via spice_utils.load_kernels().
"""

import numpy as np
import spiceypy as spice
from astropy import units as u

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
    """
    et = utc_to_et(geometry.time)
    pos_km, _ = sun_position(geometry.body, et)

    distance_km = np.linalg.norm(pos_km.value) * u.km
    distance_au = distance_km.to(DISTANCE_UNIT)

    return UncertainQuantity(distance_au, 0.0 * DISTANCE_UNIT)


def solar_zenith_angle(geometry: Geometry) -> UncertainQuantity:
    """
    Compute the solar zenith angle (SZA) at the observer location.
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

    # Map to Barycenter for position lookup to ensure compatibility with standard DE4xx kernels
    body_name = geometry.body.upper()
    if body_name == "EARTH":
        target = "EARTH BARYCENTER"
    elif body_name == "MARS":
        target = "MARS BARYCENTER"
    else:
        target = body_name

    # Sun position relative to body barycenter in J2000
    pos_j2000, _ = spice.spkpos("SUN", et, "J2000", "LT+S", target)

    # Rotate J2000 Sun vector into body-fixed frame using the surface frame name
    frame_name = f"IAU_{body_name}"
    try:
        rot = spice.pxform("J2000", frame_name, et)
        pos_bodyfixed = spice.mxv(rot, pos_j2000)
    except Exception:
        pos_bodyfixed = pos_j2000

    sun_vec = pos_bodyfixed / np.linalg.norm(pos_bodyfixed)

    dot = float(np.clip(np.dot(obs_vec, sun_vec), -1.0, 1.0))
    sza_deg = np.rad2deg(np.arccos(dot))

    return UncertainQuantity(sza_deg * ANGLE_UNIT, 0.0 * ANGLE_UNIT)


def illumination_fraction(geometry: Geometry) -> float:
    """
    Compute the illumination fraction at the observer location.
    """
    sza = solar_zenith_angle(geometry)
    sza_val = sza.value.to(u.deg).value

    if sza_val <= 90.0:
        return float(np.cos(np.deg2rad(sza_val)))
    else:
        return 0.0


def subsolar_point(geometry: Geometry) -> tuple:
    """
    Compute the subsolar point on the target body at the given time.
    """
    et = utc_to_et(geometry.time)
    body_name = geometry.body.upper()
    
    if body_name == "EARTH":
        target = "EARTH BARYCENTER"
    elif body_name == "MARS":
        target = "MARS BARYCENTER"
    else:
        target = body_name

    pos_j2000, _ = spice.spkpos("SUN", et, "J2000", "LT+S", target)

    frame_name = f"IAU_{body_name}"
    try:
        rot = spice.pxform("J2000", frame_name, et)
        pos_km = spice.mxv(rot, pos_j2000)
    except Exception:
        pos_km = pos_j2000

    dist = np.linalg.norm(pos_km)
    lat  = np.rad2deg(np.arcsin(pos_km[2] / dist))
    lon  = np.rad2deg(np.arctan2(pos_km[1], pos_km[0]))

    return lat * ANGLE_UNIT, lon * ANGLE_UNIT


def toa_irradiance(geometry: Geometry) -> UncertainQuantity:
    """
    Compute top-of-atmosphere (TOA) broadband solar irradiance.
    """
    E_0 = 1361.0 * u.W / u.m**2

    dist = sun_body_distance(geometry)
    dist_au = dist.value.to(u.au).value

    irradiance = E_0 / (dist_au ** 2)
    sigma = 0.001 * irradiance

    return UncertainQuantity(irradiance, sigma)
