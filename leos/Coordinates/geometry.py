"""
leos.geometry
-------------
Geometry: a typed, validated input container describing an observer location
in the Solar System at a specific time. No physics lives here — this is
the input structure that every LEOS computation takes as its first argument.
"""

from dataclasses import dataclass, field
from typing import Optional
from astropy import units as u
from astropy.time import Time
from leos.units import ANGLE_UNIT, ELEVATION_UNIT


# Supported bodies — extended in Phase 3
SUPPORTED_BODIES = {
    "earth", "moon", "mars",           # Phase 1 (validated)
    "mercury", "venus",                # Phase 3
    "jupiter", "saturn", "uranus", "neptune",
    "europa", "ganymede", "callisto",
    "titan", "enceladus",
}


@dataclass
class Geometry:
    """
    Observer location and time in the Solar System.

    Parameters
    ----------
    body : str
        Planet or moon name, e.g. 'mars'. Case-insensitive.
    time : astropy Time
        Observation time, e.g. Time('2024-01-15T12:00:00', format='isot').
    latitude : astropy Quantity (angle)
        Planetocentric latitude in degrees. +90 = north pole.
    longitude : astropy Quantity (angle)
        Planetocentric east longitude in degrees.
    elevation : astropy Quantity (length), optional
        Surface elevation above reference ellipsoid. Defaults to 0 m.
    label : str, optional
        Human-readable label for this geometry.

    Examples
    --------
    >>> from astropy import units as u
    >>> from astropy.time import Time
    >>> from leos.geometry import Geometry
    >>> g = Geometry(
    ...     body='mars',
    ...     time=Time('2024-01-15T12:00:00', format='isot'),
    ...     latitude=18.4 * u.deg,
    ...     longitude=77.5 * u.deg,
    ...     elevation=0 * u.m,
    ...     label='Jezero Crater'
    ... )
    >>> print(g)
    Geometry: mars | Jezero Crater | 2024-01-15T12:00:00.000 | lat=18.4°, lon=77.5°, elev=0.0 m
    """

    body      : str
    time      : Time
    latitude  : u.Quantity
    longitude : u.Quantity
    elevation : u.Quantity = field(default_factory=lambda: 0.0 * ELEVATION_UNIT)
    label     : str = "Observer"

    def __post_init__(self):
        # ── Normalise body name ──────────────────────────────────────────────
        self.body = self.body.lower().strip()
        if self.body not in SUPPORTED_BODIES:
            raise ValueError(
                f"Body '{self.body}' not recognised. "
                f"Supported: {sorted(SUPPORTED_BODIES)}"
            )

        # ── Validate time ────────────────────────────────────────────────────
        if not isinstance(self.time, Time):
            raise TypeError("time must be an astropy Time object.")

        # ── Convert and validate angles ──────────────────────────────────────
        self.latitude  = self.latitude.to(ANGLE_UNIT)
        self.longitude = self.longitude.to(ANGLE_UNIT)
        self.elevation = self.elevation.to(ELEVATION_UNIT)

        if not (-90 <= self.latitude.value <= 90):
            raise ValueError(
                f"latitude must be between -90 and +90 deg. Got {self.latitude}."
            )
        if not (-360 <= self.longitude.value <= 360):
            raise ValueError(
                f"longitude must be between -360 and +360 deg. Got {self.longitude}."
            )

    # ── Representation ───────────────────────────────────────────────────────

    def __str__(self):
        return (
            f"Geometry: {self.body} | {self.label} | "
            f"{self.time.isot} | "
            f"lat={self.latitude:.1f}, "
            f"lon={self.longitude:.1f}, "
            f"elev={self.elevation:.1f}"
        )

    def __repr__(self):
        return (
            f"Geometry(body='{self.body}', time={self.time.isot}, "
            f"lat={self.latitude:.1f}, lon={self.longitude:.1f})"
        )

    # ── Utility ──────────────────────────────────────────────────────────────

    def to_spice_inputs(self):
        """
        Return a dict of raw values suitable for passing to SPICE calls.
        SpiceyPy functions expect bare floats and strings, not Quantities.
        """
        return {
            "body"     : self.body.upper(),
            "et"       : None,   # filled by spice_utils.utc_to_et(self.time)
            "latitude" : self.latitude.value,
            "longitude": self.longitude.value,
            "elevation": self.elevation.value,
            "time_isot": self.time.isot,
        }
