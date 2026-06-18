"""
tests/test_coordinates_init.py
------------------------------
Verifies structural composition, sorting, and exposures of the 
leos.Coordinates sub-package namespace API.
"""

import pytest
import leos.Coordinates

def test_coordinates_docstring_presence():
    """Ensure the coordinates package level docstring exists."""
    assert leos.Coordinates.__doc__ is not None
    assert "Coordinates and Astro-Geometry" in leos.Coordinates.__doc__


def test_coordinates_public_api_completeness():
    """Verify everything in __all__ exists in the Coordinates namespace."""
    missing_elements = [item for item in leos.Coordinates.__all__ if not hasattr(leos.Coordinates, item)]
    assert not missing_elements, f"Items listed in Coordinates.__all__ but missing: {missing_elements}"


def test_coordinates_public_api_is_sorted():
    """Style guard checking if the sub-package exposure list is strictly alphabetical."""
    assert leos.Coordinates.__all__ == sorted(leos.Coordinates.__all__), (
        "leos.Coordinates.__all__ list should be strictly sorted alphabetically."
    )
