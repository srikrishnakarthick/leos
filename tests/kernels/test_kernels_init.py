"""
tests/test_kernels_init.py
--------------------------
Verifies structural composition, sorting, and exposures of the 
leos.kernels sub-package namespace API.
"""

import pytest
import leos.kernels

def test_kernels_docstring_presence():
    """Ensure the kernels package level docstring exists."""
    assert leos.kernels.__doc__ is not None
    assert "Kernels Submodule Suite" in leos.kernels.__doc__


def test_kernels_public_api_completeness():
    """Verify everything in __all__ exists in the kernels namespace."""
    missing_elements = [item for item in leos.kernels.__all__ if not hasattr(leos.kernels, item)]
    assert not missing_elements, f"Items listed in kernels.__all__ but missing: {missing_elements}"


def test_kernels_public_api_is_sorted():
    """Style guard checking if the sub-package exposure list is strictly alphabetical."""
    assert leos.kernels.__all__ == sorted(leos.kernels.__all__), (
        "leos.kernels.__all__ list should be strictly sorted alphabetically."
    )
