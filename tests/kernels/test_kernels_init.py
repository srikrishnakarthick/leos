"""
tests/test_kernels_init.py
--------------------------
Verifies structural composition, sorting, and exposures of the 
leos.kernels sub-package namespace API.
"""

import pytest
import types
import leos.kernels

def test_kernels_docstring_presence():
    """Ensure the kernels package level docstring exists and is populated."""
    assert leos.kernels.__doc__ is not None
    assert "Kernels Submodule Suite" in leos.kernels.__doc__


def test_kernels_public_api_completeness():
    """Verify explicit presence and object types of the core API contracts."""
    # 1. Hard contract guard: Protect against accidental deletions from __all__
    expected_core_api = {
        "DATA_DIRS",
        "KERNEL_ROOT",
        "calculate_local_md5",
        "fetch_kernels",
        "fetch_remote_md5s",
        "get_dynamic_ephemeris_urls",
    }
    actual_all = set(leos.kernels.__all__)
    
    missing_from_all = expected_core_api - actual_all
    assert not missing_from_all, f"Core components dropped from __all__: {missing_from_all}"

    # 2. Namespace exposure guard
    missing_attributes = [item for item in leos.kernels.__all__ if not hasattr(leos.kernels, item)]
    assert not missing_attributes, f"Items listed in __all__ missing from namespace: {missing_attributes}"

    # 3. Structural Type Guards: Ensure signatures don't silently mutate into strings/ints
    assert isinstance(leos.kernels.DATA_DIRS, dict), "DATA_DIRS must be a dictionary configuration."
    assert isinstance(leos.kernels.fetch_kernels, types.FunctionType), "fetch_kernels must remain a callable function."


def test_kernels_public_api_is_sorted():
    """Style guard checking if the sub-package exposure list is strictly alphabetical."""
    assert leos.kernels.__all__ == sorted(leos.kernels.__all__), (
        "leos.kernels.__all__ list should be strictly sorted alphabetically."
    )
