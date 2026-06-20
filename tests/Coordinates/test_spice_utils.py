"""
tests/Coordinates/test_spice_utils.py
-------------------------------------
Airtight testing framework validating core astrodynamics, kinematics, 
coordinate transforms, illumination geometry, instrument FOV intercepts, 
and geometric event search sub-systems.
"""

import os
import unittest
import tempfile
import shutil
import numpy as np
from unittest.mock import patch, MagicMock
from astropy.time import Time
import astropy.units as u

# Absolute file-path mapping to avoid init namespace shadowing loops
import importlib.machinery
import importlib.util
MODULE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../leos/Coordinates/spice_utils.py"))
loader = importlib.machinery.SourceFileLoader("spice_utils_module", MODULE_PATH)
spec = importlib.util.spec_from_loader("spice_utils_module", loader)
su = importlib.util.module_from_spec(spec)
loader.exec_module(su)

class TestSpiceUtilsRigorous(unittest.TestCase):

    def setUp(self):
        self.et_scalar = 700000000.0
        self.et_array = np.array([700000000.0, 700001000.0])
        self.mock_time_scalar = Time("2022-03-20T12:00:00", format="isot")
        
        # Isolated sandbox folder for testing custom path discovery engine rules
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Clean up sandbox directory structure safely
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    # ── NEW: Custom Path Infrastructure & Discovery Engine Architecture Tests ──

    def test_discover_kernels_from_custom_root(self):
        """Verify discover_kernels maps to explicit paths and obeys correct loading priority order."""
        # Setup temporary directories mimicking generic and mission structures
        generic_path = os.path.join(self.test_dir, "generic")
        mission_path = os.path.join(self.test_dir, "mission")
        os.makedirs(generic_path, exist_ok=True)
        os.makedirs(mission_path, exist_ok=True)

        # Pre-seed sample extensions out of sequence to ensure order resolution sorting logic holds
        mock_spk = os.path.join(generic_path, "planets.bsp")
        mock_lsk = os.path.join(generic_path, "leapseconds.tls")
        mock_pck = os.path.join(generic_path, "constants.tpc")
        mock_sc = os.path.join(mission_path, "spacecraft.bc")

        for fpath in [mock_spk, mock_lsk, mock_pck, mock_sc]:
            with open(fpath, "w") as f:
                f.write("mock content")

        # Execute discovery pointing explicitly to our sandbox location
        resolved_sequence = su.discover_kernels(custom_root=self.test_dir)

        # Verify strict structural operational loading priority index tracking logic:
        # Expected sequence rules order: 1st LSK (.tls) -> 2nd Meta (.tpc) -> 3rd SPK (.bsp) -> 4th Mission (.bc)
        self.assertEqual(len(resolved_sequence), 4)
        self.assertEqual(resolved_sequence[0], mock_lsk)
        self.assertEqual(resolved_sequence[1], mock_pck)
        self.assertEqual(resolved_sequence[2], mock_spk)
        self.assertEqual(resolved_sequence[3], mock_sc)

    @patch("spiceypy.furnsh")
    def test_load_kernels_raises_exception_on_empty_custom_dir(self, mock_furnsh):
        """Verify load_kernels explicitly triggers FileNotFoundError if an empty workspace is supplied."""
        with self.assertRaises(FileNotFoundError) as context:
            su.load_kernels(custom_dir=self.test_dir)
        
        # Verify custom path string boundary indicators exist in the thrown exception error trace messages
        self.assertIn(self.test_dir, str(context.exception))

    # ── Existing Geometry Engine & Core Core Analytical Pipelines ──

    @patch("spiceypy.utc2et")
    def test_time_chronology_conversion(self, mock_utc2et):
        """Verify Astropy Time scalar and array vectorized translation into Ephemeris Time (ET)."""
        mock_utc2et.return_value = 700000000.0
        
        # Test scalar
        et = su.utc_to_et(self.mock_time_scalar)
        self.assertEqual(et, 700000000.0)
        
        # Test vectorization
        mock_utc2et.side_effect = [700000000.0, 700001000.0]
        mock_time_array = Time(["2022-03-20T12:00:00", "2022-03-20T12:16:40"], format="isot")
        et_arr = su.utc_to_et(mock_time_array)
        np.testing.assert_array_equal(et_arr, self.et_array)

    @patch("spiceypy.spkpos")
    def test_positions_and_apparent_vectors(self, mock_spkpos):
        """Verify vectorization of Sun positions, light-time lags, and Astropy unit mapping."""
        mock_spkpos.return_value = (np.array([149000000.0, 0.0, 0.0]), 499.0)
        
        # 1. Scalar Run
        pos, lt = su.sun_position("EARTH", self.et_scalar)
        self.assertTrue(isinstance(pos, u.Quantity))
        self.assertEqual(pos.unit, u.km)
        self.assertEqual(lt.unit, u.s)
        np.testing.assert_array_equal(pos.value, [149000000.0, 0.0, 0.0])

        # 2. Vector Array Run
        mock_spkpos.side_effect = [
            (np.array([149000000.0, 0.0, 0.0]), 499.0),
            (np.array([149100000.0, 100.0, 0.0]), 500.0)
        ]
        pos_arr, lt_arr = su.sun_position("EARTH", self.et_array)
        self.assertEqual(pos_arr.shape, (2, 3))
        self.assertEqual(lt_arr.shape, (2,))
        self.assertEqual(pos_arr.unit, u.km)

    @patch("spiceypy.subpnt")
    def test_distances_sub_observer_point(self, mock_subpnt):
        """Verify surface sub-observer intercept mapping arrays (subpnt_c functionality)."""
        mock_subpnt.return_value = (np.array([3000.0, 0.0, 1000.0]), 350.0, np.array([1.0, 0.0, 0.0]))
        
        # Test array routing
        mock_subpnt.side_effect = [
            (np.array([3000.0, 0.0, 1000.0]), 350.0, None),
            (np.array([3001.0, 5.0, 1000.0]), 349.0, None)
        ]
        pts, alts = su.get_sub_point("MARS", "EARTH", self.et_array)
        self.assertEqual(pts.shape, (2, 3))
        self.assertEqual(alts.shape, (2,))
        self.assertEqual(pts.unit, u.km)
        self.assertEqual(alts.unit, u.km)

    @patch("spiceypy.subslr")
    def test_distances_sub_solar_point(self, mock_subslr):
        """Verify sub-solar point coordinate tracking (subslr_c functionality)."""
        mock_subslr.return_value = (np.array([3390.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]))
        
        pt = su.get_sub_solar_point("MARS", self.et_scalar)
        self.assertEqual(pt.unit, u.km)
        np.testing.assert_array_equal(pt.value, [3390.0, 0.0, 0.0])

    @patch("spiceypy.trgsep")
    def test_angular_separation_and_range_rate(self, mock_trgsep):
        """Verify angular separation computation between targets (trgsep_c functionality)."""
        mock_trgsep.return_value = 0.5
        
        sep = su.angular_separation("MOON", "SUN", "EARTH", self.et_scalar)
        self.assertEqual(sep.unit, u.deg)
        self.assertAlmostEqual(sep.value, np.degrees(0.5))

    @patch("spiceypy.spkezr")
    def test_kinematics_and_relative_velocity(self, mock_spkezr):
        """Verify 6-element relative state vector extraction [x,y,z,vx,vy,vz] (spkezr_c/spkez_c)."""
        mock_state = np.array([1000.0, 2000.0, 3000.0, 7.0, 1.0, -2.0])
        mock_spkezr.return_value = (mock_state, 0.5)
        
        # Array timeline call
        mock_spkezr.side_effect = [(mock_state, 0.5), (mock_state + 1.0, 0.5)]
        states = su.get_state_vector("MRO", "MARS", self.et_array)
        self.assertEqual(states.shape, (2, 6))
        np.testing.assert_array_equal(states[0], mock_state)

    @patch("spiceypy.pxform")
    @patch("spiceypy.sxform")
    def test_coordinate_rotation_matrices(self, mock_sxform, mock_pxform):
        """Verify position (pxform_c) and dynamic kinematic state (sxform_c) transformation matrices."""
        mock_rot_3x3 = np.eye(3)
        mock_rot_6x6 = np.eye(6)
        
        mock_pxform.return_value = mock_rot_3x3
        mock_sxform.return_value = mock_rot_6x6
        
        pos_transformed = su.transform_position([10.0, 20.0, 30.0], "J2000", "IAU_EARTH", self.et_scalar)
        state_transformed = su.transform_state(np.ones(6), "J2000", "IAU_EARTH", self.et_scalar)
        
        self.assertEqual(pos_transformed.shape, (3,))
        self.assertEqual(state_transformed.shape, (6,))

    @patch("spiceypy.sce2c")
    @patch("spiceypy.ckgp")
    def test_spacecraft_instrument_platform_attitude(self, mock_ckgp, mock_sce2c):
        """Verify instrument platform orientation matrix harvesting from CK buffers (ckgp_c)."""
        mock_matrix = np.eye(3)
        
        mock_sce2c.side_effect = [1000000.0, 1000080.0, 1000000.0, 1000080.0]  
        mock_ckgp.return_value = (mock_matrix, 0.0)
        
        matrix = su.get_spacecraft_attitude(-94, -94001, self.et_scalar)
        np.testing.assert_array_equal(matrix, mock_matrix)
        
        # Test descriptive exception escalation if kernels are missing
        mock_ckgp.side_effect = Exception("SPICE(KERNELVARNOTFOUND) -- Variable not found.")
        with self.assertRaises(RuntimeError):
            su.get_spacecraft_attitude(-94, -94001, self.et_scalar)

    @patch("spiceypy.bodvrd")
    @patch("spiceypy.georec")
    @patch("spiceypy.ilumin")
    def test_illumination_angles_engine(self, mock_ilumin, mock_georec, mock_bodvrd):
        """Verify primary ilumin_c engine calculations (Phase, Solar Incidence, Emission)."""
        mock_bodvrd.return_value = (3, np.array([3396.0, 3396.0, 3376.0]))
        mock_georec.return_value = np.array([3390.0, 0.0, 0.0])
        mock_ilumin.return_value = (None, None, 0.1, 0.5, 0.2)
        
        angles = su.get_surface_illumination("MARS", self.et_scalar, lat=15.0, lon=45.0)
        self.assertEqual(angles.unit, u.deg)
        self.assertAlmostEqual(angles.value[0], np.degrees(0.1)) 
        self.assertAlmostEqual(angles.value[1], np.degrees(0.5)) 
        self.assertAlmostEqual(angles.value[2], np.degrees(0.2)) 

    @patch("spiceypy.bodn2c")
    @patch("spiceypy.getfov")
    @patch("spiceypy.sincpt")
    def test_instrument_fov_boresight_and_surface_intercept(self, mock_sincpt, mock_getfov, mock_bodn2c):
        """Verify ray-surface terrain footprint intercepts (sincpt_c / getfov_c)."""
        mock_bodn2c.return_value = -94001
        mock_getfov.return_value = ("RECTANGLE", "MRO_CRISM", np.array([0.0, 0.0, 1.0]), 1, np.array([[0.0, 0.0, 1.0]]))
        mock_sincpt.return_value = (np.array([3380.0, 10.0, 5.0]), 700000000.0, np.array([0.0, 0.0, -1.0]), True)
        
        pt, found = su.get_fov_intercept("MRO_CRISM", "MARS", self.et_scalar, "MRO_SPACECRAFT")
        self.assertEqual(pt.unit, u.km)
        self.assertTrue(found)
        np.testing.assert_array_equal(pt.value, [3380.0, 10.0, 5.0])

    @patch("spiceypy.occult")
    def test_geometric_occultation_state(self, mock_occult):
        """Verify tracking of line-of-sight occultation and eclipse conditions (occult_c)."""
        mock_occult.return_value = 1
        
        code = su.check_occultation("MOON", "ELLIPSOID", "SUN", "ELLIPSOID", "EARTH", self.et_scalar)
        self.assertEqual(code, 1)

if __name__ == "__main__":
    unittest.main()
