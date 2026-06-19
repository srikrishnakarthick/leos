import os
import unittest
import tempfile
import shutil
import hashlib
from unittest.mock import patch, MagicMock
import importlib.machinery
import importlib.util

# Absolute file-path import to completely bypass package/init namespace shadowing
MODULE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../leos/kernels/fetch_kernels.py"))
loader = importlib.machinery.SourceFileLoader("fetch_kernels_module", MODULE_PATH)
spec = importlib.util.spec_from_loader("fetch_kernels_module", loader)
fk = importlib.util.module_from_spec(spec)
loader.exec_module(fk)

class TestFetchKernelsPipeline(unittest.TestCase):
    # Keep the rest of your TestFetchKernelsPipeline class code completely identical!

    def setUp(self):
        # Create an isolated temporary directory for every test run
        self.test_dir = tempfile.mkdtemp()
        
        # FIX: Directly patch the dictionary that the script uses for routing files
        self.saved_generic_dir = fk.DATA_DIRS["generic"]
        fk.DATA_DIRS["generic"] = self.test_dir

        self.dummy_content = b"Mock SPICE Data Footprint"
        self.valid_md5 = hashlib.md5(self.dummy_content).hexdigest()

        # Build dynamic manifest string where ALL assets perfectly match the data hash
        self.mock_checksums_txt = (
            f"{self.valid_md5}  de442.bsp\n"
            f"{self.valid_md5}  de442_tech-comments.txt\n"
            f"{self.valid_md5}  naif0012.tls\n"
            f"{self.valid_md5}  pck00011.tpc\n"
            f"{self.valid_md5}  mars_iau2000_v1.tpc\n"
        )

    def tearDown(self):
        # Safely sweep away the temporary testing environment
        shutil.rmtree(self.test_dir)
        # FIX: Restore the original production environment directory configuration
        fk.DATA_DIRS["generic"] = self.saved_generic_dir

    def helper_create_local_file(self, filename, content):
        """Helper to inject controlled files into the sandboxed test workspace."""
        filepath = os.path.join(self.test_dir, filename)
        with open(filepath, "wb") as f:
            f.write(content)
        return filepath

    def test_calculate_local_md5(self):
        """Verify the chunked local MD5 hashing utility functions exactly as expected."""
        filepath = self.helper_create_local_file("test_calc.bsp", self.dummy_content)
        computed_hash = fk.calculate_local_md5(filepath)
        self.assertEqual(computed_hash, self.valid_md5)

    @patch("leos.kernels.fetch_kernels.requests.get")
    def test_fetch_kernels_cold_start(self, mock_get):
        """Scenario A: Verify complete sequential download and verification when directory is empty."""
        mock_manifest_resp = MagicMock()
        mock_manifest_resp.text = self.mock_checksums_txt
        mock_manifest_resp.raise_for_status = MagicMock()
        
        mock_file_resp = MagicMock()
        mock_file_resp.iter_content = lambda chunk_size: [self.dummy_content]
        mock_file_resp.raise_for_status = MagicMock()

        # Chain behaviors for sequential requests.get calls
        mock_get.side_effect = [mock_manifest_resp] + [mock_file_resp] * 10

        fk.fetch_kernels()

        # Check that files were created down in the sandboxed target path
        queue = fk.get_dynamic_ephemeris_urls()
        for filename in fk.STATIC_KERNELS.keys():
            queue[filename] = fk.STATIC_KERNELS[filename]
            
        for filename in queue.keys():
            self.assertTrue(os.path.exists(os.path.join(self.test_dir, filename)))

    @patch("leos.kernels.fetch_kernels.requests.get")
    def test_fetch_kernels_warm_start(self, mock_get):
        """Scenario B: Smart caching skip checks function properly when valid assets exist."""
        mock_manifest_resp = MagicMock()
        mock_manifest_resp.text = self.mock_checksums_txt
        mock_get.return_value = mock_manifest_resp

        # Pre-seed the sandboxed directory with correct intact assets
        queue = fk.get_dynamic_ephemeris_urls()
        for filename in fk.STATIC_KERNELS.keys():
            queue[filename] = fk.STATIC_KERNELS[filename]
            
        for filename in queue.keys():
            self.helper_create_local_file(filename, self.dummy_content)

        fk.fetch_kernels()
            
        # Caching check means get was only called ONCE total (manifest file lookup only)
        self.assertEqual(mock_get.call_count, 1)

    @patch("leos.kernels.fetch_kernels.requests.get")
    def test_fetch_kernels_corruption_handling_and_recovery(self, mock_get):
        """Verify that a corrupted local file (MD5 mismatch) triggers a dynamic repair download."""
        mock_manifest_resp = MagicMock()
        mock_manifest_resp.text = self.mock_checksums_txt
        
        mock_file_resp = MagicMock()
        mock_file_resp.iter_content = lambda chunk_size: [self.dummy_content]
        
        mock_get.side_effect = [mock_manifest_resp] + [mock_file_resp] * 10

        # Pre-seed files with invalid text data to trigger integrity mismatches
        queue = fk.get_dynamic_ephemeris_urls()
        for filename in fk.STATIC_KERNELS.keys():
            queue[filename] = fk.STATIC_KERNELS[filename]
            
        for filename in queue.keys():
            self.helper_create_local_file(filename, b"Corrupted Data")

        fk.fetch_kernels()
        
        # Verify it went past the manifest to perform restorative downloads
        self.assertTrue(mock_get.call_count > 1)

    @patch("leos.kernels.fetch_kernels.requests.get")
    def test_fetch_kernels_raises_value_error_on_bad_download(self, mock_get):
        """Ensure an explicit ValueError is raised if a download finishes but the hash is still wrong."""
        mock_manifest_resp = MagicMock()
        mock_manifest_resp.text = self.mock_checksums_txt
        
        mock_file_resp = MagicMock()
        mock_file_resp.iter_content = lambda chunk_size: [b"Defective download stream content"]
        
        mock_get.side_effect = [mock_manifest_resp, mock_file_resp]

        # Force a failure because downloaded stream doesn't match the manifest expectations
        with self.assertRaises(ValueError):
            fk.fetch_kernels()

if __name__ == "__main__":
    unittest.main()
