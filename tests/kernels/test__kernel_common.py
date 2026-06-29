"""
tests/kernels/test__kernel_common.py

Tests for leos.kernels._kernel_common.
"""
import hashlib
import os

import pytest
from astropy.time import Time

from leos.kernels import _kernel_common as kc


class TestConstants:
    def test_kernel_root_paths_are_consistent(self):
        assert kc._DEFAULT_KERNEL_ROOT == kc.KERNEL_ROOT
        assert kc._CMT_CACHE_DIR == os.path.join(kc.KERNEL_ROOT, "_cmt_cache")

    def test_naif_subdirs_have_trailing_slash(self):
        for key, path in kc._NAIF_SUBDIRS.items():
            assert path.endswith("/"), f"{key} subdir should end with '/'"

    def test_checksum_manifest_urls_built_from_subdirs(self):
        for subdir, path in kc._NAIF_SUBDIRS.items():
            expected = kc._NAIF_BASE + path + "aa_checksums.txt"
            assert kc._CHECKSUM_MANIFEST_URL[subdir] == expected

    def test_subdirs_with_checksums_is_subset_of_naif_subdirs(self):
        assert kc._SUBDIRS_WITH_CHECKSUMS <= set(kc._NAIF_SUBDIRS.keys())


class TestTimeHelpers:
    def test_to_time_or_none_with_none(self):
        assert kc._to_time_or_none(None) is None

    def test_to_time_or_none_with_time_instance_returns_same_object(self):
        t = Time("2020-01-01")
        assert kc._to_time_or_none(t) is t

    def test_to_time_or_none_with_string(self):
        result = kc._to_time_or_none("2020-01-01")
        assert isinstance(result, Time)
        assert result.isot.startswith("2020-01-01")

    def test_normalize_window_no_args_returns_none_none(self):
        assert kc._normalize_window() == (None, None)

    def test_normalize_window_with_time_range(self):
        lo, hi = kc._normalize_window(time_range=("2020-01-01", "2021-01-01"))
        assert isinstance(lo, Time) and isinstance(hi, Time)
        assert lo < hi

    def test_normalize_window_with_single_time_sets_lo_eq_hi(self):
        lo, hi = kc._normalize_window(time="2020-06-01")
        assert lo == hi
        assert isinstance(lo, Time)

    def test_normalize_window_time_range_takes_priority_over_time(self):
        lo, hi = kc._normalize_window(
            time="2020-06-01", time_range=("2020-01-01", "2021-01-01")
        )
        assert lo == Time("2020-01-01")
        assert hi == Time("2021-01-01")

    def test_window_contains_both_bounds_none_always_true(self):
        assert kc._window_contains(Time("2020-01-01"), Time("2020-01-01"), None, None)

    def test_window_contains_request_within_coverage(self):
        assert kc._window_contains(
            Time("2020-06-01"), Time("2020-06-01"), "2000-01-01", "2050-01-01"
        )

    def test_window_contains_request_starts_before_coverage(self):
        assert not kc._window_contains(
            Time("1999-01-01"), Time("2020-01-01"), "2000-01-01", "2050-01-01"
        )

    def test_window_contains_request_ends_after_coverage(self):
        assert not kc._window_contains(
            Time("2020-01-01"), Time("2060-01-01"), "2000-01-01", "2050-01-01"
        )

    def test_window_contains_no_request_bounds_is_true(self):
        assert kc._window_contains(None, None, "2000-01-01", "2050-01-01")


class TestSelectTimeFilteredKernels:
    def test_unbounded_entries_always_included_regardless_of_time(self):
        entries = [("a.tls", "lsk", None, None), ("b.tpc", "pck", None, None)]
        result = kc._select_time_filtered_kernels(entries, time="2020-01-01")
        assert result == [("a.tls", "lsk"), ("b.tpc", "pck")]

    def test_unbounded_entries_included_with_no_time_request(self):
        entries = [("a.tls", "lsk", None, None)]
        result = kc._select_time_filtered_kernels(entries)
        assert result == [("a.tls", "lsk")]

    def test_bounded_entry_included_when_no_request_given(self):
        entries = [("de442.bsp", "spk_planets", "1549-12-31", "2650-01-25")]
        result = kc._select_time_filtered_kernels(entries)
        assert result == [("de442.bsp", "spk_planets")]

    def test_bounded_entry_included_when_request_matches(self):
        entries = [("de442.bsp", "spk_planets", "1549-12-31", "2650-01-25")]
        result = kc._select_time_filtered_kernels(entries, time="2020-01-01")
        assert result == [("de442.bsp", "spk_planets")]

    def test_bounded_entry_excluded_and_raises_when_request_outside_window(self):
        entries = [("de442.bsp", "spk_planets", "1549-12-31", "2650-01-25")]
        with pytest.raises(ValueError, match="No registered kernel"):
            kc._select_time_filtered_kernels(entries, time="3000-01-01")

    def test_mixed_bounded_and_unbounded_raises_if_bounded_never_matches(self):
        entries = [
            ("always.tls", "lsk", None, None),
            ("narrow.bsp", "spk_planets", "1990-01-01", "1991-01-01"),
        ]
        with pytest.raises(ValueError):
            kc._select_time_filtered_kernels(entries, time="2050-01-01")

    def test_mixed_bounded_and_unbounded_succeeds_when_bounded_matches(self):
        entries = [
            ("always.tls", "lsk", None, None),
            ("wide.bsp", "spk_planets", "1900-01-01", "2100-01-01"),
        ]
        result = kc._select_time_filtered_kernels(entries, time="2050-01-01")
        assert ("always.tls", "lsk") in result
        assert ("wide.bsp", "spk_planets") in result

    def test_context_label_appears_in_error_message(self):
        entries = [("x.bsp", "spk_planets", "1990-01-01", "1991-01-01")]
        with pytest.raises(ValueError, match="for 'MARS'"):
            kc._select_time_filtered_kernels(entries, time="2050-01-01", context_label="for 'MARS'")

    def test_time_range_filters_entries(self):
        entries = [
            ("old.bsp", "spk_planets", "1900-01-01", "1950-01-01"),
            ("new.bsp", "spk_planets", "1990-01-01", "2050-01-01"),
        ]
        result = kc._select_time_filtered_kernels(
            entries, time_range=("2000-01-01", "2010-01-01")
        )
        assert result == [("new.bsp", "spk_planets")]

    def test_prefers_shortest_timespan_coverage_fit(self):
        """Verify that out of multiple windows matching a timeframe, the tightest wins."""
        entries = [
            ("mar099.bsp", "spk_satellites", "1600-01-01", "2600-01-01"),
            ("mar099s.bsp", "spk_satellites", "1995-01-01", "2050-01-01"),
        ]
        result = kc._select_time_filtered_kernels(entries, time="2026-01-01")
        assert result == [("mar099s.bsp", "spk_satellites")]


class TestCitations:
    def setup_method(self):
        kc.reset_citations()

    def teardown_method(self):
        kc.reset_citations()

    def test_starts_empty_after_reset(self):
        assert kc.get_citations()["kernels"] == []

    def test_log_citation_appends_entry(self):
        kc._log_citation("naif0012.tls", "http://example.com/naif0012.tls", "leap seconds")
        citations = kc.get_citations()
        assert len(citations["kernels"]) == 1
        entry = citations["kernels"][0]
        assert entry == {
            "filename": "naif0012.tls",
            "url": "http://example.com/naif0012.tls",
            "context": "leap seconds",
        }

    def test_toolkit_citations_always_present(self):
        citations = kc.get_citations()
        assert kc._SPICE_CITATION in citations["toolkit"]
        assert kc._SPICEYPY_CITATION in citations["toolkit"]


class TestFetchRemoteMd5s:
    def test_parses_well_formed_manifest(self, monkeypatch):
        manifest_text = (
            "d41d8cd98f00b204e9800998ecf8427e  FILE_ONE.bsp\n"
            "0CC175B9C0F1B6A831C399E269772661  file_two.bsp\n"
        )

        class FakeResponse:
            text = manifest_text
            def raise_for_status(self): pass

        monkeypatch.setattr(kc.requests, "get", lambda url, timeout=10: FakeResponse())
        result = kc.fetch_remote_md5s("spk_satellites")
        assert result == {
            "file_one.bsp": "d41d8cd98f00b204e9800998ecf8427e",
            "file_two.bsp": "0cc175b9c0f1b6a831c399e269772661",
        }


class TestCalculateLocalMd5:
    def test_matches_hashlib_reference(self, tmp_path):
        f = tmp_path / "sample.bin"
        content = b"some kernel-ish binary content" * 1000
        f.write_bytes(content)
        expected = hashlib.md5(content).hexdigest()
        assert kc.calculate_local_md5(str(f)) == expected


class TestInferSubdir:
    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("naif0012.tls", "lsk"),
            ("pck00011.tpc", "pck"),
            ("de442.bsp", "spk_planets"),
            ("mar099.bsp", "spk_satellites"),
        ],
    )
    def test_known_extensions_map_correctly(self, filename, expected):
        assert kc._infer_subdir(filename) == expected
# Force-sync timestamp update
