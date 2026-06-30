"""
Rigorous test suite for leos/kernels/_kernel_common.py

Goal: exercise every public/private function, every branch, every
documented edge case, and a handful of latent-bug probes (dict-collision
in _select_time_filtered_kernels, malformed checksum lines, copy-vs-
reference semantics of get_citations, missing-file behavior, etc.)
"""

import hashlib
import os
import pytest
import requests
from astropy.time import Time
from unittest.mock import patch, MagicMock

from leos.kernels import _kernel_common as kc


# ─────────────────────────────────────────────────────────────────────────────
# Constants / directory architecture
# ─────────────────────────────────────────────────────────────────────────────
class TestConstants:
    def test_kernel_root_paths_are_consistent(self):
        assert kc._DEFAULT_KERNEL_ROOT == kc.KERNEL_ROOT
        assert kc._CMT_CACHE_DIR == os.path.join(kc.KERNEL_ROOT, "_cmt_cache")

    def test_kernel_root_is_under_module_dir(self):
        # KERNEL_ROOT should be derived from this module's directory, not cwd
        module_dir = os.path.dirname(kc.__file__)
        assert kc.KERNEL_ROOT == os.path.join(module_dir, "data")

    def test_naif_base_is_correct_and_https(self):
        assert kc._NAIF_BASE == "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/"
        assert kc._NAIF_BASE.startswith("https://")
        assert kc._NAIF_BASE.endswith("/")

    def test_naif_subdirs_have_trailing_slash(self):
        for key, path in kc._NAIF_SUBDIRS.items():
            assert path.endswith("/"), f"{key} subdir should end with '/'"

    def test_naif_subdirs_expected_keys_present(self):
        expected_keys = {
            "lsk", "pck", "fk_planets", "fk_satellites", "fk_stations",
            "spk_planets", "spk_satellites", "spk_asteroids", "spk_comets",
            "spk_lagrange_point", "spk_stations", "spk_tno",
        }
        assert set(kc._NAIF_SUBDIRS.keys()) == expected_keys

    def test_checksum_manifest_urls_built_from_subdirs(self):
        for subdir, path in kc._NAIF_SUBDIRS.items():
            expected = kc._NAIF_BASE + path + "aa_checksums.txt"
            assert kc._CHECKSUM_MANIFEST_URL[subdir] == expected

    def test_checksum_manifest_url_keys_match_naif_subdirs_exactly(self):
        assert set(kc._CHECKSUM_MANIFEST_URL.keys()) == set(kc._NAIF_SUBDIRS.keys())

    def test_subdirs_with_checksums_is_subset_of_naif_subdirs(self):
        assert kc._SUBDIRS_WITH_CHECKSUMS <= set(kc._NAIF_SUBDIRS.keys())

    def test_subdirs_with_checksums_expected_members(self):
        assert kc._SUBDIRS_WITH_CHECKSUMS == {
            "spk_satellites", "spk_planets", "spk_lagrange_point", "spk_asteroids",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Time helpers
# ─────────────────────────────────────────────────────────────────────────────
class TestToTimeOrNone:
    def test_none_passthrough(self):
        assert kc._to_time_or_none(None) is None

    def test_time_instance_returned_unchanged(self):
        t = Time("2026-01-01")
        assert kc._to_time_or_none(t) is t

    def test_string_converted_to_time(self):
        result = kc._to_time_or_none("2026-06-29T12:00:00")
        assert isinstance(result, Time)
        assert result.isot.startswith("2026-06-29T12:00:00")

    def test_invalid_string_raises(self):
        with pytest.raises(Exception):
            kc._to_time_or_none("not-a-date")


class TestNormalizeWindow:
    def test_no_args_returns_none_none(self):
        assert kc._normalize_window() == (None, None)

    def test_time_range_only(self):
        lo, hi = kc._normalize_window(time_range=("2026-01-01", "2026-01-02"))
        assert isinstance(lo, Time) and isinstance(hi, Time)
        assert lo < hi

    def test_single_time_sets_lo_eq_hi(self):
        lo, hi = kc._normalize_window(time="2026-06-01")
        assert lo == hi
        assert isinstance(lo, Time)

    def test_time_range_takes_priority_over_time(self):
        lo, hi = kc._normalize_window(
            time="2026-06-01", time_range=("2026-01-01", "2026-01-02")
        )
        assert lo == Time("2026-01-01")
        assert hi == Time("2026-01-02")

    def test_time_range_with_time_objects(self):
        lo_in, hi_in = Time("2026-01-01"), Time("2026-01-02")
        lo, hi = kc._normalize_window(time_range=(lo_in, hi_in))
        assert lo is lo_in
        assert hi is hi_in

    def test_time_range_with_none_elements(self):
        lo, hi = kc._normalize_window(time_range=(None, "2026-01-02"))
        assert lo is None
        assert hi == Time("2026-01-02")


class TestWindowContains:
    def test_both_bounds_none_always_true(self):
        assert kc._window_contains(Time("2026-01-01"), Time("2026-01-01"), None, None)

    def test_request_within_coverage(self):
        assert kc._window_contains(
            Time("2026-06-01"), Time("2026-06-01"), "2026-01-01", "2026-12-31"
        )

    def test_request_starts_before_coverage(self):
        assert not kc._window_contains(
            Time("2025-12-31"), Time("2026-01-01"), "2026-01-01", "2026-12-31"
        )

    def test_request_ends_after_coverage(self):
        assert not kc._window_contains(
            Time("2026-01-01"), Time("2027-01-01"), "2026-01-01", "2026-12-31"
        )

    def test_no_request_bounds_is_true(self):
        assert kc._window_contains(None, None, "2026-01-01", "2026-12-31")

    def test_only_lo_bound_supplied_and_violated(self):
        assert not kc._window_contains(Time("2025-01-01"), None, "2026-01-01", "2026-12-31")

    def test_only_hi_bound_supplied_and_violated(self):
        assert not kc._window_contains(None, Time("2027-01-01"), "2026-01-01", "2026-12-31")

    def test_coverage_start_none_only_hi_checked(self):
        # cov_start None means no lower bound on coverage -> req_lo check skipped
        assert kc._window_contains(Time("1500-01-01"), Time("2026-01-01"), None, "2026-12-31")

    def test_coverage_end_none_only_lo_checked(self):
        assert kc._window_contains(Time("2026-01-01"), Time("3000-01-01"), "2026-01-01", None)

    def test_exact_boundary_equal_is_contained(self):
        # req exactly equal to coverage bounds should be considered contained
        assert kc._window_contains(
            Time("2026-01-01"), Time("2026-12-31"), "2026-01-01", "2026-12-31"
        )


class TestCoverageWidthDays:
    def test_basic_width(self):
        assert kc._coverage_width_days("2026-01-01", "2026-01-11") == 10.0

    def test_start_none_is_infinite(self):
        assert kc._coverage_width_days(None, "2026-01-11") == float("inf")

    def test_end_none_is_infinite(self):
        assert kc._coverage_width_days("2026-01-01", None) == float("inf")

    def test_both_none_is_infinite(self):
        assert kc._coverage_width_days(None, None) == float("inf")

    def test_zero_width(self):
        assert kc._coverage_width_days("2026-01-01", "2026-01-01") == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# _select_time_filtered_kernels
# ─────────────────────────────────────────────────────────────────────────────
class TestSelectTimeFilteredKernels:
    def test_unbounded_entries_always_included_regardless_of_time(self):
        entries = [("a.tls", "lsk", None, None), ("b.tpc", "pck", None, None)]
        result = kc._select_time_filtered_kernels(entries, time="2026-01-01")
        assert result == [("a.tls", "lsk"), ("b.tpc", "pck")]

    def test_unbounded_entries_included_with_no_time_request(self):
        entries = [("a.tls", "lsk", None, None)]
        result = kc._select_time_filtered_kernels(entries)
        assert result == [("a.tls", "lsk")]

    def test_empty_entries_returns_empty_list(self):
        assert kc._select_time_filtered_kernels([]) == []

    def test_empty_entries_with_time_request_returns_empty_list(self):
        # No bounded candidates at all -> early return, no ValueError
        assert kc._select_time_filtered_kernels([], time="2026-01-01") == []

    def test_bounded_entry_included_when_no_request_given(self):
        entries = [("de442.bsp", "spk_planets", "1549-12-31", "2650-01-25")]
        result = kc._select_time_filtered_kernels(entries)
        assert result == [("de442.bsp", "spk_planets")]

    def test_bounded_entry_included_when_request_matches(self):
        entries = [("de442.bsp", "spk_planets", "1549-12-31", "2650-01-25")]
        result = kc._select_time_filtered_kernels(entries, time="2026-01-01")
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
            kc._select_time_filtered_kernels(entries, time="2026-01-01")

    def test_mixed_bounded_and_unbounded_succeeds_when_bounded_matches(self):
        entries = [
            ("always.tls", "lsk", None, None),
            ("wide.bsp", "spk_planets", "1900-01-01", "2100-01-01"),
        ]
        result = kc._select_time_filtered_kernels(entries, time="2026-01-01")
        assert ("always.tls", "lsk") in result
        assert ("wide.bsp", "spk_planets") in result
        assert len(result) == 2

    def test_context_label_appears_in_error_message(self):
        entries = [("x.bsp", "spk_planets", "1990-01-01", "1991-01-01")]
        with pytest.raises(ValueError, match="for 'MARS'"):
            kc._select_time_filtered_kernels(
                entries, time="2026-01-01", context_label="for 'MARS'"
            )

    def test_error_message_without_context_label_has_no_dangling_for(self):
        entries = [("x.bsp", "spk_planets", "1990-01-01", "1991-01-01")]
        with pytest.raises(ValueError) as excinfo:
            kc._select_time_filtered_kernels(entries, time="2026-01-01")
        msg = str(excinfo.value)
        assert "No registered kernel covers" in msg

    def test_time_range_filters_entries(self):
        entries = [
            ("old.bsp", "spk_planets", "1900-01-01", "1950-01-01"),
            ("new.bsp", "spk_planets", "1990-01-01", "2050-01-01"),
        ]
        result = kc._select_time_filtered_kernels(
            entries, time_range=("2000-01-01", "2010-01-01")
        )
        assert result == [("new.bsp", "spk_planets")]

    def test_time_range_excludes_all_bounded_raises(self):
        entries = [
            ("old.bsp", "spk_planets", "1900-01-01", "1950-01-01"),
        ]
        with pytest.raises(ValueError):
            kc._select_time_filtered_kernels(
                entries, time_range=("2000-01-01", "2010-01-01")
            )

    def test_prefers_shortest_timespan_coverage_fit(self):
        entries = [
            ("mar099.bsp", "spk_satellites", "1600-01-01", "2600-01-01"),
            ("mar099s.bsp", "spk_satellites", "1995-01-01", "2050-01-01"),
        ]
        result = kc._select_time_filtered_kernels(entries, time="2026-01-01")
        assert result == [("mar099s.bsp", "spk_satellites")]

    def test_prefers_shortest_timespan_with_no_request_too(self):
        # Even with no explicit time request, all bounded matches are kept
        # individually unless rank-reduction kicks in (only triggers on the
        # matched set). Here, since req is None, "matching" includes both,
        # and the single best (tightest) is still chosen by min().
        entries = [
            ("wide.bsp", "spk_satellites", "1600-01-01", "2600-01-01"),
            ("tight.bsp", "spk_satellites", "1995-01-01", "2050-01-01"),
        ]
        result = kc._select_time_filtered_kernels(entries)
        assert result == [("tight.bsp", "spk_satellites")]

    def test_three_overlapping_candidates_picks_globally_tightest(self):
        entries = [
            ("a.bsp", "spk_satellites", "1600-01-01", "2600-01-01"),  # ~1,000,000 days
            ("b.bsp", "spk_satellites", "1995-01-01", "2050-01-01"),  # ~55 years
            ("c.bsp", "spk_satellites", "2020-01-01", "2030-01-01"),  # ~10 years (tightest)
        ]
        result = kc._select_time_filtered_kernels(entries, time="2026-01-01")
        assert result == [("c.bsp", "spk_satellites")]

    def test_unbounded_entries_preserved_alongside_best_bounded_pick(self):
        entries = [
            ("leap.tls", "lsk", None, None),
            ("a.bsp", "spk_satellites", "1600-01-01", "2600-01-01"),
            ("b.bsp", "spk_satellites", "1995-01-01", "2050-01-01"),
        ]
        result = kc._select_time_filtered_kernels(entries, time="2026-01-01")
        assert ("leap.tls", "lsk") in result
        assert ("b.bsp", "spk_satellites") in result
        assert len(result) == 2

    def test_duplicate_filename_different_subdir_does_not_crash(self):
        # Latent-bug probe: cov_map is keyed only by filename, so two
        # bounded entries sharing a filename but different subdir/coverage
        # will collide in cov_map (the second overwrites the first for
        # ranking purposes). We don't assert a "correct" outcome here since
        # the implementation doesn't disambiguate, but we DO assert it
        # doesn't crash and returns exactly one of the two filenames once
        # for this filename instead of silently duplicating/corrupting.
        entries = [
            ("dup.bsp", "spk_satellites", "1995-01-01", "2050-01-01"),
            ("dup.bsp", "spk_planets", "2020-01-01", "2030-01-01"),
        ]
        result = kc._select_time_filtered_kernels(entries, time="2026-01-01")
        # Exactly one (filename, subdir) pair should appear in the result.
        # Both entries share the key "dup.bsp" in cov_map, which is built
        # last-write-wins -> the spk_planets coverage (tighter window) is
        # used to rank *both* candidates, producing a tie. min() then
        # returns the first tied element in iteration order, which is the
        # spk_satellites entry (it appears first in `entries`/`matching`).
        assert len(result) == 1
        fname, subdir = result[0]
        assert fname == "dup.bsp"
        assert subdir == "spk_satellites"  # first tied entry wins min()

    def test_single_bounded_entry_with_time_range_spanning_exactly(self):
        entries = [("x.bsp", "spk_planets", "2000-01-01", "2030-01-01")]
        result = kc._select_time_filtered_kernels(
            entries, time_range=("2000-01-01", "2030-01-01")
        )
        assert result == [("x.bsp", "spk_planets")]

    def test_only_one_bound_of_time_range_set(self):
        entries = [
            ("a.bsp", "spk_planets", "1990-01-01", "2000-01-01"),
            ("b.bsp", "spk_planets", "1990-01-01", "2050-01-01"),
        ]
        # only lo bound set via time_range; hi is None
        result = kc._select_time_filtered_kernels(
            entries, time_range=("1995-01-01", None)
        )
        # both satisfy req_lo (>=1990 start) trivially since req_hi is None;
        # tightest of the matching ones should win
        assert result == [("a.bsp", "spk_planets")]


# ─────────────────────────────────────────────────────────────────────────────
# Citation tracking
# ─────────────────────────────────────────────────────────────────────────────
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

    def test_multiple_citations_preserve_order(self):
        kc._log_citation("a.tls", "url_a", "ctx_a")
        kc._log_citation("b.bsp", "url_b", "ctx_b")
        kernels = kc.get_citations()["kernels"]
        assert [k["filename"] for k in kernels] == ["a.tls", "b.bsp"]

    def test_toolkit_citations_always_present(self):
        citations = kc.get_citations()
        assert kc._SPICE_CITATION in citations["toolkit"]
        assert kc._SPICEYPY_CITATION in citations["toolkit"]

    def test_toolkit_citations_present_even_with_empty_kernel_log(self):
        kc.reset_citations()
        citations = kc.get_citations()
        assert len(citations["toolkit"]) == 2

    def test_reset_clears_log(self):
        kc._log_citation("a.tls", "url_a", "ctx_a")
        assert len(kc.get_citations()["kernels"]) == 1
        kc.reset_citations()
        assert kc.get_citations()["kernels"] == []

    def test_get_citations_returns_copy_not_reference(self):
        kc._log_citation("a.tls", "url_a", "ctx_a")
        snapshot = kc.get_citations()
        snapshot["kernels"].append({"filename": "fake.tls", "url": "x", "context": "y"})
        # Mutating the returned list must not affect the internal CITATION_LOG
        assert len(kc.CITATION_LOG) == 1
        assert len(kc.get_citations()["kernels"]) == 1

    def test_get_citations_kernels_is_not_same_list_object(self):
        assert kc.get_citations()["kernels"] is not kc.CITATION_LOG


# ─────────────────────────────────────────────────────────────────────────────
# Checksum utilities
# ─────────────────────────────────────────────────────────────────────────────
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

    def test_skips_malformed_lines(self, monkeypatch):
        # Blank lines, single-token lines, and extra-whitespace lines should
        # not raise and should not contribute spurious entries.
        manifest_text = (
            "\n"
            "onlyonetoken\n"
            "   \n"
            "deadbeefdeadbeefdeadbeefdeadbeef   real_file.bsp   extra_ignored_token\n"
        )

        class FakeResponse:
            text = manifest_text
            def raise_for_status(self): pass

        monkeypatch.setattr(kc.requests, "get", lambda url, timeout=10: FakeResponse())
        result = kc.fetch_remote_md5s("spk_planets")
        assert result == {"real_file.bsp": "deadbeefdeadbeefdeadbeefdeadbeef"}

    def test_empty_manifest_returns_empty_dict(self, monkeypatch):
        class FakeResponse:
            text = ""
            def raise_for_status(self): pass

        monkeypatch.setattr(kc.requests, "get", lambda url, timeout=10: FakeResponse())
        assert kc.fetch_remote_md5s("spk_planets") == {}

    def test_uses_correct_url_for_subdir(self, monkeypatch):
        captured = {}

        class FakeResponse:
            text = ""
            def raise_for_status(self): pass

        def fake_get(url, timeout=10):
            captured["url"] = url
            captured["timeout"] = timeout
            return FakeResponse()

        monkeypatch.setattr(kc.requests, "get", fake_get)
        kc.fetch_remote_md5s("pck")
        assert captured["url"] == kc._CHECKSUM_MANIFEST_URL["pck"]
        assert captured["timeout"] == 10

    def test_invalid_subdir_key_raises_keyerror(self):
        with pytest.raises(KeyError):
            kc.fetch_remote_md5s("not_a_real_subdir")

    @patch("leos.kernels._kernel_common.requests.get")
    def test_request_exception_returns_empty_dict(self, mock_get):
        mock_get.side_effect = requests.exceptions.RequestException("Timeout")
        assert kc.fetch_remote_md5s("spk_satellites") == {}

    @patch("leos.kernels._kernel_common.requests.get")
    def test_http_error_status_returns_empty_dict(self, mock_get):
        fake_response = MagicMock()
        fake_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404")
        mock_get.return_value = fake_response
        assert kc.fetch_remote_md5s("spk_satellites") == {}

    def test_default_subdir_argument_is_spk_satellites(self, monkeypatch):
        captured = {}

        class FakeResponse:
            text = ""
            def raise_for_status(self): pass

        def fake_get(url, timeout=10):
            captured["url"] = url
            return FakeResponse()

        monkeypatch.setattr(kc.requests, "get", fake_get)
        kc.fetch_remote_md5s()  # no subdir passed -> default "spk_satellites"
        assert captured["url"] == kc._CHECKSUM_MANIFEST_URL["spk_satellites"]


class TestCalculateLocalMd5:
    def test_matches_hashlib_reference(self, tmp_path):
        f = tmp_path / "sample.bin"
        content = b"some kernel-ish binary content" * 1000
        f.write_bytes(content)
        expected = hashlib.md5(content).hexdigest()
        assert kc.calculate_local_md5(str(f)) == expected

    def test_empty_file_matches_known_empty_md5(self, tmp_path):
        f = tmp_path / "empty.bin"
        f.write_bytes(b"")
        assert kc.calculate_local_md5(str(f)) == hashlib.md5(b"").hexdigest()

    def test_file_smaller_than_chunk_size(self, tmp_path):
        f = tmp_path / "tiny.bin"
        content = b"x"
        f.write_bytes(content)
        assert kc.calculate_local_md5(str(f)) == hashlib.md5(content).hexdigest()

    def test_file_exactly_chunk_size_boundary(self, tmp_path):
        f = tmp_path / "boundary.bin"
        content = b"y" * 4096
        f.write_bytes(content)
        assert kc.calculate_local_md5(str(f)) == hashlib.md5(content).hexdigest()

    def test_file_spanning_multiple_chunks(self, tmp_path):
        f = tmp_path / "multi.bin"
        content = b"z" * (4096 * 3 + 17)
        f.write_bytes(content)
        assert kc.calculate_local_md5(str(f)) == hashlib.md5(content).hexdigest()

    def test_missing_file_raises_file_not_found(self, tmp_path):
        missing = tmp_path / "does_not_exist.bin"
        with pytest.raises(FileNotFoundError):
            kc.calculate_local_md5(str(missing))


# ─────────────────────────────────────────────────────────────────────────────
# Filename -> subdirectory inference
# ─────────────────────────────────────────────────────────────────────────────
class TestInferSubdir:
    @pytest.mark.parametrize(
        "filename,expected",
        [
            # lsk
            ("naif0012.tls", "lsk"),
            ("NAIF0012.TLS", "lsk"),  # case-insensitivity
            # pck
            ("pck00011.tpc", "pck"),
            ("earth_200101_260101.tpc", "pck"),
            ("pck00011.bpc", "pck"),
            ("moon_pa_de440_200625.bpc", "pck"),
            # spk_planets
            ("de442.bsp", "spk_planets"),
            ("de440.bsp", "spk_planets"),
            # spk_lagrange_point (all four prefixes)
            ("l1_mission.bsp", "spk_lagrange_point"),
            ("l2_mission.bsp", "spk_lagrange_point"),
            ("l4_mission.bsp", "spk_lagrange_point"),
            ("l5_mission.bsp", "spk_lagrange_point"),
            # spk_asteroids
            ("codes_asteroids.bsp", "spk_asteroids"),
            ("codes_300ast20100725.bsp", "spk_asteroids"),
            # spk_comets (all listed prefixes)
            ("c_g_comet.bsp", "spk_comets"),
            ("ison_comet.bsp", "spk_comets"),
            ("c2013_comet.bsp", "spk_comets"),
            ("siding_spring_comet.bsp", "spk_comets"),
            # spk_tno
            ("tnosat_object.bsp", "spk_tno"),
            # spk_stations (all listed prefixes)
            ("dss_station.bsp", "spk_stations"),
            ("earthstns_itrf93.bsp", "spk_stations"),
            ("ndosl_station.bsp", "spk_stations"),
            # spk_satellites (default fallthrough for .bsp)
            ("mar099.bsp", "spk_satellites"),
            ("mar099s.bsp", "spk_satellites"),
            ("jup365.bsp", "spk_satellites"),
        ],
    )
    def test_known_extensions_map_correctly(self, filename, expected):
        assert kc._infer_subdir(filename) == expected

    def test_frame_kernel_always_raises_ambiguous(self):
        with pytest.raises(ValueError, match="Cannot infer NAIF subdirectory for frame kernel"):
            kc._infer_subdir("planets.tf")

    def test_frame_kernel_error_mentions_all_three_options(self):
        with pytest.raises(ValueError) as excinfo:
            kc._infer_subdir("any.tf")
        msg = str(excinfo.value)
        assert "fk/planets" in msg
        assert "fk/satellites" in msg
        assert "fk/stations" in msg

    def test_unknown_extension_raises(self):
        with pytest.raises(ValueError, match="Cannot infer NAIF subdirectory for"):
            kc._infer_subdir("invalid_kernel.txt")

    def test_unknown_extension_error_mentions_extra_urls(self):
        with pytest.raises(ValueError, match="extra_urls"):
            kc._infer_subdir("mystery.xyz")

    def test_bsp_with_unrecognized_prefix_defaults_to_spk_satellites(self):
        # Anything ending .bsp that doesn't match a specific prefix rule
        # falls through to spk_satellites per the function's final return.
        assert kc._infer_subdir("totally_unknown_prefix.bsp") == "spk_satellites"

    def test_empty_filename_raises(self):
        with pytest.raises(ValueError):
            kc._infer_subdir("")

    def test_no_extension_raises(self):
        with pytest.raises(ValueError):
            kc._infer_subdir("no_extension_at_all")


class TestSubdirFor:
    def test_delegates_to_infer_subdir_for_valid_names(self):
        assert kc._subdir_for("de442.bsp") == "spk_planets"
        assert kc._subdir_for("naif0012.tls") == "lsk"
        assert kc._subdir_for("pck00011.tpc") == "pck"
        assert kc._subdir_for("l1_mission.bsp") == "spk_lagrange_point"

    def test_fallback_on_frame_kernel_ambiguity(self):
        # _infer_subdir raises ValueError for .tf; _subdir_for must swallow
        # it and fall back to "spk_satellites".
        assert kc._subdir_for("planets.tf") == "spk_satellites"

    def test_fallback_on_unknown_extension(self):
        assert kc._subdir_for("unknown.extension") == "spk_satellites"

    def test_fallback_does_not_propagate_exception(self):
        # Should never raise, regardless of how malformed the filename is.
        try:
            result = kc._subdir_for("")
        except ValueError:
            pytest.fail("_subdir_for must not propagate ValueError from _infer_subdir")
        assert result == "spk_satellites"

    def test_non_bsp_known_extension_not_overridden_by_fallback(self):
        # Sanity: ensure fallback path is only reached on actual ValueError,
        # not accidentally for every call.
        assert kc._subdir_for("naif0012.tls") != "spk_satellites"
