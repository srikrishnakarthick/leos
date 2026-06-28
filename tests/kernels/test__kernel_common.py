"""
tests/kernels/test__kernel_common.py

Tests for leos.kernels._kernel_common.

No network access and no real kernel files are touched: requests.get is
mocked everywhere, and any filesystem use goes through tmp_path /
monkeypatch'd cache dirs.
"""
import hashlib
import os

import pytest
from astropy.time import Time

from leos.kernels import _kernel_common as kc


# ──────────────────────────────────────────────────────────────────────────
# Module-level constants / maps
# ──────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────
# _to_time_or_none / _normalize_window / _window_contains
# ──────────────────────────────────────────────────────────────────────────

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
        # req_lo/req_hi None means "don't filter on this side"
        assert kc._window_contains(None, None, "2000-01-01", "2050-01-01")


# ──────────────────────────────────────────────────────────────────────────
# _select_time_filtered_kernels
# ──────────────────────────────────────────────────────────────────────────

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
        # has_bounded True, bounded never matches, request given -> raises,
        # even though an unbounded entry "matched" trivially.
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


# ──────────────────────────────────────────────────────────────────────────
# Citation tracking
# ──────────────────────────────────────────────────────────────────────────

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

    def test_get_citations_returns_a_copy_not_live_reference(self):
        kc._log_citation("a", "url-a", "ctx")
        snapshot = kc.get_citations()
        kc._log_citation("b", "url-b", "ctx")
        assert len(snapshot["kernels"]) == 1  # snapshot unaffected by later log

    def test_reset_citations_clears_log(self):
        kc._log_citation("a", "url-a", "ctx")
        kc.reset_citations()
        assert kc.get_citations()["kernels"] == []


# ──────────────────────────────────────────────────────────────────────────
# Checksum utilities
# ──────────────────────────────────────────────────────────────────────────

class TestFetchRemoteMd5s:
    def test_parses_well_formed_manifest(self, monkeypatch):
        manifest_text = (
            "d41d8cd98f00b204e9800998ecf8427e  FILE_ONE.bsp\n"
            "0CC175B9C0F1B6A831C399E269772661  file_two.bsp\n"  # uppercase hash
            "\n"  # blank line should be ignored
        )

        class FakeResponse:
            text = manifest_text

            def raise_for_status(self):
                pass

        monkeypatch.setattr(kc.requests, "get", lambda url, timeout=10: FakeResponse())

        result = kc.fetch_remote_md5s("spk_satellites")
        assert result == {
            "file_one.bsp": "d41d8cd98f00b204e9800998ecf8427e",
            "file_two.bsp": "0cc175b9c0f1b6a831c399e269772661",
        }

    def test_returns_empty_dict_on_network_failure(self, monkeypatch, capsys):
        def raise_err(url, timeout=10):
            raise ConnectionError("boom")

        monkeypatch.setattr(kc.requests, "get", raise_err)
        result = kc.fetch_remote_md5s("spk_satellites")
        assert result == {}
        assert "Warning" in capsys.readouterr().out

    def test_uses_correct_manifest_url_for_subdir(self, monkeypatch):
        captured = {}

        class FakeResponse:
            text = ""

            def raise_for_status(self):
                pass

        def fake_get(url, timeout=10):
            captured["url"] = url
            return FakeResponse()

        monkeypatch.setattr(kc.requests, "get", fake_get)
        kc.fetch_remote_md5s("pck")
        assert captured["url"] == kc._CHECKSUM_MANIFEST_URL["pck"]

    def test_invalid_subdir_key_raises_keyerror(self):
        with pytest.raises(KeyError):
            kc.fetch_remote_md5s("not_a_real_subdir")


class TestCalculateLocalMd5:
    def test_matches_hashlib_reference(self, tmp_path):
        f = tmp_path / "sample.bin"
        content = b"some kernel-ish binary content" * 1000
        f.write_bytes(content)

        expected = hashlib.md5(content).hexdigest()
        assert kc.calculate_local_md5(str(f)) == expected

    def test_empty_file_matches_known_md5(self, tmp_path):
        f = tmp_path / "empty.bin"
        f.write_bytes(b"")
        assert kc.calculate_local_md5(str(f)) == hashlib.md5(b"").hexdigest()


# ──────────────────────────────────────────────────────────────────────────
# Filename -> subdirectory inference
# ──────────────────────────────────────────────────────────────────────────

class TestInferSubdir:
    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("naif0012.tls", "lsk"),
            ("pck00011.tpc", "pck"),
            ("moon_pa_de440_200625.bpc", "pck"),
            ("de442.bsp", "spk_planets"),
            ("L1_de441.bsp", "spk_lagrange_point"),
            ("L2_de441.bsp", "spk_lagrange_point"),
            ("l4_de441.bsp", "spk_lagrange_point"),
            ("l5_de441.bsp", "spk_lagrange_point"),
            ("codes_300ast_20100725.bsp", "spk_asteroids"),
            ("c_g_1000012_2012_2017.bsp", "spk_comets"),
            ("ison.bsp", "spk_comets"),
            ("c2013a1_s105_merged.bsp", "spk_comets"),
            ("siding_spring_extra.bsp", "spk_comets"),
            ("tnosat_v01.bsp", "spk_tno"),
            ("dss_75_240126.bsp", "spk_stations"),
            ("earthstns_fx_240126.bsp", "spk_stations"),
            ("ndosl_v01.bsp", "spk_stations"),
            ("mar099.bsp", "spk_satellites"),  # fallback default for .bsp
            ("jup365.bsp", "spk_satellites"),
        ],
    )
    def test_known_extensions_map_correctly(self, filename, expected):
        assert kc._infer_subdir(filename) == expected

    def test_is_case_insensitive(self):
        assert kc._infer_subdir("DE442.BSP") == "spk_planets"

    def test_tf_extension_raises(self):
        with pytest.raises(ValueError, match="frame kernel"):
            kc._infer_subdir("moon_de440_250416.tf")

    def test_unknown_extension_raises(self):
        with pytest.raises(ValueError, match="Cannot infer"):
            kc._infer_subdir("mystery_file.xyz")


class TestSubdirFor:
    def test_delegates_to_infer_subdir_for_resolvable_names(self):
        assert kc._subdir_for("de442.bsp") == "spk_planets"

    def test_falls_back_to_spk_satellites_for_tf_files(self):
        # _infer_subdir raises ValueError for .tf; _subdir_for swallows it.
        assert kc._subdir_for("moon_de440_250416.tf") == "spk_satellites"

    def test_falls_back_to_spk_satellites_for_unknown_extension(self):
        assert kc._subdir_for("totally_unknown.xyz") == "spk_satellites"
