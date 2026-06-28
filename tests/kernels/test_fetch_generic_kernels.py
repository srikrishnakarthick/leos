"""
tests/kernels/test_fetch_generic_kernels.py

Tests for leos.kernels.fetch_generic_kernels.

No real NAIF network calls or multi-GB kernel downloads happen here:
requests.get / the module's own comment-fetching helpers are mocked out,
and any cache-dir usage is monkeypatched to tmp_path.
"""
import os

import pytest
from astropy.time import Time

from leos.kernels import fetch_generic_kernels as fgk


# ──────────────────────────────────────────────────────────────────────────
# _normalize_name
# ──────────────────────────────────────────────────────────────────────────

class TestNormalizeName:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("S/2020 S 49", "S2020S49"),
            ("S2020_s_49", "S2020S49"),
            ("S2020_s49", "S2020S49"),
            ("Himalia", "HIMALIA"),
            ("1 Ceres", "1CERES"),
            (65297, "65297"),
        ],
    )
    def test_strips_separators_and_uppercases(self, raw, expected):
        assert fgk._normalize_name(raw) == expected


# ──────────────────────────────────────────────────────────────────────────
# parse_kernel_comment
# ──────────────────────────────────────────────────────────────────────────

class TestParseKernelComment:
    def test_parses_fk_style_name_code_blocks(self):
        text = (
            "NAIF_BODY_NAME += ( 'HIMALIA' )\n"
            "NAIF_BODY_CODE += ( 506 )\n"
            "NAIF_BODY_NAME += ( 'ELARA' )\n"
            "NAIF_BODY_CODE += ( 507 )\n"
        )
        result = fgk.parse_kernel_comment(text, "jup365.bsp")
        assert result["bodies"]["HIMALIA"] == 506
        assert result["bodies"]["ELARA"] == 507

    def test_parses_body_table_style_listing(self):
        text = (
            "   Name        ID     GM            Lat  Lon  Flag\n"
            "   ELARA       507    1.3e21         12   34   AAA\n"
            "   PASIPHAE    508    5.0E20          1    2   BBB\n"
        )
        result = fgk.parse_kernel_comment(text, "jup347.bsp")
        assert result["bodies"]["ELARA"] == 507
        assert result["bodies"]["PASIPHAE"] == 508

    def test_skips_header_row_tokens_name_system_number(self):
        # Defensive filter: even if a header-like row slipped through the
        # body-table regex, NAME/SYSTEM/NUMBER must never become "bodies".
        text = "NAME 99 1.0 1 1 X\n"
        result = fgk.parse_kernel_comment(text, "whatever.bsp")
        assert "NAME" not in result["bodies"]

    def test_coverage_from_begin_end_time_block(self):
        text = (
            "SPK_KERNEL = test_moon.bsp\n"
            "Some other comment lines here.\n"
            "BEGIN_TIME = 1980 JAN 01 00:00:00.000\n"
            "more filler\n"
            "END_TIME = 2030 DEC 31 00:00:00.000\n"
        )
        result = fgk.parse_kernel_comment(text, "test_moon.bsp")
        begin, end = result["coverage"]
        assert begin == Time("1980-01-01")
        assert end == Time("2030-12-31")

    def test_coverage_falls_back_to_timespan_line(self):
        text = (
            "Timespan from JED 2440000.5(01-JAN-1980) "
            "to JED 2470000.5(31-DEC-2030)\n"
        )
        result = fgk.parse_kernel_comment(text, "unrelated.bsp")
        begin, end = result["coverage"]
        assert begin == Time("1980-01-01")
        assert end == Time("2030-12-31")

    def test_no_coverage_info_returns_none_none(self):
        text = "Nothing useful in here.\n"
        result = fgk.parse_kernel_comment(text, "x.bsp")
        assert result["coverage"] == (None, None)

    def test_bc_dates_treated_as_unbounded(self):
        text = (
            "SPK_KERNEL = backup.bsp\n"
            "BEGIN_TIME = 9999 B.C. JAN 01 00:00:00.000\n"
            "filler\n"
            "END_TIME = 2030 DEC 31 00:00:00.000\n"
        )
        result = fgk.parse_kernel_comment(text, "backup.bsp")
        begin, end = result["coverage"]
        assert begin is None
        assert end == Time("2030-12-31")

# ──────────────────────────────────────────────────────────────────────────
# select_common_kernels / _select_body_kernels / _select_named_static_kernel
# ──────────────────────────────────────────────────────────────────────────

class TestSelectCommonKernels:
    def test_returns_all_common_kernels_with_no_time_filter(self):
        result = select = fgk.select_common_kernels()
        names = {fname for fname, _ in select}
        assert names == {"naif0012.tls", "pck00011.tpc", "de442.bsp"}

    def test_filters_de442_out_of_range(self):
        with pytest.raises(ValueError):
            # 1549-2650 is de442's window; way outside it should raise since
            # only de442 is bounded and it won't match.
            fgk._select_time_filtered_kernels(
                [("de442.bsp", "spk_planets", "1549-12-31", "2650-01-25")],
                time="3000-01-01",
            )

    def test_de442_included_for_in_range_time(self):
        result = fgk.select_common_kernels(time="2024-01-01")
        names = {fname for fname, _ in result}
        assert "de442.bsp" in names


class TestSelectBodyKernels:
    def test_mars_returns_its_registered_kernels(self):
        result = fgk._select_body_kernels("mars")
        names = {fname for fname, _ in result}
        assert "mars_iau2000_v1.tpc" in names
        assert "mar099s.bsp" in names
        assert "mar099.bsp" in names

    def test_earth_returns_empty_list(self):
        assert fgk._select_body_kernels("EARTH") == []

    def test_unknown_body_raises_with_known_bodies_listed(self):
        with pytest.raises(ValueError, match="No registered kernel set"):
            fgk._select_body_kernels("TATOOINE")

    def test_is_case_and_whitespace_insensitive(self):
        assert fgk._select_body_kernels("  mars  ") == fgk._select_body_kernels("MARS")

    def test_mar099s_excluded_for_out_of_range_time(self):
        # mar099s.bsp covers 1995-2050; request a time only mar099.bsp covers.
        result = fgk._select_body_kernels("MARS", time="1700-01-01")
        names = {fname for fname, _ in result}
        assert "mar099s.bsp" not in names
        assert "mar099.bsp" in names


class TestSelectNamedStaticKernel:
    def test_returns_filename_and_subdir_when_in_window(self):
        entry = fgk.LAGRANGE_KERNELS["EARTH-MOON L1"]
        fname, subdir = fgk._select_named_static_kernel(entry, time="2020-01-01")
        assert fname == "L1_de441.bsp"
        assert subdir == "spk_lagrange_point"

    def test_raises_when_outside_window(self):
        entry = fgk.LAGRANGE_KERNELS["EARTH-MOON L1"]
        with pytest.raises(ValueError, match="does not cover"):
            fgk._select_named_static_kernel(entry, time="1800-01-01")

    def test_label_included_in_error_message(self):
        entry = fgk.COMET_KERNELS["ISON"]
        with pytest.raises(ValueError, match="ISON"):
            fgk._select_named_static_kernel(entry, time="1800-01-01", label="ISON")


# ──────────────────────────────────────────────────────────────────────────
# Comment fetching / caching
# ──────────────────────────────────────────────────────────────────────────

class TestCommentCachePath:
    def test_builds_path_under_cache_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(fgk, "_CMT_CACHE_DIR", str(tmp_path))
        path = fgk._comment_cache_path("jup365.bsp")
        assert path == os.path.join(str(tmp_path), "jup365.bsp.cmt.txt")


class TestFetchCommentText:
    def test_downloads_and_caches_on_first_call(self, tmp_path, monkeypatch):
        monkeypatch.setattr(fgk, "_CMT_CACHE_DIR", str(tmp_path))
        calls = {"n": 0}

        class FakeResponse:
            text = "some comment text"

            def raise_for_status(self):
                pass

        def fake_get(url, timeout=15):
            calls["n"] += 1
            return FakeResponse()

        monkeypatch.setattr(fgk.requests, "get", fake_get)

        first = fgk._fetch_comment_text("jup365.bsp", subdir="spk_satellites")
        assert first == "some comment text"
        assert calls["n"] == 1

        # Second call should hit the on-disk cache, not the network.
        second = fgk._fetch_comment_text("jup365.bsp", subdir="spk_satellites")
        assert second == "some comment text"
        assert calls["n"] == 1

    def test_force_bypasses_cache(self, tmp_path, monkeypatch):
        monkeypatch.setattr(fgk, "_CMT_CACHE_DIR", str(tmp_path))
        calls = {"n": 0}

        class FakeResponse:
            def __init__(self, text):
                self.text = text

            def raise_for_status(self):
                pass

        responses = ["first", "second"]

        def fake_get(url, timeout=15):
            calls["n"] += 1
            return FakeResponse(responses[calls["n"] - 1])

        monkeypatch.setattr(fgk.requests, "get", fake_get)

        first = fgk._fetch_comment_text("jup365.bsp", force=True)
        second = fgk._fetch_comment_text("jup365.bsp", force=True)
        assert first == "first"
        assert second == "second"
        assert calls["n"] == 2

    def test_returns_empty_string_and_caches_it_on_network_failure(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(fgk, "_CMT_CACHE_DIR", str(tmp_path))

        def raise_err(url, timeout=15):
            raise ConnectionError("404 simulated")

        monkeypatch.setattr(fgk.requests, "get", raise_err)

        result = fgk._fetch_comment_text("missing.bsp")
        assert result == ""
        assert "Could not fetch comment" in capsys.readouterr().out

    def test_uses_correct_url_for_filename_and_subdir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(fgk, "_CMT_CACHE_DIR", str(tmp_path))
        captured = {}

        class FakeResponse:
            text = ""

            def raise_for_status(self):
                pass

        def fake_get(url, timeout=15):
            captured["url"] = url
            return FakeResponse()

        monkeypatch.setattr(fgk.requests, "get", fake_get)
        fgk._fetch_comment_text("sat441.bsp", subdir="spk_satellites")
        assert captured["url"] == (
            fgk._NAIF_BASE + fgk._NAIF_SUBDIRS["spk_satellites"] + "sat441.cmt"
        )


class TestFetchAsteroidTfText:
    def test_downloads_and_caches(self, tmp_path, monkeypatch):
        monkeypatch.setattr(fgk, "_CMT_CACHE_DIR", str(tmp_path))
        calls = {"n": 0}

        class FakeResponse:
            text = "tf body"

            def raise_for_status(self):
                pass

        def fake_get(url, timeout=15):
            calls["n"] += 1
            return FakeResponse()

        monkeypatch.setattr(fgk.requests, "get", fake_get)

        first = fgk._fetch_asteroid_tf_text()
        second = fgk._fetch_asteroid_tf_text()
        assert first == second == "tf body"
        assert calls["n"] == 1

    def test_returns_empty_string_on_failure(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(fgk, "_CMT_CACHE_DIR", str(tmp_path))

        def raise_err(url, timeout=15):
            raise ConnectionError("boom")

        monkeypatch.setattr(fgk.requests, "get", raise_err)
        result = fgk._fetch_asteroid_tf_text()
        assert result == ""
        assert "Could not fetch asteroid TF" in capsys.readouterr().out


# ──────────────────────────────────────────────────────────────────────────
# _parse_asteroid_tf
# ──────────────────────────────────────────────────────────────────────────

class TestParseAsteroidTf:
    def test_parses_numbered_and_plain_names(self):
        text = (
            "NAIF_BODY_NAME += ( '1 CERES' )\n"
            "NAIF_BODY_CODE += ( 2000001 )\n"
            "NAIF_BODY_NAME += ( 'VESTA' )\n"
            "NAIF_BODY_CODE += ( 2000004 )\n"
        )
        result = fgk._parse_asteroid_tf(text)
        assert result["CERES"] == 2000001
        assert result["VESTA"] == 2000004

    def test_empty_text_returns_empty_dict(self):
        assert fgk._parse_asteroid_tf("") == {}


# ──────────────────────────────────────────────────────────────────────────
# resolve_moon_kernel
# ──────────────────────────────────────────────────────────────────────────

class TestResolveMoonKernel:
    @pytest.fixture
    def fake_candidates(self, monkeypatch):
        fake = {
            "JUPITER": ["jup_small.bsp", "jup_big.bsp"],
            "SATURN": ["sat_small.bsp"],
        }
        monkeypatch.setattr(fgk, "PLANET_CANDIDATE_KERNELS", fake)
        return fake

    def test_finds_body_by_name(self, fake_candidates, monkeypatch):
        comments = {
            "jup_small.bsp": (
                "NAIF_BODY_NAME += ( 'HIMALIA' )\nNAIF_BODY_CODE += ( 506 )\n"
            ),
            "jup_big.bsp": "",
        }
        monkeypatch.setattr(fgk, "_fetch_comment_text", lambda fname, **kw: comments.get(fname, ""))

        matches = fgk.resolve_moon_kernel("Himalia")
        assert matches == [("JUPITER", "jup_small.bsp")]

    def test_finds_body_by_numeric_id(self, fake_candidates, monkeypatch):
        comments = {
            "jup_small.bsp": (
                "NAIF_BODY_NAME += ( 'HIMALIA' )\nNAIF_BODY_CODE += ( 506 )\n"
            ),
        }
        monkeypatch.setattr(fgk, "_fetch_comment_text", lambda fname, **kw: comments.get(fname, ""))
        matches = fgk.resolve_moon_kernel(506)
        assert matches == [("JUPITER", "jup_small.bsp")]

    def test_restricts_search_to_given_planet(self, fake_candidates, monkeypatch):
        # Body name exists in a SATURN file, but caller restricts to JUPITER.
        comments = {
            "sat_small.bsp": (
                "NAIF_BODY_NAME += ( 'MIMAS' )\nNAIF_BODY_CODE += ( 601 )\n"
            ),
        }
        monkeypatch.setattr(fgk, "_fetch_comment_text", lambda fname, **kw: comments.get(fname, ""))
        with pytest.raises(ValueError):
            fgk.resolve_moon_kernel("MIMAS", planet="JUPITER")

    def test_orders_results_by_registry_order(self, fake_candidates, monkeypatch):
        # Same body name appears (hypothetically) listed in both files;
        # result order should follow PLANET_CANDIDATE_KERNELS list order.
        comments = {
            "jup_small.bsp": (
                "NAIF_BODY_NAME += ( 'HIMALIA' )\nNAIF_BODY_CODE += ( 506 )\n"
            ),
            "jup_big.bsp": (
                "NAIF_BODY_NAME += ( 'HIMALIA' )\nNAIF_BODY_CODE += ( 506 )\n"
            ),
        }
        monkeypatch.setattr(fgk, "_fetch_comment_text", lambda fname, **kw: comments.get(fname, ""))
        matches = fgk.resolve_moon_kernel("HIMALIA")
        assert matches == [("JUPITER", "jup_small.bsp"), ("JUPITER", "jup_big.bsp")]

    def test_raises_when_not_found_anywhere(self, fake_candidates, monkeypatch):
        monkeypatch.setattr(fgk, "_fetch_comment_text", lambda fname, **kw: "")
        with pytest.raises(ValueError, match="Could not find a kernel"):
            fgk.resolve_moon_kernel("NONEXISTENT MOON")

    def test_raises_when_time_window_excludes_match(self, fake_candidates, monkeypatch):
        comments = {
            "jup_small.bsp": (
                "NAIF_BODY_NAME += ( 'HIMALIA' )\nNAIF_BODY_CODE += ( 506 )\n"
                "SPK_KERNEL = jup_small.bsp\n"
                "BEGIN_TIME = 1980 JAN 01 00:00:00.000\n"
                "END_TIME = 2000 JAN 01 00:00:00.000\n"
            ),
        }
        monkeypatch.setattr(fgk, "_fetch_comment_text", lambda fname, **kw: comments.get(fname, ""))
        with pytest.raises(ValueError):
            fgk.resolve_moon_kernel("HIMALIA", time="2050-01-01")

    def test_skips_candidates_with_unfetchable_comments(self, fake_candidates, monkeypatch):
        # Empty string ("" -- as returned by _fetch_comment_text on failure)
        # must not crash resolution; it should just be skipped.
        monkeypatch.setattr(fgk, "_fetch_comment_text", lambda fname, **kw: "")
        with pytest.raises(ValueError):
            fgk.resolve_moon_kernel("HIMALIA")


# ──────────────────────────────────────────────────────────────────────────
# resolve_asteroid_kernel
# ──────────────────────────────────────────────────────────────────────────

class TestResolveAsteroidKernel:
    def test_numeric_minor_planet_number_in_range(self):
        fname, subdir = fgk.resolve_asteroid_kernel(1)  # Ceres
        assert fname == fgk.ASTEROID_KERNEL_FILE[0]
        assert subdir == fgk.ASTEROID_KERNEL_FILE[1]

    def test_numeric_full_naif_id_in_range(self):
        fname, subdir = fgk.resolve_asteroid_kernel(2000300)
        assert fname == fgk.ASTEROID_KERNEL_FILE[0]

    def test_numeric_id_out_of_range_raises(self):
        with pytest.raises(ValueError, match="outside the range"):
            fgk.resolve_asteroid_kernel(2000301)

    def test_name_found_in_tf_text(self, monkeypatch):
        monkeypatch.setattr(
            fgk, "_fetch_asteroid_tf_text",
            lambda **kw: "NAIF_BODY_NAME += ( '4 VESTA' )\nNAIF_BODY_CODE += ( 2000004 )\n",
        )
        fname, subdir = fgk.resolve_asteroid_kernel("VESTA")
        assert fname == fgk.ASTEROID_KERNEL_FILE[0]

    def test_name_with_leading_number_prefix_resolves(self, monkeypatch):
        monkeypatch.setattr(
            fgk, "_fetch_asteroid_tf_text",
            lambda **kw: "NAIF_BODY_NAME += ( 'CERES' )\nNAIF_BODY_CODE += ( 2000001 )\n",
        )
        fname, subdir = fgk.resolve_asteroid_kernel("1 CERES")
        assert fname == fgk.ASTEROID_KERNEL_FILE[0]

    def test_unknown_name_raises(self, monkeypatch):
        monkeypatch.setattr(
            fgk, "_fetch_asteroid_tf_text",
            lambda **kw: "NAIF_BODY_NAME += ( 'VESTA' )\nNAIF_BODY_CODE += ( 2000004 )\n",
        )
        with pytest.raises(ValueError, match="not found"):
            fgk.resolve_asteroid_kernel("NOTAREALASTEROID")

    def test_tf_fetch_failure_raises(self, monkeypatch):
        monkeypatch.setattr(fgk, "_fetch_asteroid_tf_text", lambda **kw: "")
        with pytest.raises(ValueError, match="tf fetch failed"):
            fgk.resolve_asteroid_kernel("VESTA")

    def test_time_window_outside_coverage_raises_before_name_lookup(self, monkeypatch):
        # Should fail fast on the time check without even hitting the network.
        called = {"n": 0}
        monkeypatch.setattr(
            fgk, "_fetch_asteroid_tf_text",
            lambda **kw: called.update(n=called["n"] + 1) or "",
        )
        with pytest.raises(ValueError, match="does not cover"):
            fgk.resolve_asteroid_kernel("VESTA", time="1700-01-01")
        assert called["n"] == 0


# ──────────────────────────────────────────────────────────────────────────
# get_generic_kernel_urls (public entry point)
# ──────────────────────────────────────────────────────────────────────────

class TestGetGenericKernelUrls:
    def test_raises_when_neither_body_nor_filenames_given(self):
        with pytest.raises(ValueError, match="needs at least one of"):
            fgk.get_generic_kernel_urls()

    def test_body_in_BODY_KERNELS_includes_common_and_body_specific(self):
        urls = fgk.get_generic_kernel_urls(body="MARS")
        # common kernels always included
        assert "naif0012.tls" in urls
        assert "pck00011.tpc" in urls
        assert "de442.bsp" in urls
        # mars-specific kernels included
        assert "mars_iau2000_v1.tpc" in urls
        assert "mar099.bsp" in urls
        assert urls["mar099.bsp"] == (
            fgk._NAIF_BASE + fgk._NAIF_SUBDIRS["spk_satellites"] + "mar099.bsp"
        )

    def test_body_earth_has_only_common_kernels(self):
        urls = fgk.get_generic_kernel_urls(body="EARTH")
        assert set(urls.keys()) == {"naif0012.tls", "pck00011.tpc", "de442.bsp"}

    def test_body_lagrange_point(self):
        urls = fgk.get_generic_kernel_urls(body="EARTH-MOON L1", time="2020-01-01")
        assert "L1_de441.bsp" in urls

    def test_body_comet(self):
        urls = fgk.get_generic_kernel_urls(body="ISON", time="2013-01-01")
        assert "ison.bsp" in urls

    def test_filenames_string_comma_separated(self):
        urls = fgk.get_generic_kernel_urls(filenames="naif0012.tls, pck00011.tpc")
        assert set(urls.keys()) == {"naif0012.tls", "pck00011.tpc"}
        assert urls["naif0012.tls"] == fgk._NAIF_BASE + fgk._NAIF_SUBDIRS["lsk"] + "naif0012.tls"

    def test_filenames_list(self):
        urls = fgk.get_generic_kernel_urls(filenames=["de442.bsp"])
        assert urls == {"de442.bsp": fgk._NAIF_BASE + fgk._NAIF_SUBDIRS["spk_planets"] + "de442.bsp"}

    def test_body_and_filenames_combined(self):
        urls = fgk.get_generic_kernel_urls(body="EARTH", filenames="de442.bsp")
        assert "naif0012.tls" in urls  # from body=EARTH common kernels
        assert "de442.bsp" in urls     # from filenames (overwrites/coexists fine)

    def test_unresolvable_body_raises_with_both_attempts_reported(self, monkeypatch):
        monkeypatch.setattr(
            fgk, "resolve_asteroid_kernel",
            lambda *a, **kw: (_ for _ in ()).throw(ValueError("asteroid: not found")),
        )
        monkeypatch.setattr(
            fgk, "resolve_moon_kernel",
            lambda *a, **kw: (_ for _ in ()).throw(ValueError("moon: not found")),
        )
        with pytest.raises(ValueError) as excinfo:
            fgk.get_generic_kernel_urls(body="MYSTERY OBJECT")
        msg = str(excinfo.value)
        assert "asteroid: not found" in msg
        assert "moon: not found" in msg

    def test_dynamic_asteroid_resolution_path_used_for_unknown_body(self, monkeypatch):
        monkeypatch.setattr(
            fgk, "resolve_asteroid_kernel",
            lambda *a, **kw: ("codes_300ast_20100725.bsp", "spk_asteroids"),
        )
        urls = fgk.get_generic_kernel_urls(body="VESTA")
        assert "codes_300ast_20100725.bsp" in urls
        assert urls["codes_300ast_20100725.bsp"] == (
            fgk._NAIF_BASE + fgk._NAIF_SUBDIRS["spk_asteroids"] + "codes_300ast_20100725.bsp"
        )

    def test_dynamic_moon_resolution_path_used_when_asteroid_lookup_fails(self, monkeypatch):
        monkeypatch.setattr(
            fgk, "resolve_asteroid_kernel",
            lambda *a, **kw: (_ for _ in ()).throw(ValueError("not an asteroid")),
        )
        monkeypatch.setattr(
            fgk, "resolve_moon_kernel",
            lambda *a, **kw: [("JUPITER", "jup365.bsp")],
        )
        urls = fgk.get_generic_kernel_urls(body="HIMALIA")
        assert "jup365.bsp" in urls
        assert urls["jup365.bsp"] == (
            fgk._NAIF_BASE + fgk._NAIF_SUBDIRS["spk_satellites"] + "jup365.bsp"
        )

    def test_unknown_body_in_lagrange_window_mismatch_raises(self):
        with pytest.raises(ValueError, match="does not cover"):
            fgk.get_generic_kernel_urls(body="EARTH-MOON L1", time="1800-01-01")
