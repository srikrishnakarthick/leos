"""
tests/kernels/missions/Mars/test_fetch_maven_kernels.py
Tests for leos.kernels.missions.Mars.fetch_maven_kernels.
"""
import pytest
from astropy.time import Time
from leos.kernels.missions.Mars import fetch_maven_kernels as fmk

class TestConstants:
    def test_maven_base_url_format(self):
        assert fmk._MAVEN_BASE.startswith("https://naif.jpl.nasa.gov/pub/naif/MAVEN/")
        assert fmk._MAVEN_BASE.endswith("/")

class TestResolveMavenCk:
    SAMPLE_LISTING = """
    <a href="mvn_sc_rel_140101_140107_v01.bc">mvn_sc_rel_140101_140107_v01.bc</a>
    <a href="archived/mvn_sc_rel_141006_141012_v01.bc">archived/mvn_sc_rel_141006_141012_v01.bc</a>
    <a href="mvn_sc_rel_141006_141012_v02.bc">mvn_sc_rel_141006_141012_v02.bc</a>
    <a href="mvn_sc_rel_141013_141019_v01.bc">mvn_sc_rel_141013_141019_v01.bc</a>
    <a href="mvn_app_rel_141013_141019_v01.bc">mvn_app_rel_141013_141019_v01.bc</a>
    <a href="mvn_sc_rel_150101_150107_v02.bc">mvn_sc_rel_150101_150107_v02.bc</a>
    """

    @pytest.fixture(autouse=True)
    def mock_requests(self, monkeypatch):
        class FakeResponse:
            text = self.SAMPLE_LISTING
            def raise_for_status(self): pass
        monkeypatch.setattr(fmk.requests, "get", lambda url, timeout=15: FakeResponse())

    def test_filters_by_structure_sc(self):
        # Asking for a single day inside the weekly kernel span so it passes containment
        result = fmk.resolve_maven_ck(time_range=("2014-10-08", "2014-10-09"), structure="sc")
        assert result == ["mvn_sc_rel_141006_141012_v02.bc"]

    def test_filters_by_structure_app(self):
        result = fmk.resolve_maven_ck(time_range=("2014-10-15", "2014-10-16"), structure="app")
        assert result == ["mvn_app_rel_141013_141019_v01.bc"]

    def test_excludes_archived_files(self):
        # Ensure v02 is picked up and the one inside archived/ directory context is skipped
        result = fmk.resolve_maven_ck(time_range=("2014-10-08", "2014-10-09"), structure="sc")
        assert "mvn_sc_rel_141006_141012_v01.bc" not in result
        assert "mvn_sc_rel_141006_141012_v02.bc" in result

    def test_results_sorted_oldest_first(self):
        # No time window should pull out all non-archived files, sorted
        result = fmk.resolve_maven_ck(structure="sc")
        assert result == [
            "mvn_sc_rel_140101_140107_v01.bc",
            "mvn_sc_rel_141006_141012_v02.bc",
            "mvn_sc_rel_141013_141019_v01.bc",
            "mvn_sc_rel_150101_150107_v02.bc",
        ]

    def test_no_window_returns_all_matching_structure(self):
        result = fmk.resolve_maven_ck(structure="sc")
        assert len(result) == 4

    def test_returns_empty_list_on_network_failure(self, monkeypatch):
        def mock_fail(*args, **kwargs): raise Exception("Network down")
        monkeypatch.setattr(fmk.requests, "get", mock_fail)
        assert fmk.resolve_maven_ck(structure="sc") == []

    def test_no_matches_in_window_returns_empty_list(self):
        assert fmk.resolve_maven_ck(time_range=("2099-01-01", "2099-01-02"), structure="sc") == []
