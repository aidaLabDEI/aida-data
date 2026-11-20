"""Fixtures shared by the offline tests of `export` and the CLI."""

from pathlib import Path

import pytest

from aida_data import _mteb, dense, graph, timeseries

MTEB = "mteb:T:org/m"


@pytest.fixture
def roots(tmp_path, monkeypatch):
    root = tmp_path / "cache"
    for module in (dense, graph, timeseries):
        monkeypatch.setattr(module, "DATASETS_DIR", root)
        monkeypatch.setattr(module, "_DATASETS_INFO", dict(module._DATASETS_INFO))
    monkeypatch.setattr(dense, "_ALIASES", dict(dense._ALIASES))
    dense.register(dense.DatasetInfo("fake", "https://e.com/f.zip", None, "euclidean", "fake-cache"))
    dense.register(dense.DatasetInfo("fake-native", "https://e.com/n.hdf5", None, "euclidean"))
    graph.register(graph.DatasetInfo("fake-graph", "file:///e", "file:///c", None))
    timeseries.register(timeseries.DatasetInfo("fake-ts", "file:///t", None))
    dense.register_alias("t-alias", MTEB)

    def no_download(url, path, *args, **kwargs):
        # The real one returns when the file is there.
        if not Path(path).exists():
            raise AssertionError("no download expected")

    for module in (dense, graph, timeseries):
        monkeypatch.setattr(module, "_download", no_download)
    monkeypatch.setattr(_mteb, "fetch_hosted", lambda *a, **k: False)
    return root


@pytest.fixture
def work(tmp_path, monkeypatch):
    """A working directory outside the cache."""
    path = tmp_path / "work"
    path.mkdir()
    monkeypatch.chdir(path)
    return path


