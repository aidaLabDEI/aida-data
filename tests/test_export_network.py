"""`aida-data export` with real downloads, into a temporary cache root
(deselected by default, run with `-m network`)."""

import pytest


@pytest.fixture
def real_root(tmp_path, monkeypatch):
    from aida_data import dense, graph, timeseries

    root = tmp_path / "cache"
    for module in (dense, graph, timeseries):
        monkeypatch.setattr(module, "DATASETS_DIR", root)
    return root


@pytest.mark.network
def test_export_native_hdf5_network(real_root, tmp_path):
    import h5py

    from aida_data import cli

    assert cli.main(["-q", "export", "mnist-784-euclidean", str(tmp_path)]) == 0
    link = tmp_path / "mnist-784-euclidean.hdf5"
    assert link.is_symlink()
    with h5py.File(link) as hfp:
        assert hfp["train"].shape[1] == 784


@pytest.mark.network
def test_export_converted_hdf5_network(real_root, tmp_path):
    import h5py

    from aida_data import cli

    assert cli.main(["-q", "export", "wine", str(tmp_path)]) == 0
    with h5py.File(tmp_path / "wine.hdf5") as hfp:
        assert hfp["train"].shape[0] == hfp["colors"].shape[0] > 0
