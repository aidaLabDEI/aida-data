"""Raw downloads of cached datasets: skipped once cached, deleted after parse."""

import numpy as np
import pytest

from aida_data import dense
from aida_data.dense import Colors, DatasetInfo, load, register

FEATURES = np.arange(12, dtype=np.float32).reshape(4, 3)
COLORS = Colors(np.array([[0], [1], [0], [1]]), ("c",), (("a", "b"),))


def _fake_loader(cache_name):
    def loader(path):
        def build(path):
            assert path.read_text() == "raw"
            return FEATURES.copy(), COLORS

        data, colors = dense._cached(path, cache_name, build)
        return data, None, None, colors

    return loader


@pytest.fixture(autouse=True)
def fake_datasets(tmp_path, monkeypatch):
    monkeypatch.setattr(dense, "DATASETS_DIR", tmp_path)
    monkeypatch.setattr(dense, "KEEP_RAW", False)
    # Snapshot the registry, so that the fake datasets do not leak.
    monkeypatch.setattr(dense, "_DATASETS_INFO", dict(dense._DATASETS_INFO))
    downloads = []

    def download(url, destination):
        downloads.append(url)
        if not destination.is_file():
            destination.write_text("raw")

    monkeypatch.setattr(dense, "_download", download)
    for name, url, cache_name in [
        ("fake", "https://example.com/fake.zip", "fake"),
        ("fake-a", "https://example.com/shared.zip", "fake-a"),
        ("fake-b", "https://example.com/shared.zip", "fake-b"),
    ]:
        register(
            DatasetInfo(name, url, _fake_loader(cache_name), "euclidean", cache_name)
        )
    return downloads


def test_raw_deleted_after_parse(tmp_path):
    ds = load("fake")
    np.testing.assert_array_equal(ds.dataset, FEATURES)
    assert (tmp_path / "fake.parquet").is_file()
    assert not (tmp_path / "fake.zip").exists()


def test_keep_raw(tmp_path, monkeypatch):
    monkeypatch.setattr(dense, "KEEP_RAW", True)
    load("fake")
    assert (tmp_path / "fake.zip").is_file()


def test_no_redownload_once_cached(monkeypatch, fake_datasets):
    load("fake")
    assert len(fake_datasets) == 1

    def fail(url, destination):
        raise AssertionError("must not download")

    monkeypatch.setattr(dense, "_download", fail)
    ds = load("fake")
    np.testing.assert_array_equal(ds.dataset, FEATURES)
    assert ds.colors.labels == COLORS.labels


def test_migration_keeps_raw(tmp_path):
    import h5py

    with h5py.File(tmp_path / "fake.hdf5", "w") as hfp:
        hfp["X"] = FEATURES
    (tmp_path / "fake.zip").write_text("raw")
    ds = load("fake")
    np.testing.assert_array_equal(ds.dataset, FEATURES)
    assert ds.colors is None
    assert (tmp_path / "fake.parquet").is_file()
    assert not (tmp_path / "fake.hdf5").exists()
    assert (tmp_path / "fake.zip").is_file()


def test_shared_raw_file_kept(tmp_path):
    load("fake-a")
    assert (tmp_path / "fake-a.parquet").is_file()
    assert (tmp_path / "shared.zip").is_file()
    load("fake-b")
    assert (tmp_path / "shared.zip").is_file()


_REAL_CACHED = [
    "pamap2",
    "biokdd",
    "metropt3",
    "household-power",
    "covertype",
    "census1990",
    "phones",
    "higgs",
    "higgs-highlevel",
]


@pytest.mark.parametrize("name", _REAL_CACHED)
def test_registered_cache_names_match_loaders(tmp_path, monkeypatch, name):
    def fail(url, destination):
        raise AssertionError("must not download")

    monkeypatch.setattr(dense, "_download", fail)
    data = np.arange(3 * 28, dtype=np.float32).reshape(3, 28)
    colors = Colors(np.array([[0], [1], [1]]), ("label",), (("0", "1"),))
    cache_name = dense._DATASETS_INFO[name].cache_name
    dense._write_parquet_cache(tmp_path / f"{cache_name}.parquet", data, colors)
    ds = load(name)
    expected = data[:, -7:] if name == "higgs-highlevel" else data
    np.testing.assert_array_equal(ds.dataset, expected)


def test_prune_raw(tmp_path, monkeypatch):
    monkeypatch.setattr(dense, "KEEP_RAW", True)
    load("fake")
    load("fake-a")
    (tmp_path / "unrelated.csv").write_text("x")
    assert dense.prune_raw() == [tmp_path / "fake.zip"]
    assert (tmp_path / "fake.zip").is_file()
    assert dense.prune_raw(dry_run=False) == [tmp_path / "fake.zip"]
    assert not (tmp_path / "fake.zip").exists()
    assert (tmp_path / "shared.zip").is_file()
    assert (tmp_path / "unrelated.csv").is_file()
