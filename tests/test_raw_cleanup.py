"""Raw downloads of cached datasets: skipped once cached, deleted after parse."""

import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from aida_data import dense
from aida_data.dense import Colors, DatasetInfo, load, register

FEATURES = np.arange(12, dtype=np.float32).reshape(4, 3)
COLORS = Colors(np.array([[0], [1], [0], [1]]), ("c",), (("a", "b"),))


def _fake_parser(path):
    assert path.read_text() == "raw"
    return FEATURES.copy(), COLORS


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
            DatasetInfo(name, url, _fake_parser, "euclidean", cache_name)
        )
    return downloads


@pytest.fixture
def dense_dir(tmp_path):
    """Where the dense datasets go under the patched `DATASETS_DIR`."""
    directory = tmp_path / "dense"
    directory.mkdir()
    return directory


def test_raw_deleted_after_parse(dense_dir):
    ds = load("fake")
    np.testing.assert_array_equal(ds.dataset, FEATURES)
    assert (dense_dir / "fake.parquet").is_file()
    assert not (dense_dir / "fake.zip").exists()


def test_keep_raw(dense_dir, monkeypatch):
    monkeypatch.setattr(dense, "KEEP_RAW", True)
    load("fake")
    assert (dense_dir / "fake.zip").is_file()


def test_no_redownload_once_cached(monkeypatch, fake_datasets):
    load("fake")
    assert len(fake_datasets) == 1

    def fail(url, destination):
        raise AssertionError("must not download")

    monkeypatch.setattr(dense, "_download", fail)
    ds = load("fake")
    np.testing.assert_array_equal(ds.dataset, FEATURES)
    assert ds.colors.labels == COLORS.labels


def test_migration_keeps_raw(dense_dir):
    import h5py

    with h5py.File(dense_dir / "fake.hdf5", "w") as hfp:
        hfp["X"] = FEATURES
    (dense_dir / "fake.zip").write_text("raw")
    ds = load("fake")
    np.testing.assert_array_equal(ds.dataset, FEATURES)
    assert ds.colors is None
    assert (dense_dir / "fake.parquet").is_file()
    assert not (dense_dir / "fake.hdf5").exists()
    assert (dense_dir / "fake.zip").is_file()


def test_shared_raw_file_kept(dense_dir):
    load("fake-a")
    assert (dense_dir / "fake-a.parquet").is_file()
    assert (dense_dir / "shared.zip").is_file()
    load("fake-b")
    assert (dense_dir / "shared.zip").is_file()


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
def test_registered_cache_names_match_loaders(dense_dir, monkeypatch, name):
    def fail(url, destination):
        raise AssertionError("must not download")

    monkeypatch.setattr(dense, "_download", fail)
    data = np.arange(3 * 28, dtype=np.float32).reshape(3, 28)
    colors = Colors(np.array([[0], [1], [1]]), ("label",), (("0", "1"),))
    cache_name = dense._DATASETS_INFO[name].cache_name
    dense._write_parquet_cache(dense_dir / f"{cache_name}.parquet", data, colors)
    ds = load(name)
    expected = data[:, -7:] if name == "higgs-highlevel" else data
    np.testing.assert_array_equal(ds.dataset, expected)


def test_prune_raw(dense_dir, monkeypatch):
    monkeypatch.setattr(dense, "KEEP_RAW", True)
    load("fake")
    load("fake-a")
    (dense_dir / "unrelated.csv").write_text("x")
    assert dense.prune_raw() == [dense_dir / "fake.zip"]
    assert (dense_dir / "fake.zip").is_file()
    assert dense.prune_raw(dry_run=False) == [dense_dir / "fake.zip"]
    assert not (dense_dir / "fake.zip").exists()
    assert (dense_dir / "shared.zip").is_file()
    assert (dense_dir / "unrelated.csv").is_file()


def test_every_dense_dataset_is_cached_or_hdf5():
    # `cache_name is None` is what marks the HDF5 datasets. Checked in a
    # fresh interpreter, because other tests leave fake datasets registered.
    code = (
        "from aida_data import dense\n"
        "bad = [n for n, i in dense._DATASETS_INFO.items() if i.cache_name is None"
        " and not dense._raw_name(i).endswith('.hdf5')]\n"
        "assert not bad, bad\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_athlete_cached_on_first_load(dense_dir, monkeypatch):
    csv = (
        '"ID","Name","Sex","Age","Height","Weight","Team"\n'
        '"1","A","M",24,180,80,"China"\n'
        '"2","B","F",NA,170,60,"China"\n'
        '"3","C","F",21,165,55.5,"Italy"\n'
    )

    def download(url, destination):
        if not destination.is_file():
            destination.write_text(csv)

    monkeypatch.setattr(dense, "_download", download)
    first = load("athlete")
    assert (dense_dir / "athlete.parquet").is_file()
    assert not (dense_dir / "athlete_events.csv").exists()

    def fail(*args):
        raise AssertionError("must not download or parse")

    monkeypatch.setattr(dense, "_download", fail)
    info = dense._DATASETS_INFO["athlete"]
    monkeypatch.setitem(
        dense._DATASETS_INFO, "athlete", replace(info, loader_function=fail)
    )
    second = load("athlete")
    np.testing.assert_array_equal(first.dataset, second.dataset)
    np.testing.assert_array_equal(first.colors.values, second.colors.values)
    assert first.colors.labels == second.colors.labels
    assert first.dataset.shape == (2, 3)
