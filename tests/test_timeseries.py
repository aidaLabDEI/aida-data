import dataclasses
import gzip

import numpy as np
import pyarrow.parquet as pq
import pytest

from aida_data import timeseries
from aida_data.timeseries import DatasetInfo, TimeSeries, load, register

VALUES = np.array([[0.1, 1.0 / 3.0], [np.nan, -2.5], [1e-300, 7.0], [3.0, np.inf]])
TIME = np.array(
    [
        "2020-01-01T00:00:00",
        "2020-01-01T00:00:01",
        "2020-01-01T00:01:00",
        "2021-06-01T12:00:00",
    ],
    dtype="datetime64[ms]",
)


def _loader_uni(path):
    assert path.read_text() == "raw"
    return VALUES[:, :1].copy(), None, None


def _loader_multi(path):
    return VALUES.copy(), ["a", "b"], None


def _loader_time(path):
    return VALUES.copy(), ["a", "b"], TIME.copy()


def _loader_two_files(path1, path2):
    return VALUES.copy(), None, None


@pytest.fixture(autouse=True)
def tiny_datasets(tmp_path, monkeypatch):
    monkeypatch.setattr(timeseries, "DATASETS_DIR", tmp_path)
    monkeypatch.setattr(timeseries, "KEEP_RAW", False)
    monkeypatch.setattr(timeseries, "_DATASETS_INFO", dict(timeseries._DATASETS_INFO))
    downloads = []

    def download(url, destination):
        downloads.append(url)
        if not destination.is_file():
            destination.write_text("raw")

    monkeypatch.setattr(timeseries, "_download", download)
    for name, loader in (
        ("tiny", _loader_uni),
        ("tiny-multi", _loader_multi),
        ("tiny-time", _loader_time),
    ):
        register(DatasetInfo(name, f"file:///{name}.txt", loader), force=True)
    register(
        DatasetInfo(
            "tiny-two",
            ("file:///a/part.txt", "file:///b/part2.txt"),
            _loader_two_files,
            filename=("one.txt", "two.txt"),
        ),
        force=True,
    )
    return downloads


def test_unknown_dataset_raises():
    with pytest.raises(KeyError, match="not available"):
        load("does-not-exist")
    with pytest.raises(KeyError, match="not available"):
        timeseries.cache_path("does-not-exist")


def test_phase_1_datasets_are_registered():
    assert {"astro", "ecg", "freezer", "gap", "humany", "steamgen"} <= set(
        timeseries.available_datasets()
    )


def test_register_rejects_duplicates_and_mismatched_filenames():
    with pytest.raises(ValueError, match="already registered"):
        register(DatasetInfo("tiny", "file:///x.txt", _loader_uni))
    with pytest.raises(ValueError, match="one file name per URL"):
        register(
            DatasetInfo("bad", ("file:///x", "file:///y"), _loader_uni, filename="x")
        )


# --- parsers


def test_load_values_gz_one_value_per_line_with_nan_and_blank_lines(tmp_path):
    path = tmp_path / "s.txt.gz"
    # CRLF line endings and blank lines (skipped, as pyattimo does) as in ECG
    with gzip.open(path, "wt", newline="") as f:
        f.write("1.5\r\n-2e-3\r\n\r\n\r\nnan\r\n4\r\n")
    values, dim_names, time = timeseries._load_values_gz(path)
    assert values.shape == (4, 1) and values.dtype == np.float64
    np.testing.assert_array_equal(values[:, 0], [1.5, -0.002, np.nan, 4.0])
    assert dim_names is None and time is None


def test_load_values_gz_plain_text_without_trailing_newline(tmp_path):
    path = tmp_path / "s.csv"
    path.write_text("1\n2\n3")
    values, _, _ = timeseries._load_values_gz(path)
    np.testing.assert_array_equal(values, [[1.0], [2.0], [3.0]])


def test_load_csv_keeps_column_names_order_and_nans(tmp_path):
    path = tmp_path / "s.csv"
    path.write_text("drum pressure,excess oxygen\n320.5,2.5\n,2.75\n321.25,\n")
    values, dim_names, time = timeseries._load_csv(path)
    assert dim_names == ["drum pressure", "excess oxygen"]
    np.testing.assert_array_equal(
        values, [[320.5, 2.5], [np.nan, 2.75], [321.25, np.nan]]
    )
    assert values.dtype == np.float64 and time is None


# --- cache


def test_cache_round_trip_is_bit_identical(tmp_path):
    path = tmp_path / "rt.parquet"
    timeseries._write_parquet_cache(path, VALUES, ("a", "b"), TIME, "1s", "http://x")
    values, dim_names, time = timeseries._read_parquet_cache(path)
    assert values.dtype == np.float64
    np.testing.assert_array_equal(values.view(np.uint64), VALUES.view(np.uint64))
    assert dim_names == ("a", "b")
    assert time.dtype == np.dtype("datetime64[ms]")
    np.testing.assert_array_equal(time, TIME)
    assert not path.with_suffix(".parquet.tmp").exists()


def test_cache_round_trip_univariate_without_time(tmp_path):
    path = tmp_path / "rt.parquet"
    x = np.array([[np.nan], [-0.0], [0.0], [5e-324]])
    timeseries._write_parquet_cache(path, x)
    values, dim_names, time = timeseries._read_parquet_cache(path)
    np.testing.assert_array_equal(values.view(np.uint64), x.view(np.uint64))
    assert dim_names is None and time is None


def test_cache_schema_and_metadata(tmp_path):
    path = tmp_path / "s.parquet"
    timeseries._write_parquet_cache(path, VALUES, ("a", "b"), TIME, "1s", "http://x")
    schema = pq.read_schema(path)
    assert schema.names == ["x0", "x1", "t"]
    assert [str(schema.field(c).type) for c in schema.names] == [
        "double",
        "double",
        "timestamp[ms]",
    ]
    meta = {
        "version": 1,
        "n_samples": 4,
        "n_dims": 2,
        "dim_names": ["a", "b"],
        "has_time": True,
        "sampling": "1s",
        "source": "http://x",
    }
    import json

    assert json.loads(schema.metadata[b"aida_data"]) == meta


def test_empty_series_round_trips(tmp_path):
    path = tmp_path / "e.parquet"
    timeseries._write_parquet_cache(path, np.empty((0, 3)))
    values, _, _ = timeseries._read_parquet_cache(path)
    assert values.shape == (0, 3)


def test_unknown_cache_version_raises(tmp_path):
    path = tmp_path / "old.parquet"
    timeseries._write_parquet_cache(path, VALUES)
    table = pq.read_table(path)
    pq.write_table(
        table.replace_schema_metadata({b"aida_data": b'{"version": 99}'}), path
    )
    with pytest.raises(ValueError, match="not a cache in a known format"):
        timeseries._read_parquet_cache(path)


def test_cache_without_metadata_raises(tmp_path):
    path = tmp_path / "plain.parquet"
    timeseries._write_parquet_cache(path, VALUES)
    pq.write_table(pq.read_table(path).replace_schema_metadata(None), path)
    with pytest.raises(ValueError, match="not a cache in a known format"):
        timeseries._read_parquet_cache(path)


# --- load


def test_load_univariate():
    ts = load("tiny")
    assert isinstance(ts, TimeSeries)
    assert ts.values.shape == (4, 1) and ts.values.dtype == np.float64
    assert (ts.n_samples, ts.n_dims) == (4, 1)
    assert ts.dim_names is None and ts.time is None
    assert ts.univariate.shape == (4,)
    np.testing.assert_array_equal(ts.univariate, VALUES[:, 0])


def test_load_multivariate_keeps_row_order_nan_and_inf():
    ts = load("tiny-multi")
    np.testing.assert_array_equal(ts.values, VALUES)
    assert ts.dim_names == ("a", "b")
    assert np.isnan(ts.values[1, 0]) and np.isinf(ts.values[3, 1])


def test_load_with_time():
    ts = load("tiny-time")
    np.testing.assert_array_equal(ts.time, TIME)


def test_univariate_raises_for_several_dims():
    with pytest.raises(ValueError, match="2 dimensions"):
        _ = load("tiny-multi").univariate


def test_time_series_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        load("tiny").name = "other"


def test_loader_output_is_validated(monkeypatch):
    def check(loader, match):
        register(DatasetInfo("bad", "file:///bad.txt", loader), force=True)
        with pytest.raises(ValueError, match=match):
            load("bad")
        assert not timeseries.cache_path("bad").exists()

    check(lambda p: (VALUES, ["only-one"], None), "dimension names")
    check(lambda p: (VALUES, None, TIME[:2]), "time must have shape")
    check(lambda p: (np.zeros((3, 2, 2)), None, None), "expected \\(n, d\\)")


def test_one_dimensional_loader_output_becomes_a_column():
    register(
        DatasetInfo("flat", "file:///flat.txt", lambda p: (VALUES[:, 0], None, None)),
        force=True,
    )
    assert load("flat").values.shape == (4, 1)


def test_first_and_later_calls_are_identical():
    first, second = load("tiny-time"), load("tiny-time")
    np.testing.assert_array_equal(first.values, second.values)
    np.testing.assert_array_equal(first.time, second.time)


def test_second_call_does_not_parse_or_download(monkeypatch, tiny_datasets):
    first = load("tiny")
    assert len(tiny_datasets) == 1

    def fail(*args):
        raise AssertionError("must not parse or download")

    monkeypatch.setattr(timeseries, "_download", fail)
    monkeypatch.setitem(
        timeseries._DATASETS_INFO,
        "tiny",
        dataclasses.replace(timeseries._DATASETS_INFO["tiny"], loader_function=fail),
    )
    np.testing.assert_array_equal(first.values, load("tiny").values)


# --- raw files


def test_paths(tmp_path):
    assert timeseries.cache_path("tiny") == tmp_path / "timeseries" / "tiny.parquet"
    assert timeseries.local_paths("tiny") == (tmp_path / "timeseries" / "tiny.txt",)
    assert timeseries.local_paths("tiny-two") == (
        tmp_path / "timeseries" / "one.txt",
        tmp_path / "timeseries" / "two.txt",
    )
    # default file names come from the URLs
    register(
        DatasetInfo(
            "u", ("file:///a/part.txt", "file:///b/part2.txt"), _loader_two_files
        ),
        force=True,
    )
    assert [p.name for p in timeseries.local_paths("u")] == ["part.txt", "part2.txt"]


def test_several_files_are_downloaded_and_all_deleted(tiny_datasets):
    load("tiny-two")
    assert tiny_datasets == ["file:///a/part.txt", "file:///b/part2.txt"]
    assert not any(path.exists() for path in timeseries.local_paths("tiny-two"))


def test_raw_deleted_after_parse():
    load("tiny")
    assert timeseries.cache_path("tiny").is_file()
    assert not any(path.exists() for path in timeseries.local_paths("tiny"))


def test_keep_raw(monkeypatch):
    monkeypatch.setattr(timeseries, "KEEP_RAW", True)
    load("tiny")
    assert all(path.is_file() for path in timeseries.local_paths("tiny"))


def test_raw_left_alone_when_cache_exists(monkeypatch):
    monkeypatch.setattr(timeseries, "KEEP_RAW", True)
    load("tiny")
    monkeypatch.setattr(timeseries, "KEEP_RAW", False)
    load("tiny")
    assert all(path.is_file() for path in timeseries.local_paths("tiny"))


def test_shared_raw_file_kept_until_all_users_are_cached():
    register(DatasetInfo("a", "file:///shared.txt", _loader_uni), force=True)
    register(DatasetInfo("b", "file:///shared.txt", _loader_uni), force=True)
    (shared,) = timeseries.local_paths("a")
    load("a")
    assert shared.is_file()  # b still needs it
    assert timeseries.prune_raw() == []
    load("b")
    assert not shared.exists()


def test_prune_raw():
    (path,) = timeseries.local_paths("tiny")
    path.parent.mkdir(parents=True)
    path.write_text("raw")
    # No cache yet: nothing to prune.
    assert timeseries.prune_raw(dry_run=False) == []
    assert path.is_file()
    timeseries._write_parquet_cache(timeseries.cache_path("tiny"), VALUES)
    assert timeseries.prune_raw() == [path]
    assert path.is_file()  # dry run
    timeseries.prune_raw(dry_run=False)
    assert not path.exists()


# --- real downloads


@pytest.mark.network
@pytest.mark.parametrize(
    "name, n, d",
    [
        ("astro", 1151350, 1),
        ("ecg", 7824879, 1),
        ("freezer", 7430755, 1),
        ("gap", 2049280, 1),
        ("humany", 26415045, 1),
        ("steamgen", 9600, 4),
    ],
)
def test_phase_1_sources(monkeypatch, name, n, d):
    from aida_data._download import download

    monkeypatch.setattr(timeseries, "_download", download)
    ts = load(name)
    assert ts.values.shape == (n, d)
    assert np.isfinite(ts.values).all()
    assert not any(path.exists() for path in timeseries.local_paths(name))
    if name == "steamgen":
        assert ts.dim_names == (
            "drum pressure",
            "excess oxygen",
            "water level",
            "steam flow",
        )
    # a second call reads the cache and returns the same values
    np.testing.assert_array_equal(load(name).values, ts.values)
