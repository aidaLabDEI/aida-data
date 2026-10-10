import dataclasses
import gzip
import sys

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


def test_phase_1_and_2_datasets_are_registered():
    assert {
        "astro",
        "ecg",
        "freezer",
        "gap",
        "humany",
        "steamgen",
        "dishwasher",
        "npo141",
        "arrhythmia",
        "foetal-ecg",
        "evaporator",
        "ruth",
        "oikolab-weather",
        "fl010",
    } <= set(timeseries.available_datasets())


def test_register_rejects_duplicates_and_mismatched_filenames():
    with pytest.raises(ValueError, match="already registered"):
        register(DatasetInfo("tiny", "file:///x.txt", _loader_uni))
    with pytest.raises(ValueError, match="one file name per URL"):
        register(
            DatasetInfo("bad", ("file:///x", "file:///y"), _loader_uni, filename="x")
        )


# --- parsers


def test_load_matrix_one_value_per_line_with_nan_and_blank_lines(tmp_path):
    path = tmp_path / "s.txt.gz"
    # CRLF line endings and blank lines (skipped, as pyattimo does) as in ECG
    with gzip.open(path, "wt", newline="") as f:
        f.write("1.5\r\n-2e-3\r\n\r\n\r\nnan\r\n4\r\n")
    values, dim_names, time = timeseries._load_matrix(path)
    assert values.shape == (4, 1) and values.dtype == np.float64
    np.testing.assert_array_equal(values[:, 0], [1.5, -0.002, np.nan, 4.0])
    assert dim_names is None and time is None


def test_load_matrix_plain_text_without_trailing_newline(tmp_path):
    path = tmp_path / "s.csv"
    path.write_text("1\n2\n3")
    values, _, _ = timeseries._load_matrix(path)
    np.testing.assert_array_equal(values, [[1.0], [2.0], [3.0]])


def test_load_matrix_options(tmp_path):
    path = tmp_path / "s.dat"
    path.write_text("t,a,b\n0.0,1,2\n0.5,3,4\n")
    values, dim_names, time = timeseries._load_matrix(
        path, delimiter=",", skiprows=1, drop_first_columns=1
    )
    np.testing.assert_array_equal(values, [[1.0, 2.0], [3.0, 4.0]])
    assert dim_names is None and time is None


def test_load_matrix_whitespace_separated_keeps_every_row(tmp_path):
    # No header: the first row is data. One row and one column stay 2-D.
    path = tmp_path / "s.dat"
    path.write_text("  1.0e+00\t -2.5e+00 \n 3.0e+00\t 4.0e+00\n")
    values, _, _ = timeseries._load_matrix(path)
    np.testing.assert_array_equal(values, [[1.0, -2.5], [3.0, 4.0]])
    path.write_text("7 8 9\n")
    assert timeseries._load_matrix(path)[0].shape == (1, 3)


TSF = """\
# a comment
@relation Tiny
@attribute series_name string
@attribute start_timestamp date
@attribute type string
@frequency hourly
@missing true
@equallength true
@data
T1:2010-01-01 00-00-00:temperature:1.5,?,3,4
T2:2010-01-01 00-00-00:pressure:10,20,30,40.25
"""


def test_load_tsf_one_dimension_per_series(tmp_path):
    path = tmp_path / "s.tsf"
    path.write_text(TSF)
    values, dim_names, time = timeseries._load_tsf(path)
    assert dim_names == ["temperature", "pressure"]
    np.testing.assert_array_equal(
        values, [[1.5, 10], [np.nan, 20], [3, 30], [4, 40.25]]
    )
    assert values.dtype == np.float64
    np.testing.assert_array_equal(
        time,
        np.array(
            [
                "2010-01-01T00:00",
                "2010-01-01T01:00",
                "2010-01-01T02:00",
                "2010-01-01T03:00",
            ],
            dtype="datetime64[ms]",
        ),
    )


def test_load_tsf_falls_back_to_series_names_and_skips_time(tmp_path):
    path = tmp_path / "s.tsf"
    text = TSF.replace("type", "kind").replace("hourly", "monthly")
    path.write_text(text.replace("temperature", "same").replace("pressure", "same"))
    _, dim_names, time = timeseries._load_tsf(path)
    assert dim_names == ["T1", "T2"]
    assert time is None


def test_load_tsf_rejects_unequal_lengths_and_data_before_tag(tmp_path):
    path = tmp_path / "s.tsf"
    path.write_text(TSF.replace("10,20,30,40.25", "10,20"))
    with pytest.raises(ValueError, match="different lengths"):
        timeseries._load_tsf(path)
    path.write_text("@attribute a string\nT1:a:1,2\n")
    with pytest.raises(ValueError, match="before the @data tag"):
        timeseries._load_tsf(path)


# start_timestamp is the last attribute, as in the Monash files
TSF_LAST = """\
@attribute series_name string
@attribute type string
@attribute start_timestamp date
@frequency hourly
@data
T1:temperature:2010-01-01 00-00-00:1.5,?,3
T2:pressure:2010-01-01 00-00-00:10,20,30
"""


def test_load_tsf_colon_clock_in_last_attribute(tmp_path):
    path = tmp_path / "s.tsf"
    path.write_text(TSF_LAST.replace("00-00-00", "00:00:00"))
    values, dim_names, time = timeseries._load_tsf(path)
    assert dim_names == ["temperature", "pressure"]
    np.testing.assert_array_equal(values, [[1.5, 10], [np.nan, 20], [3, 30]])
    assert time[0] == np.datetime64("2010-01-01T00:00:00")
    assert time[2] == np.datetime64("2010-01-01T02:00:00")


def test_load_tsf_colon_and_dash_clocks_give_identical_output(tmp_path):
    path = tmp_path / "s.tsf"
    path.write_text(TSF_LAST)
    dashes = timeseries._load_tsf(path)
    path.write_text(TSF_LAST.replace("00-00-00", "00:00:00"))
    colons = timeseries._load_tsf(path)
    np.testing.assert_array_equal(dashes[0], colons[0])
    assert dashes[1] == colons[1]
    np.testing.assert_array_equal(dashes[2], colons[2])


def test_load_tsf_date_only_start_timestamp_starts_at_midnight(tmp_path):
    path = tmp_path / "s.tsf"
    path.write_text(TSF_LAST.replace(" 00-00-00", ""))
    _, _, time = timeseries._load_tsf(path)
    assert time[0] == np.datetime64("2010-01-01T00:00:00")
    assert time[1] == np.datetime64("2010-01-01T01:00:00")


def test_load_tsf_colon_in_non_last_attribute_raises(tmp_path):
    path = tmp_path / "s.tsf"
    path.write_text(TSF_LAST.replace("T2:pressure", "T2:pres:sure"))
    with pytest.raises(ValueError, match=r"line 7 .*start_timestamp"):
        timeseries._load_tsf(path)


def test_load_tsf_too_few_fields_raises_with_line_number(tmp_path):
    path = tmp_path / "s.tsf"
    path.write_text(TSF_LAST.replace("T2:pressure:", "T2:"))
    with pytest.raises(ValueError, match="line 7"):
        timeseries._load_tsf(path)


def test_load_tsf_without_attributes(tmp_path):
    path = tmp_path / "s.tsf"
    path.write_text("@frequency hourly\n@data\n1,2,3\n")
    values, dim_names, time = timeseries._load_tsf(path)
    np.testing.assert_array_equal(values, [[1], [2], [3]])
    assert dim_names is None and time is None
    path.write_text("@data\nT1:1,2,3\n")
    with pytest.raises(ValueError, match="line 2"):
        timeseries._load_tsf(path)


def test_load_wfdb_reads_a_record(tmp_path):
    wfdb = pytest.importorskip("wfdb")
    signal = np.column_stack([np.linspace(-1, 1, 50), np.sin(np.arange(50) / 5)])
    wfdb.wrsamp(
        "rec",
        fs=100,
        units=["g", "g"],
        sig_name=["acc", "gyro"],
        p_signal=signal,
        fmt=["16", "16"],
        write_dir=str(tmp_path),
    )
    values, dim_names, time = timeseries._load_wfdb(
        tmp_path / "rec.hea", tmp_path / "rec.dat"
    )
    assert values.shape == (50, 2) and values.dtype == np.float64
    np.testing.assert_allclose(values, signal, atol=1e-3)
    assert list(dim_names) == ["acc", "gyro"] and time is None


def test_load_wfdb_without_wfdb_explains_how_to_install(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "wfdb", None)  # import raises ImportError
    with pytest.raises(ImportError, match=r"aida-data\[wfdb\]"):
        timeseries._load_wfdb(tmp_path / "rec.hea", tmp_path / "rec.dat")


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
    assert list(tmp_path.glob("*.tmp")) == []


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
        ("dishwasher", 245152, 1),
        ("npo141", 269287, 1),
        ("arrhythmia", 650000, 1),
        ("foetal-ecg", 2500, 8),
        ("evaporator", 6305, 6),
        ("ruth", 14859, 32),
        ("oikolab-weather", 100057, 8),
        ("fl010", 25132289, 6),
    ],
)
def test_real_sources(monkeypatch, name, n, d):
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
    if name == "fl010":
        assert ts.dim_names[0] == "v-acceleration"
        assert not any(path.exists() for path in timeseries.local_paths(name))
    if name == "oikolab-weather":
        assert ts.dim_names[:2] == ("temperature", "dewpoint_temperature")
        assert ts.time[0] == np.datetime64("2010-01-01T00:00")
        assert (np.diff(ts.time) == np.timedelta64(1, "h")).all()
    # a second call reads the cache and returns the same values
    np.testing.assert_array_equal(load(name).values, ts.values)
