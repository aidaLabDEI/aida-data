"""Conversion of the parquet caches to HDF5, offline: round trips, freshness,
disk space, pruning, and the declarative layout list."""

import collections
import json
import os
import shutil

import h5py
import numpy as np
import pytest

from aida_data import _hdf5, _mteb, dense, graph, timeseries
from aida_data.dense import Colors

NAME = "mteb:T:org/m"
N = 10
BATCH = 3  # fewer than the rows, so that several batches are written

POINTS = np.arange(N * 4, dtype=np.float32).reshape(N, 4) / 7
COLORS = Colors(
    np.arange(N).reshape(-1, 1) % 3 * np.array([[1, 100]]),
    ("a", "b"),
    (("x", "y", "z"), tuple(f"l{i}" for i in range(201))),
)
IDS = np.array([f"id{i}" for i in range(N)], dtype=object)

EDGES = np.array([[0, 1], [0, 3], [1, 2], [2, 4], [4, 0], [6, 5]])
GRAPH_COLORS = np.array([7, 3, 12, -1, 7, 3, 3, 0], dtype=np.int64)  # node 7 has no edges

VALUES = np.arange(N * 2, dtype=np.float64).reshape(N, 2) * 1.5
VALUES[3, 1] = np.nan
TIME = np.datetime64("2020-01-01T00:00:00", "ms") + np.arange(N) * 1000


@pytest.fixture(autouse=True)
def roots(tmp_path, monkeypatch):
    for module in (dense, graph, timeseries):
        monkeypatch.setattr(module, "DATASETS_DIR", tmp_path)
        monkeypatch.setattr(module, "_DATASETS_INFO", dict(module._DATASETS_INFO))
    dense.register(dense.DatasetInfo("fake", "https://e.com/f.zip", None, "euclidean", "fake-cache"))
    graph.register(graph.DatasetInfo("fake", "file:///e", "file:///c", None))
    timeseries.register(timeseries.DatasetInfo("fake", "file:///t", None))
    timeseries.register(timeseries.DatasetInfo("fake-nt", "file:///t", None))

    def no_download(*args, **kwargs):
        raise AssertionError("no download expected")

    monkeypatch.setattr(dense, "_download", no_download)
    monkeypatch.setattr(graph, "_download", no_download)
    monkeypatch.setattr(timeseries, "_download", no_download)
    return tmp_path


def _dense_cache(roots):
    (roots / "dense").mkdir(exist_ok=True)
    path = roots / "dense" / "fake-cache.parquet"
    dense._write_parquet_cache(path, POINTS, COLORS, IDS)
    return path


def _retrieval_result(n_queries=4):
    return _mteb.Encoded(
        POINTS,
        IDS,
        None,
        POINTS[:n_queries] * 2,
        np.array([f"q{i}" for i in range(n_queries)], dtype=object),
        _mteb.build_qrels({"q0": {"id1": 2, "id2": 1}, "q3": {"id9": 1, "id0": 0}}),
        {"similarity": "cosine", "task_type": "Retrieval"},
    )


def _written(hfp):
    """The datasets (paths) and root attributes written in `hfp`."""
    objects = set(hfp.attrs)
    hfp.visititems(
        lambda path, obj: objects.add(path) if isinstance(obj, h5py.Dataset) else None
    )
    return objects


def _check_layout(hfp, layout):
    """Every object written is in the layout list, every required one is there."""
    declared = {obj.path for obj in _hdf5.layout_objects(layout)}
    written = _written(hfp)
    assert written <= declared, written - declared
    assert {o.path for o in _hdf5.layout_objects(layout) if o.required} <= written


def _strings(dataset):
    return list(dataset.asstr()[:])


# --- round trips


def test_dense_round_trip(roots):
    _dense_cache(roots)
    path = _hdf5.ensure_hdf5("dense", "fake", batch_size=BATCH)
    assert path == roots / "dense" / "hdf5" / "fake.hdf5"
    ds = dense.load("fake", deduplicate=False)
    with h5py.File(path) as hfp:
        _check_layout(hfp, "dense")
        assert hfp["train"].dtype == np.float32 and hfp["train"].chunks is None
        np.testing.assert_array_equal(hfp["train"][:], ds.dataset)
        np.testing.assert_array_equal(hfp["colors"][:], ds.colors.values)
        assert hfp["colors"].dtype == np.uint8
        assert _strings(hfp["ids"]) == list(IDS)
        assert "test" not in hfp and "qrels" not in hfp
        assert hfp.attrs["distance"] == "euclidean"
        assert hfp.attrs["dimension"] == 4
        assert list(hfp.attrs["color_names"]) == ["a", "b"]
        assert json.loads(hfp.attrs["color_labels"]) == [list(l) for l in COLORS.labels]
        assert hfp.attrs["module"] == "dense" and hfp.attrs["name"] == "fake"
        assert hfp.attrs["aida_data_hdf5_version"] == _hdf5._HDF5_VERSION


def test_mteb_retrieval_round_trip(tmp_path):
    result = _retrieval_result()
    _mteb.write_cache(result, _mteb.cache_dir("T", "org/m"))
    path = _hdf5.ensure_hdf5("dense", NAME, batch_size=BATCH)
    assert path == tmp_path / "dense" / "hdf5" / "mteb" / "T" / "org__m.hdf5"
    ds = dense.load(NAME, deduplicate=False)
    qrels = dense.load_qrels(NAME)
    with h5py.File(path) as hfp:
        _check_layout(hfp, "dense")
        np.testing.assert_array_equal(hfp["train"][:], ds.dataset)
        np.testing.assert_array_equal(hfp["test"][:], ds.queries)
        assert _strings(hfp["ids"]) == list(ds.ids)
        assert _strings(hfp["test_ids"]) == list(ds.query_ids)
        assert "colors" not in hfp
        from_file = collections.defaultdict(dict)
        for q, d, s in zip(
            _strings(hfp["qrels/query"]), _strings(hfp["qrels/doc"]), hfp["qrels/score"][:]
        ):
            from_file[q][d] = int(s)
        assert dict(from_file) == qrels == {"q0": {"id1": 2, "id2": 1}, "q3": {"id9": 1}}
        assert hfp.attrs["distance"] == "cosine"
        assert hfp.attrs["name"] == NAME


def test_mteb_clustering_round_trip(tmp_path):
    _, colors = _mteb.clustering_colors_v1([(["a", "b"], [0, 1]), (["b", "c"], [5, 5])])
    result = _mteb.Encoded(
        POINTS[:3], None, colors, None, None, None,
        {"similarity": "euclidean", "task_type": "Clustering"},
    )
    _mteb.write_cache(result, _mteb.cache_dir("T", "org/m"))
    path = _hdf5.ensure_hdf5("dense", NAME, batch_size=2)
    with h5py.File(path) as hfp:
        _check_layout(hfp, "dense")
        np.testing.assert_array_equal(hfp["colors"][:], colors.values)
        assert not {"test", "ids", "test_ids", "qrels"} & set(hfp)
        assert list(hfp.attrs["color_names"]) == list(colors.names)


def test_graph_round_trip(roots):
    (roots / "graphs").mkdir()
    graph._write_parquet_cache(graph.cache_path("fake"), EDGES, GRAPH_COLORS)
    path = _hdf5.ensure_hdf5("graph", "fake", batch_size=BATCH)
    assert path == roots / "graphs" / "hdf5" / "fake.hdf5"
    g = graph.load_edge_list("fake", directed=True)
    with h5py.File(path) as hfp:
        _check_layout(hfp, "graph")
        offsets, neighbors = hfp["offsets"][:], hfp["neighbors"][:]
        assert offsets.dtype == np.int64 and len(offsets) == len(g.colors) + 1
        rebuilt = np.array(
            [
                (v, u)
                for v in range(len(g.colors))
                for u in neighbors[offsets[v] : offsets[v + 1]]
            ]
        )
        np.testing.assert_array_equal(rebuilt, g.edges)
        for v in range(len(g.colors)):  # sorted ascending per node
            assert (np.diff(neighbors[offsets[v] : offsets[v + 1]]) > 0).all()
        np.testing.assert_array_equal(hfp["colors"][:], g.colors)
        assert hfp.attrs["n_nodes"] == g.n_nodes and hfp.attrs["n_edges"] == g.n_edges


def test_timeseries_round_trip(roots):
    (roots / "timeseries").mkdir()
    timeseries._write_parquet_cache(
        timeseries.cache_path("fake"), VALUES, ["u", "v"], TIME, "1s", "file:///t"
    )
    path = _hdf5.ensure_hdf5("timeseries", "fake", batch_size=BATCH)
    ts = timeseries.load("fake")
    with h5py.File(path) as hfp:
        _check_layout(hfp, "timeseries")
        np.testing.assert_array_equal(hfp["values"][:], ts.values)
        assert hfp["time"].dtype == np.int64
        np.testing.assert_array_equal(hfp["time"][:].astype("datetime64[ms]"), ts.time)
        assert list(hfp.attrs["dim_names"]) == ["u", "v"]
        assert hfp.attrs["sampling"] == "1s" and hfp.attrs["source"] == "file:///t"


def test_timeseries_without_time_or_names(roots):
    (roots / "timeseries").mkdir()
    timeseries._write_parquet_cache(timeseries.cache_path("fake-nt"), VALUES)
    path = _hdf5.ensure_hdf5("timeseries", "fake-nt", batch_size=BATCH)
    with h5py.File(path) as hfp:
        _check_layout(hfp, "timeseries")
        assert not {"time", "dim_names", "sampling", "source"} & (set(hfp) | set(hfp.attrs))


# --- freshness


def test_second_ensure_does_not_rewrite(roots):
    _dense_cache(roots)
    path = _hdf5.ensure_hdf5("dense", "fake")
    assert _hdf5.is_fresh("dense", "fake")
    os.utime(path, ns=(10**9, 10**9))
    assert _hdf5.ensure_hdf5("dense", "fake") == path
    assert path.stat().st_mtime_ns == 10**9


def test_touching_the_cache_rebuilds(roots):
    cache = _dense_cache(roots)
    path = _hdf5.ensure_hdf5("dense", "fake")
    os.utime(path, ns=(10**9, 10**9))
    stat = cache.stat()
    os.utime(cache, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5))
    assert not _hdf5.is_fresh("dense", "fake")
    _hdf5.ensure_hdf5("dense", "fake")
    assert path.stat().st_mtime_ns != 10**9
    with h5py.File(path) as hfp:
        assert hfp.attrs["source_mtime_ns"] == cache.stat().st_mtime_ns


def test_new_hdf5_version_rebuilds(roots, monkeypatch):
    _dense_cache(roots)
    path = _hdf5.ensure_hdf5("dense", "fake")
    os.utime(path, ns=(10**9, 10**9))
    monkeypatch.setattr(_hdf5, "_HDF5_VERSION", 2)
    assert not _hdf5.is_fresh("dense", "fake")
    _hdf5.ensure_hdf5("dense", "fake")
    with h5py.File(path) as hfp:
        assert hfp.attrs["aida_data_hdf5_version"] == 2


def test_is_fresh_without_cache_or_file(roots):
    assert not _hdf5.is_fresh("dense", "fake")  # no cache, nothing downloaded
    _dense_cache(roots)
    assert not _hdf5.is_fresh("dense", "fake")  # cache but no conversion
    assert not (roots / "dense" / "hdf5").exists()


# --- location, native datasets


def test_converted_file_is_not_a_legacy_cache(roots):
    _dense_cache(roots)
    path = _hdf5.ensure_hdf5("dense", "fake")
    assert path != roots / "dense" / "fake-cache.hdf5"
    assert not (roots / "dense" / "fake-cache.hdf5").exists()
    dense.load("fake")
    assert path.is_file()  # `load` neither migrated nor deleted it


def test_native_dataset_is_returned_as_is(roots, monkeypatch):
    dense.register(
        dense.DatasetInfo("native", "https://e.com/native.hdf5", dense._load_hdf5, "angular")
    )
    downloads = []
    monkeypatch.setattr(dense, "_download", lambda url, dest: downloads.append(url))
    native = dense.local_path("native")
    native.parent.mkdir(parents=True)
    with h5py.File(native, "w") as hfp:
        hfp["train"] = POINTS
    assert _hdf5.hdf5_path("dense", "native") == native
    assert _hdf5.is_fresh("dense", "native")
    assert _hdf5.ensure_hdf5("dense", "native") == native
    assert not (roots / "dense" / "hdf5").exists()  # not converted
    assert _hdf5.prune_hdf5(dry_run=True) == []


# --- disk space


def test_not_enough_space(roots, monkeypatch):
    _dense_cache(roots)
    Usage = collections.namedtuple("Usage", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda path: Usage(100, 100, 0))
    with pytest.raises(OSError, match="--format parquet"):
        _hdf5.ensure_hdf5("dense", "fake")
    assert list((roots / "dense" / "hdf5").iterdir()) == []


def test_conversion_error_leaves_no_temporary(roots):
    _dense_cache(roots)
    # A cache with a column missing fails in the middle of the stream.
    import pyarrow.parquet as pq

    path = roots / "dense" / "fake-cache.parquet"
    table = pq.read_table(path)
    pq.write_table(table.drop_columns(["x3"]).replace_schema_metadata(table.schema.metadata), path)
    with pytest.raises(Exception):
        _hdf5.ensure_hdf5("dense", "fake")
    assert list((roots / "dense" / "hdf5").iterdir()) == []


# --- pruning


def test_prune_lists_converted_files_only(roots):
    _dense_cache(roots)
    (roots / "graphs").mkdir()
    graph._write_parquet_cache(graph.cache_path("fake"), EDGES, GRAPH_COLORS)
    (roots / "dense" / "native.hdf5").write_bytes(b"native download")
    converted = [
        _hdf5.ensure_hdf5("dense", "fake"),
        _hdf5.ensure_hdf5("graph", "fake"),
    ]
    _mteb.write_cache(_retrieval_result(), _mteb.cache_dir("T", "org/m"))
    converted.append(_hdf5.ensure_hdf5("dense", NAME))
    stale = roots / "dense" / "hdf5" / "mteb" / "x.hdf5.abcd.hdf5.tmp"
    stale.write_bytes(b"x")
    os.utime(stale, (1, 1))
    fresh_tmp = roots / "dense" / "hdf5" / "y.hdf5.abcd.hdf5.tmp"
    fresh_tmp.write_bytes(b"y")  # a conversion in progress

    assert sorted(_hdf5.prune_hdf5(dry_run=True)) == sorted([*converted, stale])
    assert all(p.exists() for p in converted)
    _hdf5.prune_hdf5(dry_run=False)
    assert not any(p.exists() for p in [*converted, stale])
    assert fresh_tmp.exists()
    assert (roots / "dense" / "native.hdf5").exists()
    assert (roots / "dense" / "fake-cache.parquet").exists()
    assert graph.cache_path("fake").exists()
    assert (_mteb.cache_dir("T", "org/m") / "corpus.parquet").exists()


def test_prune_raw_lists_stale_hdf5_temporaries(roots):
    stale = []
    for directory in ("dense/hdf5/mteb/T", "graphs/hdf5", "timeseries/hdf5"):
        (roots / directory).mkdir(parents=True)
        path = roots / directory / "a.hdf5.abcd.hdf5.tmp"
        path.write_bytes(b"x")
        os.utime(path, (1, 1))
        stale.append(path)
    assert dense.prune_raw() == [stale[0]]
    assert graph.prune_raw() == [stale[1]]
    assert timeseries.prune_raw() == [stale[2]]


# --- MTEB without encoding


@pytest.fixture
def no_encode(monkeypatch):
    def encode(*args, **kwargs):
        raise AssertionError("encode_task must never be called")

    monkeypatch.setattr(_mteb, "encode_task", encode)


def test_ensure_mteb_cached(no_encode, monkeypatch):
    def fetch(*args):
        raise AssertionError("cached: no fetch")

    monkeypatch.setattr(_mteb, "fetch_hosted", fetch)
    directory = _mteb.cache_dir("T", "org/m")
    _mteb.write_cache(_retrieval_result(), directory)
    assert dense._ensure_mteb(NAME) == directory


def test_ensure_mteb_hosted(no_encode, monkeypatch):
    def fetch(task, model, dest):
        _mteb.write_cache(_retrieval_result(), dest)
        return True

    monkeypatch.setattr(_mteb, "fetch_hosted", fetch)
    directory = dense._ensure_mteb(NAME)
    assert (directory / "corpus.parquet").is_file()


def test_ensure_mteb_not_available(no_encode, monkeypatch):
    monkeypatch.setattr(_mteb, "fetch_hosted", lambda *args: False)
    with pytest.raises(LookupError, match="not hosted and not cached; encode it first") as e:
        dense._ensure_mteb(NAME)
    assert NAME in str(e.value) and "aida-data[mteb]" in str(e.value)
    with pytest.raises(LookupError):
        _hdf5.ensure_hdf5("dense", NAME)


# --- layout list


def test_layout_list_is_consistent():
    keys = [(o.layout, o.path, o.attribute) for o in _hdf5.LAYOUT]
    assert len(keys) == len(set(keys))
    assert {o.layout for o in _hdf5.LAYOUT} == {"native", "dense", "graph", "timeseries"}
    assert {o.ctype for o in _hdf5.LAYOUT} <= {"float", "double", "int64_t", "std::string"}
    native = {o.path for o in _hdf5.layout_objects("native")}
    assert native == {"train", "test", "neighbors", "distances", "distance"}
