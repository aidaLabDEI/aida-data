"""Offline tests for the MTEB dense datasets: ids in caches and filters."""

import os
import time

import numpy as np

from aida_data import dense
from aida_data._cache import write_table
from aida_data.dense import Colors, _process


def _colors(values):
    return Colors(np.asarray(values).reshape(-1, 1), ("c",), (("a", "b", "c"),))


def test_parquet_round_trip_with_ids_and_meta(tmp_path):
    data = np.arange(12, dtype=np.float32).reshape(4, 3)
    ids = np.array(["d0", "d1", "d2", "d3"], dtype=object)
    colors = _colors([0, 1, 2, 1])
    path = tmp_path / "corpus.parquet"
    dense._write_parquet_cache(path, data, colors, ids, {"similarity": "cosine"})
    out, out_colors, out_ids, meta = dense._read_parquet_cache_full(path)
    np.testing.assert_array_equal(out, data)
    np.testing.assert_array_equal(out_colors.values, colors.values)
    assert list(out_ids) == list(ids)
    assert meta["similarity"] == "cosine" and meta["has_ids"] is True
    assert meta["version"] == dense._CACHE_VERSION
    # the two-value reader still works
    out2, colors2 = dense._read_parquet_cache(path)
    np.testing.assert_array_equal(out2, data)
    assert colors2 is not None


def test_no_ids_gives_none(tmp_path):
    data = np.ones((3, 2), dtype=np.float32)
    path = tmp_path / "x.parquet"
    dense._write_parquet_cache(path, data, None)
    _, colors, ids, meta = dense._read_parquet_cache_full(path)
    assert colors is None and ids is None and meta["has_ids"] is False


def test_old_cache_without_new_keys_loads(tmp_path):
    import pyarrow as pa

    path = tmp_path / "old.parquet"
    table = pa.table({"x0": np.array([1, 2], dtype=np.float32)})
    write_table(path, table, {"version": 1, "n_features": 1,
                              "color_names": None, "color_labels": None})
    data, colors, ids, meta = dense._read_parquet_cache_full(path)
    assert data.shape == (2, 1) and colors is None and ids is None
    assert "has_ids" not in meta


def _run(train, ids, colors=None, test=None, query_ids=None, distance="euclidean",
         deduplicate=True):
    return _process(distance, train, ids, colors, test, query_ids, None, None,
                    deduplicate)


def test_dedup_keeps_first_id_and_ignores_ids():
    train = np.array([[1, 1], [2, 2], [1, 1], [3, 3]], dtype=np.float32)
    ids = np.array(["a", "b", "c", "d"], dtype=object)
    ds = _run(train, ids)
    assert list(ds.ids) == ["a", "b", "d"]
    assert ds.dataset.shape[0] == 3
    ds = _run(train, ids, deduplicate=False)
    assert list(ds.ids) == ["a", "b", "c", "d"]


def test_dedup_respects_colors_with_ids():
    train = np.array([[1, 1], [1, 1], [1, 1]], dtype=np.float32)
    ids = np.array(["a", "b", "c"], dtype=object)
    ds = _run(train, ids, _colors([0, 1, 0]))
    assert list(ds.ids) == ["a", "b"]
    np.testing.assert_array_equal(ds.colors.values[:, 0], [0, 1])


def test_nan_filter_aligns_ids_and_query_ids():
    train = np.array([[1, 1], [np.nan, 2], [3, 3]], dtype=np.float32)
    test = np.array([[np.inf, 0], [5, 5]], dtype=np.float32)
    ds = _run(
        train,
        np.array(["a", "b", "c"], dtype=object),
        _colors([0, 1, 2]),
        test,
        np.array(["q0", "q1"], dtype=object),
    )
    assert list(ds.ids) == ["a", "c"]
    np.testing.assert_array_equal(ds.colors.values[:, 0], [0, 2])
    assert list(ds.query_ids) == ["q1"]
    assert ds.queries.shape[0] == 1


def test_zero_row_filter_cosine_aligns_ids():
    train = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
    test = np.array([[1, 1], [0, 0]], dtype=np.float32)
    args = (train, np.array(["a", "b", "c"], dtype=object), None, test,
            np.array(["q0", "q1"], dtype=object))
    ds = _run(*args, distance="cosine")
    assert list(ds.ids) == ["b", "c"] and list(ds.query_ids) == ["q0"]
    # euclidean keeps zero rows
    ds = _run(*args)
    assert list(ds.ids) == ["a", "b", "c"] and list(ds.query_ids) == ["q0", "q1"]


def test_ids_none_stay_none():
    ds = _run(np.ones((2, 2), dtype=np.float32), None, test=np.ones((1, 2), np.float32))
    assert ds.ids is None and ds.query_ids is None


def test_prune_raw_finds_stale_temporary_under_mteb(tmp_path, monkeypatch):
    monkeypatch.setattr(dense, "DATASETS_DIR", tmp_path)
    directory = tmp_path / "dense" / "mteb" / "TaskX" / "model-slug"
    directory.mkdir(parents=True)
    stale = directory / "corpus.parquet.abc.parquet.tmp"
    fresh = directory / "queries.parquet.def.parquet.tmp"
    real = directory / "corpus.parquet"
    for path in (stale, fresh, real):
        path.write_text("x")
    old = time.time() - 25 * 3600
    os.utime(stale, (old, old))
    os.utime(real, (old, old))
    assert dense.prune_raw() == [stale]
    dense.prune_raw(dry_run=False)
    assert not stale.exists() and fresh.exists() and real.exists()
