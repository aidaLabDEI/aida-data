"""Offline tests for the MTEB dense datasets: ids in caches and filters."""

import os
import time

import numpy as np
import pytest

from aida_data import _mteb, dense
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


# ---------------------------------------------------------------- _mteb ----


@pytest.mark.parametrize(
    "name, expected",
    [
        ("mteb:SCIDOCS:Qwen/Qwen3-Embedding-4B", ("SCIDOCS", "Qwen/Qwen3-Embedding-4B")),
        ("mteb:T:model", ("T", "model")),
    ],
)
def test_parse_name(name, expected):
    assert _mteb.parse_name(name) == expected


@pytest.mark.parametrize(
    "name",
    [
        "SCIDOCS:model",
        "dense:SCIDOCS:model",
        "mteb:SCIDOCS",
        "mteb:",
        "mteb::model",
        "mteb:SCIDOCS:",
        "mteb::",
        "mteb:SCIDOCS:a:b",
        "mteb:SCIDOCS:org//model",
        "mteb:SCIDOCS:../model",
        "mteb:../x:model",
    ],
)
def test_parse_name_bad(name):
    with pytest.raises(ValueError):
        _mteb.parse_name(name)


def test_model_slug_and_cache_dir(tmp_path, monkeypatch):
    assert _mteb.model_slug("Qwen/Qwen3-Embedding-4B") == "Qwen__Qwen3-Embedding-4B"
    monkeypatch.setattr(dense, "DATASETS_DIR", tmp_path)
    assert _mteb.cache_dir("T", "a/b") == tmp_path / "dense" / "mteb" / "T" / "a__b"


# --- clustering colors


def _rebuild(texts, colors, s):
    """The (text, label) pairs of set `s`, from its color."""
    absent = colors.labels[s].index(_mteb.ABSENT)
    col = colors.values[:, s]
    return [(t, colors.labels[s][c]) for t, c in zip(texts, col) if c != absent]


def test_clustering_colors_v1():
    sets = [
        (["a", "b", "c"], [1, 0, 1]),  # int labels
        (["c", "d"], ["x", "y"]),
        (["a", "d", "b"], [2, 2, 0]),
    ]
    texts, colors = _mteb.clustering_colors_v1(sets)
    assert texts == ["a", "b", "c", "d"]
    assert colors.names == ("set0", "set1", "set2")
    assert colors.values.shape == (4, 3) and colors.values.dtype == np.int64
    # sorted labels, "<absent>" included, str labels
    assert colors.labels[0] == ("0", "1", "<absent>")
    assert colors.labels[1] == ("<absent>", "x", "y")
    assert colors.labels[2] == ("0", "2", "<absent>")
    for s, (sentences, labels) in enumerate(sets):
        got = _rebuild(texts, colors, s)
        assert sorted(got) == sorted(zip(sentences, map(str, labels)))
    # codes are dense and 0-based
    for s in range(3):
        assert set(colors.values[:, s]) <= set(range(len(colors.labels[s])))


def test_clustering_colors_v1_length_mismatch():
    with pytest.raises(ValueError):
        _mteb.clustering_colors_v1([(["a"], [1, 2])])


def test_clustering_colors_v2_single_color():
    colors = _mteb.clustering_colors_v2([3, 1, 3, 2])
    assert colors.names == ("label",)
    assert colors.labels == (("1", "2", "3"),)
    assert list(colors.column("label")) == [2, 0, 2, 1]


def test_clustering_colors_v2_hierarchical_raises():
    with pytest.raises(NotImplementedError):
        _mteb.clustering_colors_v2([["a", "b"], ["a", "c"]])


# --- qrels


def test_build_qrels_positive_only():
    table = _mteb.build_qrels({"q1": {"d1": 1, "d2": 0, "d3": 2}, "q2": {"d1": 0}})
    assert table.column_names == ["query_id", "doc_id", "score"]
    assert table.to_pylist() == [
        {"query_id": "q1", "doc_id": "d1", "score": 1},
        {"query_id": "q1", "doc_id": "d3", "score": 2},
    ]
    assert str(table.schema.field("score").type) == "int64"
    assert str(table.schema.field("query_id").type) == "string"


def test_build_qrels_empty_and_triples():
    assert _mteb.build_qrels({}).num_rows == 0
    t = _mteb.build_qrels([("q", "d", 1), ("q", "e", -1)])
    assert t.to_pylist() == [{"query_id": "q", "doc_id": "d", "score": 1}]


# --- similarity


class _Meta:
    def __init__(self, fn):
        self.similarity_fn_name = fn


@pytest.mark.parametrize("fn", ["cosine", "euclidean", "dot"])
def test_similarity_to_distance(fn):
    assert _mteb.similarity_to_distance(_Meta(fn)) == fn


@pytest.mark.parametrize("fn", ["MaxSim", "max_sim", "manhattan", "custom"])
def test_similarity_unsupported(fn):
    with pytest.raises(NotImplementedError):
        _mteb.similarity_to_distance(_Meta(fn))


def test_similarity_none_warns(caplog):
    with caplog.at_level("WARNING", logger="aida_data.mteb"):
        assert _mteb.similarity_to_distance(_Meta(None)) == "cosine"
    assert "cosine" in caplog.text


# --- writing the cache


def _extra(task_type):
    return {
        "task": "T", "subset": "default", "split": "test", "task_type": task_type,
        "model": "org/m", "model_revision": "r1", "task_revision": "t1",
        "similarity": "cosine", "mteb_version": "2.0",
    }


def _retrieval_result():
    return _mteb.Encoded(
        np.arange(6, dtype=np.float32).reshape(3, 2),
        np.array(["d0", "d1", "d2"], dtype=object),
        None,
        np.ones((2, 2), dtype=np.float32),
        np.array(["q0", "q1"], dtype=object),
        _mteb.build_qrels({"q0": {"d1": 1, "d2": 0}}),
        _extra("Retrieval"),
    )


def test_write_cache_retrieval(tmp_path, monkeypatch):
    import pyarrow.parquet as pq

    monkeypatch.setattr(dense, "DATASETS_DIR", tmp_path)
    order = []
    real = _mteb._write_parquet_cache
    monkeypatch.setattr(
        _mteb, "_write_parquet_cache", lambda p, *a, **k: (order.append(p.name), real(p, *a, **k))
    )
    real_wt = _mteb.write_table
    monkeypatch.setattr(
        _mteb, "write_table", lambda p, *a, **k: (order.append(p.name), real_wt(p, *a, **k))
    )
    result = _retrieval_result()
    directory = _mteb.cache_dir("T", "org/m")
    _mteb.write_cache(result, directory)
    assert order[-1] == "corpus.parquet"
    assert set(order) == {"corpus.parquet", "queries.parquet", "qrels.parquet"}

    data, colors, ids, meta = dense._read_parquet_cache_full(directory / "corpus.parquet")
    np.testing.assert_array_equal(data, result.corpus)
    assert colors is None and list(ids) == ["d0", "d1", "d2"]
    for key, value in _extra("Retrieval").items():
        assert meta[key] == value
    assert meta["has_ids"] is True
    q, _, qids, _ = dense._read_parquet_cache_full(directory / "queries.parquet")
    np.testing.assert_array_equal(q, result.queries)
    assert list(qids) == ["q0", "q1"]
    qrels = pq.read_table(directory / "qrels.parquet")
    assert qrels.to_pylist() == [{"query_id": "q0", "doc_id": "d1", "score": 1}]


def test_write_cache_clustering(tmp_path):
    texts, colors = _mteb.clustering_colors_v1([(["a", "b"], [0, 1]), (["b", "c"], [5, 5])])
    result = _mteb.Encoded(
        np.ones((3, 2), dtype=np.float32), None, colors, None, None, None,
        _extra("Clustering"),
    )
    _mteb.write_cache(result, tmp_path / "x")
    assert sorted(p.name for p in (tmp_path / "x").iterdir()) == ["corpus.parquet"]
    data, out_colors, ids, meta = dense._read_parquet_cache_full(
        tmp_path / "x" / "corpus.parquet"
    )
    assert ids is None and meta["has_ids"] is False
    assert meta["task_type"] == "Clustering" and meta["similarity"] == "cosine"
    assert out_colors.names == ("set0", "set1")
    np.testing.assert_array_equal(out_colors.values, colors.values)
    assert out_colors.labels == colors.labels


def test_encode_task_without_mteb_gives_hint(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake(name, *a, **k):
        if name == "mteb" or name.startswith("mteb."):
            raise ImportError("no mteb")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(ImportError, match=r"aida-data\[mteb\]"):
        _mteb.encode_task("T", "m")


# --- hosted registry


class _Resp:
    def __init__(self, status):
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(f"{self.status_code}")


def _fake_head(monkeypatch, statuses):
    import requests

    calls = []

    def head(url, **kwargs):
        assert kwargs["allow_redirects"] is True and kwargs["timeout"]
        name = url.rsplit("/", 1)[1]
        calls.append(name)
        return _Resp(statuses[name])

    monkeypatch.setattr(requests, "head", head)
    return calls


def _fake_download(monkeypatch):
    downloaded = []

    def download(url, destination):
        downloaded.append((url, destination.name))
        destination.write_text("x")

    monkeypatch.setattr(_mteb._download, "download", download)
    return downloaded


def test_fetch_hosted_404(tmp_path, monkeypatch):
    _fake_head(monkeypatch, {"corpus.parquet": 404})
    downloaded = _fake_download(monkeypatch)
    assert _mteb.fetch_hosted("T", "a/b", tmp_path) is False
    assert downloaded == []


def test_fetch_hosted_error_raises(tmp_path, monkeypatch):
    import requests

    _fake_head(monkeypatch, {"corpus.parquet": 500})
    with pytest.raises(requests.HTTPError):
        _mteb.fetch_hosted("T", "a/b", tmp_path)

    def boom(url, **kwargs):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(requests, "head", boom)
    with pytest.raises(requests.ConnectionError):
        _mteb.fetch_hosted("T", "a/b", tmp_path)


def test_fetch_hosted_retrieval(tmp_path, monkeypatch):
    _fake_head(monkeypatch, {"corpus.parquet": 200, "qrels.parquet": 200})
    downloaded = _fake_download(monkeypatch)
    assert _mteb.fetch_hosted("T", "a/b", tmp_path / "d") is True
    assert [name for _, name in downloaded] == [
        "queries.parquet", "qrels.parquet", "corpus.parquet",
    ]
    assert downloaded[-1][0].endswith("/T/a__b/corpus.parquet")
    assert downloaded[-1][0].startswith(_mteb._MTEB_HF_BASE)


def test_fetch_hosted_clustering(tmp_path, monkeypatch):
    _fake_head(monkeypatch, {"corpus.parquet": 200, "qrels.parquet": 404})
    downloaded = _fake_download(monkeypatch)
    assert _mteb.fetch_hosted("T", "a/b", tmp_path) is True
    assert [name for _, name in downloaded] == ["corpus.parquet"]


def test_fetch_hosted_probe_error_raises(tmp_path, monkeypatch):
    import requests

    _fake_head(monkeypatch, {"corpus.parquet": 200, "qrels.parquet": 503})
    downloaded = _fake_download(monkeypatch)
    with pytest.raises(requests.HTTPError):
        _mteb.fetch_hosted("T", "a/b", tmp_path)
    assert downloaded == []


# --- loader rows


def test_doc_rows_titles():
    rows = _mteb.doc_rows(["  body one ", "body two", "body three"], ["Title", "", None])
    assert rows[0] == {"text": "Title   body one", "title": "Title", "body": "  body one "}
    assert rows[1] == {"text": "body two", "title": "", "body": "body two"}
    assert rows[2]["title"] == ""
    assert all(set(r) == {"text", "title", "body"} for r in rows)
    assert _mteb.doc_rows(["x"])[0] == {"text": "x", "title": "", "body": "x"}


def test_query_rows_instruction():
    rows = _mteb.query_rows(["q1", "q2"], ["do it", None])
    assert rows[0] == {"text": "q1 do it", "query": "q1", "instruction": "do it"}
    assert rows[1] == {"text": "q2", "query": "q2", "instruction": ""}
    assert _mteb.query_rows(["q"]) == [{"text": "q", "query": "q"}]


def test_rows_reject_non_text():
    with pytest.raises(NotImplementedError):
        _mteb.query_rows([["user: hi"]])
    with pytest.raises(NotImplementedError):
        _mteb.doc_rows([{"image": 1}])
    assert _mteb.text_rows(["a"]) == [{"text": "a"}]


def test_loader_batches():
    pytest.importorskip("torch")
    rows = _mteb.doc_rows(["a", "b", "c"], ["T", "", ""])
    batches = list(_mteb._loader(rows, 2))
    assert [len(b["text"]) for b in batches] == [2, 1]
    assert batches[0]["text"] == ["T a", "b"] and batches[0]["title"] == ["T", ""]


# --- encode_task with a fake mteb (limit, completion marker)


def _install_fake_mteb(monkeypatch, tmp_path):
    import sys
    import types
    from types import SimpleNamespace

    class _DS:  # minimal HF Dataset stand-in
        def __init__(self, cols):
            self.cols = cols
            self.column_names = list(cols)

        def __len__(self):
            return len(next(iter(self.cols.values())))

        def __getitem__(self, name):
            return self.cols[name]

        def select(self, rng):
            idx = list(rng)
            return _DS({k: [v[i] for i in idx] for k, v in self.cols.items()})

    class _Model:
        def encode(self, loader, **kwargs):
            return np.array([[float(len(r["text"])), 1.0] for r in loader], dtype=np.float32)

    task = SimpleNamespace(
        metadata=SimpleNamespace(
            name="T", modalities=["text"], type="Retrieval", eval_splits=["test"],
            revision="t1",
        ),
        dataset={"test": {
            "corpus": _DS({"id": ["d0", "d1", "d2"], "text": ["a", "bb", "ccc"]}),
            "queries": _DS({"id": ["q0", "q1"], "text": ["x", "yy"]}),
            "relevant_docs": {"q0": {"d1": 1, "d0": 0}},
        }},
        load_data=lambda: None,
    )
    fake = types.ModuleType("mteb")
    fake.__version__ = "9.9"
    fake.get_task = lambda name: task
    fake.get_model_meta = lambda *a, **k: SimpleNamespace(
        similarity_fn_name="dot", revision="r1"
    )
    fake.get_model = lambda *a, **k: _Model()
    types_mod = types.ModuleType("mteb.types")
    types_mod.PromptType = SimpleNamespace(document="document", query="query")
    fake.types = types_mod
    monkeypatch.setitem(sys.modules, "mteb", fake)
    monkeypatch.setitem(sys.modules, "mteb.types", types_mod)
    monkeypatch.setattr(_mteb, "_loader", lambda rows, batch_size: iter(rows))
    monkeypatch.setattr(dense, "DATASETS_DIR", tmp_path)
    return task


def test_encode_task_writes_cache(tmp_path, monkeypatch):
    _install_fake_mteb(monkeypatch, tmp_path)
    result = _mteb.encode_task("T", "org/m")
    directory = _mteb.cache_dir("T", "org/m")
    assert sorted(p.name for p in directory.iterdir()) == [
        "corpus.parquet", "qrels.parquet", "queries.parquet",
    ]
    _, _, ids, meta = dense._read_parquet_cache_full(directory / "corpus.parquet")
    assert list(ids) == ["d0", "d1", "d2"]
    assert meta["similarity"] == "dot" and meta["model_revision"] == "r1"
    assert meta["mteb_version"] == "9.9" and meta["task_revision"] == "t1"
    assert result.qrels.num_rows == 1


def test_encode_task_limit_does_not_write(tmp_path, monkeypatch):
    _install_fake_mteb(monkeypatch, tmp_path)
    result = _mteb.encode_task("T", "org/m", limit=2)
    assert result.corpus.shape == (2, 2) and result.queries.shape == (2, 2)
    assert not _mteb.cache_dir("T", "org/m").exists()
