import dataclasses

import numpy as np
import pyarrow.parquet as pq
import pytest

from aida_data import graph
from aida_data.graph import DatasetInfo, EdgeList, load_edge_list, register

# Node 5 is isolated and uncolored; edges contain an exact duplicate (1, 2),
# a reversed duplicate (3, 0)/(0, 3) and a self loop (4, 4).
EDGES = np.array([[1, 2], [3, 0], [0, 3], [1, 2], [4, 4], [2, 4], [0, 1]])
# Shuffled labels with non-contiguous colors; node 6 has no edges.
NODE_COLORS = np.array([[3, 12], [0, 7], [6, 3], [2, 12], [1, 3], [4, 7]])


def _loader(edges_path, colors_path):
    assert edges_path.read_text() == "raw"
    return EDGES.copy(), NODE_COLORS.copy()


def _loader_duplicate_ids(edges_path, colors_path):
    return EDGES.copy(), np.vstack([NODE_COLORS, [[0, 3]]])


@pytest.fixture(autouse=True)
def tiny_datasets(tmp_path, monkeypatch):
    monkeypatch.setattr(graph, "DATASETS_DIR", tmp_path)
    monkeypatch.setattr(graph, "KEEP_RAW", False)
    monkeypatch.setattr(graph, "_DATASETS_INFO", dict(graph._DATASETS_INFO))
    downloads = []

    def download(url, destination):
        downloads.append(url)
        if not destination.is_file():
            destination.write_text("raw")

    monkeypatch.setattr(graph, "_download", download)
    register(
        DatasetInfo("tiny", "file:///tiny.tsv", "file:///tiny_labels.tsv", _loader),
        force=True,
    )
    register(
        DatasetInfo(
            "tiny-dup-ids",
            "file:///dup.tsv",
            "file:///dup_labels.tsv",
            _loader_duplicate_ids,
        ),
        force=True,
    )
    return downloads


def test_unknown_dataset_raises():
    with pytest.raises(KeyError, match="not available"):
        load_edge_list("does-not-exist")


def test_colors_placed_by_node_id():
    g = load_edge_list("tiny")
    np.testing.assert_array_equal(g.colors, [7, 3, 12, 12, 7, -1, 3])
    assert g.colors.dtype == np.int64


def test_undirected_edges_are_canonical_and_simple():
    g = load_edge_list("tiny")
    np.testing.assert_array_equal(g.edges, [[0, 1], [0, 3], [1, 2], [2, 4]])
    assert g.edges.dtype == np.int64
    assert g.edges.shape == (4, 2)
    assert (g.edges[:, 0] < g.edges[:, 1]).all()
    assert not g.directed


def test_directed_keeps_both_orientations():
    g = load_edge_list("tiny", directed=True)
    np.testing.assert_array_equal(
        g.edges, [[0, 1], [0, 3], [1, 2], [2, 4], [3, 0]]
    )
    assert g.directed


def test_rows_are_sorted():
    g = load_edge_list("tiny", directed=True)
    order = np.lexsort((g.edges[:, 1], g.edges[:, 0]))
    np.testing.assert_array_equal(order, np.arange(len(g.edges)))


def test_remap_colors_preserves_order_and_missing():
    g = load_edge_list("tiny", remap_colors=True)
    np.testing.assert_array_equal(g.colors, [1, 0, 2, 2, 1, -1, 0])


def test_counts():
    g = load_edge_list("tiny")
    assert g.n_nodes == 7
    assert g.n_edges == 4
    assert g.n_colors == 3


def test_edge_list_is_frozen():
    g = load_edge_list("tiny")
    with pytest.raises(dataclasses.FrozenInstanceError):
        g.name = "other"


def test_duplicate_node_ids_raise():
    with pytest.raises(ValueError, match="duplicate node ids"):
        load_edge_list("tiny-dup-ids")


def test_load_tsv_pair_parses_files(tmp_path):
    edges_path = tmp_path / "g.tsv"
    colors_path = tmp_path / "g_labels.tsv"
    edges_path.write_text("0\t1\n2\t1\n1\t0\n")
    colors_path.write_text("2\t1\n0\t0\n1\t1\n")
    edges, node_colors = graph._load_tsv_pair(edges_path, colors_path)
    np.testing.assert_array_equal(edges, [[0, 1], [2, 1], [1, 0]])
    np.testing.assert_array_equal(node_colors, [[2, 1], [0, 0], [1, 1]])
    assert edges.dtype == np.int64


def test_local_paths_use_graphs_subdir(tmp_path):
    register(
        DatasetInfo(
            "remote",
            "https://example.org/data/remote.tsv",
            "https://example.org/data/remote_labels.tsv",
            _loader,
        ),
        force=True,
    )
    assert graph.local_paths("remote") == (
        tmp_path / "graphs" / "remote.tsv",
        tmp_path / "graphs" / "remote_labels.tsv",
    )


def test_cache_round_trip_is_bit_identical(tmp_path):
    # Isolated nodes at the end of the id range, missing colors, negative
    # colors, and an edge oriented from the larger to the smaller id.
    edges = np.array([[0, 2], [2, 0], [2, 3], [3, 9]])
    colors = np.array([5, -1, 300, 0, -1, -1, 7, -1, -1, -1, -1, -1])
    path = tmp_path / "rt.parquet"
    graph._write_parquet_cache(path, edges, colors)
    got_edges, got_colors = graph._read_parquet_cache(path)
    np.testing.assert_array_equal(got_edges, edges)
    np.testing.assert_array_equal(got_colors, colors)
    assert got_edges.dtype == got_colors.dtype == np.int64
    assert path.with_suffix(".parquet.tmp").exists() is False


def test_cache_without_edges(tmp_path):
    path = tmp_path / "empty.parquet"
    graph._write_parquet_cache(
        path, np.empty((0, 2), dtype=np.int64), np.array([1, 2, 3])
    )
    edges, colors = graph._read_parquet_cache(path)
    assert edges.shape == (0, 2)
    np.testing.assert_array_equal(colors, [1, 2, 3])


def test_cache_is_adjacency_list_with_delta_encoding(tmp_path):
    rng = np.random.default_rng(0)
    edges = graph._unique_rows(rng.integers(0, 5000, size=(20000, 2)))
    edges = edges[edges[:, 0] != edges[:, 1]]
    colors = rng.integers(0, 4, size=5000)
    path = tmp_path / "g.parquet"
    graph._write_parquet_cache(path, edges, colors)
    schema = pq.read_schema(path)
    assert str(schema.field("nbrs").type) == "list<element: int32>"
    assert str(schema.field("color").type) == "int8"
    encodings = pq.ParquetFile(path).metadata.row_group(0).column(0).encodings
    assert "DELTA_BINARY_PACKED" in encodings


def test_cache_has_one_row_per_node():
    g = load_edge_list("tiny")
    table = pq.read_table(graph.cache_path("tiny"))
    assert table.num_rows == g.n_nodes == 7
    assert table.column("nbrs").to_pylist() == [[1, 3], [2], [4], [0], [], [], []]


def test_first_and_later_calls_are_identical():
    first = load_edge_list("tiny", directed=True)
    second = load_edge_list("tiny", directed=True)
    np.testing.assert_array_equal(first.edges, second.edges)
    np.testing.assert_array_equal(first.colors, second.colors)


def test_second_call_does_not_parse_or_download(monkeypatch, tiny_datasets):
    first = load_edge_list("tiny")
    assert len(tiny_datasets) == 2

    def fail(*args):
        raise AssertionError("must not parse or download")

    monkeypatch.setattr(graph, "_download", fail)
    monkeypatch.setitem(
        graph._DATASETS_INFO,
        "tiny",
        dataclasses.replace(graph._DATASETS_INFO["tiny"], loader_function=fail),
    )
    second = load_edge_list("tiny")
    np.testing.assert_array_equal(first.edges, second.edges)
    np.testing.assert_array_equal(first.colors, second.colors)


def test_options_apply_to_a_cached_dataset():
    load_edge_list("tiny")
    g = load_edge_list("tiny", directed=True, remap_colors=True)
    np.testing.assert_array_equal(
        g.edges, [[0, 1], [0, 3], [1, 2], [2, 4], [3, 0]]
    )
    np.testing.assert_array_equal(g.colors, [1, 0, 2, 2, 1, -1, 0])
    g = load_edge_list("tiny")
    np.testing.assert_array_equal(g.edges, [[0, 1], [0, 3], [1, 2], [2, 4]])
    np.testing.assert_array_equal(g.colors, [7, 3, 12, 12, 7, -1, 3])


def test_raw_deleted_after_parse(tmp_path):
    load_edge_list("tiny")
    assert graph.cache_path("tiny") == tmp_path / "graphs" / "tiny.parquet"
    assert graph.cache_path("tiny").is_file()
    assert not any(path.exists() for path in graph.local_paths("tiny"))


def test_keep_raw(monkeypatch):
    monkeypatch.setattr(graph, "KEEP_RAW", True)
    load_edge_list("tiny")
    assert all(path.is_file() for path in graph.local_paths("tiny"))


def test_raw_left_alone_when_cache_exists(monkeypatch):
    monkeypatch.setattr(graph, "KEEP_RAW", True)
    load_edge_list("tiny")
    monkeypatch.setattr(graph, "KEEP_RAW", False)
    load_edge_list("tiny")
    assert all(path.is_file() for path in graph.local_paths("tiny"))


def test_prune_raw():
    edges_path, colors_path = graph.local_paths("tiny")
    edges_path.parent.mkdir(parents=True)
    edges_path.write_text("raw")
    colors_path.write_text("raw")
    # No cache yet: nothing to prune.
    assert graph.prune_raw(dry_run=False) == []
    assert edges_path.is_file()
    graph._write_parquet_cache(
        graph.cache_path("tiny"), np.array([[0, 1]]), np.array([0, 1])
    )
    assert graph.prune_raw() == [edges_path, colors_path]
    assert edges_path.is_file()  # dry run
    graph.prune_raw(dry_run=False)
    assert not edges_path.exists() and not colors_path.exists()


def test_unknown_cache_version_raises(tmp_path):
    path = tmp_path / "old.parquet"
    graph._write_parquet_cache(path, np.array([[0, 1]]), np.array([0, 1]))
    table = pq.read_table(path)
    pq.write_table(
        table.replace_schema_metadata({b"aida_data": b'{"version": 99}'}), path
    )
    with pytest.raises(ValueError, match="not a cache in a known format"):
        graph._read_parquet_cache(path)


def test_cache_without_metadata_raises(tmp_path):
    path = tmp_path / "plain.parquet"
    graph._write_parquet_cache(path, np.array([[0, 1]]), np.array([0, 1]))
    pq.write_table(pq.read_table(path).replace_schema_metadata(None), path)
    with pytest.raises(ValueError, match="not a cache in a known format"):
        graph._read_parquet_cache(path)


def test_index_dtype_falls_back_to_int64():
    assert graph._index_dtype(0) == np.int32
    assert graph._index_dtype(2**31 - 1) == np.int32
    assert graph._index_dtype(2**31) == np.int64


@pytest.mark.parametrize(
    "values, dtype",
    [
        ([-1, 0, 127], np.int8),
        ([-1, 128], np.int16),
        ([-1, 40000], np.int32),
        ([-1, 2**31], np.int64),
        ([], np.int8),
    ],
)
def test_color_dtype_is_narrowest_signed(values, dtype):
    assert graph._color_dtype(np.array(values, dtype=np.int64)) == dtype


@pytest.mark.network
def test_citeseer(monkeypatch):
    from aida_data._download import download

    monkeypatch.setattr(graph, "_download", download)
    g = load_edge_list("citeseer")
    assert isinstance(g, EdgeList)
    assert g.n_nodes == 3264
    assert g.n_colors == 6
