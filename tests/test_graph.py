import dataclasses

import numpy as np
import pytest

from aida_data import graph
from aida_data.graph import DatasetInfo, EdgeList, load_edge_list, register

# Node 5 is isolated and uncolored; edges contain an exact duplicate (1, 2),
# a reversed duplicate (3, 0)/(0, 3) and a self loop (4, 4).
EDGES = np.array([[1, 2], [3, 0], [0, 3], [1, 2], [4, 4], [2, 4], [0, 1]])
# Shuffled labels with non-contiguous colors; node 6 has no edges.
NODE_COLORS = np.array([[3, 12], [0, 7], [6, 3], [2, 12], [1, 3], [4, 7]])


def _loader(edges_path, colors_path):
    return EDGES.copy(), NODE_COLORS.copy()


def _loader_duplicate_ids(edges_path, colors_path):
    return EDGES.copy(), np.vstack([NODE_COLORS, [[0, 3]]])


@pytest.fixture(autouse=True)
def tiny_datasets(tmp_path, monkeypatch):
    monkeypatch.setattr(graph, "DATASETS_DIR", tmp_path)
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


@pytest.mark.network
def test_citeseer():
    g = load_edge_list("citeseer")
    assert isinstance(g, EdgeList)
    assert g.n_nodes == 3264
    assert g.n_colors == 6
