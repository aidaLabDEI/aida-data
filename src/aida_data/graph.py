"""\
Graph datasets with colored nodes, stored as edge lists.

Datasets come from the Sirius repository
(https://github.com/leonardopellegrina/Sirius/tree/main/data): each dataset
is a pair of TSV files, one with the edges (`u<TAB>v`) and one with the node
colors (`node<TAB>color`).

By default graphs are treated as undirected and simple: self loops and
duplicate edges are dropped, and each undirected edge is stored once as
`(u, v)` with `u < v`.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from ._download import download as _download

DATASETS_DIR = Path(os.environ.get("AIDA_DATA_DIR", "datasets"))

_LOGGER = logging.getLogger("aida_data.graph")


def _read_int_pairs(path: Path) -> np.ndarray:
    return pd.read_csv(
        path, sep="\t", header=None, dtype=np.int64, comment="#"
    ).to_numpy()


def _load_tsv_pair(edges_path: Path, colors_path: Path):
    """Read an edge list and a `(node, color)` list from two TSV files."""
    return _read_int_pairs(edges_path), _read_int_pairs(colors_path)


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    edges_url: str
    colors_url: str
    # (edges_path, colors_path) -> (edges, node_colors), both (*, 2) int arrays
    loader_function: Callable


_DATASETS_INFO: dict[str, DatasetInfo] = {}


def register(info: DatasetInfo, force=False):
    if info.name in _DATASETS_INFO and not force:
        raise ValueError(f"dataset {info.name} already registered")
    _DATASETS_INFO[info.name] = info


_SIRIUS_BASE = "https://raw.githubusercontent.com/leonardopellegrina/Sirius/main/data/"


def _register_sirius(name: str):
    register(
        DatasetInfo(
            name,
            f"{_SIRIUS_BASE}{name}.tsv",
            f"{_SIRIUS_BASE}{name}_labels.tsv",
            _load_tsv_pair,
        )
    )


for _name in (
    "abortion",
    "brexit",
    "citeseer",
    "com-dblp",
    "com-youtube",
    "combined",
    "obamacare",
    "phy_citations",
    "trivago-clicks",
    "twitter_pol",
    "uselections",
    "walmart-trips",
):
    _register_sirius(_name)


def available_datasets():
    return list(_DATASETS_INFO.keys())


def local_paths(name: str) -> tuple[Path, Path]:
    info = _DATASETS_INFO[name]
    graphs_dir = DATASETS_DIR / "graphs"
    return (
        graphs_dir / Path(urlparse(info.edges_url).path).name,
        graphs_dir / Path(urlparse(info.colors_url).path).name,
    )


@dataclass(frozen=True)
class EdgeList:
    name: str
    edges: np.ndarray  # shape (m, 2), int64, one row (u, v) per edge
    colors: np.ndarray  # shape (n,), int64, colors[v] = color of v (-1 if missing)
    directed: bool = False

    @property
    def n_nodes(self) -> int:
        return len(self.colors)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    @property
    def n_colors(self) -> int:
        return len(np.unique(self.colors[self.colors >= 0]))


def load_edge_list(
    name: str, directed: bool = False, remap_colors: bool = False
) -> EdgeList:
    """Load graph dataset `name` as an `EdgeList`.

    Processing:

    1. the raw files are downloaded (if needed) and loaded;
    2. `colors[v]` is the color of node `v`, for `v` in `0..n-1`, where `n` is
       one plus the largest node id in the edges or in the labels. Nodes
       without a color get `-1` (and a warning is logged). If
       `remap_colors=True`, colors are replaced by their rank among the
       distinct colors, so they become `0..k-1` in the original order;
    3. self loops are dropped; if `directed=False` each edge is oriented as
       `(min(u, v), max(u, v))`, so that `(u, v)` and `(v, u)` collapse;
       duplicate edges are then dropped and rows are sorted
       lexicographically.

    Example::

        g = load_edge_list("brexit")
        g.edges, g.colors
    """
    if name not in available_datasets():
        raise KeyError(
            f"Dataset `{name}` not available. Pick one of {available_datasets()}"
        )

    info = _DATASETS_INFO[name]
    edges_path, colors_path = local_paths(name)
    edges_path.parent.mkdir(parents=True, exist_ok=True)
    _download(info.edges_url, edges_path)
    _download(info.colors_url, colors_path)
    edges, node_colors = info.loader_function(edges_path, colors_path)
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    node_colors = np.asarray(node_colors, dtype=np.int64).reshape(-1, 2)

    colors = _build_colors(name, edges, node_colors, remap_colors)
    edges = _normalize_edges(name, edges, directed)
    return EdgeList(name, edges, colors, directed)


def _build_colors(
    name: str, edges: np.ndarray, node_colors: np.ndarray, remap_colors: bool
) -> np.ndarray:
    ids, labels = node_colors[:, 0], node_colors[:, 1]
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"{name}: duplicate node ids in the labels file")
    n = int(max(edges.max(initial=-1), ids.max(initial=-1))) + 1
    colors = np.full(n, -1, dtype=np.int64)
    colors[ids] = labels
    colored = colors >= 0
    n_missing = int(np.count_nonzero(~colored))
    if n_missing > 0:
        _LOGGER.warning("%s: %d nodes have no color", name, n_missing)
    if remap_colors:
        _, inv = np.unique(colors[colored], return_inverse=True)
        colors[colored] = inv
    return colors


def _normalize_edges(name: str, edges: np.ndarray, directed: bool) -> np.ndarray:
    loops = edges[:, 0] == edges[:, 1]
    n_loops = int(np.count_nonzero(loops))
    if n_loops > 0:
        _LOGGER.info("%s: dropped %d self loops", name, n_loops)
        edges = edges[~loops]
    if not directed:
        edges = np.sort(edges, axis=1)
    n_before = len(edges)
    edges = _unique_rows(edges)
    n_dups = n_before - len(edges)
    if n_dups > 0:
        _LOGGER.info("%s: dropped %d duplicate edges", name, n_dups)
    return edges


def _unique_rows(edges: np.ndarray) -> np.ndarray:
    """Unique rows of a non-negative (m, 2) array, sorted lexicographically."""
    n = int(edges.max(initial=-1)) + 1
    if n > 2**31:
        # u * n + v could overflow int64
        return np.unique(edges, axis=0).reshape(-1, 2)
    # Much faster than np.unique(axis=0): encode each row as one integer key
    # whose order matches the lexicographic order of the rows.
    keys = np.unique(edges[:, 0] * n + edges[:, 1])
    return np.stack([keys // n, keys % n], axis=1)
