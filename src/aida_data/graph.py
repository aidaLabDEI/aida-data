"""\
Graph datasets with colored nodes, returned as edge lists.

Datasets come from the Sirius repository
(https://github.com/leonardopellegrina/Sirius/tree/main/data): each dataset
is a pair of TSV files, one with the edges (`u<TAB>v`) and one with the node
colors (`node<TAB>color`).

By default graphs are treated as undirected and simple: self loops and
duplicate edges are dropped, and each undirected edge is stored once as
`(u, v)` with `u < v`.

Each dataset is parsed once into a zstd-compressed parquet cache holding an
adjacency list (see `_write_parquet_cache`); the raw TSV files are then
deleted unless `KEEP_RAW` is set.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import pyarrow as pa

from ._cache import KEEP_RAW, delete_raw, read_table, write_table
from ._download import download as _download
from ._tmp import stale_temporaries

DATASETS_DIR = Path(os.environ.get("AIDA_DATA_DIR", "datasets"))

_LOGGER = logging.getLogger("aida_data.graph")

# Layout version of the parquet caches written by `_cached`.
_CACHE_VERSION = 1


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


def cache_path(name: str) -> Path:
    """The parquet cache of dataset `name`."""
    if name not in _DATASETS_INFO:
        raise KeyError(f"Dataset `{name}` not available")
    return DATASETS_DIR / "graphs" / f"{name}.parquet"


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

    1. the raw files are downloaded (if needed) and loaded, once: the result
       is kept in a parquet cache (see `_cached`);
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

    edges, colors = _cached(name)
    if not directed:
        edges = _to_undirected(name, edges)
    if remap_colors:
        colors = _remap_colors(colors)
    return EdgeList(name, edges, colors, directed)


def _build_colors(
    name: str, edges: np.ndarray, node_colors: np.ndarray
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
    return colors


def _remap_colors(colors: np.ndarray) -> np.ndarray:
    """Replace colors by their rank among the distinct colors, keeping `-1`."""
    colors = colors.copy()
    colored = colors >= 0
    _, inv = np.unique(colors[colored], return_inverse=True)
    colors[colored] = inv
    return colors


def _to_undirected(name: str, edges: np.ndarray) -> np.ndarray:
    """Orient each edge as `(min(u, v), max(u, v))` and drop the duplicates."""
    n_before = len(edges)
    edges = _unique_rows(np.sort(edges, axis=1))
    n_dups = n_before - len(edges)
    if n_dups > 0:
        _LOGGER.info("%s: dropped %d duplicate edges", name, n_dups)
    return edges


def _clean_edges(name: str, edges: np.ndarray) -> np.ndarray:
    """Drop self loops and duplicate rows, keeping the orientation of the
    edges; rows are sorted lexicographically."""
    loops = edges[:, 0] == edges[:, 1]
    n_loops = int(np.count_nonzero(loops))
    if n_loops > 0:
        _LOGGER.info("%s: dropped %d self loops", name, n_loops)
        edges = edges[~loops]
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


def _index_dtype(n: int) -> np.dtype:
    """Integer type for node ids `0..n-1`: int32 unless `n` is too large."""
    return np.dtype(np.int32 if n <= 2**31 - 1 else np.int64)


def _color_dtype(colors: np.ndarray) -> np.dtype:
    """Narrowest signed integer type holding `colors` (which may hold `-1`)."""
    for dtype in (np.int8, np.int16, np.int32, np.int64):
        info = np.iinfo(dtype)
        if colors.min(initial=0) >= info.min and colors.max(initial=0) <= info.max:
            return np.dtype(dtype)
    raise AssertionError("unreachable: colors are int64")


def _write_parquet_cache(path: Path, edges: np.ndarray, colors: np.ndarray):
    """Write the graph to the parquet cache `path`.

    `edges` is an (m, 2) array of unique, non-loop edges sorted
    lexicographically, whose ids are below `len(colors)`. One row per node
    `v = 0..n-1`: `nbrs` is the list of out-neighbors of `v` in the
    orientation of the source, sorted ascending (an adjacency list, stored
    as parquet values plus offsets), and `color` is the color of `v`.
    """
    n = len(colors)
    counts = np.bincount(edges[:, 0], minlength=n)
    offsets = np.zeros(n + 1, dtype=np.int32)
    np.cumsum(counts, out=offsets[1:])
    nbrs = pa.ListArray.from_arrays(
        pa.array(offsets), pa.array(edges[:, 1].astype(_index_dtype(n)))
    )
    meta = {"version": _CACHE_VERSION, "n_nodes": n, "n_edges": len(edges)}
    table = pa.table({"nbrs": nbrs, "color": colors.astype(_color_dtype(colors))})
    # Almost all neighbor ids are distinct, so dictionary encoding does not
    # help for them; deltas of sorted ids are small instead.
    write_table(
        path,
        table,
        meta,
        use_dictionary=["color"],
        column_encoding={"nbrs.list.element": "DELTA_BINARY_PACKED"},
    )


def _read_parquet_cache(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a cache written by `_write_parquet_cache` as `(edges, colors)`,
    an (m, 2) and an (n,) int64 array."""
    table, _ = read_table(path, _CACHE_VERSION)
    nbrs = table.column("nbrs").combine_chunks()
    colors = table.column("color").to_numpy().astype(np.int64)
    edges = np.empty((len(nbrs.flatten()), 2), dtype=np.int64)
    edges[:, 0] = np.repeat(np.arange(len(nbrs), dtype=np.int64), np.diff(nbrs.offsets))
    edges[:, 1] = nbrs.flatten().to_numpy()
    return edges, colors


def _cached(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return `(edges, colors)` of dataset `name`, parsing it only once.

    The first call downloads the raw files, parses them with the loader of the
    dataset and stores the result in `cache_path(name)`; later calls only read
    that file. The cache does not depend on the options of `load_edge_list`:
    it holds the edges with self loops and duplicates removed, in the source
    orientation, and the colors as in the labels file (`-1` if missing).

    After a fresh parse the raw files are deleted, unless `KEEP_RAW` is set.
    """
    cache = cache_path(name)
    if not cache.is_file():
        info = _DATASETS_INFO[name]
        edges_path, colors_path = local_paths(name)
        cache.parent.mkdir(parents=True, exist_ok=True)
        _download(info.edges_url, edges_path)
        _download(info.colors_url, colors_path)
        _LOGGER.info("parsing %s into %s", edges_path, cache)
        edges, node_colors = info.loader_function(edges_path, colors_path)
        edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
        node_colors = np.asarray(node_colors, dtype=np.int64).reshape(-1, 2)
        colors = _build_colors(name, edges, node_colors)
        _write_parquet_cache(cache, _clean_edges(name, edges), colors)
        # Only after a fresh parse: upgrading alone never removes files.
        if not KEEP_RAW:
            delete_raw(edges_path, colors_path)
    # Always read back from the file, so that the first and later calls
    # return identical arrays.
    return _read_parquet_cache(cache)


def prune_raw(dry_run: bool = True) -> list[Path]:
    """Raw downloads of datasets whose cache exists, which are no longer
    needed. With `dry_run=False` they are deleted. Also lists the
    `*.part` and `*.parquet.tmp` files left by downloads and cache
    writes that were killed, once they are more than 24 hours old.
    """
    paths = [
        path
        for name in _DATASETS_INFO
        if cache_path(name).is_file()
        for path in local_paths(name)
        if path.is_file()
    ]
    paths += stale_temporaries(DATASETS_DIR / "graphs")
    total = sum(path.stat().st_size for path in paths)
    _LOGGER.info(
        "%s %d raw and stale temporary files (%.1f MiB): %s",
        "would delete" if dry_run else "deleting",
        len(paths),
        total / 2**20,
        ", ".join(str(path) for path in paths),
    )
    if not dry_run:
        delete_raw(*paths)
    return paths
