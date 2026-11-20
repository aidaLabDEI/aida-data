"""\
HDF5 files converted from the parquet caches (private).

The parquet caches store one column per feature, which is awkward to read
outside Python. `ensure_hdf5` converts a cache into an HDF5 file in which
every matrix is a single contiguous row-major dataset, kept under
`<root>/<module dir>/hdf5/` next to the caches. They are not stored as
`dense/<cache_name>.hdf5`, because `dense` treats that name as a pre-0.2
cache to migrate and then deletes it.

ann-benchmarks and VIBE datasets are already HDF5: `ensure_hdf5` returns their
download untouched.

`LAYOUT` lists the objects of every layout once. The conversion writes them,
and the C++ snippets (a later step) have one block for each of them.
"""

import errno
import json
import logging
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ._cache import delete_raw, parse_meta, read_table
from ._tmp import atomic_path, stale_temporaries

_LOGGER = logging.getLogger("aida_data.hdf5")

Module = Literal["dense", "graph", "timeseries"]

# Version of the layouts below, in the root attribute `aida_data_hdf5_version`.
# Bump it when a layout changes: converted files are then rebuilt.
_HDF5_VERSION = 1

# Rows per batch when streaming a parquet file into a dataset.
_BATCH_SIZE = 65536
# Cap on `rows * columns` of a batch, so that wide embeddings (2560 columns)
# do not make a batch of hundreds of megabytes.
_MAX_BATCH_ELEMENTS = 2**25

# Converting needs this much free space, relative to the expected size.
_SPACE_MARGIN = 1.1

# Directory name of each module under the root, see `_paths.py`.
_DIRS = {"dense": "dense", "graph": "graphs", "timeseries": "timeseries"}


# --------------------------------------------------------------------------
# Layouts
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Hdf5Object:
    """A dataset or a root attribute of an HDF5 layout.

    `layout` is "dense", "graph", "timeseries" or "native" (ann-benchmarks
    and VIBE files, which are not converted). `path` is the path of the
    dataset (`qrels/query`) or the name of the attribute. `ctype` is the
    element type the C++ snippet loads it as: integers all load as
    `int64_t` (HDF5 converts on read), so the snippet does not depend on the
    narrow dtype the cache picked. Optional objects are not in every file of
    the layout.
    """

    layout: str
    path: str
    ctype: str
    required: bool = True
    attribute: bool = False


def _root_attributes(layout: str) -> tuple[Hdf5Object, ...]:
    return tuple(
        Hdf5Object(layout, key, ctype, attribute=True)
        for key, ctype in (
            ("aida_data_hdf5_version", "int64_t"),
            ("source_mtime_ns", "int64_t"),
            ("module", "std::string"),
            ("name", "std::string"),
        )
    )


LAYOUT: tuple[Hdf5Object, ...] = (
    # ann-benchmarks / VIBE files, as downloaded.
    Hdf5Object("native", "train", "float"),
    Hdf5Object("native", "test", "float"),
    Hdf5Object("native", "neighbors", "int64_t"),
    Hdf5Object("native", "distances", "float"),
    Hdf5Object("native", "distance", "std::string", attribute=True),
    # dense: `test`, `test_ids` and `qrels/*` are MTEB retrieval only, `ids`
    # MTEB retrieval, `colors` and the `color_*` attributes colored datasets.
    *_root_attributes("dense"),
    Hdf5Object("dense", "train", "float"),
    Hdf5Object("dense", "test", "float", required=False),
    Hdf5Object("dense", "colors", "int64_t", required=False),
    Hdf5Object("dense", "ids", "std::string", required=False),
    Hdf5Object("dense", "test_ids", "std::string", required=False),
    Hdf5Object("dense", "qrels/query", "std::string", required=False),
    Hdf5Object("dense", "qrels/doc", "std::string", required=False),
    Hdf5Object("dense", "qrels/score", "int64_t", required=False),
    Hdf5Object("dense", "distance", "std::string", attribute=True),
    Hdf5Object("dense", "dimension", "int64_t", attribute=True),
    Hdf5Object("dense", "color_names", "std::string", False, attribute=True),
    # JSON: a list of lists of labels, one per color.
    Hdf5Object("dense", "color_labels", "std::string", False, attribute=True),
    # graph (CSR)
    *_root_attributes("graph"),
    Hdf5Object("graph", "offsets", "int64_t"),
    Hdf5Object("graph", "neighbors", "int64_t"),
    Hdf5Object("graph", "colors", "int64_t"),
    Hdf5Object("graph", "n_nodes", "int64_t", attribute=True),
    Hdf5Object("graph", "n_edges", "int64_t", attribute=True),
    # time series
    *_root_attributes("timeseries"),
    Hdf5Object("timeseries", "values", "double"),
    Hdf5Object("timeseries", "time", "int64_t", required=False),
    Hdf5Object("timeseries", "dim_names", "std::string", False, attribute=True),
    Hdf5Object("timeseries", "sampling", "std::string", False, attribute=True),
    Hdf5Object("timeseries", "source", "std::string", False, attribute=True),
)


def layout_objects(layout: str) -> tuple[Hdf5Object, ...]:
    """The objects of `layout` (see `Hdf5Object`)."""
    return tuple(obj for obj in LAYOUT if obj.layout == layout)


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------


def _module(module: Module):
    if module not in _DIRS:
        raise ValueError(f"unknown module `{module}`, expected one of {tuple(_DIRS)}")
    return import_module(f"aida_data.{module}")


def _module_dir(module: Module) -> Path:
    """`<root>/<module dir>`, read through the module so that tests can
    patch its `DATASETS_DIR`."""
    mod = _module(module)
    if module == "dense":
        return mod._dense_dir()
    return mod.DATASETS_DIR / _DIRS[module]


def _is_native(module: Module, name: str) -> bool:
    """Whether `name` is an HDF5 dataset that is exposed as downloaded."""
    if module != "dense" or name.startswith("mteb:"):
        return False
    return _module("dense")._DATASETS_INFO[name].cache_name is None


def hdf5_path(module: Module, name: str) -> Path:
    """Where the HDF5 file of dataset `name` is (or will be). No I/O. For an
    HDF5-native dataset it is the download itself."""
    if _is_native(module, name):
        return _module("dense").local_path(name)
    directory = _module_dir(module) / "hdf5"
    if name.startswith("mteb:"):
        from . import _mteb

        task, model = _mteb.parse_name(name)
        return directory / "mteb" / task / f"{_mteb.model_slug(model)}.hdf5"
    return directory / f"{name}.hdf5"


def _source_path(module: Module, name: str) -> Path:
    """The parquet cache `name` is converted from (`corpus.parquet` for
    MTEB, always written last). No I/O."""
    mod = _module(module)
    if module == "dense":
        if name.startswith("mteb:"):
            from . import _mteb

            return _mteb.cache_dir(*_mteb.parse_name(name)) / _mteb.CORPUS_FILE
        return mod._dense_dir() / f"{mod._DATASETS_INFO[name].cache_name}.parquet"
    return mod.cache_path(name)


def _ensure_source(module: Module, name: str) -> Path:
    """Like `_source_path`, making the cache first (downloading if needed)."""
    mod = _module(module)
    if module == "dense" and name.startswith("mteb:"):
        from . import _mteb

        return mod._ensure_mteb(name) / _mteb.CORPUS_FILE
    return mod._ensure_cache(name)


# --------------------------------------------------------------------------
# Freshness
# --------------------------------------------------------------------------


def _is_fresh_for(path: Path, source: Path) -> bool:
    """Whether the converted file `path` was made by this version from the
    current `source`. Rebuilding or re-downloading the cache changes its
    mtime, so no separate index is needed."""
    if not path.is_file() or not source.is_file():
        return False
    try:
        with h5py.File(path) as hfp:
            return (
                hfp.attrs.get("aida_data_hdf5_version") == _HDF5_VERSION
                and hfp.attrs.get("source_mtime_ns") == source.stat().st_mtime_ns
            )
    except OSError:  # not an HDF5 file, e.g. truncated
        return False


def is_fresh(module: Module, name: str) -> bool:
    """Whether `ensure_hdf5` would return the file as is: for a native
    dataset, it is downloaded; otherwise the converted file is up to date.
    Neither downloads nor converts, so it is false if the cache is missing."""
    path = hdf5_path(module, name)
    if _is_native(module, name):
        return path.is_file()
    return _is_fresh_for(path, _source_path(module, name))


# --------------------------------------------------------------------------
# Conversion
# --------------------------------------------------------------------------


def _batch_rows(batch_size: int, width: int) -> int:
    return max(1, min(batch_size, _MAX_BATCH_ELEMENTS // max(width, 1)))


def _column_bytes(pf: pq.ParquetFile, field: pa.Field, n: int) -> int:
    """Size of the HDF5 dataset made from column `field` of `pf`."""
    if pa.types.is_string(field.type):
        # A vlen string takes a 16 byte reference plus its data.
        index = pf.schema_arrow.get_field_index(field.name)
        data = sum(
            pf.metadata.row_group(i).column(index).total_uncompressed_size
            for i in range(pf.metadata.num_row_groups)
        )
        return 16 * n + data
    return n * field.type.bit_width // 8


def _expected_size(module: Module, source: Path) -> int:
    """Upper-ish estimate of the bytes of the file converted from `source`
    (and the other parquet files of an MTEB directory)."""
    files = [source]
    if module == "dense":
        files += [
            f
            for f in (source.with_name("queries.parquet"), source.with_name("qrels.parquet"))
            if f.is_file()
        ]
    total = 0
    for path in files:
        pf = pq.ParquetFile(path)
        n = pf.metadata.num_rows
        for field in pf.schema_arrow:
            if pa.types.is_list(field.type) or pa.types.is_large_list(field.type):
                meta = parse_meta(pf.schema_arrow, path, _module(module)._CACHE_VERSION)
                width = field.type.value_type.bit_width // 8
                total += meta["n_edges"] * width + 8 * (n + 1)
            else:
                total += _column_bytes(pf, field, n)
    return total


def _strings(values) -> np.ndarray:
    return np.array(values, dtype=object)


def _write_points(
    hfp: h5py.File,
    path: Path,
    matrix: str,
    ids: str,
    batch_size: int,
    with_colors: bool,
) -> dict:
    """Stream the dense parquet cache `path` into the datasets `matrix`
    (float32), `colors` (if `with_colors` and the cache has them, in the
    dtype of the cache) and `ids` (if it has them). Returns its metadata."""
    from . import dense

    pf = pq.ParquetFile(path)
    meta = parse_meta(pf.schema_arrow, path, dense._CACHE_VERSION)
    n, d = pf.metadata.num_rows, meta["n_features"]
    points = hfp.create_dataset(matrix, (n, d), dtype=np.float32)
    x_columns = [f"x{j}" for j in range(d)]
    colors = color_columns = None
    if with_colors and meta.get("color_names") is not None:
        color_columns = [f"color{j}" for j in range(len(meta["color_names"]))]
        dtype = pf.schema_arrow.field(color_columns[0]).type.to_pandas_dtype()
        colors = hfp.create_dataset("colors", (n, len(color_columns)), dtype=dtype)
    id_dataset = None
    if meta.get("has_ids"):
        id_dataset = hfp.create_dataset(ids, (n,), dtype=h5py.string_dtype())

    start = 0
    for batch in pf.iter_batches(batch_size=_batch_rows(batch_size, d)):
        stop = start + batch.num_rows
        points[start:stop] = np.stack(
            [batch.column(c).to_numpy(zero_copy_only=False) for c in x_columns],
            axis=1,
        ).astype(np.float32, copy=False)
        if colors is not None:
            colors[start:stop] = np.stack(
                [batch.column(c).to_numpy(zero_copy_only=False) for c in color_columns],
                axis=1,
            )
        if id_dataset is not None:
            id_dataset[start:stop] = _strings(batch.column("id").to_pylist())
        start = stop
    return meta


def _convert_dense(hfp: h5py.File, source: Path, name: str, batch_size: int):
    from . import _mteb, dense

    meta = _write_points(hfp, source, "train", "ids", batch_size, with_colors=True)
    if name.startswith("mteb:"):
        directory = source.parent
        distance = meta["similarity"]
        if (directory / _mteb.QUERIES_FILE).exists():
            _write_points(
                hfp, directory / _mteb.QUERIES_FILE, "test", "test_ids", batch_size,
                with_colors=False,
            )
        if (directory / _mteb.QRELS_FILE).exists():
            # Retrieval judgments are small: read them whole.
            qrels, _ = read_table(directory / _mteb.QRELS_FILE, _mteb._QRELS_VERSION)
            group = hfp.create_group("qrels")
            for key, column in (("query", "query_id"), ("doc", "doc_id")):
                group.create_dataset(
                    key, data=_strings(qrels.column(column).to_pylist()),
                    dtype=h5py.string_dtype(),
                )
            group.create_dataset(
                "score", data=qrels.column("score").to_numpy().astype(np.int64)
            )
    else:
        distance = dense._DATASETS_INFO[name].distance_type
    hfp.attrs["distance"] = distance
    hfp.attrs["dimension"] = meta["n_features"]
    if meta.get("color_names") is not None:
        hfp.attrs.create("color_names", meta["color_names"], dtype=h5py.string_dtype())
        # Ragged, with no natural HDF5 form.
        hfp.attrs["color_labels"] = json.dumps(meta["color_labels"])


def _convert_graph(hfp: h5py.File, source: Path, name: str, batch_size: int):
    from . import graph

    pf = pq.ParquetFile(source)
    meta = parse_meta(pf.schema_arrow, source, graph._CACHE_VERSION)
    n, m = meta["n_nodes"], meta["n_edges"]
    schema = pf.schema_arrow
    offsets = hfp.create_dataset("offsets", (n + 1,), dtype=np.int64)
    neighbors = hfp.create_dataset(
        "neighbors", (m,), dtype=schema.field("nbrs").type.value_type.to_pandas_dtype()
    )
    colors = hfp.create_dataset(
        "colors", (n,), dtype=schema.field("color").type.to_pandas_dtype()
    )
    offsets[0] = 0
    node = edge = 0
    for batch in pf.iter_batches(batch_size=batch_size):
        nbrs = batch.column("nbrs")
        counts = pc.list_value_length(nbrs).to_numpy(zero_copy_only=False)
        values = nbrs.flatten().to_numpy(zero_copy_only=False)
        offsets[node + 1 : node + 1 + len(nbrs)] = edge + np.cumsum(
            counts, dtype=np.int64
        )
        neighbors[edge : edge + len(values)] = values
        colors[node : node + len(nbrs)] = batch.column("color").to_numpy(
            zero_copy_only=False
        )
        node += len(nbrs)
        edge += len(values)
    if node != n or edge != m:
        raise ValueError(f"{source} has {node} nodes and {edge} edges, not {n} and {m}")
    hfp.attrs["n_nodes"] = n
    hfp.attrs["n_edges"] = m


def _convert_timeseries(hfp: h5py.File, source: Path, name: str, batch_size: int):
    from . import timeseries

    pf = pq.ParquetFile(source)
    meta = parse_meta(pf.schema_arrow, source, timeseries._CACHE_VERSION)
    n, d = pf.metadata.num_rows, meta["n_dims"]
    values = hfp.create_dataset("values", (n, d), dtype=np.float64)
    time = hfp.create_dataset("time", (n,), dtype=np.int64) if meta["has_time"] else None
    x_columns = [f"x{j}" for j in range(d)]
    start = 0
    for batch in pf.iter_batches(batch_size=_batch_rows(batch_size, d)):
        stop = start + batch.num_rows
        values[start:stop] = np.stack(
            [batch.column(c).to_numpy(zero_copy_only=False) for c in x_columns],
            axis=1,
        ).astype(np.float64, copy=False)
        if time is not None:
            # Milliseconds since the epoch, as in the cache.
            time[start:stop] = (
                batch.column("t")
                .to_numpy(zero_copy_only=False)
                .astype("datetime64[ms]")
                .astype(np.int64)
            )
        start = stop
    if meta["dim_names"] is not None:
        hfp.attrs.create("dim_names", meta["dim_names"], dtype=h5py.string_dtype())
    if meta.get("sampling") is not None:
        hfp.attrs["sampling"] = meta["sampling"]
    source_urls = meta.get("source")
    if source_urls is not None:
        # Several downloads: space separated, URLs have no spaces.
        hfp.attrs["source"] = (
            source_urls if isinstance(source_urls, str) else " ".join(source_urls)
        )


_CONVERTERS: dict[str, Callable] = {
    "dense": _convert_dense,
    "graph": _convert_graph,
    "timeseries": _convert_timeseries,
}


def _convert(module: Module, name: str, source: Path, dest: Path, batch_size: int):
    """Convert the parquet cache `source` into the HDF5 file `dest`."""
    expected = _expected_size(module, source)
    _LOGGER.info("converting %s to %s (about %.1f MiB)", source, dest, expected / 2**20)
    dest.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(dest.parent).free
    if free < expected * _SPACE_MARGIN:
        raise OSError(
            errno.ENOSPC,
            f"not enough disk space to convert {name} to HDF5: needs "
            f"{expected * _SPACE_MARGIN / 2**20:.0f} MiB in {dest.parent}, "
            f"{free / 2**20:.0f} MiB free. Free some space (`prune-hdf5` "
            "deletes converted files), or use --format parquet, which needs "
            "no conversion",
        )
    # Taken before reading, so that a cache rebuilt meanwhile is not
    # mistaken for the one converted.
    mtime = source.stat().st_mtime_ns
    with atomic_path(dest, ".hdf5.tmp") as tmp:
        with h5py.File(tmp, "w") as hfp:
            _CONVERTERS[module](hfp, source, name, batch_size)
            hfp.attrs["aida_data_hdf5_version"] = _HDF5_VERSION
            hfp.attrs["source_mtime_ns"] = mtime
            hfp.attrs["module"] = module
            hfp.attrs["name"] = name


def ensure_hdf5(module: Module, name: str, *, batch_size: int = _BATCH_SIZE) -> Path:
    """The HDF5 file of dataset `name` (canonical: a registered name, or a
    full `mteb:` name) of `module`, made if missing or out of date.

    HDF5-native dense datasets (ann-benchmarks, VIBE) return their download.
    The others make sure their parquet cache exists (which may download and
    parse), then convert it, unless the converted file is fresh (see
    `is_fresh`). Raises `OSError` (`ENOSPC`) if the disk is too small for
    the conversion; nothing is left behind.
    """
    if _is_native(module, name):
        return _module("dense")._ensure_cache(name)
    source = _ensure_source(module, name)
    dest = hdf5_path(module, name)
    if not _is_fresh_for(dest, source):
        _convert(module, name, source, dest, batch_size)
    return dest


# --------------------------------------------------------------------------
# Cleanup
# --------------------------------------------------------------------------


def prune_hdf5(dry_run: bool = True) -> list[Path]:
    """The converted HDF5 files (every `.hdf5` file under the `hdf5/`
    directories of the modules) and the stale `*.hdf5.tmp` files left by
    killed conversions. With `dry_run=False` they are deleted. Native
    downloads and parquet caches are never touched. They can be made again
    by `ensure_hdf5`; links to them break."""
    paths = []
    for module in _DIRS:
        directory = _module_dir(module) / "hdf5"
        if directory.is_dir():
            paths += sorted(p for p in directory.rglob("*.hdf5") if p.is_file())
            paths += stale_temporaries(directory, recursive=True)
    total = sum(path.stat().st_size for path in paths)
    _LOGGER.info(
        "%s %d converted HDF5 and stale temporary files (%.1f MiB): %s",
        "would delete" if dry_run else "deleting",
        len(paths),
        total / 2**20,
        ", ".join(str(path) for path in paths),
    )
    if not dry_run:
        delete_raw(*paths)
    return paths
