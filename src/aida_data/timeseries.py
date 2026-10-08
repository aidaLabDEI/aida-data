"""\
Time series datasets, for motif discovery (ATTIMO, MOMENTI).

A dataset is one long, possibly multivariate, time series, returned as a
`TimeSeries` whose `values` has shape `(n, d)`: one row per time step in the
order of the source, one column per dimension. Row order is the data, so
nothing is reordered, deduplicated or normalized. Missing values are kept as
NaN.

Each dataset is parsed once into a zstd-compressed parquet cache (see
`_write_parquet_cache`); the raw download is then deleted unless `KEEP_RAW` is
set.
"""

import logging
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import pyarrow as pa

from ._cache import KEEP_RAW, delete_raw, read_table, write_table
from ._download import download as _download

DATASETS_DIR = Path(os.environ.get("AIDA_DATA_DIR", "datasets"))

_LOGGER = logging.getLogger("aida_data.timeseries")

# Layout version of the parquet caches written by `_cached`.
_CACHE_VERSION = 1

# What a loader returns: the (n, d) values, the d dimension names (or None)
# and the (n,) datetime64 time stamps (or None).
Parsed = tuple[np.ndarray, Sequence[str] | None, np.ndarray | None]
# The same as read back from the cache.
Cached = tuple[np.ndarray, tuple[str, ...] | None, np.ndarray | None]


def _load_matrix(
    path: Path,
    delimiter: str | None = None,
    skiprows: int = 0,
    drop_first_columns: int = 0,
) -> Parsed:
    """A series as a text table without header, possibly gzipped: one row per
    time step, fields separated by `delimiter` (any whitespace by default).
    A table with one value per line gives a univariate series.

    Blank lines are skipped, as `pyattimo.load_dataset` does: the ECG file has
    46991 of them, between runs of values. `drop_first_columns` drops leading
    columns that are not data, like a time index.
    """
    values = np.loadtxt(
        path, dtype=np.float64, delimiter=delimiter, skiprows=skiprows, ndmin=2
    )
    return values[:, drop_first_columns:], None, None


def _load_csv(path: Path) -> Parsed:
    """A multivariate series: a CSV file with a header of dimension names."""
    df = pd.read_csv(path, dtype=np.float64)
    return df.to_numpy(), list(df.columns), None


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    # One URL per raw file, a single URL is the common case.
    url: str | tuple[str, ...]
    # (*raw_paths) -> (values, dim_names, time), see `Parsed`
    loader_function: Callable
    # Names of the raw files; by default the last component of the URLs.
    filename: str | tuple[str, ...] | None = None
    license: str | None = None
    sampling: str | None = None

    @property
    def urls(self) -> tuple[str, ...]:
        return (self.url,) if isinstance(self.url, str) else tuple(self.url)

    @property
    def filenames(self) -> tuple[str, ...]:
        if self.filename is None:
            return tuple(Path(urlparse(url).path).name for url in self.urls)
        return (
            (self.filename,) if isinstance(self.filename, str) else tuple(self.filename)
        )


_DATASETS_INFO: dict[str, DatasetInfo] = {}


def register(info: DatasetInfo, force=False):
    if info.name in _DATASETS_INFO and not force:
        raise ValueError(f"dataset {info.name} already registered")
    if len(info.urls) != len(info.filenames):
        raise ValueError(f"dataset {info.name}: one file name per URL is needed")
    _DATASETS_INFO[info.name] = info


# The figshare article 20747617 of the ATTIMO authors. The URLs of
# `figshare.com/ndownloader` answer with a bot challenge, this host does not.
_FIGSHARE = "https://ndownloader.figshare.com/files/"
_CC_BY_4 = "CC BY 4.0"

for _name, _file_id in (
    ("astro", 36982360),
    ("ecg", 36982384),
    ("freezer", 36982390),
    ("gap", 36982396),
    ("humany", 36982399),
):
    register(
        DatasetInfo(
            _name,
            f"{_FIGSHARE}{_file_id}",
            _load_matrix,
            filename=f"{_name}.txt.gz",
            license=_CC_BY_4,
        )
    )

# Steam generator sensors (STUMPY tutorial data), used by ATTIMO and MOMENTI.
register(
    DatasetInfo(
        "steamgen",
        "https://zenodo.org/api/records/4273921/files/STUMPY_Basics_steamgen.csv/content",
        _load_csv,
        filename="steamgen.csv",
        license=_CC_BY_4,
    )
)


# Files of the Motiflets repository (GPL-3.0), pinned to a commit of its
# `pyattimo` branch. The repository does not state a licence for the data.
_MOTIFLETS = (
    "https://raw.githubusercontent.com/patrickzib/motiflets/"
    "8afb3f9d00f2782c77f1e256ff73c3e7ebbae1d5/datasets/"
)

for _name, _path, _loader in (
    # Power draw of a dishwasher.
    ("dishwasher", "original/dishwasher.txt", _load_matrix),
    # Sleep EEG of a nap at 100 Hz (PhysioNet).
    ("npo141", "original/npo141.csv", _load_matrix),
    # Channel 0 of subject 231 of an ECG arrhythmia database; the file has a
    # `"Channel 0"` header line.
    (
        "arrhythmia",
        "experiments/arrhythmia_subject231_channel0.csv",
        partial(_load_matrix, skiprows=1),
    ),
):
    register(DatasetInfo(_name, _MOTIFLETS + _path, _loader, filename=f"{_name}.txt"))

# Files of the MOMENTI repository (AGPL-3.0), pinned to a commit. The data
# files have their own origin. MOMENTI's own loaders read them with
# `pd.read_csv`, which takes the first row of these headerless files for a
# header, and drop the first column; neither is done here.
_MOMENTI = (
    "https://raw.githubusercontent.com/aidaLabDEI/MOMENTI-motifs/"
    "baa24820c0aa2aedd13fa15ac2ee54ecc8dd7694/Datasets/"
)

for _info in (
    # Eight potentials recorded on a pregnant woman (DaISy, KU Leuven); the
    # first column is the time in seconds, every 4 ms.
    DatasetInfo(
        "foetal-ecg",
        _MOMENTI + "FOETAL_ECG.dat",
        partial(_load_matrix, drop_first_columns=1),
        filename="foetal-ecg.dat",
        sampling="250 Hz",
    ),
    # Six standardized channels of an industrial evaporator (DaISy, KU
    # Leuven); there is no index column.
    DatasetInfo(
        "evaporator",
        _MOMENTI + "evaporator.dat",
        _load_matrix,
        filename="evaporator.dat",
    ),
    # 32 channels with no header. The origin is not documented in the repository.
    DatasetInfo(
        "ruth",
        _MOMENTI + "RUTH.csv",
        partial(_load_matrix, delimiter=","),
        filename="ruth.csv",
    ),
):
    register(_info)


def available_datasets():
    return list(_DATASETS_INFO.keys())


def _check_name(name: str):
    if name not in _DATASETS_INFO:
        raise KeyError(
            f"Dataset `{name}` not available. Pick one of {available_datasets()}"
        )


def local_paths(name: str) -> tuple[Path, ...]:
    """The raw downloads of dataset `name`."""
    _check_name(name)
    return tuple(
        DATASETS_DIR / "timeseries" / filename
        for filename in _DATASETS_INFO[name].filenames
    )


def cache_path(name: str) -> Path:
    """The parquet cache of dataset `name`."""
    _check_name(name)
    return DATASETS_DIR / "timeseries" / f"{name}.parquet"


@dataclass(frozen=True)
class TimeSeries:
    name: str
    values: np.ndarray  # shape (n, d), float64, one row per time step
    dim_names: tuple[str, ...] | None = None  # d names, e.g. sensors
    time: np.ndarray | None = None  # shape (n,), datetime64[ms]

    @property
    def n_samples(self) -> int:
        return len(self.values)

    @property
    def n_dims(self) -> int:
        return self.values.shape[1]

    @property
    def univariate(self) -> np.ndarray:
        """The values as an (n,) array, for series with a single dimension."""
        if self.n_dims != 1:
            raise ValueError(f"{self.name} has {self.n_dims} dimensions, not 1")
        return self.values[:, 0]


def load(name: str) -> TimeSeries:
    """Load time series dataset `name` as a `TimeSeries`.

    The raw files are downloaded (if needed) and parsed once: the result is
    kept in a parquet cache (see `_cached`). `values` is an `(n, d)` float64
    array, with `d = 1` for univariate series, in the order of the source.

    Example::

        ts = load("ecg")
        ts.values.shape  # (n, 1)
        ts.univariate    # (n,)
    """
    _check_name(name)
    return TimeSeries(name, *_cached(name))


def _write_parquet_cache(
    path: Path,
    values: np.ndarray,
    dim_names: Sequence[str] | None = None,
    time: np.ndarray | None = None,
    sampling: str | None = None,
    source: str | list[str] | None = None,
):
    """Write the series to the parquet cache `path`.

    One row per time step: float64 columns `x0..x{d-1}`, one per dimension,
    then `t` (timestamp[ms]) if `time` is given. Column names are positional,
    the dimension names are in the `aida_data` schema metadata.
    """
    columns = {f"x{j}": values[:, j] for j in range(values.shape[1])}
    if time is not None:
        columns["t"] = pa.array(time.astype("datetime64[ms]"))
    meta = {
        "version": _CACHE_VERSION,
        "n_samples": len(values),
        "n_dims": values.shape[1],
        "dim_names": None if dim_names is None else list(dim_names),
        "has_time": time is not None,
        "sampling": sampling,
        "source": source,
    }
    write_table(path, pa.table(columns), meta)


def _read_parquet_cache(path: Path) -> Cached:
    """Read a cache written by `_write_parquet_cache`."""
    table, meta = read_table(path, _CACHE_VERSION)
    # Fill a preallocated array one column at a time, rather than with
    # `np.column_stack`, so that only one column is duplicated at a time.
    values = np.empty((table.num_rows, meta["n_dims"]), dtype=np.float64)
    for j in range(values.shape[1]):
        values[:, j] = table.column(f"x{j}").to_numpy()
    dim_names = None if meta["dim_names"] is None else tuple(meta["dim_names"])
    time = table.column("t").to_numpy() if meta["has_time"] else None
    return values, dim_names, time


def _cached(name: str) -> Cached:
    """Return `(values, dim_names, time)` of dataset `name`, parsing it only
    once.

    The first call downloads the raw files, parses them with the loader of the
    dataset and stores the result in `cache_path(name)`; later calls only read
    that file.

    After a fresh parse the raw files are deleted, unless `KEEP_RAW` is set or
    another dataset that is not cached yet needs them.
    """
    cache = cache_path(name)
    if not cache.is_file():
        info = _DATASETS_INFO[name]
        paths = local_paths(name)
        cache.parent.mkdir(parents=True, exist_ok=True)
        for url, path in zip(info.urls, paths):
            _download(url, path)
        _LOGGER.info("parsing %s into %s", ", ".join(map(str, paths)), cache)
        values, dim_names, time = info.loader_function(*paths)
        values, dim_names, time = _validate(name, values, dim_names, time)
        source = info.urls[0] if len(info.urls) == 1 else list(info.urls)
        _write_parquet_cache(cache, values, dim_names, time, info.sampling, source)
        del values
        # Only after a fresh parse: upgrading alone never removes files.
        if not KEEP_RAW:
            delete_raw(*(path for path in paths if not _needed_by_others(name, path)))
    # Always read back from the file, so that the first and later calls
    # return identical arrays.
    return _read_parquet_cache(cache)


def _validate(name: str, values, dim_names, time) -> Parsed:
    """Check and normalize what a loader returned."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError(f"{name}: expected (n, d) values, got shape {values.shape}")
    if dim_names is not None and len(dim_names) != values.shape[1]:
        raise ValueError(
            f"{name}: {len(dim_names)} dimension names, {values.shape[1]} dimensions"
        )
    if time is not None:
        time = np.asarray(time)
        if time.shape != (len(values),):
            raise ValueError(f"{name}: time must have shape ({len(values)},)")
    return values, dim_names, time


def _needed_by_others(name: str, path: Path) -> bool:
    """Whether another dataset that is not cached yet reads the raw `path`."""
    return any(
        path in local_paths(other) and not cache_path(other).is_file()
        for other in _DATASETS_INFO
        if other != name
    )


def prune_raw(dry_run: bool = True) -> list[Path]:
    """Raw downloads of datasets whose cache exists, which are no longer
    needed. With `dry_run=False` they are deleted.
    """
    paths = []
    for name in _DATASETS_INFO:
        if cache_path(name).is_file():
            paths += [
                path
                for path in local_paths(name)
                if path.is_file()
                and path not in paths
                and not _needed_by_others(name, path)
            ]
    total = sum(path.stat().st_size for path in paths)
    _LOGGER.info(
        "%s %d raw files (%.1f MiB): %s",
        "would delete" if dry_run else "deleting",
        len(paths),
        total / 2**20,
        ", ".join(str(path) for path in paths),
    )
    if not dry_run:
        delete_raw(*paths)
    return paths
