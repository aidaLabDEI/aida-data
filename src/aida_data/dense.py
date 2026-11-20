"""\
Dense datasets, under different distance measures.
Datasets are collected from different sources

Some datasets carry categorical attributes ("colors", e.g. sex or race for
fair clustering). `load` returns them in `Dataset.colors` (see `Colors`),
aligned with the rows of `Dataset.dataset`; for the other datasets
`Dataset.colors` is None. Large datasets are parsed once and cached as
zstd-compressed parquet next to the raw download.
"""

import io
import json
import logging
import re
import tarfile
import zipfile
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
from sklearn.base import BaseEstimator, TransformerMixin

from . import _mteb_names
from ._cache import KEEP_RAW, delete_raw, read_table, write_table
from ._download import download as _download
from ._paths import datasets_dir
from ._tmp import stale_temporaries

DATASETS_DIR = datasets_dir()

_LOGGER = logging.getLogger("aida_data.dense")

# Layout version of the parquet caches written by `_ensure_cache`.
_CACHE_VERSION = 1


@dataclass(frozen=True)
class Colors:
    """Categorical attributes ("colors") of the rows of a dataset.

    `values[i, j]` is the code of the `j`-th color of row `i`. Codes are
    dense, 0-based and sorted by label, so `labels[j][values[i, j]]` is the
    original category. Codes are not re-densified when rows are dropped, so
    some codes may be absent from `values`.
    """

    values: np.ndarray  # (n, c) int64
    names: tuple[str, ...]  # c column names, e.g. ("sex", "race")
    labels: tuple[tuple[str, ...], ...]  # labels[j][code] = original category

    def __post_init__(self):
        if self.values.ndim != 2 or self.values.shape[1] != len(self.names):
            raise ValueError(
                f"values must have shape (n, {len(self.names)}), got {self.values.shape}"
            )
        if len(self.labels) != len(self.names):
            raise ValueError("names and labels must have the same length")

    def _index(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError:
            raise KeyError(f"no color `{name}`, pick one of {self.names}") from None

    def column(self, name: str) -> np.ndarray:
        """The (n,) codes of color `name`."""
        return self.values[:, self._index(name)]

    def n_colors(self, name: str) -> int:
        """Number of distinct codes of color `name` actually present."""
        return len(np.unique(self.column(name)))

    def take(self, rows: np.ndarray) -> "Colors":
        """Colors of the rows selected by `rows` (indices or boolean mask)."""
        return replace(self, values=self.values[rows])


def _load_hdf5(path: Path):
    with h5py.File(path) as hfp:
        return hfp["train"][:], hfp["test"][:], hfp["distances"][:]


def _write_parquet_cache(
    path: Path,
    data: np.ndarray,
    colors: Colors | None,
    ids: np.ndarray | None = None,
    extra_meta: dict | None = None,
):
    """Write `data` and `colors` to the parquet cache `path`.

    One row per point: float32 columns `x0..x{d-1}` for the features, then
    `color0..color{c-1}` for the color codes, then the string column `id`
    if `ids` is given. Column names are positional, the color names and
    labels are in the `aida_data` schema metadata, together with `has_ids`
    and the (JSON serializable) entries of `extra_meta`.
    """
    columns = {f"x{j}": data[:, j] for j in range(data.shape[1])}
    meta = {
        "version": _CACHE_VERSION,
        "n_features": data.shape[1],
        "color_names": None,
        "color_labels": None,
        "has_ids": ids is not None,
    }
    if colors is not None:
        # Store codes in the narrowest integer type, they are widened back to
        # int64 on load.
        codes = colors.values.astype(
            np.min_scalar_type(int(colors.values.max(initial=0)))
        )
        columns |= {f"color{j}": codes[:, j] for j in range(codes.shape[1])}
        meta["color_names"] = list(colors.names)
        meta["color_labels"] = [list(labels) for labels in colors.labels]
    if ids is not None:
        if len(ids) != data.shape[0]:
            raise ValueError(f"{len(ids)} ids for {data.shape[0]} rows")
        columns["id"] = pa.array([str(i) for i in ids], type=pa.string())
    if extra_meta:
        meta |= extra_meta
    write_table(path, pa.table(columns), meta)


def _read_parquet_cache_full(
    path: Path,
) -> tuple[np.ndarray, Colors | None, np.ndarray | None, dict]:
    """Read a cache written by `_write_parquet_cache` as
    `(data, colors, ids, meta)`. `ids` is None if the cache has none (also
    for caches written before ids existed); `meta` is the `aida_data`
    metadata dict, with the entries of `extra_meta`."""
    table, meta = read_table(path, _CACHE_VERSION)
    # Fill preallocated arrays one column at a time, rather than with
    # `np.column_stack`, so that only one column is duplicated at a time.
    data = np.empty((table.num_rows, meta["n_features"]), dtype=np.float32)
    for j in range(data.shape[1]):
        data[:, j] = table.column(f"x{j}").to_numpy()
    colors = None
    if meta.get("color_names") is not None:
        values = np.empty((table.num_rows, len(meta["color_names"])), dtype=np.int64)
        for j in range(values.shape[1]):
            values[:, j] = table.column(f"color{j}").to_numpy()
        colors = Colors(
            values,
            tuple(meta["color_names"]),
            tuple(tuple(labels) for labels in meta["color_labels"]),
        )
    ids = None
    if meta.get("has_ids"):
        ids = np.array(table.column("id").to_pylist(), dtype=object)
    del table
    return data, colors, ids, meta


def _read_parquet_cache(path: Path) -> tuple[np.ndarray, Colors | None]:
    """Read a cache written by `_write_parquet_cache` as `(data, colors)`."""
    data, colors, _, _ = _read_parquet_cache_full(path)
    return data, colors


def _read_hdf5_cache(path: Path) -> tuple[np.ndarray, Colors | None]:
    """Read a cache written by aida_data < 0.2 (HDF5), only for migration."""
    with h5py.File(path) as hfp:
        data = hfp["X"][:]
        if "colors" not in hfp:
            return data, None
        codes = hfp["colors"]
        colors = Colors(
            codes[:].astype(np.int64),
            tuple(json.loads(codes.attrs["names"])),
            tuple(tuple(labels) for labels in json.loads(codes.attrs["labels"])),
        )
    return data, colors


def _write_cache(path: Path, cache_name: str, build: Callable[[Path], tuple]) -> Path:
    """Make sure the parquet cache `path.parent / f"{cache_name}.parquet"`
    exists, and return its path. If it is missing it is made by parsing
    `path` with `build`, which returns `(features, colors)`, `colors`
    possibly None. The cache is not read back.

    The cache is a zstd-compressed parquet file with one row per point and
    one column per feature and per color (see `_write_parquet_cache`). It is
    keyed by dataset name rather than by the name of the raw file, because
    several datasets may share one download.

    After a fresh parse the raw file `path` is deleted, unless `KEEP_RAW`
    is set or another registered dataset with a different cache needs it.
    """
    cache = path.parent / f"{cache_name}.parquet"
    if not cache.is_file():
        legacy = path.parent / f"{cache_name}.hdf5"
        if legacy.is_file():
            # Migrate a cache written by aida_data < 0.2 without re-parsing.
            _LOGGER.info("converting %s to %s", legacy, cache)
            _write_parquet_cache(cache, *_read_hdf5_cache(legacy))
            legacy.unlink()
        else:
            _LOGGER.info("parsing %s into %s", path, cache)
            _write_parquet_cache(cache, *build(path))
            # Only after a fresh parse: upgrading alone never removes files.
            # A raw file shared by datasets with the same cache (only HIGGS
            # today) is no longer needed by any of them; one needed by
            # another cache or loader is kept.
            if not KEEP_RAW and path.is_file() and not _raw_is_shared(path, cache_name):
                delete_raw(path)
    return cache


def _cached(
    path: Path, cache_name: str, build: Callable[[Path], tuple]
) -> tuple[np.ndarray, Colors | None]:
    """`_write_cache`, then the contents of the cache. Always reads back from
    the file, so that the first and later calls return identical arrays.
    `load` does not use it: it calls `_ensure_cache` and reads the cache
    itself."""
    return _read_parquet_cache(_write_cache(path, cache_name, build))


# Names of the caches written by `_ensure_cache`, shared by the loaders and the
# registrations (`DatasetInfo.cache_name`).
_PAMAP_CACHE = "pamap"
_BIOKDD_CACHE = "biokdd"
_METROPT3_CACHE = "metropt3"
_HOUSEHOLD_POWER_CACHE = "household-power"
_COVERTYPE_CACHE = "covertype"
_CENSUS1990_CACHE = "census1990"
_PHONES_CACHE = "phones"
_HIGGS_CACHE = "higgs"


def _split_table(
    df: pd.DataFrame, features: list[str], colors: list[str]
) -> tuple[np.ndarray, Colors]:
    """Drop the rows of `df` with nulls in `features + colors`, and return the
    float32 feature matrix and the encoded colors.

    Color codes are 0-based and sorted by category (numerically for numeric
    columns); labels are the categories converted to strings.
    """
    n_rows = len(df)
    df = df[features + colors].dropna()
    n_dropped = n_rows - len(df)
    if n_dropped > 0:
        _LOGGER.info("dropped %d rows with missing values", n_dropped)
    data = df[features].to_numpy(dtype=np.float32)
    values = np.empty((len(df), len(colors)), dtype=np.int64)
    labels = []
    for j, column in enumerate(colors):
        codes, uniques = pd.factorize(df[column], sort=True)
        values[:, j] = codes
        labels.append(tuple(str(u) for u in uniques))
    return data, Colors(values, tuple(colors), tuple(labels))


def _parse_pamap(path: Path):
    with zipfile.ZipFile(path, "r") as zip_ref:
        arr = []
        for i in range(1, 10):
            zfn = f"PAMAP2_Dataset/Protocol/subject10{i}.dat"
            zf = zip_ref.open(zfn)
            for line in zf:
                line = line.decode()
                l = list(map(float, line.strip().split()))
                # remove timestamp and activity ID
                arr.append(l[2:])
        X = np.nan_to_num(np.array(arr))  # many NaNs in data, replace them with 0.
        return X.astype(np.float32), None


def _load_census(path: Path):
    raw = np.load(path)
    data = raw["X"].astype(np.float32)
    return data, None, None


def _load_densired_hard(path: Path):
    """Synthetic densired dataset (https://github.com/PhilJahn/DENSIRED):
    5000 tight, well-separated clusters (core_num=1, radius=0.15) over
    n=100000 points, dim=100.
    """
    if not path.is_file():
        try:
            from densired import datagen
        except ImportError:
            _LOGGER.error("Install the densired package first!")
            raise
        skeleton = datagen.densityDataGen(
            dim=100,
            clunum=5000,
            core_num=1,
            radius=0.15,
            momentum=0.5,
            ratio_noise=0.0,
            seed=1234,
        )
        raw = skeleton.generate_data(100_000)
        data = np.unique(raw[:, :-1].astype(np.float32), axis=0)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, X=data)
    else:
        data = np.load(path)["X"].astype(np.float32)
    return data, None, None


def _load_ht(path: Path):
    # Unzip
    with zipfile.ZipFile(path, "r") as zip_ref:
        zip_ref.extractall(path.parent)
        # Unzip the inner file "HT_Sensor_dataset.zip"
        inner_zip_path = path.parent / "HT_Sensor_dataset.zip"
        # if there is an inner zip file, extract it
        if inner_zip_path.is_file():
            with zipfile.ZipFile(inner_zip_path, "r") as inner_zip_ref:
                inner_zip_ref.extractall(path.parent)
    # Load data
    data_path = path.parent / "HT_Sensor_dataset.dat"
    data = pd.read_csv(data_path, sep=r"\s+")
    del data["id"]
    data = data.to_numpy().astype(np.float32)
    data = np.nan_to_num(data)
    return data, None, None


def _load_fbin(path: Path):
    # `.fbin` layout: uint32 number of vectors, uint32 dimension, then the
    # vectors themselves as row-major float32.
    with open(path, "rb") as fp:
        n, d = np.fromfile(fp, dtype=np.uint32, count=2)
        data = np.fromfile(fp, dtype=np.float32, count=int(n) * int(d))
    data = data.reshape(int(n), int(d))
    return data, None, None


def _load_chem(path: Path):
    # Unzip
    # Catch and handle not finding the file, try with the explicit file name
    try:
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(path.parent)
        # Load data
        data_path = (
            path.parent / "gas+sensor+array+under+dynamic+gas+mixtures/ethylene_CO.txt"
        )
        data = pd.read_csv(data_path, sep=r"\s+").to_numpy().astype(np.float32)
    except Exception:
        data_path = path.parent / "ethylene_CO.txt"
        data = pd.read_csv(data_path, sep=r"\s+").to_numpy().astype(np.float32)
    data = np.nan_to_num(data)
    return data, None, None


# Colored datasets from fair-clustering
# (https://github.com/Cecca/fair-clustering-code, `datasets.py`). Features and
# colors follow that repository; unlike it, rows with missing values in the
# selected columns are always dropped, and features are kept as float32.

_ADULT_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]


def _load_adult(path: Path):
    """UCI Adult (`adult.data` only, not `adult.test`).

    Features: `age, fnlwgt, education-num, capital-gain, hours-per-week`.
    Colors: `sex, race, marital-status`.
    """
    with zipfile.ZipFile(path) as zf, zf.open("adult.data") as fp:
        df = pd.read_csv(
            fp,
            header=None,
            names=_ADULT_COLUMNS,
            skipinitialspace=True,
            na_values="?",
        )
    features = ["age", "fnlwgt", "education-num", "capital-gain", "hours-per-week"]
    data, colors = _split_table(df, features, ["sex", "race", "marital-status"])
    return data, None, None, colors


def _load_athlete(path: Path):
    """Olympic athletes 1896-2016 (https://github.com/rgriff23/Olympic_history).

    Features: `Age, Height, Weight`. Colors: `Sex`. Rows with missing
    values are dropped. The file has one row per athlete and event, so
    deduplication in `load` removes most rows (206k complete rows -> about
    45k).
    """
    df = pd.read_csv(path, na_values=["", "NA"])
    data, colors = _split_table(df, ["Age", "Height", "Weight"], ["Sex"])
    return data, None, None, colors


def _load_diabetes(path: Path):
    """Diabetes 130-US hospitals for years 1999-2008 (UCI 296).

    Features: `age` (lower bound of the 10-year bracket), `time_in_hospital,
    num_lab_procedures, num_procedures, num_medications, diag_1, diag_2,
    diag_3, number_diagnoses`, where the ICD-9 diagnosis codes are read as
    numbers and non-numeric codes (`V..`, `E..`) become missing. Colors:
    `gender, race`. Rows with missing values are dropped.
    """
    with zipfile.ZipFile(path) as zf, zf.open("diabetic_data.csv") as fp:
        df = pd.read_csv(fp, na_values="?", low_memory=False)
    df["age"] = df["age"].str.extract(r"(\d+)", expand=False).astype(float)
    for column in ("diag_1", "diag_2", "diag_3"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    features = [
        "age",
        "time_in_hospital",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "diag_1",
        "diag_2",
        "diag_3",
        "number_diagnoses",
    ]
    data, colors = _split_table(df, features, ["gender", "race"])
    return data, None, None, colors


def _load_creditcard(path: Path):
    """Default of credit card clients (UCI 350).

    Features: `LIMIT_BAL, AGE, BILL_AMT1..6, PAY_AMT1..6`.
    Colors: `SEX, EDUCATION, MARRIAGE` (numeric codes, see the UCI page).
    """
    with zipfile.ZipFile(path) as zf, zf.open("default of credit card clients.xls") as fp:
        df = pd.read_excel(fp, header=1)
    features = (
        ["LIMIT_BAL", "AGE"]
        + [f"BILL_AMT{i}" for i in range(1, 7)]
        + [f"PAY_AMT{i}" for i in range(1, 7)]
    )
    data, colors = _split_table(df, features, ["SEX", "EDUCATION", "MARRIAGE"])
    return data, None, None, colors


def _load_kfc(path: Path, features: list[str], colors: list[str]):
    """Datasets of "KFC: A Scalable Approximation Algorithm for k-center Fair
    Clustering" (https://github.com/FaroukY/KFC-ScalableFairClustering)."""
    data, colors = _split_table(pd.read_csv(path), features, colors)
    return data, None, None, colors


# Datasets of the silhouette repository (`research/data/parseData.py`).
# Features follow that script; in addition, the natural class labels are
# exposed as colors, so that these datasets can be used for fair clustering
# too.


def _load_breast(path: Path):
    """Breast Cancer Wisconsin (Diagnostic), UCI 17 (`wdbc.data`).

    Features: the 30 real-valued attributes (`id` dropped).
    Colors: `diagnosis` (`B`/`M`).
    """
    with zipfile.ZipFile(path) as zf, zf.open("wdbc.data") as fp:
        df = pd.read_csv(fp, header=None)
    df = df.rename(columns={0: "id", 1: "diagnosis"})
    data, colors = _split_table(df, list(range(2, 32)), ["diagnosis"])
    return data, None, None, colors


def _load_wine(path: Path):
    """Wine quality, UCI 186: red wines followed by white wines.

    Features: the 11 physicochemical attributes. Colors: `type`
    (`red`/`white`) and `quality` (score from 3 to 9).
    """
    parts = []
    with zipfile.ZipFile(path) as zf:
        for kind in ("red", "white"):
            with zf.open(f"winequality-{kind}.csv") as fp:
                parts.append(pd.read_csv(fp, sep=";").assign(type=kind))
    df = pd.concat(parts, ignore_index=True)
    features = [c for c in df.columns if c not in ("quality", "type")]
    data, colors = _split_table(df, features, ["type", "quality"])
    return data, None, None, colors


def _load_shuttle(path: Path):
    """Statlog (Shuttle), UCI 148: `shuttle.trn` followed by `shuttle.tst`.

    Features: the 9 numeric attributes. Colors: `class` (1 to 7).
    `shuttle.trn.Z` is compressed with Unix `compress`, which needs the
    optional `unlzw3` package (`pip install aida-data[shuttle]`).
    """
    try:
        from unlzw3 import unlzw
    except ImportError:
        _LOGGER.error("Install the unlzw3 package first!")
        raise
    with zipfile.ZipFile(path) as zf:
        train = unlzw(zf.read("shuttle.trn.Z"))
        test = zf.read("shuttle.tst")
    df = pd.read_csv(io.BytesIO(train + test), sep=" ", header=None)
    df = df.rename(columns={9: "class"})
    data, colors = _split_table(df, list(range(9)), ["class"])
    return data, None, None, colors


def _load_rt_iot(path: Path):
    """RT-IoT2022, UCI 942.

    Features: all numeric attributes (the index column, the ports
    `id.orig_p`/`id.resp_p` and the categorical `proto`/`service` dropped).
    Colors: `Attack_type`. Unlike `parseData.py`, `fwd_last_window_size` is
    kept: dropping it looks accidental there. Without the ports most flows
    are exact duplicates, so deduplication in `load` keeps about 18k of the
    123k rows.
    """
    with zipfile.ZipFile(path) as zf, zf.open("RT_IOT2022") as fp:
        df = pd.read_csv(fp, index_col=0)
    dropped = {"id.orig_p", "id.resp_p", "proto", "service", "Attack_type"}
    features = [c for c in df.columns if c not in dropped]
    data, colors = _split_table(df, features, ["Attack_type"])
    return data, None, None, colors


def _parse_biokdd(path: Path):
    """KDD Cup 2004 protein homology (`bio_train.dat`).

    Features: the 74 attributes (block id, example id and label dropped).
    Colors: `label` (1 if the protein is homologous to the native sequence).
    """
    with tarfile.open(path) as tf, tf.extractfile("bio_train.dat") as fp:
        df = pd.read_csv(fp, sep="\t", header=None)
    df = df.rename(columns={2: "label"})
    return _split_table(df, list(range(3, 77)), ["label"])


def _parse_metropt3(path: Path):
    """MetroPT-3 air compressor, UCI 791.

    Features: the 7 analog sensors (`TP2` to `Motor_current`); the index,
    the timestamp and the 8 digital signals are dropped. No colors.
    """
    with zipfile.ZipFile(path) as zf, zf.open("MetroPT3(AirCompressor).csv") as fp:
        df = pd.read_csv(fp, index_col=0, engine="pyarrow")
    df = df.drop(columns="timestamp").iloc[:, :-8]
    return df.to_numpy(dtype=np.float32), None


def _parse_household_power(path: Path):
    """Individual household electric power consumption, UCI 235.

    Features: the 7 measurements (`Date` and `Time` dropped). No colors.
    Missing values (`?`) become NaN, so `load` drops those rows, while
    `parseData.py` replaces them with 0.
    """
    with (
        zipfile.ZipFile(path) as zf,
        zf.open("household_power_consumption.txt") as fp,
    ):
        df = pd.read_csv(fp, sep=";", na_values="?", engine="pyarrow")
    return df.drop(columns=["Date", "Time"]).to_numpy(dtype=np.float32), None


# Large datasets, parsed once and cached (see `_ensure_cache`).


def _parse_covertype(path: Path):
    """Covertype, UCI 31 (`covtype.data.gz`), as used by streaming-fair and
    FSR.

    Features: the 54 attributes (10 numeric, 4 binary wilderness areas,
    40 binary soil types). Colors: `cover_type` (1 to 7).
    """
    with zipfile.ZipFile(path) as zf, zf.open("covtype.data.gz") as fp:
        df = pd.read_csv(fp, header=None, compression="gzip")
    df = df.rename(columns={54: "cover_type"})
    return _split_table(df, list(range(54)), ["cover_type"])


_CENSUS1990_FEATURES = [
    "dAncstry1",
    "dAncstry2",
    "iAvail",
    "iCitizen",
    "iClass",
    "dDepart",
    "iDisabl1",
    "iDisabl2",
    "iEnglish",
    "iFeb55",
    "iFertil",
    "dHispanic",
    "dHour89",
    "dHours",
    "iImmigr",
    "dIncome1",
    "dIncome2",
    "dIncome3",
    "dIncome4",
    "dIncome5",
    "dIncome6",
    "dIncome7",
    "dIncome8",
    "dIndustry",
    "iKorean",
    "iLang1",
    "iLooking",
    "iMarital",
    "iMay75880",
    "iMeans",
    "iMilitary",
    "iMobility",
    "iMobillim",
    "dOccup",
    "iOthrserv",
    "iPerscare",
    "dPOB",
    "dPoverty",
    "dPwgt1",
    "iRagechld",
    "dRearning",
    "iRelat1",
    "iRelat2",
    "iRemplpar",
    "iRiders",
    "iRlabor",
    "iRownchld",
    "dRpincome",
    "iRPOB",
    "iRrelchld",
    "iRspouse",
    "iRvetserv",
    "iSchool",
    "iSept80",
    "iSubfam1",
    "iSubfam2",
    "iTmpabsnt",
    "dTravtime",
    "iVietnam",
    "dWeek89",
    "iWork89",
    "iWorklwk",
    "iWWII",
    "iYearsch",
    "iYearwrk",
    "dYrsserv",
]


_CENSUS1990_MEMBER = "USCensus1990.data.txt"


def _parse_census1990(path: Path):
    """US Census Data (1990), UCI 116, as used by fair-clustering. Not to be
    confused with the ADBench `census` dataset.

    Features: the 66 (already discretized) attributes used by
    fair-clustering. Colors: `dAge` and `iSex`, which fair-clustering
    registers as two datasets (`census1990`, `census1990_age`).
    """
    with zipfile.ZipFile(path) as zf, zf.open(_CENSUS1990_MEMBER) as fp:
        df = pd.read_csv(fp, engine="pyarrow")
    return _split_table(df, _CENSUS1990_FEATURES, ["dAge", "iSex"])


_PHONES_INNER_ZIP = "Activity recognition exp.zip"
_PHONES_MEMBER = "Activity recognition exp/Phones_accelerometer.csv"
_PHONES_COLORS = ["gt", "User", "Model", "Device"]


def _parse_phones(path: Path):
    """Heterogeneity Activity Recognition, UCI 344
    (`Phones_accelerometer.csv`), as used by streaming-fair and MACACO.

    Features: the accelerometer readings `x, y, z`. Colors: the activity
    `gt` (7 values, including `null` for readings outside any activity, as
    in streaming-fair and MACACO), and also `User`, `Model` and `Device`.
    """
    with zipfile.ZipFile(path) as zf:
        inner = zipfile.ZipFile(zf.open(_PHONES_INNER_ZIP))
        with inner, inner.open(_PHONES_MEMBER) as fp:
            # `null` is a category of `gt`, not a missing value
            df = pd.read_csv(
                fp,
                usecols=["x", "y", "z"] + _PHONES_COLORS,
                keep_default_na=False,
                na_values=[""],
                engine="pyarrow",
            )
    return _split_table(df, ["x", "y", "z"], _PHONES_COLORS)


def _parse_higgs(path: Path):
    """HIGGS, UCI 280, as used by FSR and SamRuLe.

    Features: all 28 attributes (21 low level, 7 high level). Colors:
    `label` (1 for signal, 0 for background).
    """
    # Stream the 8 GB of CSV in batches: reading it at once with pandas needs
    # about 18 GB of memory.
    from pyarrow import csv, input_stream

    columns = [str(c) for c in range(29)]
    reader = csv.open_csv(
        input_stream(str(path), compression="gzip"),
        read_options=csv.ReadOptions(column_names=columns),
        convert_options=csv.ConvertOptions(
            column_types={c: "float32" for c in columns}
        ),
    )
    batches = [
        np.column_stack([col.to_numpy() for col in batch.columns])
        for batch in reader
    ]
    table = np.concatenate(batches)
    del batches
    codes, labels = pd.factorize(table[:, 0].astype(np.int64), sort=True)
    data = np.ascontiguousarray(table[:, 1:])
    return data, Colors(
        codes[:, np.newaxis].astype(np.int64),
        ("label",),
        (tuple(str(label) for label in labels),),
    )


def _last_7_columns(
    data: np.ndarray, colors: Colors | None
) -> tuple[np.ndarray, Colors | None]:
    """The 7 high level features of HIGGS (its last 7 columns), as used by
    streaming-fair and MACACO (see `DatasetInfo.postprocess`)."""
    return np.ascontiguousarray(data[:, -7:]), colors


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    url: str
    loader_function: Callable
    distance_type: str
    # Name of the parquet cache (`<cache_name>.parquet` in the dense
    # directory), for datasets parsed once and cached; None for the others.
    # For a cached dataset `loader_function(path)` is a pure parser returning
    # `(data, colors | None)`; for the others it returns
    # `(train, test, distances[, colors])`.
    cache_name: str | None = None
    # Applied to `(data, colors)` after reading the cache. Only for
    # `higgs-highlevel`, which shares the cache of `higgs`.
    postprocess: Callable | None = None


_DATASETS_INFO: dict[str, DatasetInfo] = {}


def register(info: DatasetInfo, force=False):
    if info.name in _DATASETS_INFO and not force:
        raise ValueError(f"dataset {info.name} already registered")
    if info.name in _ALIASES and not force:
        raise ValueError(f"{info.name} is already an alias of {_ALIASES[info.name]}")
    _DATASETS_INFO[info.name] = info


# --------------------------------------------------------------------------
# Aliases: short names for `mteb:<task>:<model>` names
# --------------------------------------------------------------------------

# alias -> canonical `mteb:<task>:<model>` name
_ALIASES: dict[str, str] = {}

# Short tags of the models, for the aliases `<task_slug>-<model_tag>`. Model
# names are too irregular to abbreviate automatically.
_MODEL_TAGS: dict[str, str] = {
    "Qwen/Qwen3-Embedding-4B": "qwen3-4b",
    "Qwen/Qwen3-Embedding-0.6B": "qwen3-0.6b",
    "sentence-transformers/all-MiniLM-L6-v2": "minilm-l6",
}

# Task slugs for which the CamelCase rule of `_task_slug` gives an ugly result.
_TASK_SLUGS: dict[str, str] = {
    "ArguAna": "arguana",
    "DBPedia": "dbpedia",
    "FiQA2018": "fiqa2018",
    "NFCorpus": "nfcorpus",
    "SciFact": "scifact",
}

# A word of a task name: an acronym (uppercase letters and digits not followed
# by a lowercase letter, e.g. `SCIDOCS`, `P2P`, the `NF` of `NFCorpus`) or a
# capitalized word (e.g. `Arxiv`, `Banking77`).
_WORD_RE = re.compile(r"[A-Z0-9]+(?![a-z])|[A-Z0-9]?[a-z0-9]+")


def _task_slug(task: str) -> str:
    """Slug of an MTEB task name: its CamelCase words and acronyms,
    lowercased and joined with `-` (`ArxivClusteringP2P` gives
    `arxiv-clustering-p2p`), unless `_TASK_SLUGS` overrides it."""
    if task in _TASK_SLUGS:
        return _TASK_SLUGS[task]
    return "-".join(w.lower() for w in _WORD_RE.findall(task))


def _model_tag(model: str) -> str:
    """Tag of `model` in `_MODEL_TAGS`; `KeyError` if it has none."""
    try:
        return _MODEL_TAGS[model]
    except KeyError:
        raise KeyError(
            f"model `{model}` has no tag: add one to `aida_data.dense._MODEL_TAGS`"
        ) from None


def _default_alias(task: str, model: str) -> str:
    return f"{_task_slug(task)}-{_model_tag(model)}"


def register_alias(alias: str, target: str, force: bool = False) -> None:
    """Make `alias` another name for the MTEB dataset `target`
    (`mteb:<task>:<model>`), in `load`, `load_qrels` and
    `available_datasets`. The alias shares the cache, the hosted files and
    the `encode_kwargs` of its target; nothing is stored under its name.

    Raises `ValueError` if `target` is not a valid `mteb:` name (so aliases
    of aliases are not allowed), if `alias` starts with `mteb:` or contains
    `:`, or if `alias` is already a registered dataset or an alias, unless
    `force=True`.
    """
    _mteb_names.parse_name(target)
    if alias.startswith(_mteb_names.PREFIX) or ":" in alias:
        raise ValueError(f"alias `{alias}` must not contain `:`")
    if not alias:
        raise ValueError("alias must not be empty")
    if not force:
        if alias in _DATASETS_INFO:
            raise ValueError(f"{alias} is already a registered dataset")
        if alias in _ALIASES:
            raise ValueError(f"{alias} is already an alias of {_ALIASES[alias]}")
    _ALIASES[alias] = target


def resolve_name(name: str) -> str:
    """The canonical name of `name`: the target of an alias, else `name`
    itself. E.g. `resolve_name("scidocs-qwen3-4b")` gives
    `"mteb:SCIDOCS:Qwen/Qwen3-Embedding-4B"`."""
    return _ALIASES.get(name, name)


register(
    DatasetInfo(
        "landmark-nomic-768-normalized",
        "https://huggingface.co/datasets/vector-index-bench/vibe/resolve/main/landmark-nomic-768-normalized.hdf5?download=true",
        _load_hdf5,
        "euclidean",
    )
)

register(
    DatasetInfo(
        "imagenet-clip-512-normalized",
        "https://huggingface.co/datasets/vector-index-bench/vibe/resolve/main/imagenet-clip-512-normalized.hdf5?download=true",
        _load_hdf5,
        "euclidean",
    )
)

register(
    DatasetInfo(
        "agnews-mxbai-1024-euclidean",
        "https://huggingface.co/datasets/vector-index-bench/vibe/resolve/main/agnews-mxbai-1024-euclidean.hdf5",
        _load_hdf5,
        "euclidean",
    )
)

register(
    DatasetInfo(
        "celeba-resnet-2048-cosine",
        "https://huggingface.co/datasets/vector-index-bench/vibe/resolve/main/celeba-resnet-2048-cosine.hdf5",
        _load_hdf5,
        "cosine",
    )
)

register(
    DatasetInfo(
        "simplewiki-openai-3072-normalized",
        "https://huggingface.co/datasets/vector-index-bench/vibe/resolve/main/simplewiki-openai-3072-normalized.hdf5?download=true",
        _load_hdf5,
        "euclidean",
    )
)

register(
    DatasetInfo(
        "deep-image-96-angular",
        "http://ann-benchmarks.com/deep-image-96-angular.hdf5",
        _load_hdf5,
        "angular",
    )
)

register(
    DatasetInfo(
        "mnist-784-euclidean",
        "http://ann-benchmarks.com/mnist-784-euclidean.hdf5",
        _load_hdf5,
        "euclidean",
    )
)

register(
    DatasetInfo(
        "fashion-mnist-784-euclidean",
        "http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5",
        _load_hdf5,
        "euclidean",
    )
)

register(
    DatasetInfo(
        "glove-100-angular",
        "http://ann-benchmarks.com/glove-100-angular.hdf5",
        _load_hdf5,
        "angular",
    )
)

register(
    DatasetInfo(
        "gist-960-euclidean",
        "http://ann-benchmarks.com/gist-960-euclidean.hdf5",
        _load_hdf5,
        "euclidean",
    )
)

register(
    DatasetInfo(
        "nytimes-256-angular",
        "http://ann-benchmarks.com/nytimes-256-angular.hdf5",
        _load_hdf5,
        "angular",
    )
)

register(
    DatasetInfo(
        "sift-128-euclidean",
        "http://ann-benchmarks.com/sift-128-euclidean.hdf5",
        _load_hdf5,
        "euclidean",
    )
)

register(
    DatasetInfo(
        "pamap2",
        "http://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip",
        _parse_pamap,
        "euclidean",
        _PAMAP_CACHE,
    )
)

register(
    DatasetInfo(
        "census",
        "https://github.com/Minqi824/ADBench/raw/main/adbench/datasets/Classical/9_census.npz",
        _load_census,
        "euclidean",
    )
)

register(
    DatasetInfo(
        "ht",
        "https://archive.ics.uci.edu/static/public/362/gas+sensors+for+home+activity+monitoring.zip",
        _load_ht,
        "euclidean",
    )
)

register(
    DatasetInfo(
        "yandex-t2i",
        "https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/base.1M.fbin",
        _load_fbin,
        "euclidean",
    )
)

register(
    DatasetInfo(
        "chem",
        "https://archive.ics.uci.edu/static/public/322/gas+sensor+array+under+dynamic+gas+mixtures.zip",
        _load_chem,
        "euclidean",
    )
)

register(
    DatasetInfo(
        "adult",
        "https://archive.ics.uci.edu/static/public/2/adult.zip",
        _load_adult,
        "euclidean",
    )
)

register(
    DatasetInfo(
        "athlete",
        "https://github.com/rgriff23/Olympic_history/raw/master/data/athlete_events.csv",
        _load_athlete,
        "euclidean",
    )
)

register(
    DatasetInfo(
        "diabetes",
        "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip",
        _load_diabetes,
        "euclidean",
    )
)

register(
    DatasetInfo(
        "creditcard",
        "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip",
        _load_creditcard,
        "euclidean",
    )
)

_KFC_BASE = "https://github.com/FaroukY/KFC-ScalableFairClustering/raw/main/data/"

for _name, _file, _features, _colors in [
    ("4area", "4area.csv", [str(c) for c in range(1, 9)], ["color"]),
    ("reuter_50_50", "c50.csv", [str(c) for c in range(10)], ["color"]),
    ("victorian", "victorian.csv", [str(c) for c in range(10)], ["color"]),
    (
        "bank",
        "bank_categorized.csv",
        [
            "age",
            "balance",
            "duration",
            "job",
            "education",
            "default",
            "housing",
            "loan",
            "contact",
        ],
        ["marital"],
    ),
]:
    register(
        DatasetInfo(
            _name,
            _KFC_BASE + _file,
            partial(_load_kfc, features=_features, colors=_colors),
            "euclidean",
        )
    )

register(
    DatasetInfo(
        "breast",
        "https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip",
        _load_breast,
        "euclidean",
    )
)

register(
    DatasetInfo(
        "wine",
        "https://archive.ics.uci.edu/static/public/186/wine+quality.zip",
        _load_wine,
        "euclidean",
    )
)

register(
    DatasetInfo(
        "shuttle",
        "https://archive.ics.uci.edu/static/public/148/statlog+shuttle.zip",
        _load_shuttle,
        "euclidean",
    )
)

register(
    DatasetInfo(
        "rt-iot2022",
        "https://archive.ics.uci.edu/static/public/942/rt-iot2022.zip",
        _load_rt_iot,
        "euclidean",
    )
)

register(
    DatasetInfo(
        "biokdd",
        "https://kdd.org/cupfiles/KDDCupData/2004/data_kddcup04.tar.gz",
        _parse_biokdd,
        "euclidean",
        _BIOKDD_CACHE,
    )
)

register(
    DatasetInfo(
        "metropt3",
        "https://archive.ics.uci.edu/static/public/791/metropt+3+dataset.zip",
        _parse_metropt3,
        "euclidean",
        _METROPT3_CACHE,
    )
)

register(
    DatasetInfo(
        "household-power",
        "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip",
        _parse_household_power,
        "euclidean",
        _HOUSEHOLD_POWER_CACHE,
    )
)

register(
    DatasetInfo(
        "covertype",
        "https://archive.ics.uci.edu/static/public/31/covertype.zip",
        _parse_covertype,
        "euclidean",
        _COVERTYPE_CACHE,
    )
)

register(
    DatasetInfo(
        "census1990",
        "https://archive.ics.uci.edu/static/public/116/us+census+data+1990.zip",
        _parse_census1990,
        "euclidean",
        _CENSUS1990_CACHE,
    )
)

register(
    DatasetInfo(
        "phones",
        "https://archive.ics.uci.edu/static/public/344/heterogeneity+activity+recognition.zip",
        _parse_phones,
        "euclidean",
        _PHONES_CACHE,
    )
)

register(
    DatasetInfo(
        "higgs",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz",
        _parse_higgs,
        "euclidean",
        _HIGGS_CACHE,
    )
)

register(
    DatasetInfo(
        "higgs-highlevel",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz",
        _parse_higgs,
        "euclidean",
        _HIGGS_CACHE,
        postprocess=_last_7_columns,
    )
)

# Not a real download URL: this dataset is synthetic (see
# _load_densired_hard).
register(
    DatasetInfo(
        "densired-hard", "file:///densired-hard.npz", _load_densired_hard, "euclidean"
    )
)


def available_datasets():
    """Names accepted by `load`: the registered datasets, then the aliases of
    MTEB datasets (see `register_alias`), each group sorted. Full
    `mteb:<task>:<model>` names are accepted too but not listed."""
    return sorted(_DATASETS_INFO) + sorted(_ALIASES)


def _safe_l2_normalize_rows(data: np.ndarray, label: str) -> np.ndarray:
    """Normalize rows with L2 norm, keeping zero/invalid rows at zero."""
    out = np.array(data, dtype=np.float32, copy=True)
    norms = np.linalg.norm(out, axis=1, keepdims=True)

    # Rows with non-finite or non-positive norm cannot be normalized safely.
    invalid_rows = (~np.isfinite(norms[:, 0])) | (norms[:, 0] <= 0.0)
    invalid_count = int(np.count_nonzero(invalid_rows))
    if invalid_count > 0:
        _LOGGER.warning(
            "safe normalization for %s: %d rows had non-finite/non-positive norm and were set to zero",
            label,
            invalid_count,
        )

    np.divide(out, norms, out=out, where=(~invalid_rows)[:, np.newaxis])
    out[invalid_rows] = 0.0
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


class SafeL2Normalizer(TransformerMixin, BaseEstimator):
    """Stateless transformer scaling each row to unit L2 norm.

    Unlike `sklearn.preprocessing.Normalizer`, rows with zero or non-finite
    norm are set to zero (with a logged warning), and any remaining NaN/inf
    values are replaced by zero. The output is always float32.

    `label` is only used in the warning message.
    """

    def __init__(self, label: str = "data"):
        self.label = label

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return _safe_l2_normalize_rows(np.asarray(X), self.label)

    def __sklearn_is_fitted__(self):
        return True


def _array_health_stats(data: np.ndarray) -> tuple[int, int]:
    norms = np.linalg.norm(data, axis=1)
    zero_norm_rows = int(np.count_nonzero(norms <= 0.0))
    non_finite_values = int(np.count_nonzero(~np.isfinite(data)))
    return zero_norm_rows, non_finite_values


def _raw_name(info: DatasetInfo) -> str:
    return Path(urlparse(info.url).path).name


def _dense_dir() -> Path:
    return DATASETS_DIR / "dense"


def local_path(name: str):
    return _dense_dir() / _raw_name(_DATASETS_INFO[name])


def _ensure_cache(name: str) -> Path:
    """Make sure dataset `name` is available locally and return the file to
    read: the raw download for datasets with no `cache_name` (HDF5 and the
    ones not cached yet), else the parquet cache, made if missing.

    Making the cache downloads the raw file if needed, parses it with the
    loader, writes the cache and deletes the raw file (see `_write_cache`).
    The cache is not read back, so that exporting a large dataset does not
    hold two copies of it in memory.
    """
    info = _DATASETS_INFO[name]
    _dense_dir().mkdir(parents=True, exist_ok=True)
    raw = local_path(name)
    if info.cache_name is None:
        _download(info.url, raw)
        return raw
    # A cached dataset does not need its raw file, which may have been
    # deleted (see `_write_cache` and `prune_raw`).
    if not _cache_exists(name):
        _download(info.url, raw)
    return _write_cache(raw, info.cache_name, info.loader_function)


def _cache_exists(name: str) -> bool:
    """Whether dataset `name` is cached (parquet, or legacy HDF5 that
    `_ensure_cache` will convert), so that its raw file is not needed."""
    cache_name = _DATASETS_INFO[name].cache_name
    if cache_name is None:
        return False
    return any(
        (_dense_dir() / f"{cache_name}{suffix}").is_file()
        for suffix in (".parquet", ".hdf5")
    )


def _raw_is_shared(path: Path, cache_name: str | None) -> bool:
    """Whether a registered dataset with a cache other than `cache_name`, or
    with no cache, downloads to the same file name as `path`."""
    return any(
        _raw_name(info) == path.name and info.cache_name != cache_name
        for info in _DATASETS_INFO.values()
    )


def prune_raw(dry_run: bool = True) -> list[Path]:
    """Raw downloads of cached datasets that are no longer needed, because
    their cache exists and no other dataset uses them. With
    `dry_run=False` they are deleted. Also lists the
    `*.part` and `*.parquet.tmp` files left by downloads and cache
    writes that were killed, once they are more than 24 hours old.
    """
    paths = []
    for name, info in _DATASETS_INFO.items():
        path = local_path(name)
        if (
            info.cache_name is not None
            and _cache_exists(name)
            and path.is_file()
            and path not in paths
            and not _raw_is_shared(path, info.cache_name)
        ):
            paths.append(path)
    paths += stale_temporaries(_dense_dir())
    # MTEB caches live in `dense/mteb/<task>/<model>/`; they have no raw files.
    paths += stale_temporaries(_dense_dir() / "mteb", recursive=True)
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


@dataclass(frozen=True)
class Dataset:
    distance: str
    dataset: np.ndarray
    queries: np.ndarray | None = None
    distances: np.ndarray | None = None
    colors: Colors | None = None  # aligned with the rows of `dataset`
    ids: np.ndarray | None = None  # aligned with the rows of `dataset`
    query_ids: np.ndarray | None = None  # aligned with the rows of `queries`


def load(
    name: str,
    pipeline: TransformerMixin | None = None,
    deduplicate: bool = True,
    **encode_kwargs,
) -> Dataset:
    """Load dataset `name`, optionally transforming it with `pipeline`.

    `name` is a registered dataset (see `available_datasets`) or an MTEB
    embedding, `mteb:<task>:<model>` (e.g.
    `mteb:SCIDOCS:sentence-transformers/all-MiniLM-L6-v2`), for Retrieval and
    Clustering tasks, or an alias of one (e.g. `scidocs-qwen3-4b`, see
    `register_alias` and `resolve_name`). The cache in `<root>/dense/mteb` is used if present,
    else the pair is downloaded from the hosted registry, else it is encoded
    locally (needs the `mteb` extra) and cached. Retrieval: the corpus is
    `dataset` (with `ids`), the queries are `queries` (with `query_ids`); the
    relevance judgments are in `load_qrels`. Clustering: the unique texts are
    `dataset`, with one color per clustering set (absent texts are labeled
    `"<absent>"`). The distance is the one of the model.

    `encode_kwargs` (`batch_size`, `model_kwargs`, `limit`) are forwarded to
    the local encoder, for `mteb:` names only (any for other names raises
    `TypeError`). They are ignored, with a warning, if the cache or the hosted
    copy is used. With `limit` (for testing) only the first rows are encoded
    and nothing is cached.

    Processing order:

    1. the raw data is downloaded (if needed) and loaded;
    2. if `deduplicate=True` duplicate rows of the train set are dropped,
       keeping the first occurrence and the original row order. Rows containing
       NaN/inf are dropped from both train and test sets;
    3. if `pipeline` is not None, it is fitted on the train set
       (`pipeline.fit_transform(train)`) and then applied to the test set
       (`pipeline.transform(test)`);
    4. for `angular`/`cosine`/`normalized` distances all-zero rows are
       dropped from train and test, and duplicate rows of train (possibly
       introduced by the pipeline) are dropped again.

    For datasets with categorical attributes, `Dataset.colors` holds them
    (see `Colors`) and is kept aligned with `Dataset.dataset` through all
    the steps above. The pipeline only sees the features. Two rows are
    duplicates only if both their features and their colors are equal, so
    points with equal features but different colors are all kept.

    `Dataset.ids` and `Dataset.query_ids`, when the dataset has them, are
    kept aligned with `dataset` and `queries` through all the steps above.
    Deduplication ignores the ids: of several rows with equal features and
    colors only the first one is kept, with its id, even if the ids differ.
    With `deduplicate=False` all of them are kept.

    `pipeline` can be any object exposing `fit_transform`/`transform`,
    typically an `sklearn.pipeline.Pipeline` or a single transformer. It is
    fitted *in place*: after `load` returns its fitted state can be
    inspected or reused to transform new points. Pass
    `sklearn.base.clone(pipeline)` to keep the original object untouched.
    The pipeline must not change the number of rows, otherwise a
    `ValueError` is raised. Its output is converted with `np.asarray`, so
    pipelines configured with `set_output(transform="pandas")` are
    accepted; the dtype is whatever the pipeline produces (`StandardScaler`,
    `PCA` and `SafeL2Normalizer` preserve float32).

    Returns an instance of Dataset

    Example::

        from sklearn.decomposition import PCA
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        pipeline = make_pipeline(
            StandardScaler(with_std=False), PCA(64), SafeL2Normalizer()
        )
        ds = load("glove-100-angular", pipeline)
    """
    name = resolve_name(name)
    if name.startswith("mteb:"):
        result = _mteb_result(name, encode_kwargs)
        return _process(
            result.meta["similarity"],
            result.corpus,
            result.ids,
            result.colors,
            result.queries,
            result.query_ids,
            None,
            pipeline,
            deduplicate,
        )
    if encode_kwargs:
        raise TypeError(
            f"unexpected arguments {sorted(encode_kwargs)}: they are only "
            "accepted for `mteb:` names"
        )
    if name not in _DATASETS_INFO:
        raise KeyError(
            f"Dataset `{name}` not available. Pick one of {available_datasets()}"
        )
    info = _DATASETS_INFO[name]
    distance = info.distance_type
    path = _ensure_cache(name)
    colors: Colors | None
    if info.cache_name is not None:
        train, colors = _read_parquet_cache(path)
        if info.postprocess is not None:
            train, colors = info.postprocess(train, colors)
        test = distances = None
    else:
        train, test, distances, *rest = info.loader_function(path)
        colors = rest[0] if rest else None
    # Only some datasets (e.g. MTEB ones) have ids for their rows.
    ids: np.ndarray | None = None
    query_ids: np.ndarray | None = None
    if colors is not None and colors.values.shape[0] != train.shape[0]:
        raise ValueError(
            f"{name}: {colors.values.shape[0]} color rows for {train.shape[0]} points"
        )
    return _process(
        distance, train, ids, colors, test, query_ids, distances, pipeline, deduplicate
    )


def _mteb_result(name: str, encode_kwargs: dict):
    """The `_mteb.Encoded` of the MTEB name `name`: read from the local
    cache, else from the hosted registry, else encoded locally. With
    `limit` in `encode_kwargs` nothing is cached and the encoded result is
    returned directly."""
    from . import _mteb  # lazy: `_mteb` imports this module

    task, model = _mteb.parse_name(name)
    directory = _mteb.cache_dir(task, model)
    if not (directory / _mteb.CORPUS_FILE).exists():
        if _mteb.fetch_hosted(task, model, directory):
            _warn_unused(name, encode_kwargs, "the hosted copy")
        else:
            result = _mteb.encode_task(task, model, **encode_kwargs)
            _mteb.check_revision(result.meta, model)
            return result
    else:
        _warn_unused(name, encode_kwargs, "the local cache")
    result = _read_mteb_cache(directory)
    _mteb.check_revision(result.meta, model)
    return result


def _warn_unused(name: str, encode_kwargs: dict, source: str):
    if encode_kwargs:
        _LOGGER.warning(
            "%s: encode arguments %s ignored, using %s",
            name,
            sorted(encode_kwargs),
            source,
        )


def _read_mteb_cache(directory: Path):
    from . import _mteb

    train, colors, ids, meta = _read_parquet_cache_full(directory / _mteb.CORPUS_FILE)
    test = query_ids = qrels = None
    if (directory / _mteb.QUERIES_FILE).exists():
        test, _, query_ids, _ = _read_parquet_cache_full(directory / _mteb.QUERIES_FILE)
    if (directory / _mteb.QRELS_FILE).exists():
        qrels, _ = read_table(directory / _mteb.QRELS_FILE, _mteb._QRELS_VERSION)
    return _mteb.Encoded(train, ids, colors, test, query_ids, qrels, meta)


def load_qrels(name: str, **encode_kwargs) -> dict[str, dict[str, int]]:
    """Relevance judgments `{query_id: {doc_id: score}}` (positive scores
    only, original MTEB ids) of the MTEB retrieval dataset `name`
    (`mteb:<task>:<model>`, or an alias of one). Makes sure the cache exists
    like `load`, with the same `encode_kwargs`. Raises `ValueError` for other
    names and for clustering tasks."""
    name = resolve_name(name)
    if not name.startswith("mteb:"):
        raise ValueError(f"`{name}` is not an MTEB name, expected `mteb:<task>:<model>`")
    result = _mteb_result(name, encode_kwargs)
    if result.qrels is None:
        raise ValueError(f"`{name}` is not a retrieval task, it has no qrels")
    out: dict[str, dict[str, int]] = {}
    for q, d, score in zip(
        result.qrels.column("query_id").to_pylist(),
        result.qrels.column("doc_id").to_pylist(),
        result.qrels.column("score").to_pylist(),
    ):
        out.setdefault(q, {})[d] = score
    return out


def _process(
    distance: str,
    train: np.ndarray,
    ids: np.ndarray | None,
    colors: Colors | None,
    test: np.ndarray | None,
    query_ids: np.ndarray | None,
    distances: np.ndarray | None,
    pipeline: TransformerMixin | None,
    deduplicate: bool,
) -> Dataset:
    """The filtering and transformation steps of `load`, keeping `ids`
    aligned with `train` and `colors`, and `query_ids` with `test`."""
    orig_n_train = train.shape[0]
    orig_n_test = test.shape[0] if test is not None else 0
    if deduplicate:
        # Remove duplicate rows (if any) from train set
        train, ids, colors = _take(
            _unique_row_indices(train, colors), train, ids, colors
        )
    # Remove completely NaN and infinite values from train and test sets, don't substitute with numbers
    train, ids, colors = _take(np.isfinite(train).all(axis=1), train, ids, colors)
    if test is not None:
        test, query_ids, _ = _take(np.isfinite(test).all(axis=1), test, query_ids)

    if pipeline is not None:
        train = _apply_transform(pipeline.fit_transform, train, "train")
        if test is not None:
            test = _apply_transform(pipeline.transform, test, "test")

    if distance in ("angular", "cosine", "normalized"):
        # remove 0-rows
        train, ids, colors = _take(~((train == 0).all(axis=1)), train, ids, colors)
        if test is not None:
            test, query_ids, _ = _take(~((test == 0).all(axis=1)), test, query_ids)

    if deduplicate:
        # Remove duplicate rows that may have been (re)introduced by the pipeline
        # (e.g. PCA collapsing points, or normalization mapping collinear vectors
        # onto each other), regardless of the return path.
        train, ids, colors = _take(
            _unique_row_indices(train, colors), train, ids, colors
        )

    # We return the distances only if the data has not been preprocessed
    # and no rows have been dropped.
    # If that's the case, then the distances are meaningless.
    if (
        test is None
        or pipeline is not None
        or train.shape[0] != orig_n_train
        or test.shape[0] != orig_n_test
    ):
        distances = None
    return Dataset(distance, train, test, distances, colors, ids, query_ids)


def _take(
    rows: np.ndarray,
    data: np.ndarray,
    ids: np.ndarray | None = None,
    colors: Colors | None = None,
) -> tuple[np.ndarray, np.ndarray | None, Colors | None]:
    """Select `rows` (indices or boolean mask) of `data`, `ids` and `colors`
    together; `ids` and `colors` may be None."""
    return (
        data[rows],
        ids[rows] if ids is not None else None,
        colors.take(rows) if colors is not None else None,
    )


def _unique_row_indices(data: np.ndarray, colors: Colors | None = None) -> np.ndarray:
    """Sorted indices of the first occurrence of each distinct row of `data`
    (of `data` and `colors` together, if `colors` is given)."""
    # Hashing the columns with pandas is several times faster than
    # `np.unique(axis=0)`, which sorts the rows lexicographically (28s vs 5s
    # on census1990). Like `np.unique`, it considers -0.0 equal to 0.0.
    df = pd.DataFrame(data, copy=False)
    if colors is not None:
        for j in range(colors.values.shape[1]):
            df[f"color{j}"] = colors.values[:, j]
    return np.flatnonzero(~df.duplicated().to_numpy())


def _apply_transform(transform: Callable, data: np.ndarray, label: str) -> np.ndarray:
    out = np.asarray(transform(data))
    if out.shape[0] != data.shape[0]:
        raise ValueError(
            f"pipeline changed the number of {label} rows "
            f"({data.shape[0]} -> {out.shape[0]}); transformers must not drop rows"
        )
    return out


# Built-in aliases of the hosted MTEB pairs. Offline: only the static list is
# read.
for _task, _model in _mteb_names._HOSTED:
    register_alias(_default_alias(_task, _model), f"mteb:{_task}:{_model}")
del _task, _model
