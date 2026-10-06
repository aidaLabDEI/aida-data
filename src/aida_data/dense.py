"""\
Dense datasets, under different distance measures.
Datasets are collected from different sources
"""

import logging
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

import h5py
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

DATASETS_DIR = Path(os.environ.get("AIDA_DATA_DIR", "datasets"))

_LOGGER = logging.getLogger("aida_data.dense")


def _download(url, destination: Path):
    import requests
    from tqdm import tqdm

    if urlparse(url).scheme not in ("http", "https"):
        # Synthetic/local datasets (e.g. densired-hard) use a
        # non-fetchable placeholder URL and are generated on demand by
        # their own loader instead of being downloaded.
        return
    if not destination.is_file():
        _LOGGER.info(f"downloading {url} to {destination}")
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            total = int(response.headers.get("Content-Length", 0))
            with (
                open(destination, "wb") as out_file,
                tqdm(
                    total=total or None,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=destination.name,
                ) as progress,
            ):
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    out_file.write(chunk)
                    progress.update(len(chunk))


def _load_hdf5(path: Path):
    with h5py.File(path) as hfp:
        return hfp["train"][:], hfp["test"][:], hfp["distances"][:]


def _load_pamap(path: Path):
    def load_zipfile():
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
            return X.astype(np.float32)

    h5path = path.parent / "pamap.hdf5"
    if not h5path.is_file():
        data = load_zipfile()
        with h5py.File(h5path, "w") as hfp:
            hfp["X"] = data

    with h5py.File(h5path) as hfp:
        data = hfp["X"][:]

    return data, None, None


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


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    url: str
    loader_function: Callable
    distance_type: str


_DATASETS_INFO: dict[str, DatasetInfo] = {}


def register(info: DatasetInfo, force=False):
    if info.name in _DATASETS_INFO and not force:
        raise ValueError(f"dataset {info.name} already registered")
    _DATASETS_INFO[info.name] = info


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
        _load_pamap,
        "euclidean",
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

# Not a real download URL: this dataset is synthetic (see
# _load_densired_hard).
register(
    DatasetInfo(
        "densired-hard", "file:///densired-hard.npz", _load_densired_hard, "euclidean"
    )
)


def available_datasets():
    return list(_DATASETS_INFO.keys())


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


def local_path(name: str):
    url = _DATASETS_INFO[name].url
    return DATASETS_DIR / Path(urlparse(url).path).name


def load(
    name: str,
    pipeline: TransformerMixin | None = None,
    load_queries: bool = False,
    deduplicate: bool = True,
):
    """Load dataset `name`, optionally transforming it with `pipeline`.

    Processing order:

    1. the raw data is downloaded (if needed) and loaded;
    2. if `deduplicate=True` duplicate rows of the train set are dropped.
       Rows containing
       NaN/inf are dropped from both train and test sets;
    3. if `pipeline` is not None, it is fitted on the train set
       (`pipeline.fit_transform(train)`) and then applied to the test set
       (`pipeline.transform(test)`);
    4. for `angular`/`cosine`/`normalized` distances all-zero rows are
       dropped from train and test, and duplicate rows of train (possibly
       introduced by the pipeline) are dropped again.

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

    Returns `(distance, train)`, or `(distance, train, test, distances)` if
    `load_queries` is True (`test` and `distances` may be None).

    Example::

        from sklearn.decomposition import PCA
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        pipeline = make_pipeline(
            StandardScaler(with_std=False), PCA(64), SafeL2Normalizer()
        )
        distance, train = load("glove-100-angular", pipeline)
    """
    if name not in available_datasets():
        raise KeyError(
            f"Dataset `{name}` not available. Pick one of {available_datasets()}"
        )
    if not DATASETS_DIR.is_dir():
        DATASETS_DIR.mkdir()

    info = _DATASETS_INFO[name]
    url, loader, distance = info.url, info.loader_function, info.distance_type
    local_name = local_path(name)
    _download(url, local_name)
    train, test, distances = loader(local_name)
    orig_n_train = train.shape[0]
    orig_n_test = test.shape[0]
    if deduplicate:
        # Remove duplicate rows (if any) from train set
        train = np.unique(train, axis=0)
    # Remove completely NaN and infinite values from train and test sets, don't substitute with numbers
    train = train[~np.isnan(train).any(axis=1) & ~np.isinf(train).any(axis=1)]
    if test is not None:
        test = test[~np.isnan(test).any(axis=1) & ~np.isinf(test).any(axis=1)]

    if pipeline is not None:
        train = _apply_transform(pipeline.fit_transform, train, "train")
        if test is not None:
            test = _apply_transform(pipeline.transform, test, "test")

    if distance in ("angular", "cosine", "normalized"):
        # remove 0-rows
        train = train[~((train == 0).all(axis=1))]
        if test is not None:
            test = test[~((test == 0).all(axis=1))]

    if deduplicate:
        # Remove duplicate rows that may have been (re)introduced by the pipeline
        # (e.g. PCA collapsing points, or normalization mapping collinear vectors
        # onto each other), regardless of the return path.
        train = np.unique(train, axis=0)

    if load_queries and test is not None:
        # We return the distances only if the data has not been preprocessed
        # and no rows have been dropped.
        # If that's the case, then the distances are meaningless.
        if (
            pipeline is None
            and train.shape[0] == orig_n_train
            and test.shape[0] == orig_n_test
        ):
            return distance, train, test, distances
        else:
            return distance, train, test
    else:
        return distance, train


def _apply_transform(transform: Callable, data: np.ndarray, label: str) -> np.ndarray:
    out = np.asarray(transform(data))
    if out.shape[0] != data.shape[0]:
        raise ValueError(
            f"pipeline changed the number of {label} rows "
            f"({data.shape[0]} -> {out.shape[0]}); transformers must not drop rows"
        )
    return out
