import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from aida_data import dense
from aida_data.dense import DatasetInfo, SafeL2Normalizer, load, register

RNG = np.random.default_rng(42)
TRAIN = RNG.normal(loc=3.0, scale=2.0, size=(200, 8)).astype(np.float32)
TEST = RNG.normal(loc=3.0, scale=2.0, size=(20, 8)).astype(np.float32)
DISTANCES = RNG.random((20, 5)).astype(np.float32)


def _loader(path):
    return TRAIN.copy(), TEST.copy(), DISTANCES.copy()


def _loader_with_zeros(path):
    train = TRAIN.copy()
    train[:3] = 0.0
    test = TEST.copy()
    test[0] = 0.0
    return train, test, DISTANCES.copy()


@pytest.fixture(autouse=True)
def tiny_datasets(tmp_path, monkeypatch):
    monkeypatch.setattr(dense, "DATASETS_DIR", tmp_path)
    register(DatasetInfo("tiny", "file:///tiny.npz", _loader, "euclidean"), force=True)
    register(
        DatasetInfo("tiny-angular", "file:///tiny-angular.npz", _loader_with_zeros, "angular"),
        force=True,
    )


def test_no_pipeline_returns_untransformed_data():
    distance, train, test, distances = load("tiny", load_queries=True)
    assert distance == "euclidean"
    np.testing.assert_array_equal(train, np.unique(TRAIN, axis=0))
    np.testing.assert_array_equal(test, TEST)
    np.testing.assert_array_equal(distances, DISTANCES)


def test_return_shape_without_queries():
    result = load("tiny", StandardScaler())
    assert len(result) == 2


def test_standard_scaler_uses_train_statistics():
    scaler = StandardScaler()
    _, train, test = load("tiny", scaler, load_queries=True)
    np.testing.assert_allclose(train.mean(axis=0), 0.0, atol=1e-5)
    np.testing.assert_allclose(train.std(axis=0), 1.0, atol=1e-4)
    expected = (TEST - TRAIN.mean(axis=0)) / TRAIN.std(axis=0)
    np.testing.assert_allclose(test, expected, rtol=1e-4, atol=1e-4)


def test_pca_projects_test_with_train_fitted_pca():
    pca = PCA(n_components=3)
    _, train, test = load("tiny", pca, load_queries=True)
    assert train.shape[1] == 3
    assert test.shape[1] == 3
    np.testing.assert_allclose(test, pca.transform(TEST), rtol=1e-5, atol=1e-5)


def test_safe_l2_normalizer_and_zero_rows_removed_for_angular():
    _, train, test = load("tiny-angular", SafeL2Normalizer(), load_queries=True)
    np.testing.assert_allclose(np.linalg.norm(train, axis=1), 1.0, rtol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(test, axis=1), 1.0, rtol=1e-5)
    # the three zeroed train rows are removed
    assert train.shape[0] == TRAIN.shape[0] - 3
    assert test.shape[0] == TEST.shape[0] - 1


def test_pipeline_is_fitted_in_place():
    pipeline = make_pipeline(StandardScaler(with_std=False), PCA(4), SafeL2Normalizer())
    load("tiny", pipeline)
    check_is_fitted(pipeline)
    assert pipeline.named_steps["pca"].explained_variance_ratio_.shape == (4,)


class _DropFirstRow(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[1:]


def test_row_dropping_transformer_raises():
    with pytest.raises(ValueError, match="number of train rows"):
        load("tiny", _DropFirstRow())


def test_pandas_output_is_converted_to_array():
    pipeline = make_pipeline(StandardScaler(), PCA(2)).set_output(transform="pandas")
    assert isinstance(pipeline.fit_transform(TRAIN), pd.DataFrame)
    _, train, test = load("tiny", pipeline, load_queries=True)
    assert isinstance(train, np.ndarray)
    assert isinstance(test, np.ndarray)
