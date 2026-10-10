import zipfile

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from aida_data import dense
from aida_data.dense import Colors, DatasetInfo, SafeL2Normalizer, load, register

# Row 0 and 1: same features, same colors (exact duplicate, row 1 dropped).
# Row 2: same features as row 0, different colors (kept).
# Row 4: NaN (dropped). Row 6: inf (dropped).
FEATURES = np.array(
    [
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 2.0],
        [3.0, 4.0],
        [np.nan, 1.0],
        [5.0, 7.0],
        [np.inf, 0.0],
        [-2.0, 0.5],
    ],
    dtype=np.float32,
)
COLOR_VALUES = np.array(
    [[0, 1], [0, 1], [1, 1], [1, 0], [0, 0], [2, 0], [1, 1], [2, 1]], dtype=np.int64
)
COLORS = Colors(
    COLOR_VALUES, ("sex", "race"), (("F", "M", "X"), ("black", "white"))
)
EXPECTED_ROWS = [0, 2, 3, 5, 7]


def _colored_loader(path):
    return FEATURES.copy(), None, None, COLORS


def _colored_angular_loader(path):
    features = FEATURES.copy()
    features[3] = 0.0
    return features, None, None, COLORS


@pytest.fixture(autouse=True)
def tiny_datasets(tmp_path, monkeypatch):
    monkeypatch.setattr(dense, "DATASETS_DIR", tmp_path)
    register(
        DatasetInfo("tiny-colored", "file:///c.npz", _colored_loader, "euclidean"),
        force=True,
    )
    register(
        DatasetInfo(
            "tiny-colored-angular", "file:///ca.npz", _colored_angular_loader, "angular"
        ),
        force=True,
    )


def _assert_aligned(ds, rows):
    """Each surviving row still carries the colors of its original row."""
    assert ds.colors.values.shape == (len(rows), 2)
    np.testing.assert_array_equal(ds.colors.values, COLOR_VALUES[rows])


def test_dedup_keeps_points_with_different_colors_and_drops_nan():
    ds = load("tiny-colored")
    np.testing.assert_array_equal(ds.dataset, FEATURES[EXPECTED_ROWS])
    _assert_aligned(ds, EXPECTED_ROWS)
    assert ds.colors.names == COLORS.names
    assert ds.colors.labels == COLORS.labels
    assert ds.queries is None and ds.distances is None


def test_no_deduplicate_keeps_duplicates_but_drops_nan():
    ds = load("tiny-colored", deduplicate=False)
    rows = [0, 1, 2, 3, 5, 7]
    np.testing.assert_array_equal(ds.dataset, FEATURES[rows])
    _assert_aligned(ds, rows)


def test_pipeline_leaves_colors_untouched():
    ds = load("tiny-colored", make_pipeline(StandardScaler(), PCA(2)))
    assert ds.dataset.shape == (len(EXPECTED_ROWS), 2)
    _assert_aligned(ds, EXPECTED_ROWS)


def test_zero_rows_dropped_with_their_colors():
    ds = load("tiny-colored-angular", SafeL2Normalizer())
    rows = [0, 2, 5, 7]
    np.testing.assert_allclose(np.linalg.norm(ds.dataset, axis=1), 1.0, rtol=1e-5)
    _assert_aligned(ds, rows)


def test_second_dedup_keeps_colors_aligned():
    # Normalization maps (1, 2) and (2, 4) onto the same point: with equal
    # colors the second is dropped, with different colors both are kept.
    features = np.array([[1, 2], [2, 4], [3, 6], [1, 0]], dtype=np.float32)
    values = np.array([[0], [0], [1], [1]])
    colors = Colors(values, ("c",), (("a", "b"),))
    register(
        DatasetInfo(
            "tiny-collinear",
            "file:///col.npz",
            lambda path: (features, None, None, colors),
            "angular",
        ),
        force=True,
    )
    ds = load("tiny-collinear", SafeL2Normalizer())
    assert ds.dataset.shape[0] == 3
    np.testing.assert_array_equal(ds.colors.values[:, 0], [0, 1, 1])
    np.testing.assert_allclose(ds.dataset[2], [1.0, 0.0])


def test_uncolored_dataset_has_no_colors():
    register(
        DatasetInfo(
            "tiny-plain",
            "file:///p.npz",
            lambda path: (FEATURES[[0, 3]].copy(), None, None),
            "euclidean",
        ),
        force=True,
    )
    assert load("tiny-plain").colors is None


def test_mismatched_color_rows_raise():
    register(
        DatasetInfo(
            "tiny-bad",
            "file:///b.npz",
            lambda path: (FEATURES[:3].copy(), None, None, COLORS),
            "euclidean",
        ),
        force=True,
    )
    with pytest.raises(ValueError, match="color rows"):
        load("tiny-bad")


def test_colors_accessors():
    assert COLORS.column("sex").tolist() == COLOR_VALUES[:, 0].tolist()
    assert COLORS.n_colors("sex") == 3
    assert COLORS.n_colors("race") == 2
    # decoding
    j = COLORS.names.index("race")
    assert [COLORS.labels[j][c] for c in COLORS.column("race")[:4]] == [
        "white",
        "white",
        "white",
        "black",
    ]
    # n_colors counts only the codes present
    assert COLORS.take([0, 1, 3]).n_colors("sex") == 2
    with pytest.raises(KeyError, match="no color"):
        COLORS.column("age")


def test_colors_shape_is_validated():
    with pytest.raises(ValueError, match="shape"):
        Colors(np.zeros((3, 1), dtype=np.int64), ("a", "b"), (("x",), ("y",)))
    with pytest.raises(ValueError, match="same length"):
        Colors(np.zeros((3, 1), dtype=np.int64), ("a",), ())


def test_split_table_drops_nulls_and_sorts_codes_by_label():
    df = pd.DataFrame(
        {
            "x": [1.0, 2.0, None, 4.0, 5.0],
            "y": [0.5, 1.5, 2.5, 3.5, 4.5],
            "sex": ["M", "F", "F", None, "M"],
            "grade": [10, 2, 3, 3, 2],
            "unused": [None] * 5,
        }
    )
    data, colors = dense._split_table(df, ["x", "y"], ["sex", "grade"])
    assert data.dtype == np.float32
    np.testing.assert_array_equal(data, [[1.0, 0.5], [2.0, 1.5], [5.0, 4.5]])
    assert colors.names == ("sex", "grade")
    # numeric categories are sorted numerically, not as strings
    assert colors.labels == (("F", "M"), ("2", "10"))
    np.testing.assert_array_equal(colors.values, [[1, 1], [0, 0], [1, 0]])
    assert colors.values.dtype == np.int64


def test_cached_parses_once(tmp_path):
    raw = tmp_path / "raw.csv"
    raw.write_text("unused")
    calls = []

    def build(path):
        calls.append(path)
        return FEATURES.copy(), COLORS

    data, colors = dense._cached(raw, "mycache", build)
    cache = tmp_path / "mycache.parquet"
    mtime = cache.stat().st_mtime_ns
    data2, colors2 = dense._cached(raw, "mycache", build)
    assert len(calls) == 1
    assert cache.stat().st_mtime_ns == mtime
    for d, c in [(data, colors), (data2, colors2)]:
        np.testing.assert_array_equal(d, FEATURES)
        np.testing.assert_array_equal(c.values, COLOR_VALUES)
        assert c.values.dtype == np.int64
        assert c.names == COLORS.names
        assert c.labels == COLORS.labels
    assert list(tmp_path.glob("*.tmp")) == []


def test_cached_without_colors(tmp_path):
    data, colors = dense._cached(
        tmp_path / "raw", "plain", lambda path: (FEATURES[:2].copy(), None)
    )
    data2, colors2 = dense._cached(tmp_path / "raw", "plain", None)
    assert colors is None and colors2 is None
    np.testing.assert_array_equal(data2, FEATURES[:2])


def test_pamap_parsed_from_zip_and_cached(tmp_path):
    path = tmp_path / "PAMAP2_Dataset.zip"
    rows = {i: [[i, 1.0, 2.0, 3.0], [i + 0.5, 1.0, 4.0, float("nan")]] for i in range(1, 10)}
    with zipfile.ZipFile(path, "w") as zf:
        for i, lines in rows.items():
            zf.writestr(
                f"PAMAP2_Dataset/Protocol/subject10{i}.dat",
                "\n".join(" ".join(map(str, line)) for line in lines) + "\n",
            )
    data, _, _ = dense._load_pamap(path)
    assert data.shape == (18, 2)
    np.testing.assert_array_equal(data[:2], [[2.0, 3.0], [4.0, 0.0]])
    assert (tmp_path / "pamap.parquet").is_file()


def test_cached_round_trip_types(tmp_path):
    labels = (("", "a b", "naïve", "日本"), ("x",))
    values = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=np.int64)
    features = np.arange(12, dtype=np.float32).reshape(4, 3)
    colors = Colors(values, ("näme with space", ""), labels)
    data, out = dense._cached(tmp_path / "raw", "types", lambda path: (features, colors))
    assert data.dtype == np.float32
    assert data.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(data, features)
    assert out.values.dtype == np.int64
    np.testing.assert_array_equal(out.values, values)
    assert out.names == colors.names
    assert out.labels == labels


def test_cached_preserves_nan_and_inf(tmp_path):
    features = np.array(
        [[np.nan, 1.0], [np.inf, -np.inf], [-0.0, 2.5]], dtype=np.float32
    )
    data, _ = dense._cached(tmp_path / "raw", "nan", lambda path: (features, None))
    np.testing.assert_array_equal(data, features)
    assert np.signbit(data[2, 0])


def test_cached_wide_color_codes(tmp_path):
    import pyarrow.parquet as pq

    values = np.arange(300, dtype=np.int64)[:, np.newaxis]
    colors = Colors(values, ("c",), (tuple(str(i) for i in range(300)),))
    features = np.zeros((300, 1), dtype=np.float32)
    _, out = dense._cached(tmp_path / "raw", "wide", lambda path: (features, colors))
    assert pq.read_schema(tmp_path / "wide.parquet").field("color0").type == "uint16"
    np.testing.assert_array_equal(out.values, values)


def _write_legacy_cache(path, data, colors):
    import json

    import h5py

    with h5py.File(path, "w") as hfp:
        hfp["X"] = data
        if colors is not None:
            codes = hfp.create_dataset("colors", data=colors.values.astype(np.uint8))
            codes.attrs["names"] = json.dumps(colors.names)
            codes.attrs["labels"] = json.dumps(colors.labels)


def test_cached_migrates_legacy_hdf5(tmp_path):
    _write_legacy_cache(tmp_path / "old.hdf5", FEATURES, COLORS)

    def build(path):
        raise AssertionError("the legacy cache must not be parsed again")

    data, colors = dense._cached(tmp_path / "raw", "old", build)
    np.testing.assert_array_equal(data, FEATURES)
    np.testing.assert_array_equal(colors.values, COLOR_VALUES)
    assert colors.names == COLORS.names
    assert colors.labels == COLORS.labels
    assert (tmp_path / "old.parquet").is_file()
    assert not (tmp_path / "old.hdf5").exists()


def test_cached_unknown_version_raises(tmp_path):
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq

    meta = {"version": 2, "n_features": 1, "color_names": None, "color_labels": None}
    table = pa.table({"x0": np.zeros(2, dtype=np.float32)})
    table = table.replace_schema_metadata({b"aida_data": json.dumps(meta).encode()})
    pq.write_table(table, tmp_path / "future.parquet")
    with pytest.raises(ValueError, match="future.parquet"):
        dense._cached(tmp_path / "raw", "future", None)
