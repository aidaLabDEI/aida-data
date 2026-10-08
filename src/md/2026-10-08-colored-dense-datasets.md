# Plan: colored dense datasets in `aida_data.dense`

Goal: handle priority 1 of `2026-10-08-software-todo.md` (§11). That means
adding HIGGS, PHONES, Covertype, the fair-clustering datasets and the
silhouette UCI datasets to `aida_data.dense`. Many of them carry categorical
"color" attributes, which `dense.load` must return next to the vectors and
keep aligned with them through deduplication, NaN filtering and the
pipeline.

## 1. Sources and preprocessing (checked on 2026-10-08)

The preprocessing was read from the code of each repository:
fair-clustering `datasets.py`, streaming-fair `datasetReaders/*.java`,
MACACO `experiments/datasets.py` and silhouette `research/data/parseData.py`.
Features and colors follow those repositories, so experiments stay
comparable. All URLs except HMDA answered HTTP 200 on 2026-10-08.

### 1.1 Datasets with colors

| name | URL | features | colors | rows |
|---|---|---|---|---|
| `higgs` | `https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz` (direct `.csv.gz`, 2.6 GB; the `static/public/280/higgs.zip` URL wraps the same file in a zip) | 28 (all, as FSR/SamRuLe) | `label` (col 0, 2 values) | 11M |
| `higgs-highlevel` | same file, same parsed cache | last 7 ("high level", as streaming-fair and MACACO) | `label` | 11M |
| `phones` | `https://archive.ics.uci.edu/static/public/344/heterogeneity+activity+recognition.zip` (old direct URL `.../00344/Activity%20recognition%20exp.zip` also works) | `x, y, z` of `Phones_accelerometer.csv` | `gt` (7 activities, `null` included); optionally also `User`, `Model`, `Device` | 13M |
| `covertype` | `https://archive.ics.uci.edu/static/public/31/covertype.zip` (contains `covtype.data.gz`) | 54 (10 numeric + 4 wilderness + 40 soil binary) | cover type (last col, 1..7) | 581k |
| `adult` | `https://archive.ics.uci.edu/static/public/2/adult.zip` (`adult.data`) | `age, fnlwgt, education-num, capital-gain, hours-per-week` | `sex, race, marital-status` | 32k |
| `athlete` | `https://github.com/rgriff23/Olympic_history/raw/master/data/athlete_events.csv` | `Age, Height, Weight` (rows with nulls dropped) | `Sex` | 271k raw, far fewer after dedup |
| `diabetes` | `https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip` (**hyphenated**; the URL in fair-clustering and in the TODO now returns 404) | `age` (lower bound of the bracket), `time_in_hospital, num_lab_procedures, num_procedures, num_medications, diag_1..3` (as float, non-numeric codes -> null), `number_diagnoses` | `gender, race` | ~100k |
| `census1990` | `https://archive.ics.uci.edu/static/public/116/us+census+data+1990.zip` (or the web.archive.org `.txt`, 361 MB, used by fair-clustering) | the 66 attributes listed in fair-clustering | `dAge, iSex` | 2.4M |
| `creditcard` | `https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip` (`.xls`) | `LIMIT_BAL, AGE, BILL_AMT1..6, PAY_AMT1..6` | `SEX, EDUCATION, MARRIAGE` | 30k |
| `4area` | `https://github.com/FaroukY/KFC-ScalableFairClustering/raw/main/data/4area.csv` | columns `1..8` | `color` | |
| `reuter_50_50` | same repo, `c50.csv` | columns `0..9` | `color` | |
| `victorian` | same repo, `victorian.csv` | columns `0..9` | `color` | |
| `bank` | same repo, `bank_categorized.csv` | `age, balance, duration, job, education, default, housing, loan, contact` | `marital` | |

Notes:

- fair-clustering registers `census1990` and `census1990_age` (and selects
  among the colors of `adult` and others with `color_idx`) only because each
  of its HDF5 files supports a single color. With multi-column colors (§2.1),
  one `census1990` entry covers both.
- `census1990` must not be confused with the ADBench `census` that is already
  registered. The name `census1990` keeps them apart.
- `adult.data` has 15 columns (fair-clustering named only 14), and its string
  values start with a space (`" Male"`), so they must be stripped. Missing
  values are `?`.
- `creditcard` is an `.xls` file, so `pd.read_excel` needs `xlrd`, which is
  a new dependency (§3).
- `athlete` repeats each athlete once per event, so deduplication removes
  most of its rows. This is expected, but the docstring should say so.

### 1.2 Datasets without colors (silhouette)

| name | URL | features (as `parseData.py`) | natural label (could be a color) |
|---|---|---|---|
| `breast` | `https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip` (`wdbc.data`) | drop `id`, `diagnosis` -> 30 | `diagnosis` (M/B) |
| `wine` | `https://archive.ics.uci.edu/static/public/186/wine+quality.zip` (red + white CSVs, `;`-separated) | drop `quality` -> 11 | `quality`, red/white |
| `shuttle` | `https://archive.ics.uci.edu/static/public/148/statlog+shuttle.zip` (`shuttle.trn.Z` + `shuttle.tst`) | drop class (col 9) -> 9, trn + tst concatenated | class (7 values) |
| `rt-iot2022` | `https://archive.ics.uci.edu/static/public/942/rt-iot2022.zip` (single file `RT_IOT2022`, 123k rows, 85 cols) | drop index, `id.orig_p`, `id.resp_p`, `proto`, `service`, `Attack_type` | `Attack_type` |
| `biokdd` | `https://kdd.org/cupfiles/KDDCupData/2004/data_kddcup04.tar.gz` (`bio_train.dat`) | drop cols 0-2 (block id, example id, label) -> 74 | label (col 2) |
| `metropt3` | `https://archive.ics.uci.edu/static/public/791/metropt+3+dataset.zip` | drop index and timestamp, drop the last 8 (digital) columns | none |
| `household-power` | `https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip` | drop `Date`, `Time`; `?` -> NaN | none |

Notes:

- `parseData.py` also drops the second-to-last RT-IoT column
  (`fwd_last_window_size`). This looks accidental. The plan keeps that
  column, and the docstring records the difference.
- For `household-power`, `parseData.py` fills missing values with 0. Here
  they become NaN, and `load` drops those rows, as it does for every other
  dataset. This is also a documented difference.
- `shuttle.trn.Z` is compressed with Unix `compress` (LZW), which the
  standard library cannot read. Add the small `unlzw3` dependency
  as an optional dependency, imported inside the loader like `densired`.
- The natural labels are available at no extra cost. Exposing them as colors
  (diagnosis, wine type and quality, shuttle class, attack type, biokdd
  label) makes these datasets usable for fair clustering too, and has no
  effect on users who ignore colors.

### 1.3 Out of scope for this step

- **hmda**: the S3 URL returns 403. The 2022 modified LAR must be found
  elsewhere on <https://ffiec.cfpb.gov/data-publication/modified-lar/2022>
  first. It is also very large.
- **BEERS**: needs a Kaggle login. It could later be supported as a
  "manual download" dataset whose loader raises an error explaining where
  to put the file.
- **BLOBS/randomized** (streaming-fair): synthetic. If needed they can be
  generated, as `densired-hard` is.
- **RNA-seq** (silhouette): needs manual cleaning in vim (see `links.txt`).
- **UBER**, **Gowalla**: spatio-temporal (§8 of the TODO).
- MACACO **Wikipedia/MusixMatch** and the DANNY variants: embeddings (§2.2
  of the TODO).

## 2. API changes in `dense.py`

### 2.1 Colors in `Dataset`

Extend the frozen `Dataset` with two optional fields (defaults keep every
existing caller working):

```python
@dataclass(frozen=True)
class Colors:
    values: np.ndarray                     # (n, c) int64, codes 0..k_j-1 per column
    names: tuple[str, ...]                 # c column names, e.g. ("sex", "race")
    labels: tuple[tuple[str, ...], ...]    # labels[j][code] = original category

    def column(self, name: str) -> np.ndarray: ...   # (n,) codes of one column
    def n_colors(self, name: str) -> int: ...


@dataclass(frozen=True)
class Dataset:
    distance: str
    dataset: np.ndarray
    queries: np.ndarray | None = None
    distances: np.ndarray | None = None
    colors: Colors | None = None           # aligned with `dataset` rows
```

Design choices:

- **Multiple color columns** in one dataset, instead of one registered
  dataset per color (fair-clustering's `census1990` vs `census1990_age`).
- Codes are **dense, 0-based and sorted by label** (as `LabelEncoder`
  produces them), so `labels[j]` is enough to decode them. This matches
  `graph.load_edge_list(remap_colors=True)`. Missing categories are dropped
  at load time (rows with nulls in a color column are removed), so there is
  no `-1`.
- Colors only describe the train set (`dataset`). None of these datasets has
  a query set.

### 2.2 Loader contract

Today a loader returns `(train, test, distances)`. Colored loaders return a
4-tuple `(train, None, None, colors)`. `load` accepts both 3- and 4-tuples,
so the existing loaders and the fake loaders in the tests stay unchanged.

### 2.3 `load` keeps colors aligned

Each row-dropping step in `load` must apply the same mask to `colors`:

1. **deduplication**: `_drop_duplicate_rows` gains a variant that returns the
   kept indices. For colored datasets duplicates are computed on
   `np.hstack([train, colors.values])`. Two points with the same features
   but different colors are both kept, because dropping either would change
   the color proportions that fair clustering relies on. Exact duplicates
   (same features and same colors) are dropped as usual;
2. **NaN/inf filtering**: the same boolean mask is applied to `colors`;
3. **pipeline**: fitted on the features only. It already must not change
   the number of rows, so the colors need no change;
4. **all-zero rows** (angular/cosine/normalized only): same mask. None of
   the new datasets is angular, but the code path should be correct anyway;
5. **second deduplication** after the pipeline: same rule as step 1.

`Colors` is rebuilt with `dataclasses.replace(colors, values=values[idx])`.
Codes are not re-densified when a color disappears after filtering, so
`labels` stays valid. `n_colors` counts the codes actually present.

### 2.4 Parsing cache

Parsing HIGGS (8 GB of CSV) or PHONES (13M lines) on every `load` is too
slow. `_load_pamap` already caches its parsed data as HDF5. This is turned
into a helper:

```python
def _cached(path: Path, cache_name: str, build: Callable[[Path], tuple]) -> tuple:
    """Parse `path` once with `build` and store the arrays (and the color
    names/labels as attributes) in DATASETS_DIR / f"{cache_name}.hdf5"."""
```

The cache is keyed by **dataset name**, not by URL basename. Several
datasets share one download (`higgs` and `higgs-highlevel`, and possibly
`creditcard` with and without colors), and `local_path` already maps them to
the same raw file. `higgs-highlevel` reads the `higgs` cache and slices the
last 7 columns, so the file is parsed once. Raw files stay compressed
(`.csv.gz` is read with `pd.read_csv(..., compression="gzip")`, files inside
zips are read with `zipfile.ZipFile.open`), and nothing is extracted into
`DATASETS_DIR`, unlike `_load_ht`/`_load_chem`.

Parsing large files: `pd.read_csv` with explicit `usecols` and
`dtype=np.float32` (the pyarrow engine is already a dependency and is much
faster for HIGGS). Colors are encoded with `pd.factorize(sort=True)` or
`pd.Categorical`, so `LabelEncoder` is not needed.

### 2.5 Helper for the loaders

Most loaders are "read a table, select feature columns, select color
columns, drop rows with nulls". One helper keeps each of them to a few
lines:

```python
def _split_table(
    df: pd.DataFrame, features: list[str], colors: list[str]
) -> tuple[np.ndarray, Colors]:
    """Drop rows with nulls in `features + colors`, return the float32
    feature matrix and the encoded colors."""
```

Per-dataset loaders then handle only reading the file (`_read_adult`,
`_read_higgs`, `_read_phones`, `_read_kfc(url, features, colors)`, ...). The
four KFC datasets share one loader built with `functools.partial`.

### 2.6 Performance of deduplication

`np.unique(axis=0)` on 11M x 28 float32 (HIGGS) is slow and needs several
GB of memory. Rows can instead be viewed as a single `np.void` (or hashed
with `pd.util.hash_array` on a contiguous view) before calling `np.unique`.
This is the same idea as `graph._unique_rows`. It should be measured on
HIGGS and PHONES before deciding whether it is needed.

## 3. Dependencies

- `xlrd` (creditcard `.xls`): add as a regular dependency. It is small and
  pure Python.
- `unlzw3` (shuttle `.Z`): optional, imported lazily with the same error
  message pattern as `densired`.
- No `polars` and no `LabelEncoder`: pandas is enough.

## 4. Tests (`tests/test_colored.py`, plus additions to `test_load_pipeline.py`)

The tests use fake loaders registered with `file:///` URLs, as the existing
tests do:

- uncolored datasets still return `colors is None`, and all existing tests
  pass unchanged (3-tuple loaders);
- a colored fake loader with duplicate rows (identical features, same and
  different colors) and NaN rows: check which rows survive and that
  `colors.values[i]` still belongs to `dataset[i]`;
- a pipeline (`StandardScaler`, `PCA`) leaves the colors untouched;
- `Colors.column`, `n_colors`, `names` and `labels`, including the decoding
  `labels[j][code]`;
- `_split_table` drops rows with nulls in color columns and encodes colors
  sorted by label;
- `_cached` writes once and reads back identical arrays and attributes
  (the cache file mtime does not change on the second call);
- parsers tested on tiny synthetic files written to `tmp_path`, using the
  exact file layouts: an `adult.data` with leading spaces and `?`, a
  `HIGGS.csv.gz` with 29 columns, a zip with
  `Phones_accelerometer.csv`, a `covtype.data.gz`, the `RT_IOT2022` header
  and the `wine` CSV pair;
- network tests (`@pytest.mark.network`) for the small datasets only:
  `adult` (shape, 2 sexes), `breast` (569 x 30), `covertype` (581012 x 54,
  7 colors before deduplication).

## 5. Docs

- module docstring of `dense.py`: mention colors;
- README: a collapsible section "Dense datasets with colors" with:

  ```python
  ds = dense.load("adult")
  ds.dataset            # (n, 5) float32
  ds.colors.names       # ("sex", "race", "marital-status")
  sex = ds.colors.column("sex")
  ds.colors.labels[0]   # ("Female", "Male")
  ```

- each loader's docstring states its source repository and any deviation
  from it (RT-IoT column, household-power NaN handling, deduplication of
  `athlete`).

## 6. Steps

1. [x] Add `Colors` and `Dataset.colors`. Make `load` accept 4-tuples and keep
   colors aligned (§2.3). Write the alignment tests with fake loaders and
   run the whole suite.
2. [x] Add the `_cached` and `_split_table` helpers, then port `_load_pamap` to
   `_cached` to check the helper on an existing dataset.
3. [x] Small colored datasets: `adult`, `athlete`, `diabetes`, `creditcard`
   (+ `xlrd`), and the KFC four. Add parser tests.
4. [x] Silhouette datasets: `breast`, `wine`, `shuttle` (+ optional `unlzw3`),
   `rt-iot2022`, `biokdd`, `metropt3`, `household-power`.
5. [x] Large datasets: `covertype`, `census1990`, `phones`, `higgs`,
   `higgs-highlevel`. Measure parse time, cache size and deduplication time,
   and apply §2.6 if needed.
6. [x] Manually load every new dataset once. Record the shape, the number of
   rows dropped by deduplication and NaN filtering, and the color counts in
   an "Implementation summary" section at the end of this file.
7. [x] Update the README and tick off priority 1 in
   `2026-10-08-software-todo.md`.

## 7. Open questions

- **Duplicate points with different colors**: the plan keeps both (§2.3,
  step 1). The alternative is to drop all but the first, as uncolored
  datasets do, at the cost of changing the color distribution. Some
  algorithms may assume distinct points.
- **Natural labels as colors** for the silhouette datasets (§1.2): exposed
  by default, or not at all, to keep those datasets identical to the
  silhouette inputs?
- **`phones` color columns**: only `gt` (as streaming-fair and MACACO), or
  also `User`, `Model`, `Device`? Extra columns cost little memory (13M x 4
  int64 = 400 MB; `int8`/`int16` codes would shrink this, so should
  `Colors.values` be a narrow int dtype instead of int64?).
- **`higgs-highlevel`**: register it separately (recommended, since
  streaming-fair and MACACO use it), or document a `FunctionTransformer`
  pipeline that selects the last 7 columns?
- **Standardization**: fair-clustering has a commented-out
  `standardize`/`-std` variant. The plan registers raw features and leaves
  scaling to the `pipeline` argument (`StandardScaler`), as for the other
  dense datasets.

## 8. Implementation summary

All steps of §6 are done. `aida_data.dense` has 20 new datasets: the 13 of
§1.1 (except hmda) and the 7 of §1.2. `Dataset.colors` is a new `Colors`
object, which `load` keeps aligned with the rows. The test suite has 46
offline tests and 4 network tests, and all of them pass.

### What changed

- `Colors` (`values`, `names`, `labels`, `column`, `n_colors`, `take`) and
  `Dataset.colors`. Loaders can return a 4-tuple. Every row-dropping step
  of `load` applies the same selection to the colors. A row is a duplicate
  only if its features and its colors are both equal.
- `_cached(path, cache_name, build)` stores the features and the colors
  (names and labels as JSON attributes, codes in the narrowest int type) in
  `<cache_name>.hdf5` next to the download. It writes to a temporary file
  and renames it when done. `_load_pamap` now uses it and keeps its old
  `pamap.hdf5` cache. `_split_table(df, features, colors)` encodes the
  colors with `pd.factorize(sort=True)`.
- §2.6: deduplication now uses `pd.DataFrame.duplicated`, which hashes the
  columns. It returns the same rows as `np.unique(axis=0)` and is much
  faster: 28 s -> 5 s on census1990, 8.8 s -> 3.9 s on phones compared
  with the `np.void` view. It applies to every dataset, not only colored
  ones.
- HIGGS is read in batches with pyarrow's streaming CSV reader.
  `pd.read_csv(engine="pyarrow")` peaked at 17.8 GB of RSS, and the
  streaming reader needs 3.2 GB. Both take about 60 s, once per cache.
- Dependencies: `xlrd` (regular), and `unlzw3` (optional extra `shuttle`,
  also in the dev group so that the `.Z` test runs).
- README section "Dense datasets with colors", module docstring, and
  priority 1 ticked off in `2026-10-08-software-todo.md`.

### Decisions on the open questions (§7)

- Duplicate points with different colors are kept.
- The natural labels of the silhouette datasets are exposed as colors.
  `metropt3` and `household-power` have no colors.
- `phones` colors are `gt, User, Model, Device`. `gt` keeps `null` as a
  category. `Colors.values` stays int64 in memory, and only the cache
  stores narrow codes.
- `higgs-highlevel` is registered as its own dataset and reads the `higgs`
  cache.
- No standardization: scaling is left to `pipeline`.

### Deviations found while implementing

- The diabetes zip now holds `diabetic_data.csv` at its root, not under
  `dataset_diabetes/`.
- `creditcard` has no parser test, because writing `.xls` would need
  `xlwt`. The manual load below covers it.
- Deduplication runs before NaN filtering and treats NaN as equal to NaN.
  In `household-power` the 25,979 rows with missing values all collapse
  into one duplicate group, and the NaN filter then drops the one row that
  is left.

### Manual loads (`load(name)` with defaults, from cache)

"Parsed" is the number of rows the loader returns. Loaders drop rows with
missing values in the selected columns before this point: `athlete` drops
271,116 -> 206,165 and `diabetes` drops 101,766 -> 89,782. The other
datasets lose no rows at this point. Times are measured with a warm cache.

| dataset | parsed | dropped (dedup) | dropped (NaN/inf) | final shape | colors (count) | load (s) |
|---|---|---|---|---|---|---|
| `adult` | 32,561 | 151 | 0 | 32,410 x 5 | `sex` 2; `race` 5; `marital-status` 7 | 0.2 |
| `athlete` | 206,165 | 160,810 | 0 | 45,355 x 3 | `Sex` 2 | 0.8 |
| `diabetes` | 89,782 | 2 | 0 | 89,780 x 9 | `gender` 3; `race` 5 | 1.1 |
| `creditcard` | 30,000 | 107 | 0 | 29,893 x 14 | `SEX` 2; `EDUCATION` 7; `MARRIAGE` 4 | 1.3 |
| `4area` | 35,385 | 0 | 0 | 35,385 x 8 | `color` 4 | 0.1 |
| `reuter_50_50` | 2,500 | 0 | 0 | 2,500 x 10 | `color` 50 | 0.0 |
| `victorian` | 4,500 | 0 | 0 | 4,500 x 10 | `color` 45 | 0.0 |
| `bank` | 4,521 | 0 | 0 | 4,521 x 9 | `marital` 3 | 0.0 |
| `breast` | 569 | 0 | 0 | 569 x 30 | `diagnosis` 2 | 0.0 |
| `wine` | 6,497 | 1,177 | 0 | 5,320 x 11 | `type` 2; `quality` 7 | 0.0 |
| `shuttle` | 58,000 | 0 | 0 | 58,000 x 9 | `class` 7 | 0.2 |
| `rt-iot2022` | 123,117 | 104,838 | 0 | 18,279 x 79 | `Attack_type` 12 | 1.2 |
| `biokdd` | 145,751 | 783 | 0 | 144,968 x 74 | `label` 2 | 0.7 |
| `metropt3` | 1,516,948 | 58,793 | 0 | 1,458,155 x 7 | none | 0.8 |
| `household-power` | 2,075,259 | 168,560 | 1 | 1,906,698 x 7 | none | 1.0 |
| `covertype` | 581,012 | 0 | 0 | 581,012 x 54 | `cover_type` 7 | 1.5 |
| `census1990` | 2,458,285 | 459,793 | 0 | 1,998,492 x 66 | `dAge` 8; `iSex` 2 | 8.9 |
| `phones` | 13,062,475 | 1,631,177 | 0 | 11,431,298 x 3 | `gt` 7; `User` 9; `Model` 4; `Device` 8 | 8.1 |
| `higgs` | 11,000,000 | 278,698 | 0 | 10,721,302 x 28 | `label` 2 | 51.4 |
| `higgs-highlevel` | 11,000,000 | 278,698 | 0 | 10,721,302 x 7 | `label` 2 | 20.0 |

Notes on the numbers:

- Parsing once, then cache size: higgs 57 s, 1.2 GB; phones 10 s,
  209 MB; census1990 3 s, 654 MB; covertype 2 s, 126 MB.
- All 278,698 duplicates in HIGGS have label 0.
- `rt-iot2022` keeps only 18k rows: once the ports are dropped, most flows
  are exact duplicates, also at float64 precision.
- `athlete` has one row per athlete and event, so most of its rows are
  duplicates, as expected.
