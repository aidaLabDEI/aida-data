# AIDA-data

This repository contains code for downloading and preprocessing datasets
used across software project of the [AIDA Lab](https://aidalabdei.github.io/)
at the Department of Information Engineering of the University of Padova.

## Usage

<details>
<summary>Dense datasets.</summary>

```python
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from aida_data.dense import SafeL2Normalizer, load

# Raw (cleaned) data
ds = load("glove-100-angular")

# Preprocess with any sklearn transformer/pipeline: it is fitted on the
# train set and applied to the test set.
pipeline = make_pipeline(StandardScaler(with_std=False), PCA(64), SafeL2Normalizer())
ds = load("glove-100-angular", pipeline, load_queries=True)
print(pipeline.named_steps["pca"].explained_variance_ratio_)
```

Datasets are downloaded to `datasets/` (or `$AIDA_DATA_DIR`). Large datasets
are parsed once into a zstd-compressed `<name>.parquet` cache next to the
download. The raw download is then deleted, since it is not needed anymore;
set `AIDA_DATA_KEEP_RAW=1` to keep it. Downloads are written to
`<name>.part` and renamed once complete; a leftover `.part` file is safe to
delete. Raw files left over from earlier
versions can be removed with `prune_raw`:

```python
from aida_data import dense

dense.prune_raw()               # dry run: list the raw files that can go
dense.prune_raw(dry_run=False)  # delete them
```

</details>

<details>
<summary>Dense datasets with colors.</summary>

Several dense datasets (fair-clustering, streaming-fair and silhouette
datasets: `adult`, `athlete`, `diabetes`, `creditcard`, `census1990`,
`4area`, `reuter_50_50`, `victorian`, `bank`, `covertype`, `phones`, `higgs`,
`higgs-highlevel`, `breast`, `wine`, `shuttle`, `rt-iot2022`, `biokdd`) come
with categorical attributes, returned as `Dataset.colors` and kept aligned
with the rows through deduplication, NaN filtering and the pipeline. Rows
are duplicates only if both features and colors are equal.

```python
from aida_data import dense

ds = dense.load("adult")
ds.dataset            # (n, 5) float32
ds.colors.names       # ("sex", "race", "marital-status")
sex = ds.colors.column("sex")  # (n,) int64 codes
ds.colors.labels[0]   # ("Female", "Male"), so labels[0][code] decodes sex
ds.colors.n_colors("race")
```

`shuttle` needs the optional `unlzw3` package (`pip install aida-data[shuttle]`).

</details>

<details>
<summary>Graph datasets with colored nodes.</summary>

Graphs from [Sirius](https://github.com/leonardopellegrina/Sirius/tree/main/data),
loaded as undirected simple graphs (self loops and duplicate edges dropped,
each edge stored once as `(u, v)` with `u < v`).

```python
from aida_data import graph

print(graph.available_datasets())
g = graph.load_edge_list("brexit")
g.edges   # (m, 2) int64 array
g.colors  # (n,) int64 array, g.colors[v] is the color of node v (-1 if missing)
g.n_nodes, g.n_edges, g.n_colors

# Keep edge orientation, and remap colors to 0..k-1
g = graph.load_edge_list("brexit", directed=True, remap_colors=True)
```

Each graph is parsed once into `datasets/graphs/<name>.parquet` (or under
`$AIDA_DATA_DIR`), a zstd-compressed adjacency list with one row per node
(`nbrs`: sorted out-neighbors, `color`), roughly a tenth of the size of the
TSV files. The TSV files are then deleted; set `AIDA_DATA_KEEP_RAW=1` to keep
them, or call `graph.prune_raw(dry_run=False)` to remove the ones left over
from earlier versions (`graph.prune_raw()` is a dry run). The options of
`load_edge_list` are applied when reading the cache.

</details>

<details>
<summary>Time series for motif discovery.</summary>

Series used by ATTIMO and MOMENTI: `astro`, `ecg`, `freezer`, `gap`, `humany`
(from the [ATTIMO figshare article](https://figshare.com/articles/dataset/Datasets/20747617),
CC BY 4.0) and the four-dimensional `steamgen`
([Zenodo 4273921](https://zenodo.org/records/4273921), CC BY 4.0). From the
Motiflets repository: `dishwasher`, `npo141` (sleep EEG) and `arrhythmia`. From
the MOMENTI repository: `foetal-ecg` (8 dimensions), `evaporator` (6) and
`ruth` (32), and `oikolab-weather` (8 hourly weather series, with `ts.time`). `fl010` (6 channels at 100 Hz,
25 million samples, from the PhysioNet LTMM database) needs the optional `wfdb`
package: `pip install aida-data[wfdb]`. The files of the last two groups come from GitHub at a pinned
commit and have their own origin and terms, see the comments in
`timeseries.py`.

```python
from aida_data import timeseries

print(timeseries.available_datasets())
ts = timeseries.load("ecg")
ts.values      # (n, d) float64, one row per time step, in the order of the source
ts.univariate  # (n,) view, only for series with d == 1
ts.dim_names   # ("drum pressure", ...) for steamgen, None if the source has no names
ts.time        # (n,) datetime64[ms], None if the source has no time stamps
```

`values` is always two-dimensional, so univariate and multivariate series share
one type; transpose it if your library expects `(d, n)`. Rows are never
reordered, deduplicated or normalized, and missing values stay NaN. The one
exception is `ecg`, whose file has 46991 blank lines between runs of values:
they are skipped, as `pyattimo.load_dataset("ecg")` does.

Each series is parsed once into `datasets/timeseries/<name>.parquet` (or under
`$AIDA_DATA_DIR`), a zstd-compressed float64 file, about the size of the
gzipped text and roughly ten times faster to read than to parse. The raw files
are then deleted; set `AIDA_DATA_KEEP_RAW=1` to keep them, or call
`timeseries.prune_raw(dry_run=False)` to remove the ones left over
(`timeseries.prune_raw()` is a dry run).

</details>
