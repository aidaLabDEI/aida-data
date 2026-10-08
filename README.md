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
set `AIDA_DATA_KEEP_RAW=1` to keep it. Raw files left over from earlier
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
