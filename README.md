# AIDA-data

This repository contains code for downloading and preprocessing datasets
used across software project of the [AIDA Lab](https://aidalabdei.github.io/)
at the Department of Information Engineering of the University of Padova.

## Where the datasets are stored

Datasets are downloaded and cached once per user, in a root directory shared
by every project, notebook and working directory. The root is the first of:

1. `$AIDA_DATA_DIR`, if set and non-empty (`~` is expanded);
2. `$XDG_CACHE_HOME/aida-data`, if `XDG_CACHE_HOME` is set to an absolute path;
3. `~/.cache/aida-data`.

Everything under it can be downloaded or rebuilt again, which is why it lives
in the cache directory. Each module has its own subdirectory:

```
<root>/
  dense/        dense datasets (MTEB embeddings in dense/mteb/)
  graphs/       graphs
  timeseries/   time series
```

Raw downloads are deleted once they are parsed into a cache; set
`AIDA_DATA_KEEP_RAW=1` to keep them.

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

Datasets are downloaded to `<root>/dense` (see [Where the datasets are
stored](#where-the-datasets-are-stored)). Every dataset that is not an HDF5 file is parsed once into a
zstd-compressed `<name>.parquet` cache next to the download. The raw download is then deleted, since it is not needed anymore;
set `AIDA_DATA_KEEP_RAW=1` to keep it. Downloads and caches are written to
a temporary file with a unique name (`<name>.<random>.part` or
`<name>.parquet.<random>.parquet.tmp`) and renamed once complete, so
concurrent loads of the same dataset do not clash. A process that is killed
can leave such a `*.part` or `*.tmp` file behind; it is safe to delete, and
`prune_raw` removes the ones older than 24 hours. Raw files left over from
earlier versions can be removed with `prune_raw` too:

```python
from aida_data import dense

dense.prune_raw()               # dry run: list the raw files that can go
dense.prune_raw(dry_run=False)  # delete them
```

</details>

<details>
<summary>MTEB embeddings.</summary>

Any text-only MTEB Retrieval or Clustering task, encoded with any MTEB model,
loads through `dense.load` as `mteb:<task>:<model>`. The `default`
subset (or the first one, if there is no `default`) and the first evaluation
split of the task are used. The distance is the model's MTEB similarity
(`cosine`, `euclidean` or `dot`; multi-vector models are not supported).

```python
from aida_data import dense

# Retrieval: the corpus is ds.dataset, the queries are ds.queries
name = "mteb:SCIDOCS:sentence-transformers/all-MiniLM-L6-v2"
ds = dense.load(name)
ds.ids         # (n,) object array of corpus ids, aligned with ds.dataset
ds.query_ids   # (q,) object array of query ids, aligned with ds.queries
qrels = dense.load_qrels(name)  # {query_id: {doc_id: score}}, positive scores only

# Clustering: the unique texts are ds.dataset, one color per clustering set
ds = dense.load("mteb:ArxivClusteringP2P:sentence-transformers/all-MiniLM-L6-v2")
ds.colors.names        # ("set0", "set1", ...)
ds.colors.labels[0]    # ("<absent>", "astro-ph", ...)
# Texts that belong to set0 (the ones not labeled "<absent>")
in_set0 = ds.colors.column("set0") != ds.colors.labels[0].index("<absent>")
```

The pair is looked up in the local cache first, then in the hosted registry on
the Hugging Face Hub, and only if neither has it the task is encoded locally,
which needs the `mteb` extra (`pip install aida-data[mteb]`) and can take a
long time for large models. Downloading a hosted copy does not need the extra.
A failed connection to the registry raises an error instead of starting a
local encode. Local encodes are cached but never uploaded.

`ids`, `query_ids` and `colors` stay aligned with the rows through the usual
processing. Deduplication ignores the ids: of several identical vectors (with
equal colors, for clustering) only the first id is kept; pass
`deduplicate=False` to keep all of them. Texts that are absent from a
clustering set get the label `"<absent>"` in that set's color.

`load` and `load_qrels` accept encoding options, forwarded to the local
encoder: `batch_size` (default 64), `model_kwargs` (passed to `mteb.get_model`,
e.g. `{"device": "cuda:1"}`) and `limit` (for testing: encodes only the first
rows and does not write the cache). They are ignored, with a warning, if the
local cache or a hosted copy is used, and are only accepted for `mteb:` names.

```python
ds = dense.load(name, batch_size=128, model_kwargs={"device": "cuda:1"})
```

The hosted pairs also have short aliases, listed by
`dense.available_datasets()` and accepted by `load` and `load_qrels`:

| Alias                           | Task                 | Model                     |
| ------------------------------- | -------------------- | ------------------------- |
| `arxiv-clustering-p2p-qwen3-4b` | `ArxivClusteringP2P` | `Qwen/Qwen3-Embedding-4B` |
| `scidocs-qwen3-4b`              | `SCIDOCS`            | `Qwen/Qwen3-Embedding-4B` |

An alias is `<task_slug>-<model_tag>`: the task's CamelCase words and acronyms,
lowercased and joined with `-` (`ArxivClusteringP2P` gives
`arxiv-clustering-p2p`, with a few manual exceptions such as `nfcorpus`), then
a short tag of the model from a fixed table (`qwen3-4b`, `qwen3-0.6b`,
`minilm-l6`). Aliases are matched exactly, so they must be written in
lowercase. An alias is only another name for its pair: it uses the same cache,
the same hosted files and the same encoding options. Any pair can get an
alias, hosted or not:

```python
ds = dense.load("scidocs-qwen3-4b")
dense.resolve_name("scidocs-qwen3-4b")  # "mteb:SCIDOCS:Qwen/Qwen3-Embedding-4B"
dense.register_alias("scidocs-minilm", "mteb:SCIDOCS:sentence-transformers/all-MiniLM-L6-v2")
```

`register_alias` raises `ValueError` if the alias is already a dataset or an
alias (pass `force=True` to override), contains `:`, or if the target is not a
full `mteb:<task>:<model>` name.

The cache of a pair is `<root>/dense/mteb/<task>/<model_slug>/`, where
`model_slug` is the model name with `/` replaced by `__`. It contains
`corpus.parquet` (the vectors, with the `id` column for retrieval and the
`color*` columns for clustering, plus the task, model, revision and similarity
as metadata; written last, so its presence marks a complete cache) and, for
retrieval, `queries.parquet` and `qrels.parquet`. The texts are not stored.

Maintainers can build and publish caches with `scripts/build_mteb.py` (encodes
a name into the local cache) and `scripts/upload_mteb.py` (uploads it to the
hosted registry; needs `HF_TOKEN`), both run with `uv run`. The upload
script refuses models without a tag in `dense._MODEL_TAGS` and prints the line
to add to `_HOSTED` in `src/aida_data/_mteb_names.py`, which gives the new pair
its alias.

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

Each graph is parsed once into `<root>/graphs/<name>.parquet`, a zstd-compressed adjacency list with one row per node
(`nbrs`: sorted out-neighbors, `color`), roughly a tenth of the size of the
TSV files. The TSV files are then deleted; set `AIDA_DATA_KEEP_RAW=1` to keep
them, or call `graph.prune_raw(dry_run=False)` to remove the ones left over
from earlier versions, along with stale `*.part`/`*.tmp` files (`graph.prune_raw()`
is a dry run). The options of
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

Each series is parsed once into `<root>/timeseries/<name>.parquet`, a zstd-compressed float64 file, about the size of the
gzipped text and roughly ten times faster to read than to parse. The raw files
are then deleted; set `AIDA_DATA_KEEP_RAW=1` to keep them, or call
`timeseries.prune_raw(dry_run=False)` to remove the ones left over, along with
stale `*.part`/`*.tmp` files
(`timeseries.prune_raw()` is a dry run).

</details>
