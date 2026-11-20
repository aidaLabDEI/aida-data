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

Each of them also has an `hdf5/` subdirectory (`dense/hdf5/`, `graphs/hdf5/`,
`timeseries/hdf5/`, and `dense/hdf5/mteb/<task>/` for MTEB) with the HDF5 files
converted by [`aida-data export`](#command-line). They can be deleted with
`aida-data prune-hdf5`.

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

## Command line

The `aida-data` command (also `python -m aida_data`) downloads a dataset into
the cache and exposes it at a path, so that programs written in other
languages (C++, Rust, Julia...) can read it without going through Python. The
file is HDF5 by default, or parquet with `--format parquet`.

```console
$ aida-data export glove-100-angular ~/data/
/home/u/data/glove-100-angular.hdf5 -> /home/u/.cache/aida-data/dense/glove-100-angular.hdf5

$ aida-data export adult                          # converted, into the current directory
/home/u/adult.hdf5 -> /home/u/.cache/aida-data/dense/hdf5/adult.hdf5

$ aida-data export scidocs-qwen3-4b ./scidocs.h5  # link path given explicitly

$ aida-data export scidocs-qwen3-4b --format parquet
$ aida-data export timeseries/ecg --copy /scratch/  # a real copy, not a link

$ aida-data path higgs --format parquet           # print the cached path, nothing else
/home/u/.cache/aida-data/dense/higgs.parquet

$ aida-data list --module graph --cached
NAME     MODULE  NATIVE   PARQUET  HDF5
brexit   graph   parquet  yes      no

$ aida-data prune-hdf5          # dry run: list the converted files and their size
$ aida-data prune-hdf5 --yes    # delete them

$ aida-data snippet adult > load_adult.cpp   # C++ program that reads adult.hdf5
```

The commands:

- `export NAME [DEST]` makes sure that the dataset is in the cache in the
  requested format (downloading, parsing and converting if needed) and links
  it at `DEST`. It prints `<link> -> <target>`, or just the path with `--copy`.
- `path NAME` does the same without linking and prints the absolute path of
  the cached file, so that `$(aida-data path glove-100-angular)` works in
  shell scripts. Progress and log messages go to stderr.
- `list [--module dense|graph|timeseries] [--cached]` prints one row per
  dataset: its native format (`hdf5` for ann-benchmarks and VIBE files,
  `parquet`, or `parquet-dir` for MTEB), and whether the parquet and HDF5
  files are cached (`yes`, `no`, or `-` if the format is not available). It
  never downloads or converts. MTEB aliases are listed, full `mteb:<task>:<model>`
  names are not.
- `prune-hdf5` lists the converted HDF5 files and their total size, and
  deletes them (and stale temporary files) with `--yes`. It never touches the
  HDF5 datasets that are downloaded as such.
- `snippet NAME` prints C++ code, see [below](#c).

`NAME` is a dataset name of any module (names are unique across modules, and
an optional `module/` prefix such as `timeseries/ecg` is accepted), an MTEB
alias, or a full `mteb:<task>:<model>` name. A name that is not found gets
suggestions. An MTEB pair is only exposed if it is cached or hosted: `export`
never encodes, so encode it first with `dense.load`.

### Destination

`DEST` works like the target of `ln -s`:

- omitted: the link is created in the current directory;
- an existing directory (or a symlink to one): the link is created inside it;
- anything else: `DEST` is the path of the link. Its parent must exist.

The name of the link is the dataset name plus the suffix of the format
(`glove-100-angular.hdf5`, `pamap2.parquet`), not the name of the cached file.
An MTEB pair exported as parquet is a directory, named after its alias
(`scidocs-qwen3-4b/`), or `<task>__<model_slug>` if it has none.

If something is already at the link path, `export` fails, unless you pass
`--force`, which replaces a file or a symlink. It never replaces a directory,
nor a link to the cached file itself. Links and copies are created atomically.

| Option | Effect |
| --- | --- |
| `--format {hdf5,parquet}` | File to expose (default `hdf5`). The ann-benchmarks and VIBE datasets exist only as HDF5, so `--format parquet` is an error for them. |
| `--force` | Replace an existing file or symlink. |
| `--copy` | Copy instead of making a symlink (a directory is copied recursively). |
| `--hardlink` | Make a hard link instead (one per file for a directory). It fails across filesystems: use `--copy`. |
| `-q`, `--quiet` | Only warnings and errors, no progress bars. |
| `--debug` | Show the traceback of errors. |

`-q` and `--debug` are accepted by every command. Exit codes: `0` on success,
`1` on a runtime error (unknown name, existing destination, format not
available, not enough disk space, network error), `2` on a usage error.
Errors are printed on stderr as `aida-data: error: <message>`.

Symlinks point into the cache, so they break if the cache is deleted or
rebuilt, or if the converted file is removed with `prune-hdf5`. (The `prune_raw`
functions never remove caches.) Use `--copy` to get a file that is independent
of the cache.

### HDF5 layouts

Datasets are contiguous and uncompressed, so that a reader needs one `H5Dread`
(or a memory map) and no filter plugins. Every matrix is row-major with one
row per point. Every converted file has the root attributes
`aida_data_hdf5_version`, `source_mtime_ns`, `module` and `name`. Converted
files are rebuilt when the parquet cache they come from changes.

Dense datasets. The ann-benchmarks and VIBE datasets are exposed as
downloaded, with `train`, `test`, `neighbors`, `distances` and the attribute
`distance`. The other datasets are converted to a compatible layout, so that
tools that read `train`, `test` and `attrs["distance"]` work unchanged, but
**they have no `neighbors` and `distances`**: aida-data does not compute
ground truth for them.

| Object | Shape and type | Present |
| --- | --- | --- |
| `train` | `(n, d)` float32 | always (for MTEB, the corpus) |
| `test` | `(q, d)` float32 | MTEB retrieval (the queries) |
| `colors` | `(n, c)` unsigned integers | colored datasets |
| `ids` | `(n,)` string | MTEB |
| `test_ids` | `(q,)` string | MTEB retrieval |
| `qrels/query`, `qrels/doc` | `(r,)` string | MTEB retrieval |
| `qrels/score` | `(r,)` int64 | MTEB retrieval |
| attributes `distance`, `dimension` | string, integer | always |
| attribute `color_names` | array of strings | colored datasets |
| attribute `color_labels` | JSON string: a list with the list of labels of each color | colored datasets |

Graphs, in CSR form. The out-neighbors of node `v` are
`neighbors[offsets[v]:offsets[v+1]]`, sorted ascending. Like the parquet
cache, the graph is the one that `load_edge_list` reads before its options
are applied.

| Object | Shape and type |
| --- | --- |
| `offsets` | `(n + 1,)` int64 |
| `neighbors` | `(m,)` integers |
| `colors` | `(n,)` integers, `-1` if missing |
| attributes `n_nodes`, `n_edges` | integer |

Time series.

| Object | Shape and type | Present |
| --- | --- | --- |
| `values` | `(n, d)` float64, NaN for missing values | always |
| `time` | `(n,)` int64, milliseconds since the Unix epoch | if the source has time stamps |
| attribute `dim_names` | array of strings | if the source has names |
| attributes `sampling`, `source` | string | if known |

Integers are stored in the narrowest type that fits; HDF5 converts them when
you read them as `int64`, which is what the C++ snippets do.

### C++

`aida-data snippet NAME` prints a complete C++17 program that opens the HDF5
file of `NAME` with [H5Easy](https://github.com/highfive-devs/highfive), the
one-call interface of the header-only HighFive library, loads every object of
the file and prints their shapes. It is a starting point to copy and adapt.
It never downloads or converts anything, so it works offline: if the HDF5
file is not cached, the program checks with `file.exist` the objects that not
every dataset of the module has; if it is, the program loads exactly what the
file has, with its shapes in comments. `--file PATH` sets the path of the file
in the code (default `NAME.hdf5`, the name that `export` gives it).

```console
$ aida-data export glove-100-angular && aida-data snippet glove-100-angular > load.cpp
```

```cpp
#include <highfive/H5Easy.hpp>

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

int main() {
    H5Easy::File file("glove-100-angular.hdf5", H5Easy::File::ReadOnly);

    // One row per point.
    auto train = H5Easy::load<std::vector<std::vector<float>>>(file, "/train");

    // Ids of the nearest neighbors of each query, in train.
    auto neighbors = H5Easy::load<std::vector<std::vector<int64_t>>>(file, "/neighbors");

    // Distance of the dataset, e.g. "euclidean" or "angular".
    auto distance = H5Easy::loadAttribute<std::string>(file, "/", "distance");
    // ...
}
```

The output also has the loads of `test` and `distances`, and comments on how
to read a large matrix into one contiguous buffer (`read_raw`, or an Eigen
row-major matrix). Floats load as `float` (`double` for time series) and
integers as `int64_t`.

HighFive is a header-only library on top of the HDF5 library. The snippets
target HighFive 3 (the header comment says which version was tested). Install
it from a package manager (`apt install libhighfive-dev`, `conda install
highfive`, `vcpkg install highfive`) or as a git submodule, then build with
either of:

```console
$ h5c++ -std=c++17 -I/path/to/highfive/include load.cpp -o load
```

```cmake
find_package(HighFive REQUIRED)
target_link_libraries(load HighFive::HighFive)
```

With HighFive 2 the CMake target is `HighFive` and `DataSet::read_raw(T*)` is
`DataSet::read(T*)`.

### Parquet layouts

With `--format parquet` you get the cache of the dataset, which is read with
any parquet library. It has one column per feature, so wide embeddings have
thousands of columns, and the color names are only in the file metadata. HDF5
is easier to read outside Python.

| Cache | Columns |
| --- | --- |
| dense | `x0..x{d-1}` (float32), then `color0..color{c-1}` (integer codes) for colored datasets, and `id` (string) if the dataset has ids. The color names and labels, the number of features and other metadata are in the `aida_data` key of the schema metadata, as JSON. |
| graph | one row per node `v`: `nbrs` (list of out-neighbors, sorted) and `color` (integer, `-1` if missing) |
| time series | one row per time step: `x0..x{d-1}` (float64), and `t` (timestamp, ms) if the source has time stamps |
| MTEB | a directory: `corpus.parquet` (the dense layout, with `id` for retrieval and `color*` for clustering), and for retrieval `queries.parquet` (the dense layout, with `id`) and `qrels.parquet` (`query_id`, `doc_id`: strings, `score`: int64) |

### Disk space

Converted HDF5 files are uncompressed copies of the zstd-compressed parquet
caches, and can be much larger: `higgs` takes about 1.2 GB, and the Qwen3-4B
embeddings of a large corpus (8.8 million vectors of 2560 floats) about 90 GB.
Before converting, `export` logs the expected size and fails if the
filesystem has less than that plus 10% free, suggesting `--format parquet`.
The converted files are kept so that they can be linked, and
`aida-data prune-hdf5 --yes` frees the space (they are rebuilt on the next
`export`).
