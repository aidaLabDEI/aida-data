# Plan: `aida-data export` command line tool

Not implemented yet.

Goal: a command line tool that downloads a dataset into the usual cache
(`<root>/dense`, `<root>/graphs`, `<root>/timeseries`, see `_paths.py`) and
exposes it at a path the user chooses, so that programs written in other
languages (C++, Rust, Julia...) can read it without going through Python.

The exposed file is HDF5 by default, or parquet with `--format parquet`:

| Dataset | `--format hdf5` (default) | `--format parquet` |
|---|---|---|
| ann-benchmarks / VIBE dense | the downloaded `.hdf5`, untouched | error: only available as HDF5 |
| any other dense, graph, time series | an HDF5 file converted from the parquet cache (§3.5) | the parquet cache |
| hosted MTEB pair | one HDF5 file holding corpus, queries and qrels (§3.5) | the directory of parquet files |

HDF5 is the default because it is the easiest format to read outside Python:
every matrix is a single contiguous row-major dataset. A C++ program reads
it with one `H5Dread` call and needs neither Arrow nor JSON metadata, and the
dense layout matches what ann-benchmarks tools already read. The parquet
caches store one column per feature (2560 columns for the Qwen3-4B
embeddings), and their colors can only be read through JSON metadata.

## 1. Decisions

Settled in the interview of 2026-11-20:

| Question | Decision |
|---|---|
| Scope | Every dataset of every module, including the aida-data parquet caches. |
| Default format | HDF5; `--format parquet` exposes the parquet cache. |
| Dense datasets without a cache (npz/csv/zip/fbin) | Give them a parquet cache in `load()` too (§3.1). |
| `higgs-highlevel`, which shares `higgs.parquet` | Gets its own cache, derived from `higgs.parquet` (§3.2). |
| MTEB pair, with `--format parquet` | One link to the cache directory. |
| `DEST` | Works like `ln -s`: an existing directory gets the link inside it. Otherwise `DEST` is the path of the link. Defaults to `.`. |
| Something already at the link path | Error, unless `--force`. `--force` never replaces a directory. |
| Names | One flat namespace across modules, with an optional `module/` prefix. A test checks that names are unique. |
| MTEB pair neither cached nor hosted | Error that suggests `dense.load(...)`. `export` never encodes. |
| Extra commands and options | `list`, `path`, `--copy`, `--hardlink`. |
| C++ starting point | `aida-data snippet NAME` prints a C++ program that loads the HDF5 export with H5Easy (§2.7). |

Decided while planning, no input needed:

- `argparse`, so no new dependency (`h5py` already is one). The `aida-data`
  entry point already declared in `pyproject.toml` (`aida_data:main`, which
  prints a placeholder today) becomes the CLI.
- Symlinks are absolute. The cache root is already absolute (`datasets_dir()`
  calls `.absolute()`).
- **Default link name**: the dataset name plus the suffix of the exported
  format, not the base name of the cached file. Cached names can be
  misleading: `pamap2` is cached as `pamap.parquet`, and an MTEB directory is
  named after the model (`Qwen__Qwen3-Embedding-4B`). For example:
  `glove-100-angular.hdf5`, `pamap2.hdf5`, `scidocs-qwen3-4b.hdf5`, or with
  `--format parquet` `pamap2.parquet` and `scidocs-qwen3-4b/`. A full
  `mteb:<task>:<model>` name uses its default alias if it has one, else
  `<task>__<model_slug>`. This is the only departure from `ln -s`, which
  would use the target's base name.
- **Converted HDF5 files are kept in the cache** next to the parquet cache,
  under `<root>/<module>/hdf5/` (§3.5), so that they can be symlinked like
  any other cached file. They are not stored as `dense/<cache_name>.hdf5`,
  because `dense._cached` treats that name as a pre-0.2 cache to migrate and
  then deletes it.
- **ann-benchmarks/VIBE datasets with `--format parquet`**: an error, not a
  conversion. Their parquet form would lose `test`, `neighbors` and
  `distances`, so it is not worth a second representation.
- **Converted files can be pruned**, since they duplicate the parquet caches
  and can be large (§3.5): `export.prune_hdf5()` and `aida-data prune-hdf5`.
- The logic lives in a library module, `aida_data/export.py`, so it is usable
  from Python too. `cli.py` only parses arguments and prints.

## 2. Behaviour

```console
$ aida-data export glove-100-angular ~/data/
~/data/glove-100-angular.hdf5 -> ~/.cache/aida-data/dense/glove-100-angular.hdf5

$ aida-data export adult                             # converted, into the current directory
./adult.hdf5 -> ~/.cache/aida-data/dense/hdf5/adult.hdf5

$ aida-data export scidocs-qwen3-4b ./scidocs.h5     # link path given explicitly
./scidocs.h5 -> ~/.cache/aida-data/dense/hdf5/mteb/SCIDOCS/Qwen__Qwen3-Embedding-4B.hdf5

$ aida-data export scidocs-qwen3-4b --format parquet
./scidocs-qwen3-4b -> ~/.cache/aida-data/dense/mteb/SCIDOCS/Qwen__Qwen3-Embedding-4B

$ aida-data export timeseries/ecg --copy /scratch/   # a real copy, not a link
/scratch/ecg.hdf5

$ aida-data path higgs --format parquet              # print the cached path (builds it if needed)
/home/u/.cache/aida-data/dense/higgs.parquet

$ aida-data list --module graph --cached
NAME            MODULE  NATIVE   PARQUET  HDF5
amazon          graph   parquet  yes      no
...

$ aida-data prune-hdf5            # dry run: list the converted files and their size
$ aida-data prune-hdf5 --yes      # delete them

$ aida-data snippet adult > load_adult.cpp   # C++ (H5Easy) program that reads adult.hdf5
```

### 2.1 `aida-data export NAME [DEST] [--format {hdf5,parquet}] [--force] [--copy | --hardlink] [-q]`

1. Resolve `NAME` (§4.1) to `(module, canonical name)`.
2. Make sure the file is in the cache in the requested format, downloading,
   parsing and converting it if needed (§3, §4.2). Downloads use the
   existing `_download.download`, which writes atomically and shows a tqdm
   bar on stderr.
3. Compute the link path:
   - `DEST` omitted: `./<default name>`;
   - `DEST` is an existing directory, or a symlink to one (as with `ln -s`
     without `-n`): `DEST/<default name>`;
   - otherwise `DEST` itself. Its parent must exist, as for `ln`. Missing
     parents are not created, because that would hide typos.
4. If something exists at the link path (checked with `lexists`, so broken
   symlinks count):
   - without `--force`: exit 1 with `<path> exists, pass --force to replace it`;
   - with `--force`: a symlink or regular file is replaced; a directory is
     always an error, even an empty one (`--force` never deletes trees).
   - If the link path resolves to the cached file itself, for example when
     exporting into the cache directory, exit 1 whatever the flags, so the
     cache never gets overwritten by a link to itself.
5. Create the link or copy **atomically**: write it under a unique temporary
   name next to the destination, then `os.replace` it into place. For a
   directory (MTEB parquet, copied or hard-linked), build the temporary
   directory, then remove the old symlink/file and rename. Reuse
   `_tmp._create_unique` (or a sibling helper that returns a fresh name
   without creating the file, since `os.symlink` needs the path to be free).
   - default: `os.symlink(target, tmp)`;
   - `--copy`: `shutil.copy2` for a file, `shutil.copytree` for an MTEB
     directory;
   - `--hardlink`: `os.link`; for an MTEB directory, a real directory with
     one hard link per file. `EXDEV` (another filesystem) becomes the error
     `cannot hard-link across filesystems, use --copy`.
6. Print `<link> -> <target>` (or just `<link>` with `--copy`) on stdout.

### 2.2 `aida-data path NAME [--format {hdf5,parquet}]`

Steps 1 and 2 of `export`, then print the absolute cached path, nothing else,
so that `$(aida-data path glove-100-angular)` works in shell scripts. All
logging and progress output goes to stderr.

### 2.3 `aida-data list [--module M] [--cached]`

One row per name: `NAME MODULE NATIVE PARQUET HDF5`. NATIVE is the format
of the cache: `hdf5` (ann-benchmarks/VIBE), `parquet`, or `parquet-dir`
(MTEB). PARQUET and HDF5 are `yes`, `no`, or `-` when the format is not
available. All of them are checked without downloading or converting.
`--cached` keeps the rows with at least one `yes`. Names are
`dense.available_datasets()` (including the MTEB aliases), then
`graph.available_datasets()`, then `timeseries.available_datasets()`. Full
`mteb:` names are not listed, since there are infinitely many. Plain
whitespace-aligned columns, so the output can be read with `awk` and `cut`.

### 2.4 `aida-data prune-hdf5 [--yes]`

Lists (by default) or deletes (`--yes`) every converted file under
`<root>/*/hdf5/`, plus stale `*.hdf5.tmp` files, with their total size, like
`prune_raw`. Never touches the native HDF5 downloads, which live elsewhere.
Symlinks already exported to these files break; the README says so.

### 2.5 Errors and exit codes

- `0` success; `1` runtime error (unknown name, existing destination, MTEB
  pair not available, format not available, network error); `2` usage error
  (argparse default).
- Unknown name: `unknown dataset 'glove-100'; did you mean: glove-100-angular?`
  (`difflib.get_close_matches` over all names).
- `--format parquet` on an HDF5-only dataset: `glove-100-angular is only
  available as hdf5`.
- MTEB pair neither cached nor hosted: `mteb:X:Y is not hosted and not cached;
  encode it first with python -c 'from aida_data import dense; dense.load("...")'
  (needs aida-data[mteb])`.
- Errors are printed as `aida-data: error: <message>` on stderr, without a
  traceback. `--debug` (global) re-raises, to see the traceback.

### 2.6 Logging

The CLI calls `logging.basicConfig(level=INFO, stream=stderr,
format="%(message)s")`, so the existing `downloading ...` and `parsing ...
into ...` messages are shown, plus a new `converting ... to ...`. `-q/--quiet`
sets WARNING and disables tqdm (`TQDM_DISABLE=1` in the environment before
the first import of tqdm).

### 2.7 `aida-data snippet NAME [--file PATH]`

Prints on stdout a complete, compilable C++17 program that opens the HDF5
export of `NAME` (§3.5) with [H5Easy](https://github.com/highfive-devs/highfive),
the one-call interface of the header-only HighFive library, loads every
object of the file and prints their shapes. It is a starting point to copy,
not a library: the code is plain and commented, one `H5Easy::load` per
object.

- **No download, no conversion.** `snippet` never fetches or builds
  anything, so it is instant and works offline.
  - If the HDF5 file is already cached and fresh (`is_cached(entry,
    "hdf5")`), the snippet is **tailored**: it loads only the objects that
    the file has, and a comment above each load gives its shape and dtype,
    read with `h5py` (e.g. `// (1183514, 100) float32`).
  - Otherwise it is **generic** for the module: the optional objects
    (`test`, `colors`, `ids`, qrels, `time`, ...) are wrapped in
    `if (file.exist("/colors")) { ... }`, so that the program is correct for
    any dataset of that module. A header comment says that it was generated
    without the file, and how to get it (`aida-data export NAME`).
- `--file PATH` sets the path in the `H5Easy::File` constructor. The default
  is the default link name of `export` (`<link_stem>.hdf5`), so
  `aida-data export NAME && aida-data snippet NAME > load.cpp` works from the
  same directory. The path is a C++ string literal, so `"` and `\` are
  escaped.
- **Dtypes are fixed by the snippet, not by the file.** Floats load as
  `float` (dense) or `double` (time series). All integers (colors, offsets,
  neighbors, time, qrels scores) load as `int64_t`, and HDF5 converts them
  on read. That way the snippet does not change when the cache picks a
  narrower type for a different dataset.
- Matrices load as `std::vector<std::vector<T>>`, the H5Easy type that needs
  no third-party dependency. A comment shows the contiguous alternatives for
  large data: `Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
  Eigen::RowMajor>` (needs `H5_USE_EIGEN`), or HighFive's
  `DataSet::read_raw` into a preallocated `std::vector<float>` of `n * d`.
- Attributes load with `H5Easy::loadAttribute<T>(file, "/", key)`.
  `color_labels` is a JSON string, so the snippet loads it as `std::string`
  with a comment pointing to a JSON library. Parsing it would add a
  dependency to the snippet.
- A header comment gives the build commands:
  `h5c++ -std=c++17 -I<highfive>/include load.cpp -o load`, or with CMake:
  `find_package(HighFive REQUIRED)` and
  `target_link_libraries(load HighFive::HighFive)`. HighFive itself comes
  from a package manager (`apt install libhighfive-dev`, `conda install
  highfive`, `vcpkg install highfive`) or a git submodule.
- The snippet targets HighFive v3 and names the tested version in its header
  comment. The H5Easy calls used (`File`, `load`, `loadAttribute`,
  `File::exist`) are the same in v2, but the CMake target name (`HighFive`
  in v2, `HighFive::HighFive` in v3) and `read_raw` (`read(T*)` in v2) are
  not, so the comment says so.

Example, tailored (`glove-100-angular`, cached):

```cpp
// Load glove-100-angular.hdf5, exported by `aida-data export glove-100-angular`.
// Generated by aida-data 0.5.0 for HighFive 3.x (H5Easy).
// Build: h5c++ -std=c++17 -I/path/to/highfive/include load.cpp -o load
#include <highfive/H5Easy.hpp>

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

int main() {
    H5Easy::File file("glove-100-angular.hdf5", H5Easy::File::ReadOnly);

    // (1183514, 100) float32, one row per point.
    auto train = H5Easy::load<std::vector<std::vector<float>>>(file, "/train");
    // (10000, 100) float32, one row per query.
    auto test = H5Easy::load<std::vector<std::vector<float>>>(file, "/test");
    // (10000, 100) int32, ids of the 100 nearest neighbors of each query in train.
    auto neighbors = H5Easy::load<std::vector<std::vector<int64_t>>>(file, "/neighbors");
    // (10000, 100) float32, distances to those neighbors.
    auto distances = H5Easy::load<std::vector<std::vector<float>>>(file, "/distances");
    auto distance = H5Easy::loadAttribute<std::string>(file, "/", "distance");

    // For large data, read into one contiguous row-major buffer instead:
    //   HighFive::DataSet ds = file.getDataSet("/train");
    //   auto dims = ds.getDimensions();
    //   std::vector<float> flat(dims[0] * dims[1]);
    //   ds.read_raw(flat.data());

    std::cout << "train " << train.size() << " x " << train[0].size()
              << ", test " << test.size() << ", distance " << distance << "\n";
}
```

The generic dense snippet also loads `colors` (`int64_t` matrix, plus the
`color_names` and `color_labels` attributes), `ids` and `test_ids`
(`std::vector<std::string>`) and the three `qrels/*` vectors, each inside a
`file.exist` check. The graph snippet loads `offsets`, `neighbors` and
`colors` as `std::vector<int64_t>`, with a loop that shows how to walk the
neighbors of node 0. The time series snippet loads `values` as
`std::vector<std::vector<double>>`, `time` as `std::vector<int64_t>`
(milliseconds since the epoch) and the `dim_names` attribute.

## 3. Library changes needed first

### 3.1 Uniform parquet caching of dense datasets

Today `_cached(path, cache_name, build)` is called *inside* 8 loaders
(pamap2, biokdd, metropt3, household-power, covertype, census1990, phones,
higgs/higgs-highlevel). The other 17 non-HDF5 datasets are parsed again on
every `load`: census, ht, yandex-t2i, chem, adult, athlete, diabetes,
creditcard, 4area, reuter_50_50, victorian, bank, breast, wine, shuttle,
rt-iot2022 and densired-hard.

Change:

1. Loaders become pure parsers: `loader(path) -> (data, colors | None)`,
   with no call to `_cached`. All the cached datasets return
   `test = distances = None` today, so nothing is lost. HDF5 datasets keep
   `_load_hdf5`, which returns `(train, test, distances)`. The two kinds are
   told apart by `cache_name`, as today.
2. `DatasetInfo.cache_name` defaults to the dataset name for every non-HDF5
   dataset. Existing cache names are kept (`pamap`, `biokdd`, `higgs`, ...),
   so that caches already on disk stay valid. No migration is needed: the
   file format does not change.
3. New `dense._ensure_cache(name) -> Path`: if `cache_name is None`, download
   and return `local_path(name)` (the HDF5). Otherwise return the parquet
   path; if it is missing, download the raw file and run the existing body
   of `_cached` (parse, write, delete raw) **without reading the cache back**.
   This avoids holding two copies of HIGGS in memory just to export it.
4. `load()` calls `_ensure_cache(name)`, then reads with
   `_read_parquet_cache` (for a cache) or the loader (for HDF5).
   `_cached` is removed or reduced to `_ensure_cache` + read.
5. Invariant, checked by a test: every registered dense dataset has either a
   `cache_name` or a raw file ending in `.hdf5`.

Notes and risks:

- `densired-hard` is generated rather than downloaded. Its `.npz` becomes a
  raw file that is deleted after caching (unless `KEEP_RAW`). Rebuilding the
  cache then needs the `densired` package again. The seed is fixed, so the
  result is the same. Acceptable.
- `ht` and `chem` extract files next to the download
  (`HT_Sensor_dataset.dat`, `ethylene_CO.txt`, ...), which `delete_raw` does
  not remove. Extract them into a temporary directory instead (small, but
  in scope, otherwise caching leaves junk behind).
- First-time users of these 17 datasets now pay the disk space of a parquet
  file. All of them are small except `yandex-t2i` (1M × 200 float32, about
  760 MB, compressed by zstd). In exchange, every later `load` is faster.
- CHANGELOG: under *Features*, "dense: cache every non-HDF5 dataset as
  parquet".

### 3.2 Derived caches (`higgs-highlevel`)

- New field `DatasetInfo.derived_from: str | None = None` and a
  `derive(data, colors) -> (data, colors)` function. For `higgs-highlevel`:
  `derived_from="higgs"`, `derive=lambda d, c: (d[:, -7:], c)`, and
  `cache_name="higgs-highlevel"`.
- `_ensure_cache` of a derived dataset: ensure the parent's cache, read it,
  apply `derive`, write the dataset's own cache. It never touches the raw
  file directly.
- `_raw_is_shared` and `prune_raw` skip derived datasets. Otherwise the 2.6 GB
  `HIGGS.csv.gz` would be kept forever, because `higgs-highlevel` would
  count as another user of it.
- `_cache_exists(name)` of a derived dataset is true if its own cache **or**
  its parent's exists. In the second case the raw file is not needed, so
  `load()` must not download it again.

### 3.3 Graph and time series

Both already cache everything as parquet. Split their `_cached(name)` the
same way: `_ensure_cache(name) -> Path` (download, parse, write, delete
raw) and the read. `cache_path(name)` already exists in both modules.

### 3.4 MTEB

New `dense._ensure_mteb(name) -> Path`, after `resolve_name`:
`_mteb.cache_dir(task, model)` if its `corpus.parquet` exists, else
`fetch_hosted(...)`, else raise the error of §2.5. Never call `encode_task`.
The model-revision check that `load` does on the cache is not repeated here:
export only exposes the files.

### 3.5 HDF5 conversion (`src/aida_data/_hdf5.py`)

New `ensure_hdf5(entry) -> Path`: converts the parquet cache of §3.1–3.4
into an HDF5 file and returns its path, or returns the native download for
ann-benchmarks/VIBE datasets.

**Location.** `<root>/dense/hdf5/<name>.hdf5`,
`<root>/dense/hdf5/mteb/<task>/<model_slug>.hdf5`,
`<root>/graphs/hdf5/<name>.hdf5`, and `<root>/timeseries/hdf5/<name>.hdf5`.
They are keyed by dataset name, not by cache name, so that `higgs` and
`higgs-highlevel` get separate files.

**Freshness.** A converted file is up to date when it exists, its root
attribute `aida_data_hdf5_version` equals `_HDF5_VERSION` (starting at 1),
and its attribute `source_mtime_ns` equals the `st_mtime_ns` of the parquet
cache it was made from (for MTEB, the one of `corpus.parquet`, which is
always written last). Otherwise it is rebuilt. A rebuilt or re-downloaded
cache therefore invalidates the converted file, without a separate index.

**Writing.**
- Atomic: write to `atomic_path(path, ".hdf5.tmp")`, then rename. Add
  `".hdf5.tmp"` to `_tmp.TEMPORARY_SUFFIXES`, and make each module's
  `prune_raw` also scan `hdf5/` recursively for stale temporaries.
- Streaming, so that memory stays bounded for HIGGS (11M × 28) and large
  MTEB corpora: read the parquet file with
  `pq.ParquetFile.iter_batches(batch_size=65536)`, build a row-major
  `(batch, d)` block with `np.column_stack`, and write it into the
  preallocated dataset with a slice assignment.
- Datasets are contiguous and uncompressed (as in ann-benchmarks). That way
  readers get one `H5Dread` and can memory-map the file, and need no filter
  plugins. The cost is disk space (see below).
- Integer dtypes are kept as in the parquet file (uint8/16/32 color codes,
  narrow neighbor ids). HDF5 readers convert them on read when they ask for
  `H5T_NATIVE_INT64`.
- Strings (ids, labels) are UTF-8 variable-length strings
  (`h5py.string_dtype()`). JSON is used only where a ragged structure has
  no natural HDF5 form (`color_labels`).

**Layouts.** Every file has the root attributes `aida_data_hdf5_version`,
`source_mtime_ns`, `module` and `name`.

Dense (ann-benchmarks compatible: tools reading `train`, `test` and
`attrs["distance"]` work unchanged):

| Object | Shape / type | Present |
|---|---|---|
| `train` | `(n, d)` float32 | always |
| `test` | `(q, d)` float32 | MTEB retrieval (the queries) |
| `colors` | `(n, c)` uint | colored datasets |
| `ids` | `(n,)` string | MTEB |
| `test_ids` | `(q,)` string | MTEB retrieval |
| `qrels/query`, `qrels/doc`, `qrels/score` | `(r,)` string, string, int32 | MTEB retrieval |
| attrs `distance`, `dimension` | string, int | always |
| attrs `color_names` | string array | colored datasets |
| attrs `color_labels` | JSON string, a list of lists | colored datasets |

There are no `neighbors` or `distances` datasets: aida-data does not compute
ground truth for parsed datasets. This is documented, so that ann-benchmarks
tooling doesn't fail in a confusing way.

Graph (CSR, the parquet adjacency list made flat):

| Object | Shape / type |
|---|---|
| `offsets` | `(n + 1,)` int64; the out-neighbors of `v` are `neighbors[offsets[v]:offsets[v+1]]`, sorted ascending |
| `neighbors` | `(m,)` uint32 or int64, as in the cache |
| `colors` | `(n,)` int, `-1` if missing |
| attrs `n_nodes`, `n_edges` | int |

Time series:

| Object | Shape / type |
|---|---|
| `values` | `(n, d)` float64, NaN for missing |
| `time` | `(n,)` int64, milliseconds since the Unix epoch, if the source has time stamps |
| attrs `dim_names` | string array, if the source has names |
| attrs `sampling`, `source` | string, if known |

**Disk space.** Converted files are uncompressed copies, so they can be much
larger than the zstd parquet caches. Examples: HIGGS about 1.2 GB,
`yandex-t2i` about 760 MB, and an MSMARCO Qwen3-4B corpus (8.8M × 2560
float32) about 90 GB. Mitigations:
- `export` logs the expected size before converting, and refuses when the
  filesystem has less than that plus 10% free (`shutil.disk_usage`), with a
  message that suggests `--format parquet`;
- `prune-hdf5` (§2.4) frees the space;
- `--format parquet` keeps working for anyone who doesn't need HDF5.

## 4. New modules

### 4.1 `src/aida_data/export.py`

```python
Module = Literal["dense", "graph", "timeseries"]
Format = Literal["hdf5", "parquet"]

@dataclass(frozen=True)
class Entry:
    name: str          # canonical: registered name, or full mteb: name
    module: Module
    native: Literal["hdf5", "parquet", "parquet-dir"]
    link_stem: str     # default name at the destination, without suffix (§1)

def resolve(name: str) -> Entry
def cached_path(entry: Entry, format: Format) -> Path   # no I/O
def is_cached(entry: Entry, format: Format) -> bool     # includes freshness for hdf5
def ensure(entry: Entry, format: Format = "hdf5") -> Path
def export(name: str, dest: Path | None = None, *, format: Format = "hdf5",
           mode="symlink", force=False) -> Path         # returns the created path
def entries(module: Module | None = None) -> list[Entry]  # for `list`
def prune_hdf5(dry_run: bool = True) -> list[Path]
```

`resolve`:

1. If `name` has the form `<module>/<rest>` and `<module>` is one of the
   three modules, look `<rest>` up only in that module. `mteb:` names
   contain `/` in the model (`Qwen/Qwen3-...`), so check the `mteb:` prefix
   first.
2. `mteb:` names and dense aliases go to `dense` (through `resolve_name`).
3. Otherwise look the name up in the registries of the three modules.
   Exactly one match is guaranteed by the uniqueness test (§5). If two
   modules ever match, raise with both qualified names.

`ensure(entry, "parquet")` calls the module's `_ensure_cache` or
`_ensure_mteb`, and raises for HDF5-only datasets. `ensure(entry, "hdf5")`
calls `_hdf5.ensure_hdf5`.

Imports of `graph` and `timeseries` are local to the functions, so
`aida-data export glove-100-angular` does not pay for parsing modules it
does not use.

### 4.2 `src/aida_data/cli.py`

`main(argv=None) -> int`: argparse with subparsers `export`, `path`, `list`,
`prune-hdf5` and `snippet`, plus the global `-q` and `--debug`. It calls `export.py`
and maps exceptions (`KeyError`, `FileExistsError`, `OSError`, `requests`
errors) to the messages of §2.5. `aida_data/__init__.py`'s `main` becomes
`sys.exit(cli.main())`. The `[project.scripts]` entry stays as it is. Add
`python -m aida_data` support with a two-line `__main__.py`.

### 4.3 `src/aida_data/_snippets.py`

```python
def cpp_snippet(entry: Entry, file: str | None = None) -> str
```

- One template per module (`dense`, `graph`, `timeseries`), kept as Python
  string constants rather than package data files. Then nothing depends on
  `uv_build` including non-Python files in the wheel.
- The templates are made of blocks, one per HDF5 object of §3.5, plus the
  native ann-benchmarks objects (`neighbors`, `distances`). The tailored
  mode keeps the blocks whose object is in the file and fills in the shape
  comments. The generic mode keeps them all and wraps the optional ones in
  `file.exist`. Both modes share the same block text, so they cannot drift
  apart.
- The objects of the HDF5 layouts are listed once, in `_hdf5.py` (name,
  module, required or optional, C++ type). `_snippets.py` and the
  conversion code both use that list, and a test checks that every object
  written by the conversion has a block, so a new object cannot be added
  without its snippet.

## 5. Tests (`tests/test_export.py`, `tests/test_hdf5.py`, `tests/test_snippets.py`, `tests/test_cli.py`)

All offline, with `DATASETS_DIR` patched to `tmp_path` and `_download`
replaced by a fake, as in `test_raw_cleanup.py`:

- names are unique across the three modules (the invariant of the flat
  namespace);
- every dense dataset has a `cache_name` or an `.hdf5` raw file (§3.1);
- `resolve`: plain name, `module/` prefix, alias, full `mteb:` name with `/`
  in the model, unknown name with suggestions, wrong module prefix;
- link path rules: no `DEST`, existing directory, symlink to a directory,
  explicit path, missing parent;
- conflicts: existing file without/with `--force`, broken symlink with
  `--force`, directory with `--force` (error), link onto the cache itself;
- modes: symlink is absolute and resolves to the cache; `--copy` gives an
  independent file (and directory, for MTEB parquet); `--hardlink` gives the
  same inode; `EXDEV` gives the `--copy` hint (monkeypatched `os.link`);
- atomicity: a failure in the middle leaves neither a temporary file nor a
  half-written destination;
- HDF5 conversion, one round-trip test per layout: a dense cache with colors
  and ids, an MTEB retrieval directory (`train`, `test`, `ids`, `test_ids`,
  qrels), a graph and a time series with `time`. Each is checked with
  `h5py` against what `load` / `load_edge_list` / `load_qrels` return for
  the same cache, with a `batch_size` smaller than the row count so that
  more than one batch is written;
- HDF5 freshness: a second `ensure` does not rewrite the file (mtime
  unchanged); touching the parquet cache, or changing
  `aida_data_hdf5_version`, makes it rebuild;
- the converted file of a dense dataset is not at `dense/<cache_name>.hdf5`,
  and `load` does not migrate or delete it;
- `--format parquet` on an HDF5-only dataset → error; `--format hdf5` on one
  → link to the native download, with no conversion;
- not enough disk space (monkeypatched `shutil.disk_usage`) → error that
  mentions `--format parquet`, and no temporary file is left;
- `prune_hdf5`: lists the converted files only, never native downloads or
  parquet caches;
- derived cache: `higgs-highlevel` is built from `higgs.parquet` without
  calling `_download`, has 7 feature columns, and `prune_raw` treats the
  raw HIGGS file as unused once `higgs.parquet` exists;
- one of the 17 newly cached datasets (`athlete`, a small CSV fixture):
  the first `load` writes the parquet and deletes the raw file, the second
  does not call the loader, and the result is identical;
- MTEB: cached directory → link (parquet) or converted file (hdf5); not
  cached and `fetch_hosted` returns False → error message, `encode_task`
  never called;
- snippets: for each module, the generic snippet, and the tailored snippet
  of a converted fixture file (dense with colors and ids, MTEB retrieval,
  graph, time series, and a native ann-benchmarks-like file), are compared
  with golden files in `tests/snippets/`. The tailored ones load exactly the
  objects of the file, and every object of the layout has a block. `--file`
  with `"` and `\` is escaped. `snippet` never calls `_download`, `ensure`
  or the conversion. A stale HDF5 file gives the generic snippet;
- compile-and-run test, marked `cpp` and skipped unless `h5c++` and the
  HighFive headers are found (`HIGHFIVE_INCLUDE` or the default include
  path): each golden snippet is compiled and run against its fixture file,
  and the shapes it prints are compared with those read by `h5py`. Register
  the marker next to `network` in `pyproject.toml`, deselected by default;
- CLI: `path` prints only the path on stdout; `list --cached` filters;
  exit codes 0, 1 and 2; `-q` silences the logs.

`@pytest.mark.network` tests: `aida-data export mnist-784-euclidean` into
`tmp_path`, then open the link with `h5py`; `aida-data export wine` (small
UCI download), then check that `train` and `colors` have the same number of
rows.

## 6. Documentation

- README: new section "Command line", with the examples of §2, and:
  - the HDF5 layouts of §3.5, and `aida-data snippet` as the way to get C++
    code that reads them (the example of §2.7, and how to install HighFive);
  - the parquet layouts, for `--format parquet`:
    - dense cache: columns `x0..x{d-1}` (float32), `color0..` (integer
      codes), optional `id`; color names and labels in the `aida_data`
      schema metadata;
    - graph cache: `nbrs` (list of neighbor ids), `color`;
    - time series cache: `x0..x{d-1}` (float64), optional `t`;
    - MTEB: `corpus.parquet` / `queries.parquet` / `qrels.parquet`, as in
      the `_mteb.py` docstring;
  - a note that symlinks break if the cache is deleted, rebuilt or pruned
    (`prune_raw` never removes caches, but `prune-hdf5` removes converted
    files); use `--copy` to avoid that;
  - the disk cost of HDF5 conversion, and `prune-hdf5`.
- CHANGELOG: "cli: `aida-data export`, `path`, `list`, `prune-hdf5` and
  `snippet`; HDF5 export of every dataset".
- Mark this plan as implemented, with a summary at the end, as for the
  previous plans.

## 7. Order of work (one commit each)

1. `refactor(dense)`: pure loaders + `_ensure_cache`, with no change in
   behaviour yet (the 8 datasets that are already cached). Existing tests
   must pass unchanged.
2. `feat(dense)`: cache the 17 other non-HDF5 datasets; `ht`/`chem`
   extraction into a temporary directory.
3. `feat(dense)`: derived cache for `higgs-highlevel`.
4. `refactor(graph, timeseries)`: `_ensure_cache`.
5. `feat`: `_hdf5.py` (conversion, freshness, layouts) + tests.
6. `feat`: `export.py` + tests.
7. `feat(cli)`: `cli.py`, `__main__.py`, `main` + CLI tests.
8. `feat(cli)`: `_snippets.py`, `snippet` subcommand, golden and `cpp`
   tests.
9. `docs`: README, CHANGELOG, plan marked as implemented.
