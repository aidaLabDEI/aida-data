# Local file format for the dataset caches: HDF5 vs NPZ vs Parquet

Goal: `datasets/` takes 6.4 GB on disk. This note measures how much each
storage format would save, and what it would cost in parse (write) and load
(read) time. All numbers were measured on 2026-10-08 on the files in
`datasets/`.

## 1. What takes the space

There are three kinds of files in `datasets/`:

1. **Raw downloads** (zip, csv.gz, tar.gz, csv), about 4.2 GB. For the
   large datasets they are needed only once, to build the cache.
2. **Parse caches** written by `dense._cached` (`<name>.hdf5`), 2.27 GiB.
   They are contiguous, **uncompressed** HDF5: `X` as float32, `colors` as
   uint8 codes.
3. **Upstream HDF5** (ann-benchmarks / vibe, e.g.
   `fashion-mnist-784-euclidean.hdf5`). These are downloaded as they are,
   also uncompressed.

| dataset | raw download | current cache |
|---|---|---|
| higgs (+ higgs-highlevel) | `HIGGS.csv.gz` 2.8 GB | `higgs.hdf5` 1.2 GB |
| phones | `heterogeneity+activity+recognition.zip` 822 MB | `phones.hdf5` 209 MB |
| metropt3 | `metropt+3+dataset.zip` 218 MB | `metropt3.hdf5` 42 MB |
| census1990 | `us+census+data+1990.zip` 169 MB | `census1990.hdf5` 654 MB |
| biokdd | `data_kddcup04.tar.gz` 66 MB | `biokdd.hdf5` 43 MB |
| household-power | `...electric+power+consumption.zip` 21 MB | `household-power.hdf5` 58 MB |
| covertype | `covertype.zip` 11 MB | `covertype.hdf5` 126 MB |

For some datasets the uncompressed cache is several times larger than the
compressed raw file (census1990 is 4x, covertype is 11x).

## 2. Benchmark

Script: each existing `*.hdf5` in `datasets/` (the 7 parse caches and
fashion-mnist) was loaded into memory and written in each format. It was
then read back into numpy arrays and checked to be bit-identical to the
original (NaN-aware for household-power).

Formats:

- **hdf5 (current)**: `hfp["X"] = data`, contiguous, no filter.
- **hdf5 gzip-4 + shuffle / lzf + shuffle**: chunked (~1 MiB chunks of
  whole rows), built-in h5py filters, so no extra dependency.
- **npz**: `np.savez` and `np.savez_compressed` (deflate).
- **parquet** (pyarrow 25): one column per feature (`c0..c{d-1}`), one
  `.parquet` file per array, read back with
  `np.column_stack([c.to_numpy() for c in table.columns])`. Tried with snappy
  (the default), zstd level 3, and zstd 3 with `byte_stream_split` (which
  turns off dictionary encoding for float columns).

Read times are with a warm page cache. They measure decoding plus the
conversion to a numpy array, not disk speed (see §3.3).

### 2.1 Size (MiB, % of current)

| dataset | shape | hdf5 (current) | hdf5 gzip-4 | hdf5 lzf | npz | npz compressed | parquet snappy | **parquet zstd-3** | parquet zstd-3 + BSS |
|---|---|---|---|---|---|---|---|---|---|
| biokdd | 145,751x74 | 41 | 27 (65%) | 35 (84%) | 41 | 24 (58%) | 16 (39%) | **15 (37%)** | 25 (62%) |
| census1990 | 2,458,285x66 | 624 | 73 (12%) | 144 (23%) | 624 | 65 (10%) | 41 (7%) | **30 (5%)** | 56 (9%) |
| covertype | 581,012x54 | 120 | 12 (10%) | 16 (13%) | 120 | 13 (10%) | 7 (6%) | **6 (5%)** | 6 (5%) |
| fashion-mnist | 60,000x784 (+ test, gt) | 217 | **40 (19%)** | 53 (24%) | 217 | 48 (22%) | 53 (24%) | 42 (20%) | 52 (24%) |
| higgs | 11,000,000x28 | 1185 | 947 (80%) | 1065 (90%) | 1185 | 970 (82%) | 657 (55%) | **621 (52%)** | 880 (74%) |
| household-power | 2,075,259x7 | 55 | 21 (38%) | 30 (54%) | 55 | 15 (28%) | 10 (17%) | **9 (15%)** | 17 (30%) |
| metropt3 | 1,516,948x7 | 41 | 18 (43%) | 24 (58%) | 41 | 15 (37%) | 11 (28%) | **10 (25%)** | 15 (36%) |
| phones | 13,062,475x3 | 199 | 89 (45%) | 111 (56%) | 199 | 78 (39%) | 66 (33%) | **61 (30%)** | 91 (46%) |

### 2.2 Totals over the 8 files

| format | total size | vs current | write (s) | read (s) |
|---|---|---|---|---|
| hdf5 (current, uncompressed) | 2,483 MiB | 100% | 0.9 | 0.7 |
| hdf5 gzip-4 + shuffle | 1,227 MiB | 49% | 93.7 | 17.4 |
| hdf5 lzf + shuffle | 1,477 MiB | 59% | 21.0 | 12.1 |
| npz (`np.savez`) | 2,483 MiB | 100% | 2.0 | 2.2 |
| npz (`np.savez_compressed`) | 1,228 MiB | 49% | 207.1 | 14.0 |
| parquet snappy | 861 MiB | 35% | 21.3 | 6.6 |
| **parquet zstd-3** | **794 MiB** | **32%** | 23.3 | 7.0 |
| parquet zstd-3 + byte_stream_split | 1,142 MiB | 46% | 17.1 | 6.9 |

Largest file (higgs, 1.2 GB): read takes 0.3 s now, 3.5 s with parquet
zstd, 5.9 s with hdf5 lzf, 8.7 s with npz compressed and 10.8 s with hdf5
gzip.

## 3. Discussion

### 3.1 Why parquet compresses best

Most columns have few distinct values: census1990 is integer-coded
categories, covertype has 44 binary columns, and the sensor and CSV-derived
floats (higgs, phones, metropt3, household-power) have limited printed
precision. Parquet stores each column separately and encodes it with a
**dictionary** before compressing, so a column with few distinct values
becomes small integer indices. HDF5 and npz compress the row-major bytes as
one generic stream, so they never see that structure.

The same reasoning explains two other results:

- `byte_stream_split` makes parquet worse here. It turns off dictionary
  encoding for floats and only pays off on truly continuous floats.
- fashion-mnist (784 pixel columns, many of them near zero) is the only file
  where hdf5 gzip beats parquet, and only by a little. Its size is dominated
  by dense, high-entropy data that no format exploits much.

### 3.2 Format by format

**HDF5 + filter (gzip / lzf)**

- Smallest code change. Only the write in `_cached` becomes
  `create_dataset("X", data=..., chunks=..., compression="gzip", shuffle=True)`.
  `_cached`'s read side, `_load_hdf5`, and existing uncompressed caches keep
  working unchanged, because decompression is transparent in h5py.
- Keeps partial/sliced reads and the attrs used for the color names and
  labels.
- Saves only about 50% (gzip) or 40% (lzf). gzip is the slowest format to
  read. A faster codec (zstd/blosc) would need the `hdf5plugin` dependency,
  and the files would then not be readable by plain h5py.

**NPZ**

- Uncompressed `savez` saves nothing over HDF5.
- `savez_compressed` gives about the same size as hdf5 gzip, but it is the
  slowest to write (207 s in total, 113 s for higgs, because it uses
  single-threaded zlib on whole arrays) and cannot be read partially or
  memory-mapped.
- It has no attributes, so the color names and labels would need to be
  stored as an extra JSON string array.
- Not worth it. Its only advantage, needing no h5py, does not apply here.

**Parquet (zstd)**

- Best size: 32% of today's total, and 5% for census1990 and covertype.
  Reads are 2-3x faster than any other compressed option, and writes are as
  fast as lzf.
- `pyarrow` is already a dependency (it is used by pandas).
- Costs:
  - It is tabular, so each 2-D array has to become `d` named columns.
    Several arrays (train/test/neighbors/distances) need one file each, or
    a directory.
  - Color names and labels go in the schema metadata
    (`table.replace_schema_metadata`).
  - Building the numpy matrix with `column_stack` briefly needs about 2x
    the array's memory (about 2.4 GB peak for higgs).
  - `_cached` and its reader must be rewritten. Old `.hdf5` caches would be
    left orphaned unless they are deleted or converted.
- It is a natural fit for the colored tabular datasets (columns could even
  keep their real names). It fits less naturally for the ANN-style
  train/test/ground-truth files, which come from upstream as HDF5 anyway.

### 3.3 Caveats

- Read times are from a warm page cache on a local NVMe disk. On a cold
  cache or a slower or network filesystem, the uncompressed files take
  longer to read (1.2 GB for higgs), and the compressed formats get closer
  to them or overtake them.
- The upstream HDF5 downloads (ann-benchmarks, vibe) could be recompressed
  after download, but then the files no longer match upstream, and the code
  that detects whether a file was already downloaded (`is_file()`) would
  need care. This is not recommended.

## 4. Recommendation

1. **Largest saving, independent of the format**: the raw archives of the
   cached datasets (`HIGGS.csv.gz`, the phones zip, metropt, census1990,
   biokdd, household-power, covertype: about 4.1 GB) are never read again
   once the cache exists. Deleting them after a successful parse, for
   example with an opt-out `keep_raw` flag, saves more than any change of
   format. The cost is a re-download if the parser changes.
2. **For the caches, switch to parquet zstd-3** if a rewrite of `_cached`
   is acceptable: 2.48 GB becomes 0.79 GB, and loading the largest dataset
   takes about 3.5 s instead of 0.3 s.
3. **Cheap alternative**: add `compression="lzf", shuffle=True` (or
   `"gzip"`) to the HDF5 write in `_cached`. It is a one-line change, fully
   backward compatible, and gives 41-51% savings, at a higher read cost than
   parquet.
4. Skip NPZ.

With (1) + (2), the dense part of `datasets/` would go from about 6.4 GB to
about 1.1 GB: 0.75 GB of parquet caches, 0.23 GB of upstream HDF5
(fashion-mnist), and about 0.07 GB of small raw files and graphs.

## 5. Implementation plan

The plan implements recommendations (1) and (2) of §4 as two independent
parts:

- **Part A** switches the parse caches from HDF5 to parquet zstd-3.
- **Part B** deletes raw downloads once their cache exists.

Part A comes first, because Part B depends on knowing where a dataset's
cache is. Each part is one commit and leaves the test suite green.

Only the 8 datasets that go through `dense._cached` are in scope: `pamap2`,
`biokdd`, `metropt3`, `household-power`, `covertype`, `census1990`,
`phones`, `higgs` and `higgs-highlevel` (the last two share one cache, which
makes 9 names for 8 caches). The small datasets parsed on every load (`adult`,
`athlete`, `breast`, ...), the upstream HDF5 downloads (ann-benchmarks,
vibe), the ADBench `.npz` files and `aida_data.graph` do not change.

### Part A: parquet caches

#### A.1 Cache file layout

> **Status: done** (implemented in `_write_parquet_cache`).

One file per cache, `DATASETS_DIR / f"{cache_name}.parquet"`, holding one
table with one row per point:

| column | type | content |
|---|---|---|
| `x0` ... `x{d-1}` | float32 | the features, `X[:, j]` |
| `color0` ... `color{c-1}` | narrowest unsigned int (`np.min_scalar_type`, as today) | the color codes, `colors.values[:, j]` |

Colors share the table with the features because they always have the same
number of rows, so one file stays self-contained. Column names are positional
on purpose: the real color names can contain any character, and feature
names are not tracked today.

The schema metadata has one key, `b"aida_data"`. Its value is a JSON object:

```json
{"version": 1, "n_features": 28, "color_names": ["label"], "color_labels": [["0", "1"]]}
```

`color_names` and `color_labels` are `null` when the dataset has no colors.
`version` lets a later layout change be detected instead of misread.

Write options: `compression="zstd", compression_level=3`, with dictionary
encoding left at the default (it is the main source of the savings, see
§3.1) and no `byte_stream_split`. The default `row_group_size` (1Mi rows) is
fine. It gives 11 row groups for higgs, so pyarrow decodes them in parallel.

#### A.2 Rewrite `_cached` in `src/aida_data/dense.py`

> **Status: done.** The layout version is the module constant `_CACHE_VERSION`.

Split today's `_cached` (lines 82-124) into three helpers, and keep its
signature and return value unchanged so that no loader changes:

1. `_write_parquet_cache(path: Path, data: np.ndarray, colors: Colors | None)`
   - Build the table with
     `pa.table({f"x{j}": data[:, j] for j in range(d)} | {f"color{j}": codes[:, j] ...})`,
     with `codes = colors.values.astype(np.min_scalar_type(int(colors.values.max(initial=0))))`.
     This is the same narrowing as today, so the "widened back to int64"
     comment still applies.
   - Attach the metadata with `table.replace_schema_metadata({b"aida_data": json.dumps(...)})`.
   - Write to `path.with_suffix(".parquet.tmp")` and then `tmp.replace(path)`.
     This keeps the current guarantee that an interrupted parse leaves no
     truncated cache.
   - Import `pyarrow` and `pyarrow.parquet` at module top level, next to
     `h5py`. pyarrow is already a declared dependency in `pyproject.toml`.

2. `_read_parquet_cache(path: Path) -> tuple[np.ndarray, Colors | None]`
   - `table = pq.read_table(path)`, then parse the metadata and check
     `version == 1` (otherwise raise `ValueError` naming the file and telling
     the user to delete it).
   - Build `X` without the 2x peak memory of `column_stack` noted in §3.2:
     preallocate `data = np.empty((table.num_rows, d), dtype=np.float32)` and
     fill it one column at a time with
     `data[:, j] = table.column(f"x{j}").to_numpy()`. Only one column is
     duplicated at a time, and `data` is C-contiguous, which the callers and
     the `_unique_row_indices` hashing rely on.
   - Build the colors the same way, into an int64 `(n, c)` array, and return
     `Colors(values, tuple(color_names), tuple(map(tuple, color_labels)))`.
   - `del table` before returning, so that the Arrow buffers are freed.

3. `_read_hdf5_cache(path)`: today's read branch, moved unchanged. It is used
   only for migration (A.3).

New `_cached` flow:

```python
def _cached(path, cache_name, build):
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
    return _read_parquet_cache(cache)
```

The legacy HDF5 branch makes the switch free for existing users. The 2.3 GB
of HDF5 caches are converted the first time each dataset is loaded (about
13 s for higgs), and the raw files are not touched or needed. This branch
can be removed in a later release.

After a fresh parse, `_cached` reads the data back from the file instead of
returning the arrays from `build` directly. This keeps a single code path,
so the first and later calls return identical arrays (dtypes,
contiguity). It costs one extra read (about 3.5 s for higgs) only on the
first load.

`_load_hdf5`, which reads the upstream ann-benchmarks/vibe files, stays as
it is. `h5py` stays a dependency for it.

#### A.3 Docs to update

> **Status: done**, including the optional README line.

- `dense.py` module docstring (line 8): "cached as HDF5 next to the raw
  download" becomes "cached as zstd-compressed parquet".
- The `_cached` docstring: the file name and the layout from A.1 (in short).
- `README.md`: there is no mention of the cache format today, so nothing
  changes. Optionally add one line under "Dense datasets" saying that large
  datasets are parsed once into `<name>.parquet`.

#### A.4 Tests (`tests/test_colored.py`, `tests/test_dense_parsers.py`)

> **Status: done.** `test_higgs_and_highlevel_share_the_cache` also needed
> `higgs.hdf5` -> `higgs.parquet`. 51 tests pass.

Update the existing ones:

- `test_cached_parses_once`: `mycache.hdf5` becomes `mycache.parquet`, and
  `mycache.hdf5.tmp` becomes `mycache.parquet.tmp`.
- `test_pamap_parsed_from_zip_and_cached`: assert `pamap.parquet`.
- `test_biokdd`: assert `biokdd.parquet`.
- `test_cached_without_colors` needs no change. It already checks that a
  second call with `build=None` reads the cache.

Add new ones:

- **Round trip of types**: features come back float32 and C-contiguous,
  colors int64. Labels with non-ASCII characters, spaces and empty strings
  are preserved exactly.
- **NaN and inf are preserved** in the features. `household-power` relies on
  this, because `load` drops those rows only after the cache.
- **Wide color codes**: a color with more than 255 categories is stored as
  uint16 and read back correctly.
- **Migration**: write a legacy `.hdf5` cache by hand, in today's format
  (`X`, plus `colors` with `names`/`labels` JSON attrs). Then call `_cached`
  with a `build` that raises. Check that it returns the same arrays, creates
  `.parquet` and removes `.hdf5`.
- **Unknown version**: a parquet cache with `version: 2` in its metadata
  raises `ValueError`.

Run `uv run pytest`. The network tests (`-m network`) are not needed for
this part.

#### A.5 Check on the real data

> **Status: done**, see "Results of A.5" just below. `pamap2` was not
> checked: neither its raw file nor its cache is in `datasets/`.

1. Copy `datasets/` aside, or rely on the migration branch, which deletes
   each `.hdf5` only after the parquet file is written.
2. Load each cached dataset (`load(name)` for the 9 names listed above) and
   compare `dataset` and `colors` with the output of the same call on the
   commit before Part A. Save the arrays with `np.save` beforehand, and
   compare with `np.array_equal(..., equal_nan=True)`.
3. Check the sizes against §2.1: about 794 MiB in total for the 7 parse
   caches plus fashion-mnist, minus fashion-mnist (still HDF5), so about
   750 MiB. Check that `load("higgs")` takes about 3-4 s with a warm cache.
4. Write the measured numbers in a short "Results" note at the end of this
   file.

#### Results of A.5 (measured 2026-10-08)

The outputs of `load(name)` before Part A were saved with `np.save`. Then
the new code was run on the real `datasets/`, so each `.hdf5` cache went
through the migration branch. For all 8 names (`biokdd`, `metropt3`,
`household-power`, `covertype`, `census1990`, `phones`, `higgs`,
`higgs-highlevel`), `dataset`, `colors.values` (dtype included),
`colors.names` and `colors.labels` are identical to before
(`np.array_equal(..., equal_nan=True)`).

| cache | hdf5 (MiB) | parquet (MiB) | parquet read, warm (s) |
|---|---|---|---|
| biokdd | 41 | 15.2 | 0.14 |
| census1990 | 624 | 29.7 | 1.96 |
| covertype | 120 | 6.2 | 0.40 |
| higgs | 1185 | 620.8 | 3.25 |
| household-power | 55 | 8.6 | 0.11 |
| metropt3 | 41 | 10.0 | 0.08 |
| phones | 199 | 60.8 | 0.41 |
| **total** | **2,266** | **751.2** | |

The sizes match §2.1 (expected about 750 MiB). Reading the higgs cache
takes 3.25 s, as expected. Note that the full `load("higgs")` takes about
55 s, both before and after Part A: most of it is deduplication
(`_unique_row_indices`) on 11M rows, not the cache read. The one-time
migration (read HDF5, write parquet, read back) took about 68 s for higgs.

### Part B: delete raw downloads once cached

#### B.1 The problem to solve first

> **Status: done** (solved by B.2 and B.3).

`load()` calls `_download(url, local_name)` before the loader, and
`_download` checks only `destination.is_file()`. If the raw file is deleted,
the next `load("higgs")` downloads 2.8 GB again, even though the cache is
there. So `load` has to know, before downloading, whether the dataset has a
cache and whether that cache already exists.

#### B.2 Register the cache name

> **Status: done.** `pamap2` uses the cache name `"pamap"` (the name its
> loader already used), so there are 8 constants for 9 registrations. The
> raw file name is computed by a new helper `_raw_name(info)`, also used
> by `local_path`.

- Add a field `cache_name: str | None = None` to `DatasetInfo`. Because it
  has a default, the existing `register(DatasetInfo(...))` calls and the test
  fixtures that build `DatasetInfo` positionally keep working.
- Set it in the 9 registrations of the cached datasets. `higgs` and
  `higgs-highlevel` both get `"higgs"`.
- Add a constant for each name, for example `_HIGGS_CACHE = "higgs"`, and use
  it both in the registration and in the loader's `_cached(path, ..., build)`
  call. The two then cannot drift apart. This is the only change to the
  loaders.
- Add `_cache_exists(name) -> bool` to the module. It returns True if
  `info.cache_name` is set and `DATASETS_DIR / f"{cache_name}.parquet"`
  exists, or the legacy `.hdf5` exists (until A.2's migration branch is
  removed).

#### B.3 Skip the download in `load`

> **Status: done.**

In `load` (around line 1150) download only when needed:

```python
if not _cache_exists(name):
    _download(url, local_name)
```

The loader still receives `local_name`. `_cached` derives the cache path from
`path.parent` and never opens `path` when the cache exists, so a missing raw
file is fine. The loader docstrings say nothing about the raw file, so they
need no change.

#### B.4 Delete the raw file after a successful parse

> **Status: done.** The guard is `_raw_is_shared(path, cache_name)`. It
> compares file names (not full paths), so it also works when `_cached` is
> called with a path outside `DATASETS_DIR`.

- Add a module setting next to `DATASETS_DIR`, following the same
  environment-variable pattern:
  `KEEP_RAW = os.environ.get("AIDA_DATA_KEEP_RAW", "0") == "1"`.
- In `_cached`, in the "fresh parse" branch only and only after
  `tmp.replace(cache)` has succeeded, do
  `if not KEEP_RAW: path.unlink(missing_ok=True)`, and log the freed size at
  INFO level. Never delete in the migration branch, so that upgrading alone
  never removes user files.
- Raw files shared by two datasets: only HIGGS is shared, and both of its
  datasets use the same cache, so deleting after the first parse is safe.
  Add a comment saying so, because a future dataset that shares a raw file
  but not a cache would break this assumption.
  - Guard: before deleting, check that no *other* registered dataset with a
    different `cache_name` (or no cache) has the same `local_path`. If one
    does, keep the file.

#### B.5 Clean up existing raw files

> **Status: done**, including the README section.

Users who already have the caches still keep their raw files, because B.4
only acts on a new parse. Add `dense.prune_raw(dry_run: bool = True) -> list[Path]`:

- For each registered dataset with a `cache_name` whose cache exists, and
  whose `local_path` is a file not needed by another dataset (same check as
  in B.4), collect that file.
- With `dry_run=True` it only returns and logs the list and the total size.
  With `dry_run=False` it deletes the files.
- Document it in the README under "Dense datasets", with the
  `AIDA_DATA_KEEP_RAW` setting.

On the current `datasets/` it should list `HIGGS.csv.gz`,
`heterogeneity+activity+recognition.zip`, `metropt+3+dataset.zip`,
`us+census+data+1990.zip`, `data_kddcup04.tar.gz`,
`individual+household+electric+power+consumption.zip` and `covertype.zip`
(about 4.1 GB).

#### B.6 Tests

> **Status: done**, in the new file `tests/test_raw_cleanup.py`. 66 tests pass.

All tests monkeypatch `DATASETS_DIR` to `tmp_path`, as `tiny_datasets` does.
Register a small fake dataset with a `cache_name`, whose loader goes through
`_cached`.

- **Raw deleted after parse**: after the first `load`, the cache exists and
  the raw file is gone.
- **`KEEP_RAW`**: with `monkeypatch.setattr(dense, "KEEP_RAW", True)` the raw
  file is kept.
- **No re-download**: after the first load, monkeypatch `dense._download` to
  raise, and check that a second `load` succeeds.
- **Migration keeps raw**: with a legacy `.hdf5` and a raw file present,
  `load` converts the cache and keeps the raw file.
- **Shared raw file kept**: register two datasets with the same URL but
  different `cache_name`s. After loading one of them the raw file still
  exists.
- **Registered names match the loaders**: for each of the 9 real cached
  datasets, write a tiny parquet cache under its `cache_name` into
  `tmp_path`, monkeypatch `_download` to raise, and check that `load(name)`
  returns that data. This catches any mismatch between `DatasetInfo.cache_name`
  and the name passed to `_cached`.
- **`prune_raw`**: the dry run lists the right files and deletes nothing.
  `dry_run=False` deletes exactly those files.

#### B.7 Check on the real data

> **Status: done.**
>
> 1. The dry run listed exactly the 7 files of B.5, 3,932 MiB in total.
> 2. After `prune_raw(dry_run=False)`, `du -sh datasets` gives 1.1 GB
>    (4.9 GB before the prune, 6.4 GB before Part A).
> 3. With `dense._download` replaced by a function that raises (this
>    simulates a disconnected network), all 8 cached datasets that are
>    present load without downloading. `pamap2`, which is not cached
>    locally, does try to download, as it should.

1. `prune_raw()` (dry run): the list matches B.5.
2. `prune_raw(dry_run=False)`, then `du -sh datasets`: it should be about
   1.1 GB (§4).
3. With the network disconnected, `load` each of the 9 cached datasets:
   none should try to download.

## 6. Summary

Both parts of §5 are implemented, each in one commit, and the test suite
passes (66 tests, up from 46).

- **Part A** (`773381d`, *store parse caches as zstd-compressed parquet*):
  `_cached` now writes `<cache_name>.parquet` (zstd-3, one column per
  feature and per color code, color names and labels in the `aida_data`
  schema metadata, with a layout `version`). It is split into
  `_write_parquet_cache`, `_read_parquet_cache` and `_read_hdf5_cache`.
  Existing `.hdf5` caches are converted on first load, without the raw
  file. On the real data the outputs of `load` are bit-identical to before,
  the 7 caches went from 2,266 MiB to 751 MiB, and reading the higgs cache
  takes 3.25 s instead of 0.3 s.
- **Part B** (`50b4a2c`, *delete raw downloads once their cache exists*):
  `DatasetInfo.cache_name` registers each cache, `load` skips the download
  when the cache exists, `_cached` deletes the raw file after a fresh parse
  (opt-out `AIDA_DATA_KEEP_RAW=1`, kept when another dataset with a
  different cache uses it, and never deleted in the migration branch), and
  `prune_raw` cleans up the existing raw files. On the real data it
  removed 3.9 GiB.
- **Result**: `datasets/` went from 6.4 GB to 1.1 GB, as predicted in §4.

Deviations from the plan and things to know:

- `pamap2`'s cache is called `pamap`, the name its loader already used, so
  there are 8 cache-name constants for the 9 cached datasets. `pamap2` was
  not checked on real data, because it is not in `datasets/`.
- The full `load("higgs")` still takes about 55 s, both before and after
  this change. Almost all of it is deduplication of 11M rows
  (`_unique_row_indices`), not the cache read. If load time matters, that
  is the next thing to look at, not the file format.
- The raw files on this machine are now deleted. If a parser changes, the
  dataset must be downloaded again (2.8 GB for HIGGS), after deleting its
  `.parquet` cache.
- The legacy HDF5 branch in `_cached` (and `_read_hdf5_cache`, and the
  `.hdf5` check in `_cache_exists`) can be removed in a later release.
- This note itself (`src/md/2026-10-08-local-file-format.md`) is still
  untracked and was not included in the commits.
