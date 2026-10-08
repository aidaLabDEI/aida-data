# Plan: parquet caches for the graph datasets

Goal: store the graph datasets (`graph.py`) locally as zstd-compressed
parquet, as `dense.py` does since 773381d / afb16af, and delete the raw TSVs
once the cache exists. This note compares an **edge list** layout with an
**adjacency list** layout (parquet list columns), recommends one, and lays out
the changes.

## 1. Where we start

- `graph.load_edge_list` downloads two TSVs per dataset into
  `datasets/graphs/` (`<name>.tsv`, `<name>_labels.tsv`), parses them with
  `pd.read_csv` on every call, then builds `colors` and normalizes the edges.
- The Sirius files are text and large: `combined.tsv` is ~72 MB, and the
  other big ones (`com-dblp`, `com-youtube`, `phy_citations`,
  `trivago-clicks`, `walmart-trips`) are of the same order. Parsing text with
  pandas is the dominant cost of each `load_edge_list`.
- `EdgeList` already exposes `edges` as an `(m, 2)` int64 array, `colors` as
  an `(n,)` int64 array.
- Only `abortion` is downloaded locally right now (10.1 MB of TSV), so the
  measurements below come from that one graph. **They must be repeated on the
  larger ones before committing to the layout** (see §6).

## 2. Candidate layouts

### A. Edge list

`<name>.edges.parquet` with two integer columns `u`, `v`, one row per edge,
sorted lexicographically (this is already what `_normalize_edges` returns).
Colors in a second file (see §4).

### B. Adjacency list

`<name>.adj.parquet` with one row per node `v = 0..n-1` and a single column
`nbrs: list<int32>`. Row index is the node id, so nodes with no edges cost an
empty list. Two variants for undirected graphs:

- **B1, symmetric**: both `u -> v` and `v -> u` stored (what algorithms that
  scan neighborhoods want, `2m` entries).
- **B2, upper**: only neighbors `w > v` stored (`m` entries), equivalent to
  edge list A in a CSR-like form.

Parquet stores the list column as a flat values column plus repetition/
definition levels, so the on-disk form is essentially CSR (values + offsets).

## 3. Measurements

Script (not committed): load `abortion` via `load_edge_list`
(n = 279,505, m = 639,827; 10.1 MB of TSV), write each layout with
`pq.write_table(..., compression="zstd", compression_level=3)`, read back.

| layout | size | % of TSV | write (s) |
|---|---|---|---|
| A. edges int64, dictionary on (pyarrow default) | 1,605 KB | 16% | 0.09 |
| A. edges int32, dictionary on | 2,238 KB | 22% | 0.05 |
| **A. edges int32, `DELTA_BINARY_PACKED`** | **933 KB** | **9%** | 0.03 |
| B1. adjacency symmetric, int32, default | 3,056 KB | 30% | 0.09 |
| B2. adjacency upper, int32, default | 2,156 KB | 21% | 0.06 |
| B2. adjacency upper, int32, `DELTA_BINARY_PACKED` | 977 KB | 10% | 0.04 |
| colors, int8 | 15 KB | 0.1% | 0.005 |

Read, warm cache: edges to `(m, 2)` int32 array 0.05 s; adjacency to
`(values, offsets)` 0.02 s (no stacking needed). Both are far below the
`read_csv` time for the same file.

What to take from this:

1. **The encoding matters more than the layout.** The default dictionary
   encoding is bad for edge columns (almost all values distinct); int32 with
   dictionary is even *larger* than int64, because the dictionary falls back
   to plain pages. Explicit `DELTA_BINARY_PACKED` is ~1.7-2.4x smaller.
   Sorted `u` has tiny deltas, `v` is sorted within each `u`, so deltas are
   small. Note this differs from dense, where dictionary encoding wins
   (`2026-10-08-local-file-format.md` §3.1) and `use_dictionary=False` is
   *not* wanted.
2. With delta encoding, A and B2 are **within 5% of each other** in size.
   B1 (symmetric) roughly doubles the entries and costs more, with the only
   benefit being not having to symmetrize at load time.
3. Adjacency read is ~2x faster and yields CSR directly. It is, however, a
   different in-memory representation from what `EdgeList` exposes today.

## 4. Recommendation: adjacency list (B2), one file

Revised after checking one-file storage. Because the adjacency layout has one
row per node, the colors fit in the same file as a `color` column, row-aligned
with `nbrs`. The edge list cannot do this (its rows are edges), so it needs a
second file. Measured on `abortion`, one file with `nbrs` (delta-encoded) +
`color` (int8): **991 KB**, vs 948 KB for the two edge-list files (+4%).
Read 0.03 s; edges are rebuilt exactly with
`np.repeat(np.arange(n), np.diff(offsets))` for `u` and `values` for `v`
(round trip checked bit-identical, colors included).

Reasons:

- **One file per dataset**: nothing to keep in sync, no row-count mismatch
  between edges and nodes, one tmp+rename for an atomic write, one file to
  delete or copy. Node-level attributes (colors, later others) become extra
  columns.
- Size is a wash against the edge list (§3, +4% with colors).
- Reads give CSR (`offsets`, `values`) directly, which is the cheapest form
  for neighborhood access; `EdgeList.edges` is derived with one `np.repeat`.
- Isolated nodes and the node count `n` are explicit (row count), instead of
  being inferred from the largest id.

Costs to accept:

- Directed graphs: `nbrs` holds out-neighbors; in-neighbors need a transpose
  at load time. Undirected is stored once as `u < v` (B2), so a symmetric CSR
  needs a symmetrization at load time.
- The list column is read through pyarrow (`.offsets` / `.values`); other
  tools (pandas, DuckDB, polars) read it as nested lists, less convenient
  than flat columns. The metadata JSON documents the layout.
- Must set the encoding by the nested column path
  (`"nbrs.list.element"`); verify it applies on the pinned pyarrow version
  in a test (compare file size to the plain-written file).

The edge list (A) remains the fallback if the nested column turns out
fragile; it is a 4% smaller, flatter format at the price of a second file.

### File layout

In `datasets/graphs/`, one file per dataset, keyed by dataset name (as in
dense `_cached`): `<name>.parquet`.

- Row `v` is node `v`, for `v = 0..n-1`.
- `nbrs: list<int32>` (int64 if `n > 2**31 - 1`): neighbors of `v`, sorted
  ascending. Undirected: only neighbors `w > v` (each edge once).
  `compression="zstd", compression_level=3, use_dictionary=["color"]`,
  `column_encoding={"nbrs.list.element": "DELTA_BINARY_PACKED"}`.
- `color: int8` (narrowest *signed* integer holding `colors.max()` and
  `-1`, unlike dense).
- Schema metadata key `aida_data`, JSON `{"version": 1, "directed": false,
  "n_nodes": n, "n_edges": m}`. The reader raises the same "not a cache in a
  known format" `ValueError` on version mismatch.
- Writes go to `*.parquet.tmp` then `replace`, as in dense.

## 5. What goes in the cache

`load_edge_list(name, directed, remap_colors)` has three parameters that
change the result. The cache must not depend on them. Proposal:

- Cache the **result of parsing, before `directed` handling**: edges (stored
  as adjacency, §4) with
  self loops removed and exact duplicate rows removed, orientation as in the
  source, lexicographically sorted. Colors as built by `_build_colors` with
  `remap_colors=False` (so `-1` for missing, original ids).
- At load time: `directed=False` applies `np.sort(axis=1)` + `_unique_rows`
  to the cached array (cheap relative to parsing text; ~ms for 10^6 edges);
  `remap_colors` applies the existing `np.unique` remap.
- Why not cache the canonical undirected form directly: `directed=True`
  would then be unrecoverable once the raw files are deleted. Open question:
  if nobody uses `directed=True`, caching the undirected form saves that
  per-load sort. Decide before implementing (see §7).

## 6. Implementation steps

- [x] 1. **Benchmark on the real data** first. Download all 12 datasets once
   (`combined`, `com-youtube`, `com-dblp` matter most), repeat §3 with:
   int64 vs int32, delta vs dictionary, B2 vs A, zstd level 3 vs 9, and
   `row_group_size` (default 1M rows is fine unless the files are tiny).
   Update §3 and §4 if the conclusion changes (e.g. very dense id spaces
   where dictionary might win).
   *Done on 9 of 12 graphs (`abortion`, `brexit`, `citeseer`, `com-dblp`,
   `com-youtube`, `combined`, `phy_citations`, `trivago-clicks`,
   `walmart-trips`; `twitter_pol`, `obamacare`, `uselections` not fetched).
   The conclusion holds: the chosen layout is 6-18% of the TSV size
   (`combined` 76 MB -> 4.7 MB, `com-youtube` 47 MB -> 5.8 MB), 2-7% larger
   than the two-file edge list with delta encoding (37% on the 11 KB
   `citeseer`), and 1.5-2.5x smaller
   than the int64/dictionary edge list. Dictionary encoding never won.
   Skipped `row_group_size`: the biggest file has 3M entries.*
- [x] 2. `graph.py`: add `_write_parquet_cache(path, edges, colors)` and
   `_read_parquet_cache(path)`, mirroring `dense.py`. Reuse
   `_CACHE_VERSION` naming, tmp+replace, metadata JSON. Start with copying
   the dense helpers rather than abstracting; extract a shared module
   (`_parquet.py`) only if a third user appears.
- [x] 3. `graph.py`: add `_cached(info, ...)` that, if the cache exists, reads it;
   otherwise downloads, runs `info.loader_function`, runs the parse-time
   cleaning of §5, writes the cache, then deletes the raw files
   (honoring the same `KEEP_RAW` env switch as dense; reuse its definition by
   importing it rather than defining a second one).
- [x] 4. `load_edge_list` returns from the cache path. As in dense, always read the
   file back after writing, so the first and later calls return identical
   arrays.
- [x] 5. Skip the download entirely if the cache exists (the `_download` calls
   currently run unconditionally; `_download` itself is a no-op when the file
   exists, but raw files are gone after cleanup, so they would be
   re-downloaded).
- [x] 6. `local_paths(name)`: keep returning the raw TSV paths (tests rely on it);
   add `cache_path(name)` for the parquet file.
- [x] 7. A `prune_raw`-like helper for already-downloaded raw TSVs, or reuse
   `dense.prune_raw` pattern: only delete raw files if a cache exists. No
   migration of old files is needed (no previous graph cache format).
- [x] 8. Tests (`tests/test_graph.py`, `tests/test_raw_cleanup.py` pattern):
   - round trip: write then read gives bit-identical `edges`/`colors`
     (including `-1` colors and isolated nodes at the end of the id range);
   - second `load_edge_list` call does not touch the loader (monkeypatch the
     loader to raise) and gives equal arrays;
   - raw files are deleted after a fresh parse, kept with `KEEP_RAW`, and
     left alone when the cache already existed;
   - `directed=True/False` and `remap_colors` give the same results as today
     from a cached dataset;
   - bad metadata version raises `ValueError`;
   - int64 fallback when `n > 2**31 - 1` (test the dtype-selection helper
     directly, do not build such a graph).
- [x] 9. README: one line on the graph cache (as the dense commit did), and the
   `AIDA_DATA_DIR` layout.

## 7. Open questions

Decisions taken during implementation:

1. `directed=True` is supported, so the cache keeps the source orientation
   (§5) and `directed=False` symmetrizes on load. Nobody was asked; revisit
   if no consumer uses `directed=True`.
2. In-neighbors / symmetric CSR: not added, no consumer known. `nbrs` holds
   out-neighbors in the source orientation.
3. One file (adjacency) kept; nested-column encoding verified by a test.
4. zstd level 3 kept: level 9 saves under 3% and writes 2-5x slower.

## 8. Summary

Graph datasets are now parsed once into `datasets/graphs/<name>.parquet`
(zstd level 3, adjacency list) and the raw TSVs are deleted afterwards.

- `src/aida_data/graph.py`
  - `_write_parquet_cache` / `_read_parquet_cache`: one row per node, `nbrs:
    list<int32>` (delta-encoded, int64 if `n > 2**31 - 1`) and `color` (the
    narrowest signed int), `aida_data` metadata `{version, n_nodes,
    n_edges}`, tmp + rename, `ValueError` on an unknown version. The reader
    rebuilds the `(m, 2)` edges with one `np.repeat`.
  - `_cached(name)`: skips the download when the cache exists; otherwise
    downloads, parses, cleans (self loops and duplicates removed, source
    orientation kept, sorted), writes, deletes the raw files (unless
    `KEEP_RAW`, imported from `dense`) and always reads the file back.
  - `load_edge_list` applies `directed` (`_to_undirected`) and
    `remap_colors` (`_remap_colors`) after reading, so the cache does not
    depend on them. `_build_colors` no longer remaps; `_normalize_edges` is
    split into `_clean_edges` (parse time) and `_to_undirected` (load time).
  - New `cache_path(name)` and `prune_raw(dry_run=True)`; `local_paths` is
    unchanged.
- `tests/test_graph.py`: fake downloads in the fixture; tests for the round
  trip (isolated nodes, `-1` colors, empty graph), schema and delta encoding,
  no re-parse/download on the second call, options on a cached dataset, raw
  deletion / `KEEP_RAW` / cache already present, `prune_raw`, bad metadata,
  dtype helpers. The network test restores the real downloader.
- `README.md`: a paragraph on the graph cache, `AIDA_DATA_KEEP_RAW` and
  `graph.prune_raw`.
- Checked on `abortion` that all four `directed`/`remap_colors` combinations
  return arrays identical to the previous implementation. Full suite: 85
  passed; `-m network` citeseer test passes.
- Deviations: the metadata has no `directed` key (the cache is before
  `directed` handling, §5); `m` for `abortion` in the cache is 670,501
  (source orientation) rather than 639,827 (undirected).
