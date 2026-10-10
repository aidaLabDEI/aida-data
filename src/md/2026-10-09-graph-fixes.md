# Plan: 64-bit adjacency offsets for very large graphs

Follow-up to `2026-10-08-local-graph-storage.md`, from the code review of
`c8c3f90..6e3d787` (finding 2). Nothing here is implemented yet.

Goal: a graph with 2^31 or more edges is cached correctly or fails loudly,
instead of being cached with wrong edges and no error.

## 1. The problem

`_write_parquet_cache` (`src/aida_data/graph.py:264-269`) builds the
adjacency list like this:

```python
counts = np.bincount(edges[:, 0], minlength=n)
offsets = np.zeros(n + 1, dtype=np.int32)
np.cumsum(counts, out=offsets[1:])
nbrs = pa.ListArray.from_arrays(pa.array(offsets), ...)
```

`cumsum` with an `int32` `out=` casts without any error. Checked:
`np.cumsum([2**31, 1], out=int32_array)` gives `[-2147483648, -2147483647]`.
With m ≥ 2^31 edges after cleaning, the offsets wrap to negative values.
`pa.ListArray.from_arrays` then either raises a confusing error or builds a
list array that points at the wrong neighbors. In the second case the cache
is written and every later `load_edge_list` returns wrong edges.

None of the registered graphs comes close (the largest are around 10^7
edges), so this only matters for `register()`ed datasets. But the failure is
silent and permanent (the raw file is deleted after caching), so it is worth
closing. Node ids already handle this case (`_index_dtype` switches to int64
above 2^31 − 1); the offsets do not.

## 2. Design

- Compute the offsets in int64 always (`np.zeros(n + 1, dtype=np.int64)`),
  so the `cumsum` cannot wrap. The extra memory is 4 bytes per node, only
  while writing.
- Choose the arrow list type from the edge count:
  - `len(edges) <= 2**31 - 1`: `pa.ListArray.from_arrays(pa.array(offsets,
    pa.int32()), ...)`. The cache file is unchanged for every graph that
    works today, so no `_CACHE_VERSION` bump and no re-parse.
  - otherwise: `pa.LargeListArray.from_arrays(pa.array(offsets), ...)`.
  Put the threshold in a module constant, `_MAX_LIST_OFFSET = 2**31 - 1`,
  so the test can lower it.
- Checked with the locked pyarrow: a `large_list<int32>` column written with
  `column_encoding={"nbrs.list.element": "DELTA_BINARY_PACKED"}` reads back
  as `large_list` with int64 offsets. The parquet column path is the same,
  so the `write_table` call does not change.
- `_read_parquet_cache` needs no change for correctness: `nbrs.offsets`,
  `np.diff` and `flatten()` work the same on both list types, and the code
  already builds int64 arrays. While touching it, flatten once
  (`values = nbrs.flatten()`) instead of twice. This is review cleanup item
  8, and it matters most for exactly these large graphs.
- Update the `_write_parquet_cache` docstring: offsets are int32, or int64
  (`large_list`) for 2^31 or more edges.

## 3. Implementation steps

1. [ ] **Tests first** in `tests/test_graph.py`:
   - with `_MAX_LIST_OFFSET` monkeypatched to e.g. 3, a small graph with
     more edges round-trips bit-identically through `_write_parquet_cache`
     / `_read_parquet_cache`, and the stored column type
     (`pq.read_schema(path)`) is `large_list`;
   - without the patch, the same graph is stored as `list` (the file format
     of existing caches is unchanged);
   - a unit check that the offsets are computed without wrapping: give a
     helper that builds the offsets counts whose sum exceeds 2^31 (e.g.
     `counts = np.array([2**31, 1])`) and check that the last offset is
     `2**31 + 1`. This means pulling the offsets construction out into a
     small `_adjacency_offsets(counts)` function, which also keeps the test
     cheap (no 16 GB edge array).
2. [ ] **Fix `_write_parquet_cache`** as in §2, and flatten once in
   `_read_parquet_cache`.
3. [ ] Run `tests/test_graph.py` and the whole suite, then load a couple of
   real cached graphs (`abortion`, `com-dblp`) to confirm that existing
   caches still read without a re-parse.

## 4. Out of scope

- Negative node ids (review finding 3): `np.bincount` raises on them. That
  belongs in input validation in `_cached`, which is a separate change.
- Memory: a 2^31-edge graph needs about 32 GB for the int64 `(m, 2)` edge
  array alone, so in practice such a graph may not load at all on a typical
  machine. This plan only makes sure that the cache is never silently wrong.
