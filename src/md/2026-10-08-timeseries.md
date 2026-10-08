# Plan: `aida_data.timeseries`, time series stored as parquet

Goal: priority 3 of `2026-10-08-software-todo.md`. Add a `timeseries` module
for the ATTIMO (§3.1 of the todo) and MOMENTI (§3.2) datasets. As in `dense`
and `graph`, each dataset is parsed once into a zstd-compressed parquet file
under `datasets/`, the raw download is deleted afterwards, and later calls
only read the parquet file.

The layout in §3 was a proposal and step 1 checked it on real data, as the
graph plan did; the numbers are in §7 and the layout below already reflects
them.

## 1. Where we start

- `dense.py` and `graph.py` share a pattern: a registry of `DatasetInfo`, a
  `_cached` function (download if no cache, parse, write `*.parquet.tmp`,
  rename, delete raw unless `KEEP_RAW`, always read back), an `aida_data`
  schema-metadata JSON with a `version` key, `cache_path`, `local_paths`,
  `prune_raw`.
- `_download.download` is shared. `KEEP_RAW` is defined in `dense` and
  imported by `graph`.
- The graph plan said to extract a shared helper module "only if a third user
  appears". `timeseries` is the third user (§5, step 2).
- A time series is a single long sequence, not a collection of points: one
  dataset is one `(n, d)` array, `d = 1` for ATTIMO and `d >= 2` for MOMENTI.
  Row order is the data, so none of the cleaning that `dense.load` does
  (dedup, dropping non-finite rows) is allowed.

## 2. Datasets, in the order we add them

From todo §3. "Verify" means the source or format was read from the
repositories' scripts, not checked by downloading it.

| Phase | Dataset | Source | Format to parse | Notes |
|---|---|---|---|---|
| 1 | ASTRO, ECG, freezer, GAP | figshare article 20747617, files 36982360, 36982384, 36982390, 36982396 | gzipped text, one value per line | Mapping verified against `pyattimo.load_dataset`. Use `ndownloader.figshare.com`: `figshare.com/ndownloader` answers 202 to a WAF challenge. CC BY 4.0. |
| 1 | HumanY | same article, file 36982399 | same | 31 MB gz, 26.4M values. |
| 1 | steamgen | zenodo 4273921, `STUMPY_Basics_steamgen.csv` | CSV with header, 4 dims | CC BY 4.0. Shared by ATTIMO and MOMENTI: one dataset, one cache. |
| 2 | Motiflets: Arrhythmia, penguin, dishwasher, EEG-sleep (npo141) | `patrickzib/motiflets` `datasets/...` | verify (csv/npy/arff) | Pin to a commit hash, not the `pyattimo` branch. |
| 2 | MOMP `*.mat` | `patrickzib/motiflets` `datasets/momp/{name}.mat` | MATLAB | Check for v7.3 (HDF5, `h5py` is already a dependency) vs older (needs `scipy.io`). |
| 2 | FOETAL_ECG, evaporator, RUTH | MOMENTI repo `Datasets/` | CSV/text | Check the DaISy origin and licence of the first two, and the origin of RUTH. |
| 2 | oikolab weather | MOMENTI repo / Monash archive | `.tsf` | Write a small `.tsf` parser (header + `attribute:values` lines); MOMENTI's `data_loader.py` is a reference. |
| 3 | FL010 | PhysioNet record | WFDB | Needs `wfdb`, as an optional dependency (like `shuttle`). |
| 3 | quake (IU.ANMO) | ObsPy fetch | ObsPy stream | Needs `obspy`, optional. Check that the fetch is reproducible. |
| 3 | CLEAN_House1 (REFIT), whales (NOAA SanctSound) | not in repos | -- | Whales needs a pipeline built from audio clips. Both may stay out of scope; decide after phase 2. |
| -- | PAMAP time-series view | -- | -- | `dense.pamap2` treats it as points. Do not add a second copy; see §6. |
| -- | ecg-heartbeat-av, insect_b, case1, insect15, synthetic `synth-w100-*`, Whales/VCAB noised | -- | -- | Skip: tiny examples (low priority) or unpublished. Synthetic ones can be generators later. |

Phase 1 alone covers the ATTIMO benchmark datasets that have stable URLs and
exercises the whole pipeline. Phases 2 and 3 only add loaders.

## 3. Proposed storage layout

One file per dataset, `datasets/timeseries/<name>.parquet` (same
directory-per-module convention as `datasets/graphs/`).

- One row per time step, in order. Float columns `x0..x{d-1}`, one per
  dimension, named positionally like dense. Dimension names (for example
  steamgen's sensor names) go in the metadata.
- Univariate and multivariate use the same layout.
- Float dtype: **float64**, unlike dense (float32). Motif discovery compares
  subsequence distances, and step 1 showed float32 is lossy on 4 of the 5
  series measured (§7).
- Time stamps: most sources have none. If a source has them (oikolab,
  possibly steamgen), store a `t` column (`timestamp[ms]` or int64, see §7)
  only for those datasets, and keep the metadata `has_time` in sync. Not
  needed for phase 1.
- Missing values stay as NaN. Rows are never dropped, because that would
  shift the time axis.
- Schema metadata key `aida_data`, JSON
  `{"version": 1, "n_samples": n, "n_dims": d, "dim_names": [...] | null,
  "has_time": false, "sampling": null | "...", "source": "<url>"}`. Reader
  raises the same "not a cache in a known format" `ValueError` on a version
  mismatch.
- Writes go to `*.parquet.tmp` and are renamed, as in the other modules.
- Encoding: `compression="zstd", compression_level=3` and the pyarrow default
  encoding (dictionary, falling back to plain when there are too many
  distinct values). Step 1 chose it over `BYTE_STREAM_SPLIT` and `PLAIN`.

## 4. Public API

```python
from aida_data import timeseries

timeseries.available_datasets()
ts = timeseries.load("ecg")      # TimeSeries
ts.values                        # (n, d) float64
ts.dim_names                     # tuple[str, ...] | None
ts.time                          # (n,) array or None
```

- `TimeSeries(name, values, dim_names=None, time=None)`, frozen dataclass
  like `EdgeList`. `values` is always 2-D, so that univariate and
  multivariate code share one type. Add `ts.univariate` returning the 1-D
  view for `d == 1` (and raising otherwise), since ATTIMO wants 1-D arrays.
- `local_paths(name)` returns the raw download(s); `cache_path(name)` the
  parquet file; `prune_raw(dry_run=True)` as in `graph`.
- No `pipeline`, `deduplicate` or normalization arguments in the first
  version. z-normalization is done per subsequence by the algorithms, not per
  series.

## 5. Implementation steps

1. [x] **Benchmark on real data first**, phase 1 files plus one multivariate
   (steamgen). Download them once and compare: float64 vs float32;
   dictionary vs `BYTE_STREAM_SPLIT` vs plain; zstd 3 vs 9; read time vs
   parsing the gzipped text. Check round trip is bit-identical in float64.
   Record the numbers in §7 and fix §3 if they change the conclusion
   (for example if quantized data makes dictionary clearly better).
2. [x] **Extract the shared cache helpers.** Create `_cache.py` with: atomic
   write (`*.tmp` then rename), the `aida_data` metadata read with version
   check and its `ValueError`, and the `_delete_raw` helper that `graph` and
   `dense` each reimplement. Keep it small and do not change any on-disk
   format. Move `KEEP_RAW` there and re-export it from `dense`. Do this in
   its own commit with the existing 85 tests passing before any timeseries
   code lands. If it turns out to touch too much, skip it and copy the helpers
   as the graph module did, but then say so in the commit message.
3. [x] `timeseries.py`: `DatasetInfo(name, url, loader_function, ...)` plus
   `register`, `available_datasets`, `local_paths`, `cache_path`. The loader
   gets the raw path and returns `(values, dim_names, time)`. Datasets with
   several files (if any) take a tuple of URLs like `graph.DatasetInfo`.
4. [x] `_write_parquet_cache` / `_read_parquet_cache`: layout of §3, metadata,
   tmp + rename, version check. The reader fills a preallocated `(n, d)`
   array column by column, as dense does, to avoid doubling memory on large
   multivariate series.
5. [x] `_cached(name)`: skip the download when the cache exists; else download,
   parse, write, delete raw unless `KEEP_RAW`, always read back from the
   file. Make sure a raw file used by two datasets (if one appears) is not
   deleted early (`dense._raw_is_shared`); with steamgen as one dataset this
   should not occur in phase 1.
6. [x] `load(name)` returns `TimeSeries`; unknown names raise `KeyError` with the
   list of available datasets, as in the other modules.
7. [x] Phase 1 registrations and loaders: the figshare gzipped-text parser
   (`np.loadtxt` is too slow for HumanY; use `pd.read_csv` with
   `header=None` or `np.fromstring`/`np.genfromtxt` on the decompressed
   buffer, pick after measuring) and the steamgen CSV. Pin each URL, and for
   GitHub raw URLs use a commit hash.
   *Done; `np.loadtxt` won the measurement (§7), so it is used.*
8. [x] `prune_raw(dry_run=True)` for raw files left over (pattern from `graph`).
9. [x] Tests in `tests/test_timeseries.py`, with fake downloads as in the
   `test_graph.py` fixture, no network by default:
   - parser tests on tiny inline files (text one-value-per-line, CSV with
     named columns, NaNs kept in place);
   - round trip bit-identical, including `d = 1`, `d > 1`, NaN and `time`;
   - second `load` does not call the loader or the downloader;
   - raw deleted after a fresh parse, kept with `KEEP_RAW`, untouched when
     the cache already existed;
   - bad metadata version raises `ValueError`;
   - `univariate` raises for `d > 1`;
   - one `@pytest.mark.network` test per phase-1 source, checking shape and
     that the file is gone after loading.
10. [x] README: a collapsed section like the graph one (datasets, the example
    above, the cache path under `AIDA_DATA_DIR`, `KEEP_RAW`, `prune_raw`).
11. [x] Phase 2 loaders and registrations, one commit per source type (`.mat`,
    `.tsf`, plain CSV), each with a parser test on a tiny fixture.
    *Done for plain text tables and `.tsf`. No `.mat` loader: the files do not
    exist as the plan assumed (§8).*
12. [x] Phase 3: add the optional extras (`wfdb`, `obspy`) to
    `[project.optional-dependencies]`, as is done for `shuttle`, and import
    them lazily inside the loader with a clear error message. Decide whether
    CLEAN_House1 and whales are in scope.
    *Done for `wfdb` (FL010). `obspy` not added: quake not implemented. Out
    of scope: CLEAN_House1, whales, quake (§7, §8).*
13. [x] Mark priority 3 as done in `2026-10-08-software-todo.md`, listing what is
    still missing, as was done for priority 1. Fill in §7 and §8 of this
    note.

Commit granularity: step 2 alone, steps 3-10 as `feat(timeseries): ...` with
phase 1, then one commit per later phase.

## 6. Risks and things to check

- **Source stability.** figshare and zenodo are stable; GitHub raw URLs on
  branches (`pyattimo`, `pyattimo_refactor`) are not. Pin commits. Some
  files exist only inside the lab repositories; if a repo file moves, the
  loader breaks. Consider hosting a copy.
- **Licences.** DaISy, REFIT, Monash, PhysioNet and NOAA have different
  terms. Record the source and licence in each registration; do not
  redistribute from this repository.
- **Memory.** HumanY and the larger MOMENTI series are tens of MB as text but
  not a problem as arrays; the point is to not hold text plus array plus
  table at once while parsing. Measure peak memory on HumanY in step 1.
- **Overlap with dense.** `pamap2` exists in `dense` as points. If MOMENTI or
  ATTIMO need the ordered per-subject series, add a loader that reads the same
  raw files; do not convert the dense cache, whose row order and dedup may
  differ. Check this before registering PAMAP.
- **Shape convention.** MOMENTI may expect `(d, n)` and ATTIMO `(n,)`. The API
  gives `(n, d)` plus `univariate`; transposing is the caller's job. Confirm
  with both libraries before fixing the API.
- **`.tsf` generality.** Monash `.tsf` files can hold many series of
  different lengths. oikolab is a single multivariate series; a general
  parser is out of scope.

## 7. Open questions

To decide while implementing, recording the answer here:

1. float64 vs float32 for the cache. **float64.** Measured in step 1 on
   ASTRO, ECG, freezer, GAP, steamgen, HumanY (float64 round trip bit-identical
   in all cases, checked with `np.array_equal`):

   | series | n | distinct | float32 exact? | dict z3 | bss z3 | plain z3 | dict z9 |
   |---|---|---|---|---|---|---|---|
   | steamgen (d=4) | 9.6k | all | no | 0.33 MiB | 0.25 | 0.27 | 0.32 |
   | freezer | 7.4M | 2160 | yes | 2.97 | 2.51 | 3.07 | 2.79 |
   | ECG | 7.8M | 3984 | no | 3.58 | 13.56 | 4.91 | 3.41 |
   | GAP | 2.0M | 4186 | no | 2.52 | 10.67 | 3.51 | 2.43 |
   | ASTRO | 1.2M | all | no | 8.89 | 7.75 | 8.45 | 8.88 |
   | HumanY | 26.4M | -- | -- | 40.6 | 101.6 | 42.4 | -- |

   Sizes are float64. Raw gz sizes: 0.5 (csv), 2.3, 3.3, 2.8, 8.1, 29.7 MiB, so
   the parquet files are about the size of the gz text. `BYTE_STREAM_SPLIT` is
   up to 4x worse on quantized data (ECG, GAP, HumanY) and only 13% better on
   high-entropy data (ASTRO), so the default dictionary encoding is kept.
   zstd 9 gains at most 6% and writes 2-5x slower: level 3 stays. Reading the
   parquet file takes 0.02-0.08 s (HumanY 0.23 s), against 0.25-0.8 s (HumanY
   3-4 s) to parse the text.

   Parsers (gz text, one value per line), HumanY: `np.loadtxt` 2.9 s and
   305 MiB peak; `pd.read_csv` 3.3 s, 505 MiB; `np.fromstring` 4.4 s, 585 MiB;
   `pyarrow.csv` 0.9 s, 596 MiB. All give identical values. Contrary to the
   assumption of step 7, `np.loadtxt` is fine on numpy 2 and uses the least
   memory, so it is used; `pyarrow.csv` is the option if parsing speed ever
   matters.
2. Time column type and whether to keep it at all for phase 1. **Kept as an
   optional `t` column of type `timestamp[ms]`**, read back as `datetime64[ms]`.
   No phase 1 series has time stamps, so it is only exercised by tests until
   oikolab (phase 2); millisecond resolution is enough for its hourly data.
3. Whether the shared `_cache.py` extraction (step 2) is worth it. **Yes**: it
   was small (`write_table`, `read_table`, `delete_raw`, `KEEP_RAW`), removed
   more lines than it added in `dense` and `graph`, and the 85 existing tests
   passed unchanged. Each module still reads its own `KEEP_RAW` global, which
   is what the tests monkeypatch.
4. Whether CLEAN_House1 and whales are in scope. **No.** CLEAN_House1 is not
   in any repository, and whales needs a pipeline from NOAA audio clips with
   nothing to check it against. `quake` is deferred for the same reason (§8).
5. Whether to add generators for the synthetic ATTIMO datasets. **Not now**:
   they are not part of the published datasets and nobody asked for them.


## 8. Findings while implementing

- **ECG has blank lines.** The file uses CRLF line endings and contains 46991
  blank lines (7871870 lines, 7824879 values), in runs such as lines 1249-1252.
  No other phase 1 file has any. `np.loadtxt` (which `pyattimo.load_dataset`
  uses) skips them, so ATTIMO's ECG is the 7824879-value series; the loader
  does the same, so results match the library. This breaks the "never drop
  rows" rule of §3 for this one file, on purpose: turning the gaps into NaN
  would make every subsequence overlapping a gap non-finite. It is documented
  in the README. Revisit if the gaps turn out to be meaningful.
- **figshare.** The URLs `figshare.com/ndownloader/files/<id>` in `pyattimo`
  answer HTTP 202 with an empty body (AWS WAF challenge) to `requests` and
  `curl`; `ndownloader.figshare.com/files/<id>` redirects to S3 and works. The
  ids belong to article 20747617, not a collection. File md5s from the figshare
  API match the downloads. The article also holds `Whales.txt.gz` (3.1 GB) and
  `VCAB_noised.txt.gz` (9.2 GB), which are not registered.
- **Phase 2 sources.**
  - The Motiflets `pyattimo` branch has `datasets/original/{dishwasher.txt,
    npo141.csv,...}` and `datasets/experiments/arrhythmia_subject231_channel0.csv`
    (header line `"Channel 0"`); all are one value per line. `EEG.csv` there
    also has blank lines (543893 lines, 539922 values); it is not registered
    because nothing asks for it. Files are pinned to commit `8afb3f9`.
  - `penguin.txt` (78 MB) has 9 tab-separated columns and nothing says which
    ones the papers use, so it is not registered.
  - The `.mat` files (`PeVAMmotif/`) are MATLAB 5 files (`scipy.io` would do,
    not `h5py`), but they hold insect recordings (a variety name and labels),
    not the MOMP series of the plan, and `datasets/momp/` does not exist on the
    `pyattimo` branch. No `.mat` loader was written, since there is nothing to
    test it against.
  - MOMENTI's own scripts read the headerless `.dat`/`RUTH.csv` files with
    `pd.read_csv`, so the first row is taken as a header and lost, and they
    drop column 0, which is the time index of FOETAL_ECG but a real channel of
    `evaporator` (6 channels, no index). Here every row and column is kept
    (FOETAL_ECG: 2500x8, the time column dropped, `sampling="250 Hz"`;
    evaporator: 6305x6; RUTH: 14859x32). MOMENTI also smooths oikolab with a
    Savitzky-Golay filter and adds noise to RUTH in some tests; that is
    preprocessing, left to the caller.
  - oikolab: 8 series x 100057 hours from 2010-01-01T00:00, no missing values.
    The `.tsf` parser handles one series per dimension only; dimension names
    come from the `type` attribute and `time` from `start_timestamp` plus
    `@frequency`.
  - Licences: figshare, zenodo steamgen and oikolab are CC BY 4.0, LTMM is
    ODC-By 1.0. The GitHub-hosted files carry the licence of their repository
    (Motiflets GPL-3.0, MOMENTI AGPL-3.0), which does not necessarily cover the
    data; DaISy (FOETAL_ECG, evaporator) and RUTH have no confirmed terms and
    `license=None` in the registry. Nothing is redistributed from this repo.
- **Phase 3.** FL010 is `ltmm/1.0.0/FL010.{hea,dat}` of PhysioNet (ODC-By 1.0).
  From here `physionet.org` served ~30 KB/s (300 MB would take hours) while
  the AWS open data bucket `physionet-open.s3.amazonaws.com` gave ~5 MB/s with
  identical bytes, so the registration uses the bucket. Measured on the real
  record: 25132289x6, no NaN, 65 s for a first load (about 55 s of it
  download), peak RSS 3.2 GB for a 1.2 GB array, 125 MiB parquet from 288 MiB
  raw, 1.2 s to read back. `wfdb` is an optional extra (`aida-data[wfdb]`) and
  also in the dev group. `obspy` was not added because quake was not
  implemented: it is spectrogram band energies (nperseg 8, 32 bands between 1
  and 20 Hz) of an IU.ANMO trace read from a local `quake.mseed` that is in no
  repository, so a rebuilt series could not be compared with the paper's file.
- **Shared downloader (fixed).** Interrupted or stalled downloads used to leave
  truncated files; see `2026-10-08-fix-downloader.md`.

## 9. Summary

- New module `aida_data.timeseries`: `load(name) -> TimeSeries` (`values`
  `(n, d)` float64, `dim_names`, `time`, `univariate`), `available_datasets`,
  `local_paths`, `cache_path`, `prune_raw`, `register`/`DatasetInfo`. Each
  series is parsed once into `datasets/timeseries/<name>.parquet` (zstd 3,
  default dictionary encoding, float64, columns `x0..x{d-1}` and an optional
  `t` timestamp[ms], metadata `aida_data` version 1) and the raw download is
  deleted unless `AIDA_DATA_KEEP_RAW=1`.
- 14 datasets: `astro`, `ecg`, `freezer`, `gap`, `humany`, `steamgen`,
  `dishwasher`, `npo141`, `arrhythmia`, `foetal-ecg`, `evaporator`, `ruth`,
  `oikolab-weather`, `fl010`. Parsers: whitespace/CSV tables without header
  (`np.loadtxt`), CSV with header, Monash `.tsf`, WFDB (optional `wfdb`).
- New `_cache.py` shared by `dense`, `graph` and `timeseries` (atomic parquet
  write, metadata check, raw deletion, `KEEP_RAW`); formats unchanged.
- Measured choices (§7): float64 because float32 is lossy on 4 of 5 series,
  default dictionary encoding because `BYTE_STREAM_SPLIT` is up to 4x larger
  on quantized data, zstd level 3, `np.loadtxt` as the text parser.
- Deviations from the plan: ECG blank lines are skipped, as `pyattimo` does,
  instead of kept as NaN; the figshare URLs use `ndownloader.figshare.com`;
  FL010 uses the PhysioNet S3 bucket; no `.mat` loader, `penguin`, `quake`,
  CLEAN_House1 or whales (§8).
- Tests: 120 offline tests pass (35 new for `timeseries`); 14 network tests
  (one per registered source) pass against the real files. README has a
  section for the module; the software todo marks priority 3 as done and lists
  what is missing.
- Commits: `refactor: share the parquet cache helpers...`, then four
  `feat(timeseries)` commits (phase 1; Motiflets and MOMENTI text files;
  `.tsf` and oikolab; FL010 with the `wfdb` extra), then this note and the todo.
