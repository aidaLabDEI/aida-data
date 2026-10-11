# Plan: cache datasets in the XDG user cache directory

Implemented, see the summary at the end.

Goal: datasets are downloaded and cached in one per-user directory, so every
project, notebook and working directory shares the same copy, instead of a
`datasets/` directory relative to wherever Python was started. The location
stays configurable through an environment variable. Dense datasets move to
their own `dense/` subdirectory, next to `graphs/` and `timeseries/`.

## 1. Current state

- Each dataset module computes its root at import time, independently:
  - `src/aida_data/dense.py:34`
  - `src/aida_data/graph.py:33`
  - `src/aida_data/timeseries.py:32`

  all as `DATASETS_DIR = Path(os.environ.get("AIDA_DATA_DIR", "datasets"))`.
- The default `datasets` is **relative to the current working directory**:
  running a script from another directory downloads everything again (several
  GB, see `2026-10-08-local-file-format.md`).
- Layout under the root: dense files directly in the root, graph files in
  `graphs/` (`graph.py:111`, `:122`, `:356`), time series in `timeseries/`
  (`timeseries.py:371`, `:379`, `:539`).
- `dense.py` uses the root directly in four places:
  - `local_path` (`dense.py:1181`), the raw download path;
  - `_cache_exists` (`dense.py:1191`), the parquet / legacy HDF5 cache;
  - `prune_raw` (`dense.py:1223`), the stale temporaries sweep;
  - `load` (`dense.py:1300-1301`), which creates the root with
    `DATASETS_DIR.mkdir()`, without `parents=True`: it would fail for a
    nested root such as `~/.cache/aida-data/dense`.

  `_cached` puts the cache next to the raw file (`path.parent`,
  `dense.py:166`), so it follows `local_path` with no change.
- Tests replace the root with `monkeypatch.setattr(module, "DATASETS_DIR",
  tmp_path)`, per module.
- `README.md:29`, `:96`, `:139` document `datasets/` (or `$AIDA_DATA_DIR`).

## 2. Design

### 2.1 Where the datasets go

Resolution order of the root, first match wins:

1. `$AIDA_DATA_DIR`, if set and non-empty: used as is (after
   `expanduser()`). This is the existing variable.
2. `$XDG_CACHE_HOME/aida-data`, if `XDG_CACHE_HOME` is set, non-empty and an
   absolute path (the XDG Base Directory spec says relative values must be
   ignored).
3. `~/.cache/aida-data`, the XDG default.

Why the *cache* directory and not `XDG_DATA_HOME`: everything stored is
either a download that can be fetched again or a parquet cache that can be
rebuilt from it, which is exactly what the spec reserves `XDG_CACHE_HOME`
for. Backup tools commonly skip `~/.cache`, which is what we want for
multi-GB re-downloadable files.

No new dependency (`platformdirs`): the same rule gives `~/.cache/aida-data`
on macOS and `%USERPROFILE%\.cache\aida-data` on Windows, which is fine for
this package, and anyone who wants the platform-native location can set
`AIDA_DATA_DIR`.

### 2.2 Layout under the root

```
<root>/
  dense/        raw downloads and <cache_name>.parquet of dense datasets
  graphs/       unchanged
  timeseries/   unchanged
```

Each module family now owns one subdirectory and nothing lives directly in
the root. This also keeps `dense.prune_raw` from ever looking at the other
modules' files, since `stale_temporaries` is not recursive but today runs on
the root that contains `graphs/` and `timeseries/`.

### 2.3 One place that resolves the root

Add a small module `src/aida_data/_paths.py` (next to `_cache.py` and
`_tmp.py`):

```python
def datasets_dir() -> Path:
    """The root directory of the downloads and caches (see the README)."""
```

- It implements the order of 2.1 and returns an absolute path; it does not
  create the directory (callers already create what they need).
- The three modules keep their module-level `DATASETS_DIR` (the root), now
  set to `datasets_dir()` at import time. Keeping the name means:
  - the tests' `monkeypatch.setattr(module, "DATASETS_DIR", tmp_path)` keep
    working;
  - the behaviour stays "read the environment once, at import", as today.
  Resolving lazily on every call is not needed and would break the
  monkeypatching pattern.
- In `dense.py`, add a helper mirroring the other modules:

  ```python
  def _dense_dir() -> Path:
      return DATASETS_DIR / "dense"
  ```

  It is a function, not a module constant, so that monkeypatching
  `DATASETS_DIR` still moves it. Use it in the four places of section 1
  (`local_path`, `_cache_exists`, `prune_raw`, `load`), and make `load`
  create it with `mkdir(parents=True, exist_ok=True)`.

### 2.4 Existing data

No automatic migration and no runtime warning or log message: the change is
documented in `CHANGELOG.md` only, as a breaking change, with the commands
to keep existing downloads:

- old default (`./datasets`, `AIDA_DATA_DIR` unset):

  ```sh
  mkdir -p ~/.cache/aida-data/dense
  mv datasets/graphs datasets/timeseries ~/.cache/aida-data/ 2>/dev/null
  mv datasets/* ~/.cache/aida-data/dense/
  ```

- `AIDA_DATA_DIR` set: the dense files must move into `$AIDA_DATA_DIR/dense`:

  ```sh
  cd "$AIDA_DATA_DIR" && mkdir -p dense
  find . -maxdepth 1 -type f -exec mv {} dense/ \;
  ```

  (or keep using `./datasets` with `AIDA_DATA_DIR=datasets`, then the
  second recipe.)

Without moving, nothing breaks: the datasets are downloaded and parsed again
into the new location.

### 2.5 Documentation

- `README.md`: replace "downloaded to `datasets/` (or `$AIDA_DATA_DIR`)" in
  the three places with the resolution order of 2.1 and the layout of 2.2,
  stated once in the general section; the dense, graph and time series
  sections refer to `<root>/dense`, `<root>/graphs` and `<root>/timeseries`.
- Mention `AIDA_DATA_KEEP_RAW` next to it, so both environment variables are
  documented together.
- `.gitignore` keeps `/datasets`, for people who set `AIDA_DATA_DIR=datasets`.

## 3. Tests

New `tests/test_paths.py`, using `monkeypatch.setenv` / `delenv` and
`monkeypatch.setenv("HOME", ...)` so the real home is never touched:

1. `AIDA_DATA_DIR` set: it wins over `XDG_CACHE_HOME`.
2. `AIDA_DATA_DIR` with `~`: expanded.
3. `AIDA_DATA_DIR` empty string: ignored, falls through to XDG.
4. `XDG_CACHE_HOME=/some/abs`: root is `/some/abs/aida-data`.
5. `XDG_CACHE_HOME` relative (`foo`): ignored, root is
   `$HOME/.cache/aida-data`.
6. Neither set: root is `$HOME/.cache/aida-data`.
7. The result is always absolute.

Dense layout:

8. With `DATASETS_DIR` monkeypatched to `tmp_path`, `dense.local_path(name)`
   is under `tmp_path / "dense"`.
9. `dense.load` on a fresh root creates `<root>/dense` (covers the
   `parents=True` fix); can reuse the fixtures of
   `tests/test_load_pipeline.py`.

Existing tests to update for the new dense subdirectory:

- `tests/test_cache.py:108-113`: the `(dense, None)` case of
  `test_prune_raw_sweeps_stale_temporaries` becomes `(dense, "dense")`, and
  the `if subdir else tmp_path` branch goes away.
- `tests/test_raw_cleanup.py` and `tests/test_load_pipeline.py`: any file
  they create at `tmp_path / ...` for a dense dataset moves to
  `tmp_path / "dense" / ...`.
- Tests that call parsers or `dense._cached` with an explicit path
  (`tests/test_dense_parsers.py`, `tests/test_colored.py`) do not depend on
  `DATASETS_DIR` and should not change; check the ones that monkeypatch it
  (`test_dense_parsers.py:99`, `:228`, `:313`, `test_colored.py:50`).

Also check that no test depends on the old relative default (grep for
`"datasets"`).

## 4. Steps

- [x] 1. Add `_paths.py` with `datasets_dir()` and its tests.
- [x] 2. Use it in `dense.py`, `graph.py`, `timeseries.py`.
- [x] 3. Add `_dense_dir()` in `dense.py`, use it in the four places, fix the
     `mkdir` in `load`; update the affected tests.
- [x] 4. README and CHANGELOG (including the migration commands of 2.4).
- [x] 5. Run `uv run pytest`; then once by hand, with no env vars set, load a small
     dense dataset and check it lands in `~/.cache/aida-data/dense`.

## 5. Summary

Implemented as planned, with no deviations from the design.

- `src/aida_data/_paths.py`: `datasets_dir()` resolves the root as
  `$AIDA_DATA_DIR` (non-empty, `~` expanded, made absolute), then
  `$XDG_CACHE_HOME/aida-data` (absolute values only), then
  `~/.cache/aida-data`. It does not create the directory.
- `dense.py`, `graph.py`, `timeseries.py`: `DATASETS_DIR = datasets_dir()`
  at import time; the now-unused `import os` was dropped from all three.
- `dense.py`: new `_dense_dir()` (`DATASETS_DIR / "dense"`), used by
  `local_path`, `_cache_exists`, `prune_raw` and `load`; `load` now calls
  `_dense_dir().mkdir(parents=True, exist_ok=True)`.
- Tests:
  - new `tests/test_paths.py` (cases 1-7 of section 3, with `HOME` pointed at
    `tmp_path`);
  - `tests/test_load_pipeline.py`: `test_local_path_is_under_dense` (8) and
    `test_load_creates_nested_dense_dir` (9);
  - `tests/test_cache.py`: the dense case of
    `test_prune_raw_sweeps_stale_temporaries` uses `"dense"`;
  - `tests/test_raw_cleanup.py`: a `dense_dir` fixture
    (`tmp_path / "dense"`) replaces `tmp_path` in the tests' file paths.
  - `test_dense_parsers.py` and `test_colored.py` needed no change; no test
    relied on the old relative `datasets` default.
- `README.md`: new "Where the datasets are stored" section (resolution order,
  layout, `AIDA_DATA_KEEP_RAW`); the dense, graph and time series paragraphs
  refer to `<root>/dense`, `<root>/graphs`, `<root>/timeseries`.
- `CHANGELOG.md`: an `[Unreleased]` section with the breaking change and the
  migration commands of 2.4. `.gitignore` unchanged.

Verification: `uv run pytest` gives 165 passed (18 network tests
deselected). By hand, with `AIDA_DATA_DIR` and `XDG_CACHE_HOME` unset and
Python started outside the repository, `dense.load("breast")` returned
(569, 30) and downloaded to
`~/.cache/aida-data/dense/breast+cancer+wisconsin+diagnostic.zip`.
