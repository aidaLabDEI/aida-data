# Plan: safe temporary files in the downloader and the parquet cache

Follow-up to `2026-10-08-fix-downloader.md`, from the code review of
`c8c3f90..6e3d787` (findings 1, 4 and 6). Nothing here is implemented yet.

Goal: two processes loading the same dataset at once cannot corrupt or delete
each other's files, a failed cache write leaves nothing behind, and the
download length check works whatever `Content-Encoding` the server sends.

## 1. The problems

1. **Shared `.part` name** (`src/aida_data/_download.py:30`). The temporary
   file is always `<destination>.part`. When two processes (parallel pytest
   workers, two notebooks) download the same file:
   - both `open(part, "wb")` the same inode and interleave or truncate each
     other's bytes;
   - when one of them fails, its `except BaseException: part.unlink()`
     deletes the file the other is still writing, and the other's
     `part.replace(destination)` raises `FileNotFoundError`;
   - one can rename a half-written `.part` into place while the other is
     still writing into it (same inode), so `destination` ends up truncated.
2. **The parquet temporary file is never cleaned up and its name is shared**
   (`src/aida_data/_cache.py:23`). `write_table` writes
   `<name>.parquet.tmp` and renames it, but it has no `try`. A Ctrl-C or a
   full disk during `pq.write_table` leaves a possibly multi-GB `.tmp` file
   that nothing deletes, and `prune_raw` does not list it. Concurrent writers
   share the name, with the same races as in 1.
3. **The length check is too loose** (`src/aida_data/_download.py:50-57`).
   - It is skipped when there is *any* `Content-Encoding` header, including
     `identity`, so a server that sends `Content-Encoding: identity` and then
     closes early is never caught.
   - `Content-Length` counts the bytes on the wire, but `received` and the
     progress bar count the bytes after decompression. With gzip responses
     (`raw.githubusercontent.com`; `requests` asks for gzip by default), the
     bar goes past 100% and the check cannot run.

## 2. Design

### 2.1 Unique temporary names (problems 1 and 2)

- One small helper in `_cache.py` (or a new `_tmp.py` if that reads better),
  used by both modules:

  ```python
  @contextmanager
  def atomic_path(path: Path, suffix: str):
      """Yield a unique temporary path next to `path`. On a normal exit it
      replaces `path`; on any exception it is deleted."""
  ```

  It makes the name with `tempfile.mkstemp(dir=path.parent,
  prefix=path.name + ".", suffix=suffix)` and closes the descriptor straight
  away, because `open(..., "wb")` and `pq.write_table` both want a path.
  Same directory, so `os.replace` stays an atomic rename on one filesystem.
- Suffixes stay as they are (`.part` for downloads, `.parquet.tmp` for
  caches), so the files are still easy to recognize:
  `ecg.txt.gz.k3j9x_.part`, `ecg.parquet.k3j9x_.parquet.tmp`.
- `download()` and `write_table()` both go through the helper. Their logic
  does not change otherwise.
- **Concurrent writers both finish**: each one renames its own complete file
  onto `destination`, and the last one wins. The two files have the same
  content, and readers see either the old inode or the new one, never a
  partial file. The cost is a wasted second download, which is acceptable.
  We don't add a file lock (see §4).
- **Stale files from killed processes** (SIGKILL, power loss) skip the
  `except` and now have unique names, so the next run no longer overwrites
  them. Add a sweep to the shared helper: `stale_temporaries(directory)`
  lists `*.part` and `*.parquet.tmp` files whose mtime is more than 24 h old.
  Hook it into the three `prune_raw` functions so that they report and
  delete these files too. The age threshold keeps the sweep from touching a
  download that is still running in another process.
  *Decision for Matteo:* is 24 h fine, or should the sweep only happen in
  `prune_raw` (as proposed) and never automatically?

### 2.2 Length check on wire bytes (problem 3)

- Count what came over the wire with `response.raw.tell()`, which urllib3's
  `HTTPResponse` exposes (2.8.0 is locked, checked). It returns the bytes
  read from the socket before decoding, which is exactly what
  `Content-Length` counts.
- Replace the `encoded` condition with:

  ```python
  wire = response.raw.tell()
  if total and wire != total:
      raise OSError(f"incomplete download of {url}: got {wire} of {total} bytes")
  ```

- Progress bar: inside the loop, `progress.update(response.raw.tell() -
  progress.n)` instead of `len(chunk)`, so the bar and `total` count the
  same unit.
- Drop the `received` counter, which nothing else uses.

## 3. Implementation steps

1. [ ] **Tests first**, in `tests/test_download.py` and a new
   `tests/test_cache.py`:
   - `FakeResponse` gets a `raw` attribute with a `tell()` that returns the
     number of encoded bytes "read so far". It defaults to the sum of the
     chunk lengths, and a constructor argument overrides it to simulate
     compression.
   - gzip-like response: decoded chunks longer than `Content-Length`, with
     `raw.tell() == Content-Length` → success.
   - `Content-Encoding: identity` with a short body → raises, nothing left.
   - compressed and short (`raw.tell() < Content-Length`) → raises.
   - two concurrent downloads: two threads, each with its own fake response,
     where thread A blocks inside `iter_content` on a `threading.Event`
     while thread B fails. Check that B does not remove A's temporary file,
     and that A then completes and `destination` holds A's body.
   - replace `test_stale_part_is_replaced` with: an old `data.bin.part` is
     left alone by `download()` (it is no longer read), and
     `stale_temporaries` returns it once its mtime is set back 25 h
     (`os.utime`), but not when it is fresh.
   - `write_table` with `pq.write_table` monkeypatched to raise after
     creating the file → the exception propagates and no `*.tmp` file is
     left in the directory.
   - update the three existing checks for `<name>.parquet.tmp`
     (`tests/test_graph.py:150`, `tests/test_colored.py:212`,
     `tests/test_timeseries.py:254`) to glob for `*.tmp`.
2. [ ] **Add the helper** and switch `_cache.write_table` to it.
3. [ ] **Switch `_download.download`** to the helper and the wire-byte check.
   Update the docstring, which still names `<destination>.part`.
4. [ ] **Sweep stale temporaries** from `prune_raw` in `dense.py`,
   `graph.py` and `timeseries.py`, depending on the decision in §2.1.
5. [ ] Run the whole suite, then `pytest -m network` once: GitHub serves
   gzip, so this checks `raw.tell()` against a real server.
6. [ ] Note in the README (cache section) that `*.part` / `*.tmp` files left
   by killed processes are removed by `prune_raw`.

## 4. Out of scope

- **No file lock.** Concurrent `load()` calls still both download and parse.
  There is also a gap this plan does not close: process A can finish, cache
  and `delete_raw` the raw file while process B is still between `download()`
  and parsing, and B then fails or downloads the file again. Closing it needs
  a per-dataset lock (`fcntl.flock` on a `<name>.lock` file, or the
  `filelock` package). Worth doing if parallel test runs hit it.
- Truncated files written directly to `destination` by the old downloader
  (review finding 5). That needs checksums in `DatasetInfo`, as noted in
  `2026-10-08-fix-downloader.md`.
- Windows: `os.replace` onto a file that another process has open fails
  there. Not a target today.
