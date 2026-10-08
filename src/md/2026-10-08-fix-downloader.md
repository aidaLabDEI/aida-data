# Plan: make `_download.download` safe against interrupted downloads

Follow-up to `2026-10-08-timeseries.md` (§8, "Shared downloader"). Not part of
that work, so nothing here has been implemented.

Goal: a download either produces the complete file or leaves nothing at the
destination, and a stalled connection fails instead of hanging.

## 1. The problem

`src/aida_data/_download.py` has two defects, both seen while loading `fl010`
from a throttled server:

1. **Truncated files look complete.** The body is streamed straight into
   `destination`, and the function returns early `if destination.is_file()`.
   If the process is killed, the connection drops, or `raise_for_status` /
   the iteration raises half way, a partial file stays behind. The next call
   skips the download, the loader parses the partial file, and the result is
   cached as if it were the dataset. For text formats this can even succeed
   silently, giving a shorter series; the parquet cache then makes the error
   permanent until someone deletes it by hand.
2. **No timeout.** `requests.get(..., stream=True)` is called without
   `timeout`, so a stalled server blocks forever (the FL010 run sat at
   0 KiB/s for minutes with no error).

Callers (`dense._cached` via the loaders, `graph._cached`, `timeseries._cached`)
all rely on "after `download` returns, `destination` is complete", so the fix
belongs in the shared helper and needs no change in them.

## 2. Design

- Write to `destination.with_name(destination.name + ".part")`, and
  `replace()` it onto `destination` only after the whole body was received.
  On any exception remove the `.part` file and re-raise. The existing `*.tmp`
  names are used for parquet caches, so `.part` avoids clashes.
- Check completeness when the server sends `Content-Length`: if the number of
  bytes written differs, raise (`requests` does not always notice a short
  body with chunked reading). Without `Content-Length`, rely on the exception
  path only.
- `timeout=(10, 60)` on `requests.get`: 10 s to connect, 60 s without receiving
  a byte. This bounds a stall without limiting total time for big files.
  Module constant `_TIMEOUT` so tests and callers can change it.
- Stale `.part` files from a killed process are overwritten by the next
  attempt (opened with `"wb"`), never promoted. No resume support: `Range`
  requests complicate things and servers differ; revisit only if re-downloading
  the big files hurts.
- Keep the non-http early return (synthetic datasets such as `densired-hard`)
  and the `if not destination.is_file()` skip unchanged.
- Existing truncated files from earlier versions cannot be told apart from
  complete ones. Not handled; the README already says how to force a re-parse
  (delete the cache), and deleting the raw file forces a new download. Note it
  in the commit message.
- Optional, decide in step 4: verify a checksum when the registration has one
  (figshare publishes md5s). It would catch a corrupt but full-length file, but
  needs a new field on every `DatasetInfo`, so it is out of scope here.

## 3. Implementation steps

1. [x] **Write the failing tests first** in `tests/test_download.py`, with
   `requests.get` replaced by a fake response object (no network):
   - success: file has the body, no `.part` left;
   - body raises half way (`iter_content` raises `ConnectionError` after a
     chunk): `destination` does not exist, `.part` does not exist, the
     exception propagates;
   - `Content-Length` larger than the bytes received: raises, nothing left;
   - an existing complete `destination` is not touched and `requests.get` is not
     called;
   - a stale `destination.part` from an earlier run is replaced, not promoted;
   - non-http URL returns without creating anything;
   - `requests.get` is called with a `timeout` (check the kwarg).
2. [x] **Fix `_download.py`** as in §2. Keep the `tqdm` progress bar. Run the
   new tests, then the whole suite (the other modules monkeypatch
   `_download`, so they should be unaffected).
3. [x] **Check against a real stall**: serve a file with a local
   `http.server` thread that sends half the body and then sleeps, set
   `_TIMEOUT` low, and confirm the call raises within the timeout and leaves no
   files. This is one test, marked as slow if it takes more than a second or
   two.
4. [x] **Run the network tests** (`pytest -m network`) once, to make sure the
   real servers (figshare, zenodo, GitHub, PhysioNet S3) still work with the
   `.part` rename and the timeout. FL010 (300 MB) is the one that matters for
   the timeout choice.
5. [x] **Docs**: one line in the README next to `AIDA_DATA_KEEP_RAW` saying that
   downloads are written to `<name>.part` and renamed, and that a leftover
   `.part` file is safe to delete. Add a `.part` pattern to `prune_raw`? Not
   needed: a `.part` file is never a registered raw path, so `prune_raw`
   ignores it; mention it only in the README.
6. [x] Fill in §5 of this note and update `2026-10-08-timeseries.md` §8
   (the "Shared downloader (not changed)" bullet) to point here.

One commit: `fix(download): write to a .part file and set a timeout`.

## 4. Risks and things to check

- **Servers that close idle streams.** A read timeout of 60 s is generous for
  chunked streaming, but some servers pause before the first byte for large
  generated files (Kaggle-style exports). If a registered source needs more,
  make the timeout an argument of `download` rather than raising the global.
- **Filesystems without atomic rename** across directories: the `.part` file is
  in the same directory as the destination, so `replace` stays atomic.
- **Disk full** while writing: the `OSError` goes through the same cleanup
  path, which is what we want.
- **Concurrent processes** downloading the same dataset share one `.part`
  name and can corrupt each other. Before this change they could too (both wrote
  to `destination`). Using a unique suffix (`tempfile.NamedTemporaryFile` in
  the same directory) would fix it at no cost; prefer that over a fixed
  `.part` if it does not complicate the stale-file test in step 1. Decide
  while implementing and record it in §5.
- **Truncated files already on disk** (§2) stay a known gap.

## 5. Summary

Done. Changes:

- `src/aida_data/_download.py`: body is streamed to `<name>.part` and
  `replace()`d onto the destination only when complete; any exception
  (including `KeyboardInterrupt`) removes the `.part` file and re-raises.
  `requests.get` gets `timeout=_TIMEOUT` (module constant, `(10, 60)`).
  If `Content-Length` is present and the byte count differs, an `OSError` is
  raised; the check is skipped when `Content-Encoding` is set, since
  `iter_content` yields decoded bytes then.
- `tests/test_download.py`: the seven cases from step 1 with a fake response,
  plus the real-stall test (local `http.server`, timeout `(2, 0.5)`), which
  runs in well under a second so is not marked slow.
- README: one sentence about `.part` files.
- `2026-10-08-timeseries.md` §8 now points here.

Decisions:

- Fixed `.part` name rather than a unique temp name: simpler, keeps the
  stale-file behaviour deterministic. Concurrent downloads of the same dataset
  can still clash, exactly as before; revisit if it shows up.
- Checksum verification and resume not done (out of scope, as planned).
- Truncated files from earlier versions are still undetectable (known gap).

Verification: full suite 128 passed; `pytest -m network` 18 passed (~2 min,
including FL010) with the new timeout and rename. The new tests were written
before the fix but I did not run them against the old code to watch them fail.
