"""Temporary files for downloads and caches, shared by the dataset modules."""

import os
import secrets
import time
from contextlib import contextmanager
from pathlib import Path

# Suffixes of the temporary files made through `atomic_path`.
TEMPORARY_SUFFIXES = (".part", ".parquet.tmp")

# Temporaries older than this were left by a killed process: no running
# download or cache write keeps a file untouched that long.
STALE_AFTER = 24 * 3600


def _create_unique(path: Path, suffix: str) -> Path:
    """Create an empty file `<path>.<random><suffix>` next to `path`.

    Like `tempfile.mkstemp`, but with mode 0666 minus the umask (mkstemp
    uses 0600), so that the renamed file gets the mode `open` would give it.
    """
    for _ in range(100):
        tmp = path.with_name(f"{path.name}.{secrets.token_hex(4)}{suffix}")
        try:
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
        except FileExistsError:
            continue
        os.close(fd)
        return tmp
    raise FileExistsError(f"no free temporary name for {path}")


@contextmanager
def atomic_path(path: Path, suffix: str):
    """Yield a unique temporary path next to `path`. On a normal exit it
    replaces `path`; on any exception it is deleted.

    Each caller gets its own file, so concurrent writers of the same `path`
    do not touch each other's data: the last one to finish wins, and
    readers see either the old file or a complete new one.
    """
    tmp = _create_unique(path, suffix)
    try:
        yield tmp
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def stale_temporaries(directory: Path) -> list[Path]:
    """The temporary files in `directory` (not recursive) that were not
    modified for more than `STALE_AFTER` seconds, left by killed processes."""
    if not directory.is_dir():
        return []
    cutoff = time.time() - STALE_AFTER
    return sorted(
        path
        for path in directory.iterdir()
        if path.name.endswith(TEMPORARY_SUFFIXES)
        and path.is_file()
        and path.stat().st_mtime < cutoff
    )
