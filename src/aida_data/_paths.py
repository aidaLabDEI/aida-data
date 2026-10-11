"""The root directory of the downloads and caches, shared by the dataset modules."""

import os
from pathlib import Path


def datasets_dir() -> Path:
    """The root directory of the downloads and caches (see the README).

    First match wins: `$AIDA_DATA_DIR`; `$XDG_CACHE_HOME/aida-data`, if
    `XDG_CACHE_HOME` is absolute (the XDG Base Directory spec says relative
    values must be ignored); `~/.cache/aida-data`. The directory is not
    created.
    """
    root = os.environ.get("AIDA_DATA_DIR")
    if root:
        return Path(root).expanduser().absolute()
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache and Path(xdg_cache).is_absolute():
        return Path(xdg_cache) / "aida-data"
    return Path.home() / ".cache" / "aida-data"
