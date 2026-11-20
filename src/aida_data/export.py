"""Expose datasets at a path of the user's choice (the library behind the
`aida-data` command line tool).

A dataset is resolved to an `Entry`, made available in the cache in the
requested format (HDF5 by default, or the parquet cache) and then linked,
copied or hard-linked to the destination. Nothing here prints.
"""

import difflib
import errno
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from . import _hdf5, _mteb_names, _tmp, dense

Module = Literal["dense", "graph", "timeseries"]
Format = Literal["hdf5", "parquet"]

_MODULES: tuple[Module, ...] = ("dense", "graph", "timeseries")


class UnknownDataset(KeyError):
    """No (or no unique) dataset has the requested name."""

    def __str__(self) -> str:
        # `KeyError` would show the repr of the message, in quotes.
        return str(self.args[0])


class FormatNotAvailable(ValueError):
    """The dataset cannot be exposed in the requested format."""


@dataclass(frozen=True)
class Entry:
    name: str  # canonical: registered name, or full mteb: name
    module: Module
    native: Literal["hdf5", "parquet", "parquet-dir"]
    link_stem: str  # default name at the destination, without suffix


# --------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------


def _registry(module: Module) -> dict:
    # Local imports: exporting a dense dataset must not load the others.
    if module == "dense":
        return dense._DATASETS_INFO
    if module == "graph":
        from . import graph

        return graph._DATASETS_INFO
    from . import timeseries

    return timeseries._DATASETS_INFO


def _mteb_entry(name: str, alias: str | None = None) -> Entry:
    task, model = _mteb_names.parse_name(name)
    if alias is None:
        default = None
        try:
            default = dense._default_alias(task, model)
        except KeyError:  # model without a tag, so no default alias
            pass
        if default is not None and dense._ALIASES.get(default) == name:
            alias = default
        else:
            alias = f"{task}__{_mteb_names.model_slug(model)}"
    return Entry(name, "dense", "parquet-dir", alias)


def _lookup(module: Module, name: str) -> Entry | None:
    """The entry of registered (or alias) `name` in `module`, else None."""
    if module == "dense" and name in dense._ALIASES:
        return _mteb_entry(dense.resolve_name(name), alias=name)
    info = _registry(module).get(name)
    if info is None:
        return None
    native = "hdf5" if module == "dense" and info.cache_name is None else "parquet"
    return Entry(name, module, native, name)


def _all_names() -> list[str]:
    names = []
    for module in _MODULES:
        names += [e.link_stem for e in entries(module)]
    return names


def resolve(name: str) -> Entry:
    """The `Entry` for `name`: a registered name, a dense alias, a full
    `mteb:<task>:<model>` name, any of them optionally prefixed with
    `<module>/`. Raises `UnknownDataset` if nothing (or more than one
    dataset) matches."""
    module = None
    # `mteb:` names contain `/` in the model, so they are checked first.
    if not name.startswith(_mteb_names.PREFIX) and "/" in name:
        head, rest = name.split("/", 1)
        if head in _MODULES:
            module, name = head, rest
    if name.startswith(_mteb_names.PREFIX):
        if module not in (None, "dense"):
            raise UnknownDataset(f"{name} is a dense dataset, not a {module} one")
        return _mteb_entry(name)

    if module is not None:
        entry = _lookup(module, name)
        if entry is not None:
            return entry
        others = [m for m in _MODULES if m != module and _lookup(m, name)]
        if others:
            raise UnknownDataset(
                f"{name!r} is not in module {module}, but in {others[0]}: "
                f"use {others[0]}/{name}"
            )
    else:
        # Dense first: a dense name is found without importing the other
        # modules. Names are unique across modules (a test checks it), so
        # only graph and timeseries are compared with each other.
        found = [e] if (e := _lookup("dense", name)) else []
        if not found:
            found = [e for m in _MODULES[1:] if (e := _lookup(m, name))]
        if len(found) > 1:
            raise UnknownDataset(
                f"{name!r} is ambiguous: "
                + " or ".join(f"{e.module}/{name}" for e in found)
            )
        if found:
            return found[0]

    message = f"unknown dataset {name!r}"
    close = difflib.get_close_matches(name, _all_names(), n=3)
    if close:
        message += "; did you mean: " + ", ".join(close) + "?"
    raise UnknownDataset(message)


def entries(module: Module | None = None) -> list[Entry]:
    """Every registered dataset, with the dense MTEB aliases but not the
    (infinitely many) full `mteb:` names: dense, then graph, then time
    series, or only those of `module`."""
    result = []
    for m in _MODULES if module is None else (module,):
        if m == "dense":
            names = dense.available_datasets()
        else:
            names = list(_registry(m))
        result += [_lookup(m, n) for n in names]
    return result


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------


def available_formats(entry: Entry) -> tuple[Format, ...]:
    """The formats `entry` can be exposed in: both, except for the native
    HDF5 datasets (ann-benchmarks, VIBE), whose parquet form would lose
    `test`, `neighbors` and `distances`."""
    return ("hdf5",) if entry.native == "hdf5" else ("parquet", "hdf5")


def _require(entry: Entry, format: Format) -> None:
    if format not in available_formats(entry):
        raise FormatNotAvailable(f"{entry.name} is only available as hdf5")


def cached_path(entry: Entry, format: Format) -> Path:
    """Where `entry` is (or will be) cached in `format`. No I/O. For parquet
    it is the file, or the directory of an MTEB pair; for native HDF5 it is
    the download. Raises `FormatNotAvailable` for parquet of a native HDF5
    dataset."""
    _require(entry, format)
    if format == "hdf5":
        return _hdf5.hdf5_path(entry.module, entry.name)
    if entry.native == "parquet-dir":
        from . import _mteb

        return _mteb.cache_dir(*_mteb_names.parse_name(entry.name))
    if entry.module == "dense":
        return dense._dense_dir() / f"{dense._DATASETS_INFO[entry.name].cache_name}.parquet"
    if entry.module == "graph":
        from . import graph

        return graph.cache_path(entry.name)
    from . import timeseries

    return timeseries.cache_path(entry.name)


def is_cached(entry: Entry, format: Format) -> bool:
    """Whether `ensure(entry, format)` would not download or convert
    anything (a stale HDF5 file does not count). No downloads or
    conversions here either. False for a format that is not available."""
    if format not in available_formats(entry):
        return False
    if format == "hdf5":
        return _hdf5.is_fresh(entry.module, entry.name)
    path = cached_path(entry, format)
    if entry.native == "parquet-dir":
        from . import _mteb

        return (path / _mteb.CORPUS_FILE).exists()
    return path.is_file()


def ensure(entry: Entry, format: Format = "hdf5") -> Path:
    """Make `entry` available in the cache in `format` (downloading, parsing
    and converting if needed) and return its path: see `cached_path`."""
    _require(entry, format)
    if format == "hdf5":
        return _hdf5.ensure_hdf5(entry.module, entry.name)
    if entry.native == "parquet-dir":
        return dense._ensure_mteb(entry.name)
    if entry.module == "dense":
        return dense._ensure_cache(entry.name)
    if entry.module == "graph":
        from . import graph

        return graph._ensure_cache(entry.name)
    from . import timeseries

    return timeseries._ensure_cache(entry.name)


def prune_hdf5(dry_run: bool = True) -> list[Path]:
    """See `_hdf5.prune_hdf5`."""
    return _hdf5.prune_hdf5(dry_run)


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------


def _link_path(entry: Entry, format: Format, dest: Path | None) -> Path:
    """Where the link goes, like `ln -s TARGET [DEST]`: inside `dest` if it
    is a directory (or a symlink to one), else `dest` itself, whose parent
    must exist."""
    suffix = "" if entry.native == "parquet-dir" and format == "parquet" else f".{format}"
    default = entry.link_stem + suffix
    if dest is None:
        return Path(default)
    dest = Path(dest)
    if dest.is_dir():
        return dest / default
    if not dest.parent.is_dir():
        # Not created: that would hide typos.
        raise FileNotFoundError(f"{dest.parent}: no such directory")
    return dest


def _link_directory(source: Path, tmp: Path, mode: str) -> None:
    """Build the directory `tmp` as a copy of `source`, one hard link per
    file for `mode == "hardlink"`."""
    if mode == "copy":
        shutil.copytree(source, tmp)
        return
    tmp.mkdir()
    for path in sorted(source.rglob("*")):
        target = tmp / path.relative_to(source)
        if path.is_dir():
            target.mkdir()
        else:
            os.link(path, target)


def export(
    name: str,
    dest: Path | None = None,
    *,
    format: Format = "hdf5",
    mode: Literal["symlink", "copy", "hardlink"] = "symlink",
    force: bool = False,
) -> Path:
    """Make dataset `name` available in `format` (see `ensure`) and expose it
    at `dest` as a symlink (absolute, the default), a copy or hard links.

    `dest` works like the last argument of `ln -s`: an existing directory
    (or a symlink to one) gets the link inside it, named after the dataset
    (`Entry.link_stem` plus `.hdf5` or `.parquet`; an MTEB parquet directory
    has no suffix); otherwise `dest` is the path of the link, and its parent
    must exist. The default is the current directory.

    Something already at the link path is an error (`FileExistsError`),
    unless `force`, which replaces files and symlinks, never directories
    (`IsADirectoryError`). The link is created under a temporary name and
    renamed, so readers never see it half-made, and nothing is left behind
    on failure. Returns the link path (not resolved).
    """
    if mode not in ("symlink", "copy", "hardlink"):
        raise ValueError(f"unknown mode {mode!r}")
    entry = resolve(name)
    _require(entry, format)
    link = _link_path(entry, format, dest)
    target = cached_path(entry, format)

    # The parent is resolved but not the last component, so that exporting
    # again over an existing link to the cache is a normal replacement.
    if Path(os.path.realpath(link.parent)) / link.name == Path(os.path.realpath(target)):
        raise FileExistsError(f"{link} is the cached file itself, choose another destination")
    if os.path.lexists(link):
        if not force:
            raise FileExistsError(f"{link} exists, pass --force to replace it")
        if link.is_dir() and not link.is_symlink():
            raise IsADirectoryError(f"{link} is a directory, --force does not replace directories")

    # Make the cache last: errors about the destination come before a download.
    target = ensure(entry, format)
    tmp = _tmp.unused_name(link, ".part")
    try:
        if target.is_dir():
            if mode == "symlink":
                os.symlink(target.absolute(), tmp)
            else:
                _link_directory(target, tmp, mode)
            if mode != "symlink" and os.path.lexists(link):
                link.unlink()  # a directory cannot replace a file with rename
            os.replace(tmp, link)
        else:
            if mode == "symlink":
                os.symlink(target.absolute(), tmp)
            elif mode == "copy":
                shutil.copy2(target, tmp)
            else:
                os.link(target, tmp)
            os.replace(tmp, link)
    except OSError as error:
        _remove(tmp)
        if error.errno == errno.EXDEV:
            raise OSError(
                "cannot hard-link across filesystems, use --copy"
            ) from error
        raise
    except BaseException:
        _remove(tmp)
        raise
    return link


def _remove(path: Path) -> None:
    """Delete the temporary `path` (file, symlink or tree), if it exists."""
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)
