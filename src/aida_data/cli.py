"""The `aida-data` command line tool: argument parsing and printing only.
The logic lives in `aida_data.export`.

Heavy imports (pandas, tqdm, ...) stay inside the commands, so that `-q` can
set `TQDM_DISABLE` before tqdm is first imported.
"""

import argparse
import errno
import logging
import os
import sys
from pathlib import Path

PROG = "aida-data"


# --------------------------------------------------------------------------
# Commands: each takes the parsed arguments and returns the exit code
# --------------------------------------------------------------------------


def _export(args) -> int:
    from . import export

    mode = "copy" if args.copy else "hardlink" if args.hardlink else "symlink"
    link = export.export(
        args.name, args.dest, format=args.format, mode=mode, force=args.force
    )
    if mode == "copy":
        print(link)
    else:
        print(
            f"{link} -> {export.cached_path(export.resolve(args.name), args.format).absolute()}"
        )
    return 0


def _path(args) -> int:
    from . import export

    print(export.ensure(export.resolve(args.name), args.format).absolute())
    return 0


def _list(args) -> int:
    from . import export

    def flag(entry, format):
        if format not in export.available_formats(entry):
            return "-"
        return "yes" if export.is_cached(entry, format) else "no"

    rows = [("NAME", "MODULE", "NATIVE", "PARQUET", "HDF5")]
    for entry in export.entries(args.module):
        # `link_stem` is the name as listed: the alias for MTEB aliases,
        # whose `name` is the full `mteb:` name.
        row = (
            entry.link_stem,
            entry.module,
            entry.native,
            flag(entry, "parquet"),
            flag(entry, "hdf5"),
        )
        if not args.cached or "yes" in row[3:]:
            rows.append(row)
    widths = [max(len(row[i]) for row in rows) for i in range(5)]
    for row in rows:
        print("  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip())
    return 0


def _prune_hdf5(args) -> int:
    from . import export

    paths = export.prune_hdf5(dry_run=True)
    total = sum(path.stat().st_size for path in paths)
    for path in paths:
        print(path)
    verb = "deleting" if args.yes else "would delete"
    print(
        f"{verb} {len(paths)} converted HDF5 and stale temporary files ({total / 2**20:.1f} MiB)"
    )
    if args.yes:
        from ._cache import delete_raw

        # Delete exactly what was listed, not a second listing.
        delete_raw(*paths)
    elif paths:
        print("pass --yes to delete them", file=sys.stderr)
    return 0


def _snippet(args) -> int:
    from . import _snippets, export

    print(_snippets.cpp_snippet(export.resolve(args.name), args.file), end="")
    return 0


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------


def _add_export(sub, common) -> None:
    p = sub.add_parser("export", parents=[common], help="expose a dataset at a path")
    p.add_argument("name", metavar="NAME")
    p.add_argument(
        "dest",
        metavar="DEST",
        nargs="?",
        type=Path,
        help="link path or directory (default: .)",
    )
    _add_format(p)
    p.add_argument(
        "--force", action="store_true", help="replace an existing file or symlink"
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--copy", action="store_true", help="copy instead of linking")
    mode.add_argument(
        "--hardlink", action="store_true", help="hard-link instead of symlinking"
    )
    p.set_defaults(func=_export)


def _add_path(sub, common) -> None:
    p = sub.add_parser(
        "path", parents=[common], help="print the cached path of a dataset"
    )
    p.add_argument("name", metavar="NAME")
    _add_format(p)
    p.set_defaults(func=_path)


def _add_list(sub, common) -> None:
    p = sub.add_parser(
        "list", parents=[common], help="list datasets and what is cached"
    )
    p.add_argument("--module", choices=("dense", "graph", "timeseries"))
    p.add_argument(
        "--cached", action="store_true", help="only datasets cached in some format"
    )
    p.set_defaults(func=_list)


def _add_prune_hdf5(sub, common) -> None:
    p = sub.add_parser(
        "prune-hdf5", parents=[common], help="delete the converted HDF5 files"
    )
    p.add_argument("--yes", action="store_true", help="delete (default: dry run)")
    p.set_defaults(func=_prune_hdf5)


def _add_snippet(sub, common) -> None:
    p = sub.add_parser(
        "snippet", parents=[common], help="print C++ code that loads the HDF5 export"
    )
    p.add_argument("name", metavar="NAME")
    p.add_argument(
        "--file", metavar="PATH", help="path of the file in the code (default: NAME.hdf5)"
    )
    p.set_defaults(func=_snippet)


def _add_format(parser) -> None:
    parser.add_argument("--format", choices=("hdf5", "parquet"), default="hdf5")


# One function per subcommand; add new ones here.
_SUBCOMMANDS = (_add_export, _add_path, _add_list, _add_prune_hdf5, _add_snippet)


def build_parser() -> argparse.ArgumentParser:
    # SUPPRESS keeps the subparser from overwriting a value given before it.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        default=argparse.SUPPRESS,
        help="only warnings and errors, no progress bars",
    )
    common.add_argument(
        "--debug",
        action="store_true",
        default=argparse.SUPPRESS,
        help="show the traceback of errors",
    )
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Download datasets and expose them at a path.",
        parents=[common],
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    for add in _SUBCOMMANDS:
        add(sub, common)
    return parser


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def _message(error: BaseException) -> str:
    if isinstance(error, OSError) and error.errno and error.strerror:
        names = [str(f) for f in (error.filename, error.filename2) if f]
        return f"{', '.join(names)}: {error.strerror}" if names else error.strerror
    return str(error) or type(error).__name__


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    quiet = getattr(args, "quiet", False)
    if quiet:
        os.environ["TQDM_DISABLE"] = "1"  # before tqdm is first imported
    logging.basicConfig(
        level=logging.WARNING if quiet else logging.INFO,
        stream=sys.stderr,
        format="%(message)s",
        force=True,
    )
    from . import export

    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130
    except (LookupError, export.FormatNotAvailable, OSError) as error:
        # UnknownDataset is a LookupError; requests' errors are OSErrors.
        if getattr(args, "debug", False):
            raise
        print(f"{PROG}: error: {_message(error)}", file=sys.stderr)
        return 1
