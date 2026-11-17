# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "aida-data[mteb]",
# ]
#
# [tool.uv.sources]
# aida-data = { path = "..", editable = true }
# ///
"""Build the local cache of an MTEB encoding (maintainer script).

Encodes the task with the model (unless a cache or hosted copy exists) and
stores it in `<root>/dense/mteb/<task>/<model_slug>/`.

Usage (the script header points aida-data at this checkout):

    uv run scripts/build_mteb.py \\
        mteb:SciFact:sentence-transformers/all-MiniLM-L6-v2 --device cpu

"""

import argparse

from aida_data import _mteb, dense


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("name", help="mteb:<task>:<model>")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None, help='e.g. "cpu", "cuda:1"')
    args = parser.parse_args()

    kwargs = {}
    if args.batch_size is not None:
        kwargs["batch_size"] = args.batch_size
    if args.device is not None:
        kwargs["model_kwargs"] = {"device": args.device}

    ds = dense.load(args.name, **kwargs)
    task, model = _mteb.parse_name(args.name)
    print(f"cache: {_mteb.cache_dir(task, model)}")
    print(f"corpus: {ds.dataset.shape}")
    if ds.queries is not None:
        print(f"queries: {ds.queries.shape}")
    if ds.colors is not None:
        print(f"colors: {ds.colors.names}")


if __name__ == "__main__":
    main()
