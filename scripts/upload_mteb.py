# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "aida-data",
#     "huggingface_hub",
# ]
#
# [tool.uv.sources]
# aida-data = { path = "..", editable = true }
# ///
"""Upload a local MTEB cache to the lab HF dataset repo (maintainer script).

Uploads `<root>/dense/mteb/<task>/<model_slug>/*.parquet` to
`<task>/<model_slug>/` in `_mteb._MTEB_HF_REPO`. Needs `HF_TOKEN` in the
environment. Refuses models without a tag in `dense._MODEL_TAGS`, and prints
the line to add to `_mteb_names._HOSTED`, which gives the pair its alias.

Usage (the script header points aida-data at this checkout):

    HF_TOKEN=... uv run scripts/upload_mteb.py \\
        mteb:SciFact:sentence-transformers/all-MiniLM-L6-v2
"""

import argparse
import os
import sys

from huggingface_hub import HfApi

from aida_data import _mteb, dense


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("name", help="mteb:<task>:<model>")
    args = parser.parse_args()

    task, model = _mteb.parse_name(args.name)
    if model not in dense._MODEL_TAGS:
        sys.exit(
            f"model `{model}` has no tag: add one to `_MODEL_TAGS` in "
            "src/aida_data/dense.py first"
        )
    directory = _mteb.cache_dir(task, model)
    if not (directory / _mteb.CORPUS_FILE).exists():
        sys.exit(f"{directory / _mteb.CORPUS_FILE} is missing: build the cache first")
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN is not set")

    path_in_repo = f"{task}/{_mteb.model_slug(model)}"
    HfApi(token=token).upload_folder(
        repo_id=_mteb._MTEB_HF_REPO,
        repo_type="dataset",
        folder_path=str(directory),
        path_in_repo=path_in_repo,
        allow_patterns=["*.parquet"],
        commit_message=f"Add {task} / {model}",
    )
    print(f"uploaded {directory} to {_mteb._MTEB_HF_REPO}/{path_in_repo}")
    if (task, model) not in _mteb._HOSTED:
        alias = dense._default_alias(task, model)
        print(
            f"add this line to `_HOSTED` in src/aida_data/_mteb_names.py "
            f"(alias `{alias}`):\n    (\"{task}\", \"{model}\"),"
        )


if __name__ == "__main__":
    main()
