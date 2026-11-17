"""\
MTEB dataset names (private): the `mteb:<task>:<model>` form and the hosted
pairs. No dependencies, so that `dense` can import it at import time
(`_mteb` imports `dense`, and re-exports everything here).
"""

PREFIX = "mteb:"

# `(task, model)` pairs hosted in `_mteb._MTEB_HF_REPO`. `dense` registers a
# short alias for each of them (see `dense.register_alias`), so every model
# here needs a tag in `dense._MODEL_TAGS`. Update it after each upload with
# `scripts/upload_mteb.py`, which prints the line to add.
_HOSTED: list[tuple[str, str]] = [
    ("ArxivClusteringP2P", "Qwen/Qwen3-Embedding-4B"),
    ("SCIDOCS", "Qwen/Qwen3-Embedding-4B"),
]


def parse_name(name: str) -> tuple[str, str]:
    """Split `mteb:<task>:<model>` into `(task, model)`.

    The name is split on the first two `:`. Model names contain `/` but
    never `:`. Raises `ValueError` for a missing prefix, an empty task or
    model, an extra `:`, or parts that would escape the cache directory
    (`..`, empty path components).
    """
    if not name.startswith(PREFIX):
        raise ValueError(f"`{name}` is not an MTEB name, expected `mteb:<task>:<model>`")
    parts = name[len(PREFIX) :].split(":")
    if len(parts) != 2:
        raise ValueError(f"`{name}` must have the form `mteb:<task>:<model>`")
    task, model = parts
    if not task or not model:
        raise ValueError(f"`{name}`: task and model must not be empty")
    if "/" in task or "\\" in task or task in (".", ".."):
        raise ValueError(f"`{name}`: invalid task name `{task}`")
    if any(p in ("", ".", "..") for p in model.split("/")) or "\\" in model:
        raise ValueError(f"`{name}`: invalid model name `{model}`")
    return task, model


def model_slug(model: str) -> str:
    """Directory name of `model`: `/` replaced by `__`."""
    return model.replace("/", "__")
