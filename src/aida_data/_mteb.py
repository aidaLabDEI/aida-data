"""\
MTEB encodings as dense datasets (private).

Encodes an MTEB task (retrieval or clustering, text only) with an MTEB model,
and stores the vectors in the parquet cache layout of `aida_data.dense`, in
`<root>/dense/mteb/<task>/<model_slug>/`:

- `corpus.parquet`: the corpus (retrieval) or the unique texts (clustering),
  written last: its presence marks the cache as complete;
- `queries.parquet`, `qrels.parquet`: retrieval only.

Everything that needs `mteb`, `torch` or `datasets` imports them lazily, so
this module can be imported without the `mteb` extra. The pure parts (parsing
names, building rows, colors and qrels, writing the files) do not need them.
"""

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

from . import _download, dense
from ._cache import write_table
from ._mteb_names import _HOSTED, PREFIX, model_slug, parse_name  # noqa: F401
from .dense import Colors, _write_parquet_cache

_LOGGER = logging.getLogger("aida_data.mteb")

# Label of the texts that do not belong to a clustering set.
ABSENT = "<absent>"

_MTEB_HF_REPO = "aidaLabDEI/aida-data"
_MTEB_HF_BASE = f"https://huggingface.co/datasets/{_MTEB_HF_REPO}/resolve/main/"

CORPUS_FILE = "corpus.parquet"
QUERIES_FILE = "queries.parquet"
QRELS_FILE = "qrels.parquet"

# Layout version of `qrels.parquet`, in its `aida_data` schema metadata.
_QRELS_VERSION = 1

_INSTALL_HINT = "pip install aida-data[mteb]"


# --------------------------------------------------------------------------
# Names and paths
# --------------------------------------------------------------------------


def cache_dir(task: str, model: str) -> Path:
    """Local cache directory of the pair `(task, model)`."""
    return dense._dense_dir() / "mteb" / task / model_slug(model)


# --------------------------------------------------------------------------
# Hosted registry
# --------------------------------------------------------------------------


def _hosted_url(task: str, model: str, filename: str) -> str:
    return f"{_MTEB_HF_BASE}{task}/{model_slug(model)}/{filename}"


def _exists_hosted(url: str) -> bool:
    """HEAD `url`: True for 200, False for 404, raise for anything else
    (including network errors)."""
    import requests

    response = requests.head(url, allow_redirects=True, timeout=_download._TIMEOUT)
    if response.status_code == 404:
        return False
    response.raise_for_status()
    return True


def fetch_hosted(task: str, model: str, dest: Path) -> bool:
    """Download the cache of `(task, model)` from the hosted registry into
    the directory `dest`. Returns False if the registry does not have it.

    A `HEAD` request on `corpus.parquet` decides: 404 means "not hosted";
    any other error (HTTP or network) is raised, so that a failed connection
    never silently starts a long local encode.

    Whether the task is retrieval is found by probing `qrels.parquet` with
    another `HEAD`: if it exists, `queries.parquet` and `qrels.parquet` are
    downloaded first, then `corpus.parquet`; otherwise (clustering) only
    `corpus.parquet`. Probing, rather than downloading the corpus first and
    reading its `task_type`, keeps `corpus.parquet` the last file to appear,
    straight from `_download.download`, which already writes atomically. An
    interrupted download leaves no `corpus.parquet`, so the next call starts
    again (files already complete are skipped by `download`).
    """
    if not _exists_hosted(_hosted_url(task, model, CORPUS_FILE)):
        return False
    is_retrieval = _exists_hosted(_hosted_url(task, model, QRELS_FILE))
    dest.mkdir(parents=True, exist_ok=True)
    files = [QUERIES_FILE, QRELS_FILE] if is_retrieval else []
    for filename in [*files, CORPUS_FILE]:
        _download.download(_hosted_url(task, model, filename), dest / filename)
    return True


# --------------------------------------------------------------------------
# Model metadata helpers
# --------------------------------------------------------------------------


def similarity_to_distance(meta: Any) -> str:
    """The aida distance name for the model metadata `meta` (an MTEB
    `ModelMeta`, only `similarity_fn_name` is read).

    `cosine`, `euclidean` and `dot` map to themselves. `MaxSim`
    (multi-vector models) and other functions raise `NotImplementedError`.
    `None` falls back to `"cosine"`, with a warning.
    """
    value = getattr(meta, "similarity_fn_name", None)
    value = getattr(value, "value", value)  # ScoringFunction enum -> str
    if value is None:
        _LOGGER.warning("the model declares no similarity function, assuming cosine")
        return "cosine"
    value = str(value)
    if value.lower() in ("maxsim", "max_sim"):
        raise NotImplementedError(
            "multi-vector models (MaxSim similarity) are not supported"
        )
    if value in ("cosine", "euclidean", "dot"):
        return value
    raise NotImplementedError(f"similarity function `{value}` is not supported")


def _import_mteb():
    try:
        import mteb
    except ImportError as e:
        raise ImportError(
            f"encoding MTEB tasks needs the `mteb` package: {_INSTALL_HINT}"
        ) from e
    return mteb


def check_revision(meta_in_file: dict, model: str) -> None:
    """Warn if the revision of `model` pinned by MTEB differs from
    `model_revision` in `meta_in_file` (the metadata of a cache). Does
    nothing if `mteb` is not installed or does not know the model."""
    try:
        import mteb

        pinned = mteb.get_model_meta(model).revision
    except Exception:  # no mteb, unknown model, no network: nothing to compare
        return
    stored = meta_in_file.get("model_revision")
    if stored is not None and pinned is not None and stored != pinned:
        _LOGGER.warning(
            "%s: the cache was encoded with revision %s, MTEB now pins %s",
            model,
            stored,
            pinned,
        )


# --------------------------------------------------------------------------
# Pure building blocks (no mteb needed)
# --------------------------------------------------------------------------


@dataclass
class Encoded:
    """Result of `encode_task`.

    `corpus` is `(n, d)` float32 with `ids` (retrieval) and/or `colors`
    (clustering). `queries`, `query_ids` and `qrels` (a table with columns
    `query_id`, `doc_id`, `score`) are None for clustering. `meta` is the
    extra metadata written into `corpus.parquet`.
    """

    corpus: np.ndarray
    ids: np.ndarray | None
    colors: Colors | None
    queries: np.ndarray | None
    query_ids: np.ndarray | None
    qrels: pa.Table | None
    meta: dict

    @property
    def is_retrieval(self) -> bool:
        return self.queries is not None


def _check_texts(texts: Sequence) -> None:
    for t in texts:
        if not isinstance(t, str):
            raise NotImplementedError(
                "only plain text is supported (no conversational queries or "
                f"multimodal inputs), got {type(t).__name__}"
            )


def doc_rows(texts: Sequence[str], titles: Sequence[str | None] | None = None) -> list[dict]:
    """Batch rows of corpus entries, shaped like MTEB's `CorpusInput`.

    `text` is `title + " " + text` when there is a non-empty title, else the
    stripped text, as `mteb._create_dataloaders` does. Every row has `title`
    (`""` if missing) and `body`, so that the rows have the same keys.
    """
    _check_texts(texts)
    rows = []
    for i, text in enumerate(texts):
        title = (titles[i] if titles is not None else None) or ""
        full = f"{title} {text}".strip() if title else text.strip()
        rows.append({"text": full, "title": title, "body": text})
    return rows


def query_rows(
    texts: Sequence[str], instructions: Sequence[str | None] | None = None
) -> list[dict]:
    """Batch rows of queries, shaped like MTEB's `QueryInput`.

    `query` is the text; `text` is the text followed by its instruction, if
    any. If `instructions` is given every row has an `instruction` key (`""`
    where missing), so that the rows have the same keys.
    """
    _check_texts(texts)
    rows = []
    for i, text in enumerate(texts):
        row = {"text": text, "query": text}
        if instructions is not None:
            instr = instructions[i]
            if instr is not None:
                row["text"] = f"{text} {instr}"
            row["instruction"] = instr or ""
        rows.append(row)
    return rows


def text_rows(texts: Sequence[str]) -> list[dict]:
    """Batch rows of clustering texts (`TextInput`)."""
    _check_texts(texts)
    return [{"text": t} for t in texts]


def _loader(rows: list[dict], batch_size: int):
    """A `torch` DataLoader over `rows`. `default_collate` turns a list of
    `{str: str}` into `{str: list[str]}`, the batch format of MTEB's
    `BatchedInput`. The rows are wrapped in a HF `Dataset` because MTEB's
    wrappers read `loader.dataset.features` to detect image/audio/video."""
    from datasets import Dataset
    from torch.utils.data import DataLoader

    return DataLoader(Dataset.from_list(rows), batch_size=batch_size, shuffle=False)


def clustering_colors_v1(
    sets: Iterable[tuple[Sequence, Sequence]],
) -> tuple[list[str], Colors]:
    """Unique texts and colors of a clustering task in the v1 format: several
    (overlapping) sets of `(sentences, labels)`.

    Returns `(texts, colors)`: the unique texts in order of first
    appearance, and one color `set{s}` per set, with the label of each text
    in that set or `"<absent>"` if it is not in it. Labels are `str(label)`;
    codes are the sorted positions of the labels (`"<absent>"` included), as
    `pd.factorize(sort=True)` gives. If a text occurs twice in a set, the
    last label wins. Every set is rebuilt exactly as the rows whose column
    is not `"<absent>"`, up to the order of its texts.
    """
    index: dict[str, int] = {}
    texts: list[str] = []
    members: list[tuple[np.ndarray, list[str]]] = []
    for sentences, labels in sets:
        if len(sentences) != len(labels):
            raise ValueError(f"{len(sentences)} sentences but {len(labels)} labels")
        idx = np.empty(len(sentences), dtype=np.int64)
        for k, text in enumerate(sentences):
            i = index.setdefault(text, len(texts))
            if i == len(texts):
                texts.append(text)
            idx[k] = i
        members.append((idx, [str(label) for label in labels]))

    n = len(texts)
    values = np.empty((n, len(members)), dtype=np.int64)
    all_labels = []
    for s, (idx, labels) in enumerate(members):
        column = np.full(n, ABSENT, dtype=object)
        column[idx] = labels
        codes, uniques = pd.factorize(column, sort=True)
        values[:, s] = codes
        all_labels.append(tuple(str(u) for u in uniques))
    names = tuple(f"set{s}" for s in range(len(members)))
    return texts, Colors(values, names, tuple(all_labels))


def clustering_colors_v2(labels: Sequence) -> Colors:
    """Colors of a clustering task in the v2 format (one row per text): the
    single color `label`. Hierarchical (list-valued) labels raise
    `NotImplementedError`."""
    for label in labels:
        if isinstance(label, (list, tuple, np.ndarray)):
            raise NotImplementedError("hierarchical clustering labels are not supported")
    codes, uniques = pd.factorize(pd.Series([str(label) for label in labels]), sort=True)
    return Colors(
        codes.astype(np.int64)[:, np.newaxis],
        ("label",),
        (tuple(str(u) for u in uniques),),
    )


def build_qrels(
    qrels: dict[str, dict[str, int]] | Iterable[tuple[str, str, int]],
) -> pa.Table:
    """The qrels table with columns `query_id` (string), `doc_id` (string)
    and `score` (int64), keeping only the positive scores. `qrels` is either
    `{query_id: {doc_id: score}}` (MTEB's `relevant_docs`) or an iterable of
    `(query_id, doc_id, score)`."""
    if isinstance(qrels, dict):
        triples = (
            (q, d, s) for q, docs in qrels.items() for d, s in docs.items()
        )
    else:
        triples = qrels
    rows = [(str(q), str(d), int(s)) for q, d, s in triples if s > 0]
    return pa.table(
        {
            "query_id": pa.array([r[0] for r in rows], type=pa.string()),
            "doc_id": pa.array([r[1] for r in rows], type=pa.string()),
            "score": pa.array([r[2] for r in rows], type=pa.int64()),
        }
    )


def write_cache(result: Encoded, directory: Path) -> None:
    """Write `result` into `directory`: `queries.parquet` and `qrels.parquet`
    first (retrieval), `corpus.parquet` last, as the completion marker."""
    directory.mkdir(parents=True, exist_ok=True)
    if result.is_retrieval:
        _write_parquet_cache(
            directory / QUERIES_FILE, result.queries, None, result.query_ids, result.meta
        )
        write_table(
            directory / QRELS_FILE,
            result.qrels,
            {"version": _QRELS_VERSION},
        )
    _write_parquet_cache(
        directory / CORPUS_FILE, result.corpus, result.colors, result.ids, result.meta
    )


# --------------------------------------------------------------------------
# Encoding (needs mteb, torch)
# --------------------------------------------------------------------------


def encode(model, rows, task, split, subset, prompt_type, batch_size, desc="Encoding"):
    """Encode `rows` (see `doc_rows`, `query_rows`, `text_rows`) through the
    MTEB model wrapper, so that model-specific prompts (e.g. e5's
    "query: " / "passage: ") are applied.

    Encodes in chunks of `batch_size * 16` rows so that a single progress bar
    tracks the whole encode (MTEB's per-call bars are disabled)."""
    from tqdm.auto import tqdm

    chunk_size = batch_size * 16
    chunks = []
    with tqdm(total=len(rows), desc=desc, unit="text", dynamic_ncols=True) as bar:
        for lo in range(0, len(rows), chunk_size):
            chunk = rows[lo : lo + chunk_size]
            emb = model.encode(
                _loader(chunk, batch_size),
                task_metadata=task.metadata,
                hf_split=split,
                hf_subset=subset,
                prompt_type=prompt_type,
                show_progress_bar=False,
            )
            chunks.append(np.asarray(emb, dtype=np.float32))
            bar.update(len(chunk))
    if not chunks:
        return np.empty((0, 0), dtype=np.float32)
    return np.concatenate(chunks)


def _column(ds, name: str, default=None):
    """Column `name` of a HF Dataset as a list, or `default` if missing."""
    return list(ds[name]) if name in ds.column_names else default


def _first_subset_split(task, split: str):
    """The `(subset, data)` of `task.dataset` for `split`: `data` is
    `dataset[split]` for single-subset tasks, else `dataset[subset][split]`
    with subset `"default"` if present, otherwise the first one."""
    ds = task.dataset
    if split in ds:
        return "default", ds[split]
    subset = "default" if "default" in ds else next(iter(ds))
    return subset, ds[subset][split]


def encode_task(
    task: str,
    model: str,
    *,
    batch_size: int = 64,
    model_kwargs: dict | None = None,
    limit: int | None = None,
) -> Encoded:
    """Encode MTEB task `task` with model `model`.

    Supports text-only retrieval and clustering tasks; others raise
    `NotImplementedError`. The model is loaded at the revision pinned by
    MTEB (`mteb.get_model_meta`), which is recorded in the metadata, and
    `model_kwargs` go to `mteb.get_model` (e.g. `{"device": "cuda:1"}`).

    The result is written to `cache_dir(task, model)`, `corpus.parquet`
    last. With `limit` (for testing) only the first `limit` documents and
    queries (texts, for clustering) are encoded and nothing is written, so a
    partial encode can never be read back as the full dataset.
    """
    mteb = _import_mteb()
    from mteb.types import PromptType

    mteb_task = mteb.get_task(task)
    meta = mteb_task.metadata
    if list(meta.modalities) != ["text"]:
        raise NotImplementedError(
            f"{task}: only text tasks are supported, got modalities {list(meta.modalities)}"
        )
    if meta.type not in ("Retrieval", "Clustering"):
        raise NotImplementedError(
            f"{task}: only Retrieval and Clustering tasks are supported, got {meta.type}"
        )
    mteb_task.load_data()
    split = meta.eval_splits[0]

    model_meta = mteb.get_model_meta(model, fetch_from_hf=True)
    distance = similarity_to_distance(model_meta)
    revision = model_meta.revision
    if revision == "no_revision_available":
        revision = None
    encoder = mteb.get_model(model, revision=revision, **(model_kwargs or {}))

    subset, data = _first_subset_split(mteb_task, split)
    extra = {
        "task": task,
        "subset": subset,
        "split": split,
        "task_type": meta.type,
        "model": model,
        "model_revision": model_meta.revision,
        "task_revision": meta.revision,
        "similarity": distance,
        "mteb_version": mteb.__version__,
    }

    if meta.type == "Retrieval":
        result = _encode_retrieval(
            encoder, mteb_task, data, subset, split, batch_size, limit, extra, PromptType
        )
    else:
        result = _encode_clustering(
            encoder, mteb_task, data, subset, split, batch_size, limit, extra
        )

    if limit is None:
        write_cache(result, cache_dir(task, model))
    else:
        _LOGGER.info("limit=%d: not writing the cache", limit)
    return result


def _encode_retrieval(
    encoder, task, data, subset, split, batch_size, limit, extra, PromptType
) -> Encoded:
    corpus, queries = data["corpus"], data["queries"]
    if limit is not None:
        corpus = corpus.select(range(min(limit, len(corpus))))
        queries = queries.select(range(min(limit, len(queries))))
    _LOGGER.info(
        "%s/%s/%s: %d docs, %d queries",
        task.metadata.name, subset, split, len(corpus), len(queries),
    )
    ids = np.array([str(i) for i in corpus["id"]], dtype=object)
    query_ids = np.array([str(i) for i in queries["id"]], dtype=object)
    docs = doc_rows(list(corpus["text"]), _column(corpus, "title"))
    instructions = _column(queries, "instruction")
    qrows = query_rows(list(queries["text"]), instructions)
    doc_emb = encode(encoder, docs, task, split, subset, PromptType.document, batch_size, "Corpus")
    q_emb = encode(encoder, qrows, task, split, subset, PromptType.query, batch_size, "Queries")
    return Encoded(
        doc_emb, ids, None, q_emb, query_ids, build_qrels(data["relevant_docs"]), extra
    )


def _encode_clustering(
    encoder, task, data, subset, split, batch_size, limit, extra
) -> Encoded:
    in_col, label_col = task.input_column_name, task.label_column_name
    if not isinstance(in_col, str):
        raise NotImplementedError("multi-column clustering inputs are not supported")
    sentences, labels = data[in_col], data[label_col]
    if len(sentences) and isinstance(sentences[0], (list, tuple)):
        # v1: one row per clustering set
        texts, colors = clustering_colors_v1(zip(sentences, labels))
        n_sets = len(colors.names)
    else:
        # v2: one row per text
        texts, colors, n_sets = list(sentences), clustering_colors_v2(list(labels)), 1
    if limit is not None:
        texts = texts[:limit]
        colors = colors.take(np.arange(len(texts)))
    _LOGGER.info(
        "%s/%s/%s: %d sets, %d texts",
        task.metadata.name, subset, split, n_sets, len(texts),
    )
    # No prompt type: same as MTEB's clustering evaluator (instruction models
    # get the task prompt).
    emb = encode(encoder, text_rows(texts), task, split, subset, None, batch_size, "Texts")
    return Encoded(emb, None, colors, None, None, None, extra)
