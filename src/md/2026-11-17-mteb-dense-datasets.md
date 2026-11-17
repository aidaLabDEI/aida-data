# Plan: MTEB encodings as dense datasets

Goal: turn the standalone `build_encodings.py` into part of `aida_data.dense`.
Any MTEB task can be encoded with any MTEB model and loaded with
`dense.load`, with the same cleaning and pipeline as the other dense datasets.
Pairs that are already built come from a hosted registry on the HF Hub. Pairs
that are not get encoded locally on first use and cached in `<root>/dense`.

## 1. Decisions (from the interview of 2026-11-17)

| topic | decision |
|---|---|
| where encoding happens | in `load()`, on demand. First check the hosted HF registry, then encode locally |
| task types | Retrieval and Clustering |
| naming | prefixed string: `dense.load("mteb:SCIDOCS:Qwen/Qwen3-Embedding-4B")` |
| qrels | separate function: `dense.load_qrels(name)`, keyed by the original MTEB ids |
| ids | new optional `Dataset.ids` / `Dataset.query_ids`, filtered together with the rows |
| clustering sets | one color per set (`set0`, `set1`, ...) over the unique texts |
| texts absent from a set | explicit label `"<absent>"` in that color's labels (codes stay dense and 0-based) |
| distance | taken from the model's MTEB metadata (`similarity_fn_name`) |
| storage | the existing parquet cache layout (`_write_parquet_cache`) plus an `id` column. Queries and qrels go in their own parquet files |
| texts | not stored |
| hosting | a dataset repo under the AIDA lab HF organization (name kept in a constant) |
| after a local encode | cache locally only. Uploading is a separate maintainer step |
| encode options | extra `**encode_kwargs` on `load`, forwarded to the encoder for `mteb:` names only |
| model revision | encode with the revision MTEB pins. Store it in the parquet metadata and warn on mismatch. It is not part of the path |

## 2. Public API

```python
from aida_data import dense

# Retrieval: corpus -> ds.dataset, queries -> ds.queries
ds = dense.load("mteb:SCIDOCS:Qwen/Qwen3-Embedding-4B")
ds.ids            # (n,) object array of corpus ids, aligned with ds.dataset
ds.query_ids      # (q,) object array of query ids, aligned with ds.queries
qrels = dense.load_qrels("mteb:SCIDOCS:Qwen/Qwen3-Embedding-4B")
# {query_id: {doc_id: score}}, positive judgments only

# Clustering: unique texts -> ds.dataset, one color per clustering set
ds = dense.load("mteb:ArxivClusteringP2P:Qwen/Qwen3-Embedding-4B")
ds.colors.names               # ("set0", ..., "set{k-1}")
ds.colors.labels[0]           # ("<absent>", "astro-ph", ...)
in_set0 = ds.colors.column("set0") != ds.colors.labels[0].index("<absent>")

# Local encoding options (ignored, with a warning, if the cache or a
# hosted copy exists)
ds = dense.load("mteb:SCIDOCS:sentence-transformers/all-MiniLM-L6-v2",
                batch_size=128, model_kwargs={"device": "cuda:1"})
```

- `name` is split on the first two `:`, as `mteb:<task>:<model>`. Model names
  contain `/` but never `:`.
- `available_datasets()` stays the list of registered names. MTEB names are
  open-ended, so they are not listed. (An optional `dense.hosted_mteb()` that
  lists the hosted repo is left for later, see §7.)
- `Dataset` gains `ids: np.ndarray | None = None` and
  `query_ids: np.ndarray | None = None`. Both default to None, so nothing
  changes for existing datasets.
- `load_qrels(name)` makes sure the cache exists (hosted download or local
  encode, taking the same `**encode_kwargs`). It reads `qrels.parquet` and
  raises `ValueError` for clustering tasks and for names without the `mteb:`
  prefix.

## 3. Files and layout

Local cache: `<root>/dense/mteb/<task>/<model_slug>/`, where
`model_slug = model.replace("/", "__")`. The hosted repo uses the same
relative layout:
`https://huggingface.co/datasets/<ORG>/<REPO>/resolve/main/<task>/<model_slug>/<file>`.

| file | content |
|---|---|
| `corpus.parquet` | `_write_parquet_cache` layout: `x0..x{d-1}` float32, `color*` (clustering only), and a new string column `id` (retrieval only) |
| `queries.parquet` | retrieval only: same layout, `x*` + `id` |
| `qrels.parquet` | retrieval only: `query_id` (str), `doc_id` (str), `score` (int), positive scores only |

The `aida_data` schema metadata of `corpus.parquet` gets these extra keys:
`task`, `subset`, `split`, `task_type`, `model`, `model_revision`,
`task_revision`, `similarity`, `mteb_version`, `has_ids`. The reader
uses `meta.get(...)` for all of them, so `_CACHE_VERSION` stays 1 and
the existing caches remain readable.

`corpus.parquet` is written **last**, both when encoding and when
downloading. Its presence is the completion marker. Each file is already
written atomically through `write_table` / `atomic_path`.

## 4. Implementation steps

### 4.1 `src/aida_data/_mteb.py` (new, private)

It holds everything that touches `mteb`. Imports are lazy, so the package
does not depend on `mteb`.

1. `parse_name(name) -> (task, model)` and `model_slug(model)`.
2. `cache_dir(task, model) -> Path` under `_dense_dir() / "mteb"`.
3. `fetch_hosted(task, model, dest) -> bool`:
   - Send a `HEAD` request for `corpus.parquet` on the HF URL. A 404 returns
     False. Any other HTTP or network error is raised. A failed connection
     must not silently start a multi-hour 4B-model encode.
   - Otherwise, download `queries.parquet` and `qrels.parquet` if the task is
     retrieval, then `corpus.parquet`, with `_download.download`.
   - Whether a task is retrieval is known after `corpus.parquet` is
     downloaded (from `task_type` in its metadata). So download the corpus to
     a temporary name first and rename it last, or else probe
     `qrels.parquet`. Pick one at implementation time.
4. `encode_task(task, model, *, batch_size=64, model_kwargs=None, limit=None)`:
   this is the code from `build_encodings.py`, cleaned up.
   - `meta = mteb.get_model_meta(model)`, then
     `mteb.get_model(model, revision=meta.revision, **model_kwargs)`.
   - `encode(...)`: the chunked encoder with a progress bar. It builds its
     own loader and does not use the private
     `mteb._create_dataloaders.create_dataloader` (see §4.1.1).
   - Retrieval: build corpus, queries and qrels. Subset is `"default"` if
     present, otherwise the first one. Split is `eval_splits[0]`. Both are
     recorded in the metadata.
   - Clustering: unique texts plus memberships, as now. Then build the
     `Colors` (§4.3).
   - `limit` is only for testing. With `limit` set, the result is returned
     but **not** written to the cache, so a partial encode can never be read
     back as the full dataset.
   - Remove the `icecream` dependency and the debug `ic(corpus)`.
5. `similarity_to_distance(meta)`: `cosine` → `"cosine"`, `euclidean` →
   `"euclidean"`, `dot` → `"dot"`. `max_sim` (multi-vector models) raises
   `NotImplementedError`. `None` falls back to `"cosine"` with a warning.
6. `check_revision(meta_in_file, model)`: if `mteb` can be imported and the
   revision pinned by MTEB differs from the stored one, log a warning.
   Without `mteb`, it does nothing.
7. An `ImportError` with the hint `pip install aida-data[mteb]` when a
   local encode is needed and `mteb` is not installed.

### 4.1.1 Encoding through the public MTEB API only

`build_encodings.py` uses `create_dataloader` only to turn a HF `Dataset`
into a `torch.utils.data.DataLoader` of batches. For text tasks, this is all
it does (read in mteb 2.27.14, `_create_dataloaders.py`):

- **documents** (`PromptType.document`): `text = (title + " " + text).strip()`
  if the title is non-empty, else `text.strip()`. It also adds `body` (the
  original text) and `title`;
- **queries** (`PromptType.query`): `query = text`. If an `instruction`
  column exists, `text = text + " " + instruction`. Conversations (`text` is a
  list) are flattened into `"role: content; ..."`;
- **clustering** (`prompt_type=None`): batches of `{"text": [...]}` only;
- image/audio/video handling, which text tasks do not need.

The batch format is part of MTEB's public types (`mteb.types.BatchedInput`,
and the `TextInput` / `CorpusInput` / `QueryInput` TypedDicts it documents).
`EncoderProtocol.encode(inputs: DataLoader[BatchedInput], ...)` is the public
model interface. So `_mteb.py` builds the loader itself:

```python
from torch.utils.data import DataLoader

def _doc_rows(corpus):       # -> list[CorpusInput-shaped dict]
    for title, text in zip(titles_or_empty(corpus), corpus["text"]):
        title = title or ""
        full = f"{title} {text}".strip() if title else text.strip()
        yield {"text": full, "title": title, "body": text}

def _query_rows(queries):    # -> list[QueryInput-shaped dict]
    instr = queries["instruction"] if "instruction" in queries.column_names else None
    for i, text in enumerate(queries["text"]):
        row = {"text": text, "query": text}
        if instr is not None and instr[i] is not None:
            row["text"] = f"{text} {instr[i]}"
            row["instruction"] = instr[i]
        yield row

def _loader(rows: list[dict], batch_size: int) -> DataLoader:
    # default_collate turns a list of {str: str} into {str: list[str]}
    return DataLoader(rows, batch_size=batch_size, shuffle=False)
```

Notes:

- Every document row has `title` and `body`, with `""` for a missing title.
  This keeps batches uniform; `default_collate` fails on mixed keys. If any
  query in a task has an instruction, every row gets an `instruction` key
  (with `""` where it is missing), for the same reason.
- Only bi-encoder wrappers are used here, and in mteb 2.27.14 they read only
  `batch["text"]`. `query` and `instruction` are read only by rerankers. They
  are still filled in to match the documented types.
- Scope: text-only tasks (`task.metadata.modalities == ["text"]`).
  Conversational queries (`text` is a list) and multimodal tasks raise
  `NotImplementedError`.
- The MTEB API used is all public: `mteb.get_task`, `task.load_data()`,
  `task.dataset`, `task.metadata`, `mteb.get_model` / `get_model_meta`,
  `EncoderProtocol.encode`, and `mteb.types.PromptType`. `torch` comes with
  `sentence-transformers`.
- Drift check: a test marked `network` encodes a few rows of a small
  retrieval task twice, once through `mteb.evaluate`'s code path and once
  through `_loader`, and compares the vectors. This test may call the private
  `create_dataloader`, because only the test depends on it. Without the
  `mteb` extra, the test is skipped.

### 4.2 `src/aida_data/dense.py`

1. Add `Dataset.ids`, `Dataset.query_ids`.
2. Extend `_write_parquet_cache` / `_read_parquet_cache` with an optional
   `ids` array (the `id` column) and an optional `extra_meta` dict. Readers
   return `ids=None` when `has_ids` is missing or false.
3. In `load`, branch on `name.startswith("mteb:")` **before** the registry
   lookup, because `local_path` / `_DATASETS_INFO` would raise `KeyError`:
   - make sure the cache exists: local, then `fetch_hosted`, then
     `encode_task`;
   - read `train, ids, colors` from `corpus.parquet` and
     `test, query_ids` from `queries.parquet`. Distance comes from the
     metadata, and `distances = None`;
   - warn when `encode_kwargs` are given but not used (cache or hosted hit);
   - registered names with `encode_kwargs` raise `TypeError`.
4. Generalize `_take` so it filters `ids` together with `data` and `colors`.
   Apply the same masks to `test` / `query_ids` in the NaN and zero-row
   filters. Deduplication still compares only features and colors, not ids.
   Two documents with identical vectors keep only the first id. This goes in
   the docstring, and `deduplicate=False` keeps them all.
5. `load_qrels(name, **encode_kwargs)`.
6. `prune_raw`: check that `stale_temporaries(_dense_dir())` also finds stale
   files under `dense/mteb/**`. If it does not search recursively, add the
   MTEB subtree. MTEB caches have no raw files to prune.

### 4.3 Clustering colors

For each set `s`, the color column `set{s}` has `n_unique` entries. Each entry
is either the label of that text in set `s` or `"<absent>"`. The labels are
`str(label)` (MTEB labels can be ints), sorted, with `"<absent>"` among them,
and encoded with `pd.factorize(sort=True)` as the other loaders do. Every
original set can be rebuilt exactly as the rows where its column is not
`"<absent>"`.

The current code supports only the v1 format (one row per set with
`sentences` / `labels` lists). For v2 tasks (one row per text), use a single
color `label`. Hierarchical (list-valued) labels raise `NotImplementedError`
for now. The exact v2 schema needs checking against MTEB 2.x when this is
implemented.

### 4.4 Packaging

- `pyproject.toml`: add the extra `mteb = ["mteb>=2.27.14", "sentence-transformers>=6.3.0"]`.
  Check that this `mteb` version supports `requires-python >= 3.12`; the
  script header asked for 3.13.
- No private MTEB imports (§4.1.1). The usual upper bound `mteb<3` still
  applies, because the batch-format contract could change in a major
  release.
- Remove the top-level `build_encodings.py` and replace it with two
  maintainer scripts:
  - `scripts/build_mteb.py` (PEP 723, depends on `aida-data[mteb]`) calls
    `dense.load(name, **kwargs)` to build a local cache;
  - `scripts/upload_mteb.py` (PEP 723, `huggingface_hub`) uploads
    `<root>/dense/mteb/<task>/<model_slug>/` to the lab repo, given an
    `HF_TOKEN`.

### 4.5 Constants

```python
_MTEB_HF_REPO = "<lab-org>/aida-data-mteb"   # TODO: real name
_MTEB_HF_BASE = f"https://huggingface.co/datasets/{_MTEB_HF_REPO}/resolve/main/"
```

## 5. Tests (`tests/test_mteb.py`)

These run offline by default and use small synthetic data, with no `mteb`:

- `parse_name` / `model_slug`, including bad names.
- Writing and reading the parquet cache with `ids` and the extra metadata.
  Old caches without the new keys still load.
- `ids` / `query_ids` stay aligned through dedup, NaN filtering, zero-row
  filtering (`cosine`) and a pipeline.
- Building clustering colors from fake v1 sets: `"<absent>"` coding, and every
  set rebuilt exactly.
- The `load` flow with `fetch_hosted` / `encode_task` monkeypatched:
  - a local hit calls neither;
  - a hosted 404 falls back to encoding;
  - other HTTP errors are raised;
  - `limit` does not write a cache;
  - unused `encode_kwargs` log a warning;
  - registered names with `encode_kwargs` raise.
- `load_qrels`: round trip, positive scores only, `ValueError` for
  clustering.

Tests marked `network`:

- Download one hosted pair end to end.
- Encode a small task with `sentence-transformers/all-MiniLM-L6-v2` on CPU
  (needs the `mteb` extra; skip if it is not installed).

## 6. Docs

- README: a new `<details>` section "MTEB embeddings" covering the name
  format, ids and qrels, clustering colors, the `mteb` extra, and the local
  cache path.
- CHANGELOG entry.
- Mark this plan as implemented with a summary at the end, as the earlier
  plans do.

## 7. Open points / later

- Multilingual tasks with several subsets: only `default` or the first
  subset is used. A later version could support `mteb:<task>/<subset>:<model>`.
- `dense.hosted_mteb()`, which would list the hosted repo through the HF tree
  API.
- The first batch of hosted (task, model) pairs to build and upload, for
  example `ArxivClusteringP2P` and `SCIDOCS` with `Qwen/Qwen3-Embedding-4B`.
