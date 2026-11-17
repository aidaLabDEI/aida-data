# Plan: short aliases for MTEB dense datasets

Follow-up to `2026-11-17-mteb-dense-datasets.md`, which must be implemented
first.

Goal: let users write short names such as `scidocs-qwen3-4b` instead of the
full form `mteb:SCIDOCS:Qwen/Qwen3-Embedding-4B`, everywhere a dense dataset
name is accepted (`dense.load`, `dense.load_qrels`), and list them in
`dense.available_datasets()`, so the hosted MTEB encodings can be discovered
like the other dense datasets.

## 1. Behaviour

```python
from aida_data import dense

dense.available_datasets()       # [..., "arxiv-clustering-p2p-qwen3-4b", "scidocs-qwen3-4b"]
ds = dense.load("scidocs-qwen3-4b")             # same as the full mteb: name
qrels = dense.load_qrels("scidocs-qwen3-4b")

dense.resolve_name("scidocs-qwen3-4b")          # "mteb:SCIDOCS:Qwen/Qwen3-Embedding-4B"
dense.register_alias("scidocs-minilm", "mteb:SCIDOCS:sentence-transformers/all-MiniLM-L6-v2")
```

- An alias is only another name for its target. It shares the same cache
  directory (`dense/mteb/<task>/<model_slug>/`), the same hosted files and the
  same `**encode_kwargs`. Nothing is stored under the alias name.
- The full `mteb:<task>:<model>` form keeps working, for any pair.
- An alias may point to a pair that is not hosted. Loading it then encodes
  locally, as the full name would.

## 2. Naming convention

`<task_slug>-<model_tag>`, lowercase and hyphenated like the existing dense
names (`glove-100-angular`, `higgs-highlevel`).

- `task_slug(task)`: split CamelCase and acronyms into words, lowercase them
  and join with `-`. For example, `ArxivClusteringP2P` becomes
  `arxiv-clustering-p2p`, `SCIDOCS` becomes `scidocs`, and `NFCorpus`
  becomes `nfcorpus`. The rule is a small regex. The generated slugs are
  checked in a test against an explicit table (§5). Where the rule gives an
  ugly result, a manual override in `_TASK_SLUGS` wins.
- `model_tag(model)`: an explicit table only. Model names are too irregular
  to abbreviate automatically:

  ```python
  _MODEL_TAGS = {
      "Qwen/Qwen3-Embedding-4B": "qwen3-4b",
      "Qwen/Qwen3-Embedding-0.6B": "qwen3-0.6b",
      "sentence-transformers/all-MiniLM-L6-v2": "minilm-l6",
  }
  ```

  The example in the request, `scidocs-quen3-4B`, becomes `scidocs-qwen3-4b`:
  "qwen" with a w, and lowercase like every other name.

## 3. Registry

All of it lives in `dense.py`, next to `_DATASETS_INFO`.

1. `_ALIASES: dict[str, str]`, from alias to canonical `mteb:` name.
2. `register_alias(alias, target, force=False)`:
   - `target` must be a full `mteb:<task>:<model>` name. It is validated with
     `_mteb.parse_name`. Aliases of aliases are not allowed, so resolution
     takes one step and cannot loop.
   - `alias` must not start with `mteb:` and must not contain `:`.
   - It raises `ValueError` if `alias` is already a registered dataset name or
     an alias, unless `force=True`. The same rule as `register`.
   - `register` gets the symmetric check: registering a dataset under an
     existing alias name raises `ValueError`.
3. `resolve_name(name) -> str` returns `_ALIASES.get(name, name)`. It is
   public, so users can see which pair and which cache an alias uses.
4. Hosted pairs: a static list in `_mteb.py`.

   ```python
   _HOSTED = [
       ("ArxivClusteringP2P", "Qwen/Qwen3-Embedding-4B"),
       ("SCIDOCS", "Qwen/Qwen3-Embedding-4B"),
   ]
   ```

   At import time, `dense.py` registers `f"{task_slug(t)}-{model_tag(m)}"`
   for each pair. A model without a tag in `_MODEL_TAGS` is an error at
   import time, so a missing tag is caught by the test suite rather than by
   users. The list is updated whenever pairs are uploaded with
   `scripts/upload_mteb.py`. That script prints the line to add, and refuses
   to upload a model that has no tag.
5. Aliases are not checked against the network at import time. Importing
   `aida_data.dense` must stay offline.

## 4. Changes to existing functions

- `load(name, ...)` and `load_qrels(name, ...)` call `resolve_name` first.
  The rest of the code, including the `mteb:` branch, logs and error
  messages, sees only the canonical name. The `KeyError` for unknown names
  lists the aliases too.
- `available_datasets()` returns registered names followed by aliases, both
  sorted within their group. Full `mteb:` names are open-ended and are still
  not listed.
- `local_path`, `_cache_exists` and `prune_raw` iterate over
  `_DATASETS_INFO` only, so they are unchanged. Aliases have no raw files.

## 5. Tests (`tests/test_aliases.py`, offline)

- `task_slug` on a table of real MTEB task names, including
  `ArxivClusteringP2P`, `SCIDOCS`, `NFCorpus`, `MSMARCO`, `StackExchangeClustering`
  and `TwentyNewsgroupsClustering`, plus the overrides.
- Every model in `_HOSTED` has a tag, and every generated alias is unique and
  matches `^[a-z0-9.-]+$`.
- `register_alias`:
  - rejects collisions with datasets and other aliases;
  - rejects targets that are not `mteb:` names, targets that are aliases, and
    aliases that contain `:`;
  - `force=True` overrides.
- `register` rejects a name that is already an alias.
- `load` with an alias and `load` with its full name resolve to the same
  cache directory. Encoding and downloading are monkeypatched, as in the
  MTEB plan's tests.
- `available_datasets()` contains the hosted aliases.

## 6. Docs

- README, in the "MTEB embeddings" section: the alias table (alias, task,
  model) for the hosted pairs, the naming rule, `register_alias` and
  `resolve_name`.
- A CHANGELOG entry.

## 7. Open points

- Case-insensitive lookup, so that `scidocs-qwen3-4B` also works. Existing
  names are matched exactly, so this plan matches aliases exactly too.
  Lowercasing only aliases would be inconsistent.
- Aliases for pairs that are not hosted but are commonly used. For now only
  hosted pairs get built-in aliases, and users add their own with
  `register_alias`.
- Generating `_HOSTED` from the HF repo listing instead of keeping it static.
  That needs network access at import time or a cached index, which is
  rejected in §3.5 for now.
