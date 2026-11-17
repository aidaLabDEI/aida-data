"""Offline tests for the short aliases of MTEB dense datasets."""

import re
import subprocess
import sys

import pytest

# Reuse the faked fetch/encode flow of the MTEB tests.
from test_mteb import _retrieval_result, flow  # noqa: F401

from aida_data import _mteb, dense
from aida_data.dense import DatasetInfo


@pytest.fixture(autouse=True)
def _restore_registries(monkeypatch):
    """Tests may add aliases and datasets: restore both registries after."""
    monkeypatch.setattr(dense, "_ALIASES", dict(dense._ALIASES))
    monkeypatch.setattr(dense, "_DATASETS_INFO", dict(dense._DATASETS_INFO))


@pytest.mark.parametrize(
    "task, slug",
    [
        ("ArxivClusteringP2P", "arxiv-clustering-p2p"),
        ("ArxivClusteringS2S", "arxiv-clustering-s2s"),
        ("SCIDOCS", "scidocs"),
        ("MSMARCO", "msmarco"),
        ("StackExchangeClustering", "stack-exchange-clustering"),
        ("TwentyNewsgroupsClustering", "twenty-newsgroups-clustering"),
        ("HotpotQA", "hotpot-qa"),
        ("ClimateFEVER", "climate-fever"),
        ("CQADupstackAndroidRetrieval", "cqa-dupstack-android-retrieval"),
        ("Banking77Classification", "banking77-classification"),
        ("TRECCOVID", "treccovid"),
        ("NQ", "nq"),
        # overrides
        ("NFCorpus", "nfcorpus"),
        ("SciFact", "scifact"),
        ("ArguAna", "arguana"),
        ("FiQA2018", "fiqa2018"),
        ("DBPedia", "dbpedia"),
    ],
)
def test_task_slug(task, slug):
    assert dense._task_slug(task) == slug


def test_overrides_are_needed():
    """Every override differs from what the rule gives."""
    for task, slug in dense._TASK_SLUGS.items():
        assert "-".join(w.lower() for w in dense._WORD_RE.findall(task)) != slug


def test_hosted_aliases():
    aliases = [dense._default_alias(t, m) for t, m in _mteb._HOSTED]
    assert len(set(aliases)) == len(aliases)
    for (task, model), alias in zip(_mteb._HOSTED, aliases):
        assert model in dense._MODEL_TAGS
        assert re.fullmatch(r"[a-z0-9.-]+", alias)
        assert dense.resolve_name(alias) == f"mteb:{task}:{model}"
    assert "scidocs-qwen3-4b" in aliases
    assert "arxiv-clustering-p2p-qwen3-4b" in aliases


def test_model_tags_are_valid():
    tags = list(dense._MODEL_TAGS.values())
    assert len(set(tags)) == len(tags)
    assert all(re.fullmatch(r"[a-z0-9.-]+", t) for t in tags)


def test_missing_model_tag_raises():
    with pytest.raises(KeyError, match="_MODEL_TAGS"):
        dense._default_alias("SCIDOCS", "org/untagged")


def test_resolve_name():
    assert dense.resolve_name("scidocs-qwen3-4b") == "mteb:SCIDOCS:Qwen/Qwen3-Embedding-4B"
    assert dense.resolve_name("glove-100-angular") == "glove-100-angular"
    assert dense.resolve_name("mteb:T:org/m") == "mteb:T:org/m"


def test_available_datasets_lists_aliases_last():
    names = dense.available_datasets()
    registered = sorted(dense._DATASETS_INFO)
    assert names[: len(registered)] == registered
    assert names[len(registered) :] == sorted(dense._ALIASES)
    assert "scidocs-qwen3-4b" in names
    assert not any(n.startswith("mteb:") for n in names)


# --- register_alias / register


def test_register_alias():
    dense.register_alias("scidocs-minilm", "mteb:SCIDOCS:sentence-transformers/all-MiniLM-L6-v2")
    assert (
        dense.resolve_name("scidocs-minilm")
        == "mteb:SCIDOCS:sentence-transformers/all-MiniLM-L6-v2"
    )
    assert "scidocs-minilm" in dense.available_datasets()


def test_register_alias_rejects_collisions():
    with pytest.raises(ValueError, match="registered dataset"):
        dense.register_alias("glove-100-angular", "mteb:T:org/m")
    with pytest.raises(ValueError, match="already an alias"):
        dense.register_alias("scidocs-qwen3-4b", "mteb:T:org/m")


@pytest.mark.parametrize(
    "alias, target",
    [
        ("x", "glove-100-angular"),  # not an mteb: name
        ("x", "scidocs-qwen3-4b"),  # an alias
        ("x", "mteb:T"),  # malformed
        ("a:b", "mteb:T:org/m"),
        ("mteb:T:org/m", "mteb:T:org/m"),
        ("", "mteb:T:org/m"),
    ],
)
def test_register_alias_rejects_bad_names(alias, target):
    with pytest.raises(ValueError):
        dense.register_alias(alias, target)


def test_register_alias_force_overrides():
    dense.register_alias("scidocs-qwen3-4b", "mteb:T:org/m", force=True)
    assert dense.resolve_name("scidocs-qwen3-4b") == "mteb:T:org/m"
    dense.register_alias("glove-100-angular", "mteb:T:org/m", force=True)
    assert dense.resolve_name("glove-100-angular") == "mteb:T:org/m"


def test_register_rejects_alias_name():
    info = DatasetInfo("scidocs-qwen3-4b", "file:///x", lambda p: None, "euclidean")
    with pytest.raises(ValueError, match="alias"):
        dense.register(info)
    dense.register(info, force=True)
    assert "scidocs-qwen3-4b" in dense._DATASETS_INFO


def test_unknown_name_error_lists_aliases():
    with pytest.raises(KeyError, match="scidocs-qwen3-4b"):
        dense.load("no-such-dataset")


# --- load / load_qrels


def test_alias_and_full_name_share_cache(flow):  # noqa: F811
    calls, _ = flow
    dense.register_alias("t-m", "mteb:T:org/m")
    ds_alias = dense.load("t-m")
    assert calls == ["fetch", ("encode", {})]
    assert (_mteb.cache_dir("T", "org/m") / _mteb.CORPUS_FILE).exists()
    # the full name now hits the cache written for the alias
    ds_full = dense.load("mteb:T:org/m")
    assert calls == ["fetch", ("encode", {})]
    assert list(ds_alias.ids) == list(ds_full.ids)
    assert dense.load_qrels("t-m") == dense.load_qrels("mteb:T:org/m")
    assert len(calls) == 2


def test_alias_forwards_encode_kwargs(flow):  # noqa: F811
    calls, _ = flow
    dense.register_alias("t-m", "mteb:T:org/m")
    dense.load("t-m", limit=2)
    assert calls == ["fetch", ("encode", {"limit": 2})]


def test_builtin_alias_loads_its_pair(flow):  # noqa: F811
    calls, _ = flow
    _mteb.write_cache(
        _retrieval_result(), _mteb.cache_dir("SCIDOCS", "Qwen/Qwen3-Embedding-4B")
    )
    ds = dense.load("scidocs-qwen3-4b")
    assert calls == []
    assert list(ds.ids) == ["d0", "d1", "d2"]


# --- imports


@pytest.mark.parametrize("first", ["dense", "_mteb"])
def test_import_order(first):
    """Aliases are registered whichever module is imported first."""
    code = (
        f"from aida_data import {first}\n"
        "from aida_data import dense\n"
        "assert dense.resolve_name('scidocs-qwen3-4b').startswith('mteb:')\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
