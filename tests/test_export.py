"""`aida_data.export`, offline: name resolution, link path rules, conflicts,
modes, atomicity and the cache lookups, with `DATASETS_DIR` patched."""

import errno
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from aida_data import _mteb, dense, export, graph, timeseries
from aida_data.dense import Colors

MTEB = "mteb:T:org/m"
POINTS = np.arange(12, dtype=np.float32).reshape(3, 4)
IDS = np.array(["a", "b", "c"], dtype=object)


pytestmark = pytest.mark.usefixtures("roots")


def _dense_cache(root):
    (root / "dense").mkdir(parents=True, exist_ok=True)
    path = root / "dense" / "fake-cache.parquet"
    dense._write_parquet_cache(path, POINTS, Colors(np.zeros((3, 1), dtype=np.uint8), ("c",), (("x",),)), IDS)
    return path


def _mteb_cache():
    result = _mteb.Encoded(POINTS, IDS, None, None, None, None, {"similarity": "cosine", "task_type": "Clustering"})
    _mteb.write_cache(result, _mteb.cache_dir("T", "org/m"))
    return _mteb.cache_dir("T", "org/m")


def _native(root):
    (root / "dense").mkdir(parents=True, exist_ok=True)
    path = dense.local_path("fake-native")
    path.write_bytes(b"native")
    return path


# --- resolution


def test_registered_names_are_unique_across_modules(monkeypatch):
    monkeypatch.undo()  # the real registries, without the fixture's fakes
    names = [e.link_stem for e in export.entries()]
    assert len(names) == len(set(names))


def test_resolve_plain_and_prefix():
    assert export.resolve("fake") == export.Entry("fake", "dense", "parquet", "fake")
    assert export.resolve("fake-native").native == "hdf5"
    assert export.resolve("fake-graph") == export.Entry("fake-graph", "graph", "parquet", "fake-graph")
    assert export.resolve("timeseries/fake-ts").module == "timeseries"
    assert export.resolve("dense/fake").module == "dense"


def test_resolve_alias_keeps_alias_as_stem():
    assert export.resolve("t-alias") == export.Entry(MTEB, "dense", "parquet-dir", "t-alias")
    assert export.resolve("dense/t-alias").link_stem == "t-alias"


def test_resolve_full_mteb_name():
    assert export.resolve(MTEB) == export.Entry(MTEB, "dense", "parquet-dir", "T__org__m")
    assert export.resolve("dense/" + MTEB).name == MTEB
    # With a default alias, that is the stem.
    name = "mteb:SCIDOCS:Qwen/Qwen3-Embedding-4B"
    assert export.resolve(name).link_stem == "scidocs-qwen3-4b"
    with pytest.raises(export.UnknownDataset, match="not a graph one"):
        export.resolve("graph/" + MTEB)


def test_resolve_unknown_with_suggestion():
    with pytest.raises(export.UnknownDataset) as info:
        export.resolve("fake-grap")
    assert str(info.value).startswith("unknown dataset 'fake-grap'; did you mean: fake-graph")
    with pytest.raises(KeyError):
        export.resolve("zzzzzzzz")
    with pytest.raises(export.UnknownDataset, match="^unknown dataset 'zzzzzzzz'$"):
        export.resolve("zzzzzzzz")


def test_resolve_wrong_module_prefix():
    with pytest.raises(export.UnknownDataset, match="use graph/fake-graph"):
        export.resolve("dense/fake-graph")


def test_resolve_ambiguous(monkeypatch):
    graph.register(graph.DatasetInfo("fake-ts", "file:///e", "file:///c", None))
    with pytest.raises(export.UnknownDataset, match="graph/fake-ts or timeseries/fake-ts"):
        export.resolve("fake-ts")
    assert export.resolve("timeseries/fake-ts").module == "timeseries"


def test_entries_order_and_contents():
    names = [e.link_stem for e in export.entries()]
    d, g, t = (
        [e.link_stem for e in export.entries(m)] for m in ("dense", "graph", "timeseries")
    )
    assert names == d + g + t
    assert d == dense.available_datasets()
    assert "t-alias" in d and "fake-graph" in g and "fake-ts" in t
    assert not any(n.startswith("mteb:") for n in names)
    assert {e.native for e in export.entries("dense")} == {"hdf5", "parquet", "parquet-dir"}


def test_exporting_dense_does_not_import_graph_or_timeseries():
    code = (
        "import sys\n"
        "from aida_data import export\n"
        "export.resolve('glove-100-angular'); export.cached_path(export.resolve('adult'), 'parquet')\n"
        "export.cached_path(export.resolve('mteb:T:a/b'), 'hdf5')\n"
        "assert 'aida_data.graph' not in sys.modules and 'aida_data.timeseries' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


# --- cache lookups


def test_cached_path_and_is_cached(roots):
    entry = export.resolve("fake")
    assert export.cached_path(entry, "parquet") == roots / "dense" / "fake-cache.parquet"
    assert export.cached_path(entry, "hdf5") == roots / "dense" / "hdf5" / "fake.hdf5"
    assert not export.is_cached(entry, "parquet") and not export.is_cached(entry, "hdf5")
    _dense_cache(roots)
    assert export.is_cached(entry, "parquet") and not export.is_cached(entry, "hdf5")
    export.ensure(entry, "hdf5")
    assert export.is_cached(entry, "hdf5")
    assert export.available_formats(entry) == ("parquet", "hdf5")


def test_graph_and_timeseries_paths(roots):
    assert export.cached_path(export.resolve("fake-graph"), "parquet") == graph.cache_path("fake-graph")
    assert export.cached_path(export.resolve("fake-ts"), "parquet") == timeseries.cache_path("fake-ts")


def test_native_hdf5_lookups(roots):
    entry = export.resolve("fake-native")
    assert export.available_formats(entry) == ("hdf5",)
    assert export.cached_path(entry, "hdf5") == dense.local_path("fake-native")
    assert not export.is_cached(entry, "hdf5")
    _native(roots)
    assert export.is_cached(entry, "hdf5")
    assert not export.is_cached(entry, "parquet")
    with pytest.raises(export.FormatNotAvailable, match="fake-native is only available as hdf5"):
        export.cached_path(entry, "parquet")
    with pytest.raises(export.FormatNotAvailable):
        export.ensure(entry, "parquet")


def test_mteb_cached(roots):
    entry = export.resolve("t-alias")
    path = export.cached_path(entry, "parquet")
    assert path == _mteb.cache_dir("T", "org/m")
    assert not export.is_cached(entry, "parquet")
    _mteb_cache()
    assert export.is_cached(entry, "parquet")
    assert export.ensure(entry, "parquet") == path
    hdf5 = export.ensure(entry, "hdf5")
    assert hdf5 == roots / "dense" / "hdf5" / "mteb" / "T" / "org__m.hdf5"
    assert export.is_cached(entry, "hdf5")


def test_mteb_not_available(roots):
    with pytest.raises(LookupError, match="not hosted and not cached"):
        export.ensure(export.resolve("t-alias"), "parquet")


# --- link path


def test_default_destination_is_cwd(roots, work):
    _dense_cache(roots)
    link = export.export("fake", format="parquet")
    assert link == Path("fake.parquet")
    assert link.is_symlink()
    assert (work / "fake.parquet").resolve() == roots / "dense" / "fake-cache.parquet"


def test_destination_directory_and_symlink_to_directory(roots, work):
    _dense_cache(roots)
    (work / "d").mkdir()
    (work / "l").symlink_to("d")
    assert export.export("fake", work / "d") == work / "d" / "fake.hdf5"
    assert export.export("fake", work / "l", force=True) == work / "l" / "fake.hdf5"
    assert (work / "d" / "fake.hdf5").exists()


def test_explicit_destination_and_missing_parent(roots, work):
    _dense_cache(roots)
    assert export.export("fake", work / "x.h5") == work / "x.h5"
    assert (work / "x.h5").exists()
    with pytest.raises(FileNotFoundError):
        export.export("fake", work / "missing" / "x.h5")
    assert not (work / "missing").exists()


def test_mteb_directory_has_no_suffix(roots, work):
    _mteb_cache()
    assert export.export("t-alias", format="parquet") == Path("t-alias")
    assert export.export("t-alias") == Path("t-alias.hdf5")


# --- conflicts


def test_existing_file(roots, work):
    _dense_cache(roots)
    (work / "fake.hdf5").write_text("old")
    with pytest.raises(FileExistsError, match="pass --force to replace it"):
        export.export("fake")
    assert (work / "fake.hdf5").read_text() == "old"
    export.export("fake", force=True)
    assert (work / "fake.hdf5").is_symlink()


def test_broken_symlink_with_force(roots, work):
    _dense_cache(roots)
    (work / "fake.hdf5").symlink_to("nowhere")
    with pytest.raises(FileExistsError):
        export.export("fake")
    export.export("fake", force=True)
    assert (work / "fake.hdf5").exists()


def test_directory_is_never_replaced(roots, work):
    _dense_cache(roots)
    (work / "fake.hdf5").mkdir()
    with pytest.raises(IsADirectoryError):
        export.export("fake", force=True)
    assert (work / "fake.hdf5").is_dir()
    assert not [p for p in work.iterdir() if p.name != "fake.hdf5"]


def test_link_onto_the_cache_itself(roots):
    path = _dense_cache(roots)
    export.ensure(export.resolve("fake"), "hdf5")
    before = path.read_bytes()
    for force in (False, True):
        with pytest.raises(FileExistsError, match="cached file itself"):
            export.export("fake", roots / "dense" / "fake-cache.parquet", format="parquet", force=force)
        with pytest.raises(FileExistsError, match="cached file itself"):
            export.export("fake", roots / "dense" / "hdf5", force=force)
    assert path.read_bytes() == before and not path.is_symlink()


# --- modes


def test_symlink_is_absolute(roots, work):
    _dense_cache(roots)
    link = export.export("fake", format="parquet")
    assert os.path.isabs(os.readlink(link))
    assert link.resolve() == roots / "dense" / "fake-cache.parquet"


def test_copy_is_independent(roots, work):
    cache = _dense_cache(roots)
    link = export.export("fake", format="parquet", mode="copy")
    assert not link.is_symlink() and link.read_bytes() == cache.read_bytes()
    link.write_bytes(b"changed")
    assert cache.read_bytes() != b"changed"


def test_copy_directory(roots, work):
    cache = _mteb_cache()
    link = export.export("t-alias", format="parquet", mode="copy")
    assert link.is_dir() and not link.is_symlink()
    assert sorted(p.name for p in link.iterdir()) == sorted(p.name for p in cache.iterdir())
    (link / _mteb.CORPUS_FILE).write_bytes(b"x")
    assert (cache / _mteb.CORPUS_FILE).read_bytes() != b"x"


def test_hardlink(roots, work):
    cache = _dense_cache(roots)
    link = export.export("fake", format="parquet", mode="hardlink")
    assert not link.is_symlink() and link.stat().st_ino == cache.stat().st_ino


def test_hardlink_directory(roots, work):
    cache = _mteb_cache()
    link = export.export("t-alias", format="parquet", mode="hardlink")
    assert link.is_dir() and not link.is_symlink()
    for path in cache.iterdir():
        assert (link / path.name).stat().st_ino == path.stat().st_ino


def test_copy_directory_over_existing_symlink_with_force(roots, work):
    _mteb_cache()
    export.export("t-alias", format="parquet")
    link = export.export("t-alias", format="parquet", mode="copy", force=True)
    assert link.is_dir() and not link.is_symlink()


def test_hardlink_across_filesystems(roots, work, monkeypatch):
    _dense_cache(roots)

    def exdev(*args, **kwargs):
        raise OSError(errno.EXDEV, "Invalid cross-device link")

    monkeypatch.setattr(os, "link", exdev)
    with pytest.raises(OSError, match="cannot hard-link across filesystems, use --copy"):
        export.export("fake", mode="hardlink")
    assert os.listdir(work) == []


# --- atomicity


def test_failure_leaves_nothing(roots, work, monkeypatch):
    _dense_cache(roots)
    (work / "fake.hdf5").write_text("old")

    def broken(src, dst, *args, **kwargs):
        open(dst, "w").write("half")
        raise OSError(errno.EIO, "disk on fire")

    monkeypatch.setattr("shutil.copy2", broken)
    with pytest.raises(OSError, match="disk on fire"):
        export.export("fake", mode="copy", force=True)
    assert os.listdir(work) == ["fake.hdf5"]
    assert (work / "fake.hdf5").read_text() == "old"


def test_directory_failure_leaves_nothing(roots, work, monkeypatch):
    (_mteb_cache() / "extra.parquet").write_bytes(b"x")
    calls = []
    real = os.link

    def flaky(src, dst, **kwargs):
        if calls:
            raise OSError(errno.EIO, "disk on fire")
        calls.append(1)
        real(src, dst, **kwargs)

    monkeypatch.setattr(os, "link", flaky)
    with pytest.raises(OSError, match="disk on fire"):
        export.export("t-alias", format="parquet", mode="hardlink")
    assert os.listdir(work) == []


# --- formats


def test_parquet_of_native_hdf5_is_an_error(roots, work):
    _native(roots)
    with pytest.raises(export.FormatNotAvailable, match="fake-native is only available as hdf5"):
        export.export("fake-native", format="parquet")
    assert os.listdir(work) == []


def test_hdf5_of_native_links_the_download(roots, work):
    path = _native(roots)
    link = export.export("fake-native")
    assert link == Path("fake-native.hdf5")
    assert link.resolve() == path
    assert not (roots / "dense" / "hdf5").exists()  # nothing converted


def test_mteb_hdf5_is_converted(roots, work):
    _mteb_cache()
    link = export.export("t-alias")
    assert link.name == "t-alias.hdf5"
    assert link.resolve() == roots / "dense" / "hdf5" / "mteb" / "T" / "org__m.hdf5"


def test_mteb_parquet_links_the_directory(roots, work):
    cache = _mteb_cache()
    link = export.export("t-alias", format="parquet")
    assert link.name == "t-alias" and link.resolve() == cache


def test_prune_hdf5_delegates(roots):
    _dense_cache(roots)
    path = export.ensure(export.resolve("fake"))
    assert export.prune_hdf5() == [path] and path.exists()
    assert export.prune_hdf5(dry_run=False) == [path] and not path.exists()
