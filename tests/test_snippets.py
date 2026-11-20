"""C++ snippets (`aida-data snippet`), offline: golden files, tailored and
generic modes, escaping, and (marked `cpp`) compiling and running them."""

import os
import re
import shutil
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pytest

from aida_data import _hdf5, _mteb, _snippets, cli, dense, export, graph, timeseries
from aida_data.dense import Colors

GOLDEN = Path(__file__).parent / "snippets"
MTEB_ALIAS = "t-alias"  # of conftest, for MTEB = mteb:T:org/m

N = 10
POINTS = np.arange(N * 4, dtype=np.float32).reshape(N, 4) / 7
COLORS = Colors(
    np.arange(N).reshape(-1, 1) % 3 * np.array([[1, 100]]),
    ("a", "b"),
    (("x", "y", "z"), tuple(f"l{i}" for i in range(101))),
)
IDS = np.array([f"id{i}" for i in range(N)], dtype=object)
EDGES = np.array([[0, 1], [0, 3], [1, 2], [2, 4], [4, 0], [6, 5]])
GRAPH_COLORS = np.array([7, 3, 12, -1, 7, 3, 3, 0], dtype=np.int64)
VALUES = np.arange(N * 2, dtype=np.float64).reshape(N, 2) * 1.5
TIME = np.datetime64("2020-01-01T00:00:00", "ms") + np.arange(N) * 1000


@pytest.fixture(autouse=True)
def fixed_version(monkeypatch):
    monkeypatch.setattr(_snippets, "_version", lambda: "0.0.0")


# --------------------------------------------------------------------------
# Fixture files: name -> function that makes the cache and the HDF5 file
# --------------------------------------------------------------------------


def _dense(roots):
    (roots / "dense").mkdir(parents=True, exist_ok=True)
    dense._write_parquet_cache(roots / "dense" / "fake-cache.parquet", POINTS, COLORS, IDS)
    return _hdf5.ensure_hdf5("dense", "fake", batch_size=3)


def _mteb_retrieval(roots):
    result = _mteb.Encoded(
        POINTS,
        IDS,
        None,
        POINTS[:4] * 2,
        np.array([f"q{i}" for i in range(4)], dtype=object),
        _mteb.build_qrels({"q0": {"id1": 2, "id2": 1}, "q3": {"id9": 1, "id0": 0}}),
        {"similarity": "cosine", "task_type": "Retrieval"},
    )
    _mteb.write_cache(result, _mteb.cache_dir("T", "org/m"))
    return _hdf5.ensure_hdf5("dense", "mteb:T:org/m", batch_size=3)


def _graph(roots):
    (roots / "graphs").mkdir(parents=True, exist_ok=True)
    graph._write_parquet_cache(graph.cache_path("fake-graph"), EDGES, GRAPH_COLORS)
    return _hdf5.ensure_hdf5("graph", "fake-graph", batch_size=3)


def _timeseries(roots):
    (roots / "timeseries").mkdir(parents=True, exist_ok=True)
    timeseries._write_parquet_cache(
        timeseries.cache_path("fake-ts"), VALUES, ["u", "v"], TIME, "1s", "file:///t"
    )
    return _hdf5.ensure_hdf5("timeseries", "fake-ts", batch_size=3)


def _native(roots):
    path = dense.local_path("fake-native")
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as hfp:
        hfp["train"] = POINTS
        hfp["test"] = POINTS[:3]
        hfp["neighbors"] = np.arange(6, dtype=np.int32).reshape(3, 2)
        hfp["distances"] = POINTS[:3, :2]
        hfp.attrs["distance"] = "angular"
    return path


# (entry name, fixture) of every module; the golden files are
# `generic_<key>.cpp` (nothing cached) and `tailored_<key>.cpp`.
CASES = {
    "dense": ("fake", _dense),
    "mteb": (MTEB_ALIAS, _mteb_retrieval),
    "graph": ("fake-graph", _graph),
    "timeseries": ("fake-ts", _timeseries),
    "native": ("fake-native", _native),
}


def _check_golden(name, text):
    path = GOLDEN / name
    if os.environ.get("AIDA_DATA_UPDATE_GOLDEN"):
        path.parent.mkdir(exist_ok=True)
        path.write_text(text)
    assert text == path.read_text(), (
        f"{name} differs, rerun with AIDA_DATA_UPDATE_GOLDEN=1 to regenerate"
    )


# --------------------------------------------------------------------------
# Golden files
# --------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["dense", "mteb", "graph", "timeseries", "native"])
def test_generic_golden(roots, key):
    entry = export.resolve(CASES[key][0])
    _check_golden(f"generic_{key}.cpp", _snippets.cpp_snippet(entry))


@pytest.mark.parametrize("key", list(CASES))
def test_tailored_golden(roots, key):
    name, make = CASES[key]
    make(roots)
    _check_golden(f"tailored_{key}.cpp", _snippets.cpp_snippet(export.resolve(name)))


# --------------------------------------------------------------------------
# Blocks
# --------------------------------------------------------------------------


def test_every_layout_object_has_a_block():
    for layout in ("native", "dense", "graph", "timeseries"):
        objects = _hdf5.layout_objects(layout)
        without = {o.path for o in objects if (layout, o.path) not in _snippets._BLOCKS}
        # Only the bookkeeping attributes of the converted files have none.
        assert without == (set() if layout == "native" else set(_snippets.BOOKKEEPING))
        assert all(o.attribute for o in objects if o.path in _snippets.BOOKKEEPING)
    layouts = {(o.layout, o.path) for o in _hdf5.LAYOUT}
    assert set(_snippets._BLOCKS) <= layouts


def _loaded(snippet):
    """The datasets and the attributes a snippet loads."""
    datasets = set(re.findall(r'H5Easy::load<.*>\(file, "/([^"]+)"\)', snippet))
    attributes = set(re.findall(r'loadAttribute<.*>\(file, "/", "([^"]+)"\)', snippet))
    return datasets, attributes


def _in_file(path):
    with h5py.File(path) as hfp:
        datasets = set()
        hfp.visititems(
            lambda p, o: datasets.add(p) if isinstance(o, h5py.Dataset) else None
        )
        return datasets, set(hfp.attrs) - _snippets.BOOKKEEPING


@pytest.mark.parametrize("key", list(CASES))
def test_tailored_loads_exactly_the_objects_of_the_file(roots, key):
    name, make = CASES[key]
    path = make(roots)
    snippet = _snippets.cpp_snippet(export.resolve(name))
    assert _loaded(snippet) == _in_file(path)
    assert "file.exist" not in snippet and "hasAttribute" not in snippet
    assert "generated without the file" not in snippet.lower()


def test_tailored_has_shape_comments(roots):
    _dense(roots)
    snippet = _snippets.cpp_snippet(export.resolve("fake"))
    assert "// (10, 4) float32, one row per point." in snippet
    assert "// (10, 2) uint8, integer codes, one column per color." in snippet
    assert "// (10,) string, one id per row of train." in snippet


def test_generic_wraps_optional_objects(roots):
    snippet = _snippets.cpp_snippet(export.resolve("fake"))
    assert 'if (file.exist("/colors")) {' in snippet
    assert 'if (file.exist("/qrels/query")) {' in snippet
    assert 'if (file.hasAttribute("color_names")) {' in snippet
    assert "Generated without the file" in snippet
    assert "aida-data export fake" in snippet
    # Required ones are not guarded.
    assert 'if (file.exist("/train"))' not in snippet
    assert 'auto train = H5Easy::load<std::vector<std::vector<float>>>(file, "/train");' in snippet


def test_types(roots):
    dense_snippet = _snippets.cpp_snippet(export.resolve("fake"))
    assert (
        'auto colors = H5Easy::load<std::vector<std::vector<int64_t>>>(file, "/colors");'
        in dense_snippet
    )
    assert 'H5Easy::load<std::vector<std::string>>(file, "/ids")' in dense_snippet
    assert (
        'H5Easy::loadAttribute<std::vector<std::string>>(file, "/", "color_names")'
        in dense_snippet
    )
    assert 'H5Easy::loadAttribute<std::string>(file, "/", "color_labels")' in dense_snippet
    assert "JSON library" in dense_snippet
    graph_snippet = _snippets.cpp_snippet(export.resolve("fake-graph"))
    for name in ("offsets", "neighbors", "colors"):
        assert f'H5Easy::load<std::vector<int64_t>>(file, "/{name}")' in graph_snippet
    assert "neighbors of node 0 ->" in graph_snippet
    ts_snippet = _snippets.cpp_snippet(export.resolve("fake-ts"))
    assert 'H5Easy::load<std::vector<std::vector<double>>>(file, "/values")' in ts_snippet
    assert 'H5Easy::load<std::vector<int64_t>>(file, "/time")' in ts_snippet
    assert "read_raw" in ts_snippet and "H5_USE_EIGEN" in ts_snippet


# --------------------------------------------------------------------------
# File path, staleness, no network
# --------------------------------------------------------------------------


def test_file_option_is_escaped(roots):
    snippet = _snippets.cpp_snippet(export.resolve("fake"), 'a"b\\c\nd.hdf5')
    assert 'H5Easy::File file("a\\"b\\\\c\\nd.hdf5", H5Easy::File::ReadOnly);' in snippet
    assert "\nd.hdf5" not in snippet  # no raw newline in the literal


def test_default_file_is_the_link_name(roots):
    assert 'H5Easy::File file("fake.hdf5"' in _snippets.cpp_snippet(export.resolve("fake"))
    snippet = _snippets.cpp_snippet(export.resolve(MTEB_ALIAS))
    assert 'H5Easy::File file("t-alias.hdf5"' in snippet
    assert "aida-data export t-alias" in snippet
    assert 'H5Easy::File file("x.h5"' in _snippets.cpp_snippet(export.resolve("fake"), "x.h5")


def test_stale_hdf5_gives_the_generic_snippet(roots):
    generic = _snippets.cpp_snippet(export.resolve("fake"))
    source = _dense(roots).parent.parent / "fake-cache.parquet"
    assert _snippets.cpp_snippet(export.resolve("fake")) != generic
    stat = source.stat()
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
    assert _snippets.cpp_snippet(export.resolve("fake")) == generic


def test_snippet_never_downloads_or_converts(roots, monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise AssertionError("no download, cache or conversion expected")

    for module in (dense, graph, timeseries):
        monkeypatch.setattr(module, "_download", fail)
        monkeypatch.setattr(module, "_ensure_cache", fail, raising=False)
    monkeypatch.setattr(dense, "_ensure_mteb", fail)
    monkeypatch.setattr(export, "ensure", fail)
    monkeypatch.setattr(_hdf5, "ensure_hdf5", fail)
    monkeypatch.setattr(_hdf5, "_convert", fail)
    for name in ("fake", "fake-native", "fake-graph", "fake-ts", MTEB_ALIAS, "mteb:T:org/m"):
        assert cli.main(["snippet", name]) == 0
    assert capsys.readouterr().out.count("int main()") == 6
    assert not (roots / "dense").exists()


def test_cli_snippet(roots, capsys):
    assert cli.main(["snippet", "fake", "--file", "/data/x.hdf5"]) == 0
    out, err = capsys.readouterr()
    assert 'H5Easy::File file("/data/x.hdf5"' in out and err == ""
    assert out == _snippets.cpp_snippet(export.resolve("fake"), "/data/x.hdf5")
    assert cli.main(["snippet", "nope"]) == 1


# --------------------------------------------------------------------------
# Compile and run
# --------------------------------------------------------------------------


def _highfive_include():
    candidates = [os.environ.get("HIGHFIVE_INCLUDE"), "/usr/include", "/usr/local/include"]
    for candidate in candidates:
        if candidate and (Path(candidate) / "highfive" / "H5Easy.hpp").is_file():
            return candidate
    return None


def _compiler():
    """The command to compile against HDF5: `h5c++` if its compiler exists,
    else `g++` with the flags of `pkg-config hdf5`."""
    if shutil.which("h5c++"):
        shown = subprocess.run(["h5c++", "-show"], capture_output=True, text=True)
        if shown.returncode == 0 and shutil.which(shown.stdout.split()[0]):
            return ["h5c++"]
    if shutil.which("g++") and shutil.which("pkg-config"):
        flags = subprocess.run(
            ["pkg-config", "--cflags", "--libs", "hdf5"], capture_output=True, text=True
        )
        if flags.returncode == 0:
            return ["g++", *flags.stdout.split()]
    return None


def _expected(path, layout):
    """What the snippet prints for `path`, read with h5py: `key: value`."""
    out = {}
    with h5py.File(path) as hfp:
        for obj in _hdf5.layout_objects(layout):
            if obj.path in _snippets.BOOKKEEPING:
                continue
            if obj.attribute:
                if obj.path not in hfp.attrs:
                    continue
                value = hfp.attrs[obj.path]
                if obj.rank == 1:
                    out[obj.path] = str(len(value))
                elif (obj.layout, obj.path) in _snippets._PRINT_SIZE:
                    out[obj.path] = str(len(value))
                else:
                    out[obj.path] = str(value)
            elif obj.path in hfp:
                out[obj.path] = " x ".join(str(n) for n in hfp[obj.path].shape)
    return out


@pytest.mark.cpp
@pytest.mark.parametrize("kind", ["generic", "tailored"])
@pytest.mark.parametrize("key", list(CASES))
def test_snippet_compiles_and_prints_the_shapes(roots, tmp_path, kind, key):
    include, compiler = _highfive_include(), _compiler()
    if include is None or compiler is None:
        pytest.skip("needs h5c++ (or g++ and pkg-config hdf5) and the HighFive headers")
    name, make = CASES[key]
    fixture = make(roots)
    work = tmp_path / "build"
    work.mkdir()
    shutil.copy(fixture, work / f"{export.resolve(name).link_stem}.hdf5")
    (work / "load.cpp").write_text((GOLDEN / f"{kind}_{key}.cpp").read_text())
    build = subprocess.run(
        [*compiler, "-std=c++17", f"-I{include}", "-Wall", "load.cpp", "-o", "load"],
        cwd=work, capture_output=True, text=True,
    )
    assert build.returncode == 0, build.stderr
    run = subprocess.run(["./load"], cwd=work, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    printed = dict(
        line.split(": ", 1) for line in run.stdout.splitlines() if ": " in line
    )
    assert printed == _expected(fixture, _snippets._layout_of(export.resolve(name)))
