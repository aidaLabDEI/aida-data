"""C++ snippets that read the HDF5 export of a dataset (private).

`cpp_snippet` returns a complete C++17 program for HighFive (H5Easy) that
loads the objects of the file and prints their shapes. It never downloads or
converts anything. If the HDF5 file is already there and fresh, the program
is tailored to the objects it has, with a shape and dtype comment above each
read. Otherwise it is generic for the layout, and the optional objects are
loaded only `if (file.exist(...))`.

There is one block per object of `_hdf5.LAYOUT`, described in `_BLOCKS`; its
load statement is derived from the layout entry (`ctype`, `rank`,
`attribute`), so the snippets cannot drift from the files.
"""

from importlib.metadata import PackageNotFoundError, version

from . import _hdf5
from .export import Entry

# The HighFive release the generated code was compiled and run against.
HIGHFIVE_TESTED = "3.3.0"

# Objects with no block: bookkeeping attributes of the converted files, not
# data. Native files do not have them.
BOOKKEEPING = frozenset(
    {"aida_data_hdf5_version", "source_mtime_ns", "module", "name"}
)

# One entry per object of `_hdf5.LAYOUT`, except the bookkeeping ones: a
# phrase that goes after the shape and dtype in the comment above its load.
_BLOCKS: dict[tuple[str, str], str] = {
    ("native", "train"): "one row per point",
    ("native", "test"): "one row per query",
    ("native", "neighbors"): "ids of the nearest neighbors of each query, in train",
    ("native", "distances"): "distances from each query to those neighbors",
    ("native", "distance"): 'distance of the dataset, e.g. "euclidean" or "angular"',
    ("dense", "train"): "one row per point",
    ("dense", "test"): "one row per query (MTEB retrieval)",
    ("dense", "colors"): "integer codes, one column per color",
    ("dense", "ids"): "one id per row of train",
    ("dense", "test_ids"): "one id per row of test",
    ("dense", "qrels/query"): "query id of each relevance judgment",
    ("dense", "qrels/doc"): "document id of each relevance judgment",
    ("dense", "qrels/score"): "relevance score of each judgment",
    ("dense", "distance"): 'distance of the dataset, e.g. "euclidean" or "cosine"',
    ("dense", "dimension"): "number of columns of train",
    ("dense", "color_names"): "name of each column of colors",
    ("dense", "color_labels"): (
        "JSON text: a list of lists of labels, one list per color.\n"
        "Parsing it needs a JSON library, e.g. nlohmann/json"
    ),
    ("graph", "offsets"): (
        "CSR offsets, one per node plus one: the neighbors of node v are\n"
        "neighbors[offsets[v]] to neighbors[offsets[v + 1] - 1]"
    ),
    ("graph", "neighbors"): "neighbor lists of all the nodes, one after the other",
    ("graph", "colors"): "color of each node",
    ("graph", "n_nodes"): "number of nodes",
    ("graph", "n_edges"): "number of edges (entries of neighbors)",
    ("timeseries", "values"): "one row per time step, one column per dimension",
    ("timeseries", "time"): "milliseconds since the Unix epoch, one per row of values",
    ("timeseries", "dim_names"): "name of each column of values",
    ("timeseries", "sampling"): "sampling period of the series, e.g. 1s",
    ("timeseries", "source"): "URL of the data, several are separated by spaces",
}

# Printed as its length: it is a long text.
_PRINT_SIZE = frozenset({("dense", "color_labels")})

# The matrix the contiguous-read comment is about, per layout.
_MAIN_MATRIX = {"native": "train", "dense": "train", "timeseries": "values"}

_GRAPH_WALK = """\
    // The neighbors of node 0.
    if (offsets.size() > 1) {
        std::cout << "neighbors of node 0 ->";
        for (int64_t i = offsets[0]; i < offsets[1]; ++i) {
            std::cout << " " << neighbors[i];
        }
        std::cout << "\\n";
    }
"""


# --------------------------------------------------------------------------
# Pieces
# --------------------------------------------------------------------------


def _version() -> str:
    try:
        return version("aida-data")
    except PackageNotFoundError:  # not installed
        return "unknown"


def _literal(text: str) -> str:
    """The content of a C++ string literal that spells `text`."""
    return (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
    )


def _layout_of(entry: Entry) -> str:
    return "native" if entry.native == "hdf5" else entry.module


def _var(obj: _hdf5.Hdf5Object) -> str:
    return obj.path.replace("/", "_")


def _cpp_type(obj: _hdf5.Hdf5Object) -> str:
    if obj.attribute:
        return f"std::vector<{obj.ctype}>" if obj.rank == 1 else obj.ctype
    return "std::vector<" * obj.rank + obj.ctype + ">" * obj.rank


def _load(obj: _hdf5.Hdf5Object) -> str:
    """The statement that loads `obj` into a variable."""
    if obj.attribute:
        call = f'H5Easy::loadAttribute<{_cpp_type(obj)}>(file, "/", "{obj.path}")'
    else:
        call = f'H5Easy::load<{_cpp_type(obj)}>(file, "/{obj.path}")'
    return f"auto {_var(obj)} = {call};"


def _exists(obj: _hdf5.Hdf5Object) -> str:
    if obj.attribute:
        return f'file.hasAttribute("{obj.path}")'
    return f'file.exist("/{obj.path}")'


def _print(obj: _hdf5.Hdf5Object) -> str:
    """The statement that prints `<path>: <shape or value>`."""
    var, head = _var(obj), f'"{obj.path}: "'
    key = (obj.layout, obj.path)
    if obj.rank == 2:
        what = f"{var}.size() << \" x \" << ({var}.empty() ? 0 : {var}[0].size())"
    elif obj.rank == 1 or key in _PRINT_SIZE:
        what = f"{var}.size()"
    else:
        what = var
    return f"std::cout << {head} << {what} << \"\\n\";"


def _shape_comment(obj: _hdf5.Hdf5Object, hfp) -> str:
    """The comment above the load of `obj`, read from the open file `hfp`
    for a dataset (shape and dtype), else generic."""
    text = _BLOCKS[(obj.layout, obj.path)]
    if obj.attribute or hfp is None:
        return text[0].upper() + text[1:] + "."
    dataset = hfp[obj.path]
    dtype = "string" if dataset.dtype.kind == "O" else str(dataset.dtype)
    return f"{dataset.shape} {dtype}, {text}."


def _comment(text: str, indent: str) -> list[str]:
    return [f"{indent}// {line}" for line in text.split("\n")]


def _present(hfp, obj: _hdf5.Hdf5Object) -> bool:
    if obj.attribute:
        return obj.path in hfp.attrs
    return obj.path in hfp


def _header(entry: Entry, file: str, tailored: bool) -> list[str]:
    export_name = entry.name
    if entry.name.startswith("mteb:") and entry.link_stem != entry.name:
        from . import dense

        if dense._ALIASES.get(entry.link_stem) == entry.name:
            export_name = entry.link_stem
    lines = [
        f"// Load {_literal(file)}, exported by `aida-data export {export_name}`.",
        f"// Generated by aida-data {_version()} for HighFive 3.x (H5Easy), "
        f"tested with HighFive {HIGHFIVE_TESTED}.",
    ]
    if tailored:
        lines.append(
            "// It loads the objects of the file it was generated from; "
            "each comment gives the shape and dtype."
        )
    else:
        lines += [
            "// Generated without the file, so it checks the objects that not every "
            "dataset has",
            f"// with `file.exist`. Get the file with `aida-data export {export_name}`, "
            "then run",
            f"// `aida-data snippet {export_name}` again for a version made for it.",
        ]
    lines += [
        "//",
        "// Build: h5c++ -std=c++17 -I/path/to/highfive/include load.cpp -o load",
        "// or with CMake:",
        "//   find_package(HighFive REQUIRED)",
        "//   target_link_libraries(load HighFive::HighFive)",
        "// HighFive comes from a package manager (apt install libhighfive-dev,",
        "// conda install highfive, vcpkg install highfive) or a git submodule.",
        "// With HighFive 2.x the CMake target is `HighFive` and DataSet::read_raw(T*)",
        "// is DataSet::read(T*).",
        "#include <highfive/H5Easy.hpp>",
        "",
        "#include <cstdint>",
        "#include <iostream>",
        "#include <string>",
        "#include <vector>",
        "",
        "int main() {",
    ]
    return lines


def _contiguous(layout: str, matrix: _hdf5.Hdf5Object) -> list[str]:
    ctype = matrix.ctype
    return [
        "    // For large data, read into one contiguous row-major buffer instead:",
        f'    //   HighFive::DataSet ds = file.getDataSet("/{matrix.path}");',
        "    //   auto dims = ds.getDimensions();",
        f"    //   std::vector<{ctype}> flat(dims[0] * dims[1]);",
        "    //   ds.read_raw(flat.data());",
        "    // or, with H5_USE_EIGEN defined before including HighFive:",
        f"    //   using Matrix = Eigen::Matrix<{ctype}, Eigen::Dynamic, Eigen::Dynamic,",
        "    //                                Eigen::RowMajor>;",
        f'    //   auto m = H5Easy::load<Matrix>(file, "/{matrix.path}");',
    ]


# --------------------------------------------------------------------------
# Snippet
# --------------------------------------------------------------------------


def cpp_snippet(entry: Entry, file: str | None = None) -> str:
    """A C++17 program that loads the HDF5 export of `entry` with H5Easy
    (see the module docstring). `file` is the path it opens, default
    `<link_stem>.hdf5`. Does no download, conversion or cache building."""
    layout = _layout_of(entry)
    objects = [o for o in _hdf5.layout_objects(layout) if o.path not in BOOKKEEPING]
    path = file if file is not None else f"{entry.link_stem}.hdf5"
    tailored = _hdf5.is_fresh(entry.module, entry.name)
    hfp = None
    if tailored:
        import h5py

        hfp = h5py.File(_hdf5.hdf5_path(entry.module, entry.name))
    try:
        if hfp is not None:
            objects = [o for o in objects if _present(hfp, o)]
        lines = _header(entry, path, tailored)
        lines.append(
            f'    H5Easy::File file("{_literal(path)}", H5Easy::File::ReadOnly);'
        )
        prints = []  # at the end of main: what is in scope there
        for obj in objects:
            lines.append("")
            comment = _comment(_shape_comment(obj, hfp), "    ")
            if tailored or obj.required:
                lines += comment + [f"    {_load(obj)}"]
                prints.append(f"    {_print(obj)}")
            else:
                lines.append(f"    if ({_exists(obj)}) {{")
                lines += _comment(_shape_comment(obj, hfp), "        ")
                lines += [f"        {_load(obj)}", f"        {_print(obj)}", "    }"]
        by_path = {o.path: o for o in objects}
        main = by_path.get(_MAIN_MATRIX.get(layout, ""))
        if main is not None:
            lines += ["", *_contiguous(layout, main)]
        if layout == "graph" and {"offsets", "neighbors"} <= by_path.keys():
            lines += ["", *_GRAPH_WALK.splitlines()]
        if prints:
            lines += ["", "    // The shapes of what was loaded.", *prints]
        lines += ["}", ""]
    finally:
        if hfp is not None:
            hfp.close()
    return "\n".join(lines)
