"""`aida-data` command line, offline: output, exit codes, errors, logging."""

import logging
import os
import subprocess
import sys

import numpy as np
import pytest

from aida_data import cli, dense, export
from aida_data.dense import Colors

pytestmark = pytest.mark.usefixtures("roots")

POINTS = np.arange(12, dtype=np.float32).reshape(3, 4)
IDS = np.array(["a", "b", "c"], dtype=object)


@pytest.fixture(autouse=True)
def restore_logging():
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    root.handlers[:], root.level = handlers, level


@pytest.fixture
def cached(roots):
    """`fake` cached as parquet."""
    (roots / "dense").mkdir(parents=True, exist_ok=True)
    path = roots / "dense" / "fake-cache.parquet"
    dense._write_parquet_cache(path, POINTS, Colors(np.zeros((3, 1), dtype=np.uint8), ("c",), (("x",),)), IDS)
    return path


def run(capsys, *argv):
    code = cli.main(list(argv))
    out, err = capsys.readouterr()
    return code, out, err


def test_path_prints_only_the_path(capsys, cached, roots):
    code, out, err = run(capsys, "path", "fake", "--format", "parquet")
    assert code == 0
    assert out == f"{cached.absolute()}\n"


def test_path_converts_and_logs_on_stderr(capsys, cached):
    code, out, err = run(capsys, "path", "fake")
    assert code == 0
    assert out.strip().endswith(".hdf5") and os.path.isabs(out.strip())
    assert out.count("\n") == 1 and "converting" in err


def test_export_prints_link_and_target(capsys, cached, work):
    code, out, err = run(capsys, "export", "fake", "--format", "parquet")
    assert code == 0
    assert out == f"fake.parquet -> {cached.absolute()}\n"
    assert (work / "fake.parquet").is_symlink()


def test_export_option_after_subcommand_and_dest(capsys, cached, work):
    (work / "d").mkdir()
    code, out, _ = run(capsys, "export", "fake", "d", "--format", "parquet", "-q")
    assert code == 0 and (work / "d" / "fake.parquet").is_symlink()


def test_export_copy_prints_only_link(capsys, cached, work):
    code, out, _ = run(capsys, "export", "fake", "x.parquet", "--format", "parquet", "--copy")
    assert code == 0 and out == "x.parquet\n"
    assert not (work / "x.parquet").is_symlink()


def test_export_existing_needs_force(capsys, cached, work):
    (work / "fake.parquet").write_text("x")
    code, out, err = run(capsys, "export", "fake", "--format", "parquet")
    assert code == 1 and out == "" and "pass --force" in err
    assert run(capsys, "export", "fake", "--format", "parquet", "--force")[0] == 0


def test_list_columns(capsys, cached):
    code, out, _ = run(capsys, "list", "--module", "dense")
    lines = out.splitlines()
    assert code == 0
    assert lines[0].split() == ["NAME", "MODULE", "NATIVE", "PARQUET", "HDF5"]
    rows = {line.split()[0]: line.split() for line in lines[1:]}
    assert rows["fake"] == ["fake", "dense", "parquet", "yes", "no"]
    assert rows["fake-native"] == ["fake-native", "dense", "hdf5", "-", "no"]
    # aligned: every row has its second column at the same offset
    assert len({line.index(line.split()[1]) for line in lines}) == 1


def test_list_cached_filter(capsys, cached):
    code, out, _ = run(capsys, "list", "--cached")
    names = [line.split()[0] for line in out.splitlines()[1:]]
    assert code == 0 and names == ["fake"]


def test_prune_hdf5_dry_run_then_yes(capsys, cached):
    path = export.ensure(export.resolve("fake"))
    code, out, _ = run(capsys, "prune-hdf5")
    assert code == 0 and path.exists()
    assert str(path) in out.splitlines() and "would delete 1" in out and "MiB" in out
    code, out, _ = run(capsys, "prune-hdf5", "--yes")
    assert code == 0 and not path.exists() and "deleting 1" in out


def test_unknown_name_suggests(capsys):
    code, out, err = run(capsys, "path", "fak")
    assert code == 1 and out == ""
    assert err.startswith("aida-data: error: unknown dataset 'fak'; did you mean: fake")
    assert "Traceback" not in err


def test_format_not_available(capsys):
    code, _, err = run(capsys, "path", "fake-native", "--format", "parquet")
    assert code == 1 and "only available as hdf5" in err


def test_missing_parent_is_an_error_line(capsys, cached, work):
    code, _, err = run(capsys, "export", "fake", "no/such/x")
    assert code == 1 and err.count("\n") == 1 and err.startswith("aida-data: error:")


def test_usage_errors_exit_2(capsys):
    for argv in (["--bogus"], ["export"], [], ["export", "fake", "--copy", "--hardlink"]):
        with pytest.raises(SystemExit) as info:
            cli.main(argv)
        assert info.value.code == 2
    capsys.readouterr()


def test_debug_reraises(capsys):
    with pytest.raises(export.UnknownDataset):
        cli.main(["--debug", "path", "nope"])
    with pytest.raises(export.UnknownDataset):
        cli.main(["path", "nope", "--debug"])


def test_keyboard_interrupt(capsys, monkeypatch):
    def boom(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(export, "ensure", boom)
    code, out, err = run(capsys, "path", "fake")
    assert code == 130 and err == ""


def test_quiet_silences_info(capsys, cached, monkeypatch):
    monkeypatch.setenv("TQDM_DISABLE", "0")  # restored at the end
    monkeypatch.delenv("TQDM_DISABLE")
    _, _, err = run(capsys, "path", "fake")
    assert "converting" in err
    export.prune_hdf5(dry_run=False)
    capsys.readouterr()
    _, _, err = run(capsys, "-q", "path", "fake")
    assert err == "" and os.environ["TQDM_DISABLE"] == "1"


def test_python_dash_m_help():
    result = subprocess.run([sys.executable, "-m", "aida_data", "--help"], capture_output=True, text=True)
    assert result.returncode == 0 and "aida-data" in result.stdout

