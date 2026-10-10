import os
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from aida_data import _cache, _tmp, dense, graph, timeseries

TABLE = pa.table({"x": np.arange(5)})


def _age(path, hours):
    then = time.time() - hours * 3600
    os.utime(path, (then, then))


def test_write_table_round_trip(tmp_path):
    path = tmp_path / "t.parquet"
    _cache.write_table(path, TABLE, {"version": 1})
    table, meta = _cache.read_table(path, 1)
    assert table.column("x").to_pylist() == list(range(5))
    assert meta == {"version": 1}
    assert list(tmp_path.iterdir()) == [path]


def test_write_table_failure_leaves_nothing(tmp_path, monkeypatch):
    def fail(table, where, **options):
        # Create the file, as a write interrupted halfway would.
        with open(where, "wb") as f:
            f.write(b"PAR1 half a file")
        raise KeyboardInterrupt

    monkeypatch.setattr(pq, "write_table", fail)
    path = tmp_path / "t.parquet"
    with pytest.raises(KeyboardInterrupt):
        _cache.write_table(path, TABLE, {"version": 1})
    assert list(tmp_path.glob("*.tmp")) == []
    assert list(tmp_path.iterdir()) == []


def test_write_table_failure_keeps_existing_cache(tmp_path, monkeypatch):
    path = tmp_path / "t.parquet"
    _cache.write_table(path, TABLE, {"version": 1})
    before = path.read_bytes()

    def fail(table, where, **options):
        raise OSError("disk full")

    monkeypatch.setattr(pq, "write_table", fail)
    with pytest.raises(OSError, match="disk full"):
        _cache.write_table(path, TABLE, {"version": 2})
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_atomic_path_names(tmp_path):
    path = tmp_path / "ecg.parquet"
    with _tmp.atomic_path(path, ".parquet.tmp") as tmp:
        assert tmp.parent == tmp_path
        assert tmp.name.startswith("ecg.parquet.")
        assert tmp.name.endswith(".parquet.tmp")
        tmp.write_bytes(b"x")
    assert path.read_bytes() == b"x"
    assert list(tmp_path.iterdir()) == [path]


def test_atomic_path_unique(tmp_path):
    path = tmp_path / "data.bin"
    with _tmp.atomic_path(path, ".part") as a:
        with _tmp.atomic_path(path, ".part") as b:
            assert a != b
            a.write_bytes(b"a")
            b.write_bytes(b"b")
        assert path.read_bytes() == b"b"
    # The last one to finish wins.
    assert path.read_bytes() == b"a"
    assert list(tmp_path.iterdir()) == [path]


def test_atomic_path_mode_follows_umask(tmp_path):
    path = tmp_path / "data.bin"
    old = os.umask(0o022)
    try:
        with _tmp.atomic_path(path, ".part") as tmp:
            tmp.write_bytes(b"x")
    finally:
        os.umask(old)
    assert path.stat().st_mode & 0o777 == 0o644


def test_stale_temporaries(tmp_path):
    assert _tmp.stale_temporaries(tmp_path / "missing") == []
    old_part = tmp_path / "a.bin.x1.part"
    old_tmp = tmp_path / "a.parquet.x2.parquet.tmp"
    fresh = tmp_path / "b.bin.x3.part"
    other = tmp_path / "c.csv"
    for path in (old_part, old_tmp, fresh, other):
        path.write_text("x")
    for path in (old_part, old_tmp, other):
        _age(path, 25)
    _age(fresh, 23)
    assert _tmp.stale_temporaries(tmp_path) == [old_part, old_tmp]


@pytest.mark.parametrize(
    "module, subdir",
    [(dense, None), (graph, "graphs"), (timeseries, "timeseries")],
)
def test_prune_raw_sweeps_stale_temporaries(tmp_path, monkeypatch, module, subdir):
    monkeypatch.setattr(module, "DATASETS_DIR", tmp_path)
    directory = tmp_path / subdir if subdir else tmp_path
    directory.mkdir(exist_ok=True)
    stale = directory / "x.zip.abc.part"
    fresh = directory / "y.zip.def.part"
    stale_cache = directory / "z.parquet.ghi.parquet.tmp"
    for path in (stale, fresh, stale_cache):
        path.write_text("x")
    _age(stale, 25)
    _age(stale_cache, 48)
    assert module.prune_raw() == [stale, stale_cache]
    assert stale.is_file()  # dry run
    assert module.prune_raw(dry_run=False) == [stale, stale_cache]
    assert not stale.exists() and not stale_cache.exists()
    assert fresh.is_file()
    assert module.prune_raw() == []
