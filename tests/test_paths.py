import pytest

from aida_data._paths import datasets_dir


@pytest.fixture(autouse=True)
def clean_env(tmp_path, monkeypatch):
    """A fake home, and none of the variables set."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("AIDA_DATA_DIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)


def test_aida_data_dir_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("AIDA_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert datasets_dir() == tmp_path / "data"


def test_aida_data_dir_expands_user(tmp_path, monkeypatch):
    monkeypatch.setenv("AIDA_DATA_DIR", "~/data")
    assert datasets_dir() == tmp_path / "home" / "data"


def test_aida_data_dir_empty_is_ignored(tmp_path, monkeypatch):
    monkeypatch.setenv("AIDA_DATA_DIR", "")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert datasets_dir() == tmp_path / "xdg" / "aida-data"


def test_xdg_cache_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert datasets_dir() == tmp_path / "xdg" / "aida-data"


def test_relative_xdg_cache_home_is_ignored(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", "foo")
    assert datasets_dir() == tmp_path / "home" / ".cache" / "aida-data"


def test_default(tmp_path):
    assert datasets_dir() == tmp_path / "home" / ".cache" / "aida-data"


@pytest.mark.parametrize(
    "env",
    [{}, {"AIDA_DATA_DIR": "datasets"}, {"XDG_CACHE_HOME": "foo"}],
)
def test_always_absolute(monkeypatch, env):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert datasets_dir().is_absolute()
