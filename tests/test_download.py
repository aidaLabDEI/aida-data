import http.server
import threading
import time

import pytest
import requests

from aida_data import _download

URL = "https://example.org/data.bin"


class FakeResponse:
    def __init__(self, chunks, content_length=None, fail_after=None):
        self.chunks = chunks
        self.headers = {}
        if content_length is not None:
            self.headers["Content-Length"] = str(content_length)
        self.fail_after = fail_after

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size=None):
        for i, chunk in enumerate(self.chunks):
            if self.fail_after is not None and i >= self.fail_after:
                raise requests.ConnectionError("connection dropped")
            yield chunk


@pytest.fixture
def fake_get(monkeypatch):
    calls = []

    def install(response):
        def get(url, **kwargs):
            calls.append((url, kwargs))
            return response

        monkeypatch.setattr(requests, "get", get)
        return calls

    return install


def test_success(tmp_path, fake_get):
    dest = tmp_path / "data.bin"
    fake_get(FakeResponse([b"abc", b"def"], content_length=6))
    _download.download(URL, dest)
    assert dest.read_bytes() == b"abcdef"
    assert list(tmp_path.iterdir()) == [dest]


def test_failure_halfway_leaves_nothing(tmp_path, fake_get):
    dest = tmp_path / "data.bin"
    fake_get(FakeResponse([b"abc", b"def"], fail_after=1))
    with pytest.raises(requests.ConnectionError):
        _download.download(URL, dest)
    assert list(tmp_path.iterdir()) == []


def test_short_body_raises(tmp_path, fake_get):
    dest = tmp_path / "data.bin"
    fake_get(FakeResponse([b"abc"], content_length=10))
    with pytest.raises(Exception):
        _download.download(URL, dest)
    assert list(tmp_path.iterdir()) == []


def test_existing_file_untouched(tmp_path, fake_get):
    dest = tmp_path / "data.bin"
    dest.write_bytes(b"old")
    calls = fake_get(FakeResponse([b"new"]))
    _download.download(URL, dest)
    assert dest.read_bytes() == b"old"
    assert calls == []


def test_stale_part_is_replaced(tmp_path, fake_get):
    dest = tmp_path / "data.bin"
    (tmp_path / "data.bin.part").write_bytes(b"stale partial content")
    fake_get(FakeResponse([b"fresh"]))
    _download.download(URL, dest)
    assert dest.read_bytes() == b"fresh"
    assert list(tmp_path.iterdir()) == [dest]


def test_non_http_url(tmp_path, fake_get):
    calls = fake_get(FakeResponse([b"x"]))
    _download.download("synthetic://densired-hard", tmp_path / "x")
    assert calls == []
    assert list(tmp_path.iterdir()) == []


def test_timeout_is_passed(tmp_path, fake_get):
    calls = fake_get(FakeResponse([b"x"]))
    _download.download(URL, tmp_path / "data.bin")
    assert calls[0][1]["timeout"] == _download._TIMEOUT


def test_real_stall_times_out(tmp_path, monkeypatch):
    done = threading.Event()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", "100")
            self.end_headers()
            self.wfile.write(b"x" * 50)
            self.wfile.flush()
            done.wait(10)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(_download, "_TIMEOUT", (2, 0.5))
    dest = tmp_path / "data.bin"
    start = time.monotonic()
    try:
        with pytest.raises(requests.RequestException):
            _download.download(f"http://127.0.0.1:{server.server_port}/f", dest)
    finally:
        done.set()
        server.shutdown()
        server.server_close()
    assert time.monotonic() - start < 5
    assert list(tmp_path.iterdir()) == []
