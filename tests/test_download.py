import gzip
import http.server
import os
import threading
import time

import pytest
import requests

from aida_data import _download, _tmp

URL = "https://example.org/data.bin"


class FakeRaw:
    """Stands in for urllib3's `HTTPResponse`: `tell()` is the number of
    encoded bytes read from the socket so far."""

    def __init__(self):
        self.wire = 0

    def tell(self):
        return self.wire


class FakeResponse:
    """`chunks` are the decoded chunks. `wire_sizes`, one per chunk, are
    the encoded bytes each chunk took on the wire (default: its length),
    to simulate compression."""

    def __init__(
        self,
        chunks,
        content_length=None,
        fail_after=None,
        wire_sizes=None,
        content_encoding=None,
        before_chunk=None,
    ):
        self.chunks = chunks
        self.headers = {}
        if content_length is not None:
            self.headers["Content-Length"] = str(content_length)
        if content_encoding is not None:
            self.headers["Content-Encoding"] = content_encoding
        self.fail_after = fail_after
        self.wire_sizes = wire_sizes or [len(chunk) for chunk in chunks]
        self.before_chunk = before_chunk
        self.raw = FakeRaw()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size=None):
        for i, chunk in enumerate(self.chunks):
            if self.before_chunk is not None:
                self.before_chunk(i)
            if self.fail_after is not None and i >= self.fail_after:
                raise requests.ConnectionError("connection dropped")
            self.raw.wire += self.wire_sizes[i]
            yield chunk


@pytest.fixture
def fake_get(monkeypatch):
    """Install fake responses for `requests.get`, returned in order (the
    last one is reused)."""
    calls = []

    def install(*responses):
        queue = list(responses)
        lock = threading.Lock()

        def get(url, **kwargs):
            with lock:
                calls.append((url, kwargs))
                return queue.pop(0) if len(queue) > 1 else queue[0]

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


def test_old_part_is_left_alone(tmp_path, fake_get):
    dest = tmp_path / "data.bin"
    old = tmp_path / "data.bin.part"
    old.write_bytes(b"stale partial content")
    fake_get(FakeResponse([b"fresh"]))
    _download.download(URL, dest)
    assert dest.read_bytes() == b"fresh"
    assert sorted(tmp_path.iterdir()) == [dest, old]
    assert old.read_bytes() == b"stale partial content"
    # Fresh: possibly a download running in another process.
    assert _tmp.stale_temporaries(tmp_path) == []
    day_ago = time.time() - 25 * 3600
    os.utime(old, (day_ago, day_ago))
    assert _tmp.stale_temporaries(tmp_path) == [old]


def test_compressed_body_longer_than_content_length(tmp_path, fake_get):
    # The decoded body is longer than Content-Length, which counts the
    # gzip-encoded bytes on the wire.
    dest = tmp_path / "data.bin"
    fake_get(
        FakeResponse(
            [b"a" * 100, b"b" * 100],
            content_length=30,
            wire_sizes=[20, 10],
            content_encoding="gzip",
        )
    )
    _download.download(URL, dest)
    assert dest.read_bytes() == b"a" * 100 + b"b" * 100
    assert list(tmp_path.iterdir()) == [dest]


def test_identity_encoding_short_body_raises(tmp_path, fake_get):
    dest = tmp_path / "data.bin"
    fake_get(FakeResponse([b"abc"], content_length=10, content_encoding="identity"))
    with pytest.raises(OSError, match="incomplete download"):
        _download.download(URL, dest)
    assert list(tmp_path.iterdir()) == []


def test_compressed_short_body_raises(tmp_path, fake_get):
    # More decoded bytes than Content-Length, but fewer on the wire.
    dest = tmp_path / "data.bin"
    fake_get(
        FakeResponse(
            [b"a" * 100],
            content_length=30,
            wire_sizes=[20],
            content_encoding="gzip",
        )
    )
    with pytest.raises(OSError, match="incomplete download"):
        _download.download(URL, dest)
    assert list(tmp_path.iterdir()) == []


def test_concurrent_failure_keeps_other_download(tmp_path, fake_get):
    dest = tmp_path / "data.bin"
    a_waiting = threading.Event()
    release_a = threading.Event()

    def block_a(i):
        if i == 1:
            a_waiting.set()
            assert release_a.wait(10)

    response_a = FakeResponse([b"AAA", b"aaa"], content_length=6, before_chunk=block_a)
    response_b = FakeResponse([b"BBB", b"bbb"], fail_after=1)
    fake_get(response_a, response_b)
    errors = []

    def run_a():
        try:
            _download.download(URL, dest)
        except BaseException as e:  # pragma: no cover - reported below
            errors.append(e)

    thread_a = threading.Thread(target=run_a)
    thread_a.start()
    try:
        assert a_waiting.wait(10)
        (part_a,) = tmp_path.glob("data.bin.*.part")
        with pytest.raises(requests.ConnectionError):
            _download.download(URL, dest)
        # B cleaned up its own file only.
        assert list(tmp_path.iterdir()) == [part_a]
        assert not dest.exists()
    finally:
        release_a.set()
        thread_a.join(10)
    assert errors == []
    assert dest.read_bytes() == b"AAAaaa"
    assert list(tmp_path.iterdir()) == [dest]


def test_downloaded_file_mode_follows_umask(tmp_path, fake_get):
    dest = tmp_path / "data.bin"
    fake_get(FakeResponse([b"abc"], content_length=3))
    old = os.umask(0o022)
    try:
        _download.download(URL, dest)
    finally:
        os.umask(old)
    assert dest.stat().st_mode & 0o777 == 0o644


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


def _serve(handler):
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_real_gzip_response(tmp_path):
    body = b"0123456789" * 10_000
    encoded = gzip.compress(body)

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *args):
            pass

    server = _serve(Handler)
    dest = tmp_path / "data.bin"
    try:
        _download.download(f"http://127.0.0.1:{server.server_port}/f", dest)
    finally:
        server.shutdown()
        server.server_close()
    assert dest.read_bytes() == body
    assert list(tmp_path.iterdir()) == [dest]


def test_real_gzip_response_cut_short(tmp_path):
    # The server announces the full encoded length, then closes early.
    encoded = gzip.compress(os.urandom(50_000))

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded[: len(encoded) // 2])
            self.wfile.flush()
            self.close_connection = True

        def log_message(self, *args):
            pass

    server = _serve(Handler)
    dest = tmp_path / "data.bin"
    try:
        with pytest.raises((OSError, requests.RequestException)):
            _download.download(f"http://127.0.0.1:{server.server_port}/f", dest)
    finally:
        server.shutdown()
        server.server_close()
    assert list(tmp_path.iterdir()) == []
