"""Download helper shared by the dataset modules."""

import logging
from pathlib import Path
from urllib.parse import urlparse

from ._tmp import atomic_path

_LOGGER = logging.getLogger("aida_data.download")

# (connect, read) in seconds: the read timeout bounds the time without
# receiving a byte, not the total download time.
_TIMEOUT = (10, 60)


def download(url, destination: Path):
    """Download `url` to `destination`.

    The body is written to a temporary file with a unique name,
    `<destination>.<random>.part`, and renamed only once it was fully
    received, so `destination` is either complete or missing. Concurrent
    downloads of the same file each write their own temporary file.
    """
    import requests
    from tqdm import tqdm

    if urlparse(url).scheme not in ("http", "https"):
        # Synthetic/local datasets (e.g. densired-hard) use a
        # non-fetchable placeholder URL and are generated on demand by
        # their own loader instead of being downloaded.
        return
    if not destination.is_file():
        _LOGGER.info(f"downloading {url} to {destination}")
        with (
            atomic_path(destination, ".part") as part,
            requests.get(url, stream=True, timeout=_TIMEOUT) as response,
        ):
            response.raise_for_status()
            # Content-Length counts the bytes on the wire, before any
            # Content-Encoding is decoded, and so does `response.raw.tell()`.
            total = int(response.headers.get("Content-Length", 0))
            with (
                open(part, "wb") as out_file,
                tqdm(
                    total=total or None,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=destination.name,
                ) as progress,
            ):
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    out_file.write(chunk)
                    progress.update(response.raw.tell() - progress.n)
            wire = response.raw.tell()
            if total and wire != total:
                raise OSError(
                    f"incomplete download of {url}: got {wire} of {total} bytes"
                )
