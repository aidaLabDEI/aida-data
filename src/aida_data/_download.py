"""Download helper shared by the dataset modules."""

import logging
from pathlib import Path
from urllib.parse import urlparse

_LOGGER = logging.getLogger("aida_data.download")

# (connect, read) in seconds: the read timeout bounds the time without
# receiving a byte, not the total download time.
_TIMEOUT = (10, 60)


def download(url, destination: Path):
    """Download `url` to `destination`.

    The body is written to `<destination>.part` and renamed only once it
    was fully received, so `destination` is either complete or missing.
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
        part = destination.with_name(destination.name + ".part")
        try:
            with requests.get(url, stream=True, timeout=_TIMEOUT) as response:
                response.raise_for_status()
                total = int(response.headers.get("Content-Length", 0))
                received = 0
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
                        received += len(chunk)
                        progress.update(len(chunk))
                # Content-Length counts encoded bytes, so only compare
                # when the body was not transparently decompressed.
                encoded = response.headers.get("Content-Encoding")
                if total and not encoded and received != total:
                    raise OSError(
                        f"incomplete download of {url}: "
                        f"got {received} of {total} bytes"
                    )
            part.replace(destination)
        except BaseException:
            part.unlink(missing_ok=True)
            raise
