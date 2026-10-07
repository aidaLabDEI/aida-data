"""Download helper shared by the dataset modules."""

import logging
from pathlib import Path
from urllib.parse import urlparse

_LOGGER = logging.getLogger("aida_data.download")


def download(url, destination: Path):
    import requests
    from tqdm import tqdm

    if urlparse(url).scheme not in ("http", "https"):
        # Synthetic/local datasets (e.g. densired-hard) use a
        # non-fetchable placeholder URL and are generated on demand by
        # their own loader instead of being downloaded.
        return
    if not destination.is_file():
        _LOGGER.info(f"downloading {url} to {destination}")
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            total = int(response.headers.get("Content-Length", 0))
            with (
                open(destination, "wb") as out_file,
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
                    progress.update(len(chunk))
