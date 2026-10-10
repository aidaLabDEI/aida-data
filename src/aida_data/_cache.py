"""Helpers shared by the dataset modules for their parquet caches."""

import json
import logging
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ._tmp import atomic_path

_LOGGER = logging.getLogger("aida_data.cache")

# Keep the raw download of a cached dataset after parsing it.
KEEP_RAW = os.environ.get("AIDA_DATA_KEEP_RAW", "0") == "1"


def write_table(path: Path, table: pa.Table, meta: dict, **options):
    """Write `table` to the parquet file `path`, with `meta` as its
    `aida_data` schema metadata. `options` go to `pq.write_table`; the
    compression defaults to zstd level 3."""
    options = {"compression": "zstd", "compression_level": 3} | options
    table = table.replace_schema_metadata({b"aida_data": json.dumps(meta).encode()})
    # Write to a unique temporary file and rename it only once it is
    # complete, so that an interrupted write leaves neither a truncated
    # cache nor a temporary file behind.
    with atomic_path(path, ".parquet.tmp") as tmp_path:
        pq.write_table(table, tmp_path, **options)


def read_table(path: Path, version: int) -> tuple[pa.Table, dict]:
    """Read the parquet cache `path` with its `aida_data` metadata, raising
    `ValueError` if the metadata is missing or has a different `version`."""
    table = pq.read_table(path)
    meta = json.loads((table.schema.metadata or {}).get(b"aida_data", b"null"))
    if not isinstance(meta, dict) or meta.get("version") != version:
        raise ValueError(
            f"{path} is not a cache in a known format, delete it to parse the "
            "dataset again"
        )
    return table, meta


def delete_raw(*paths: Path):
    """Delete the raw files among `paths` that exist."""
    for path in paths:
        if path.is_file():
            size = path.stat().st_size
            path.unlink()
            _LOGGER.info("deleted %s (%.1f MiB)", path, size / 2**20)
