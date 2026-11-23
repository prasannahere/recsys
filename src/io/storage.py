"""Local filesystem and gs:// URI helpers."""

from __future__ import annotations

import io
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, Iterator

GCS_PREFIX = "gs://"


def is_gcs_uri(path: str | Path) -> bool:
    return str(path).startswith(GCS_PREFIX)


def parse_gcs_uri(uri: str) -> tuple[str, str]:
    if not is_gcs_uri(uri):
        raise ValueError(f"Not a GCS URI: {uri}")
    bucket, _, blob = uri[len(GCS_PREFIX) :].partition("/")
    if not bucket or not blob:
        raise ValueError(f"Invalid GCS URI: {uri}")
    return bucket, blob


def join_uri(base: str | Path, *parts: str) -> str:
    base_str = str(base)
    if is_gcs_uri(base_str):
        return base_str.rstrip("/") + "/" + "/".join(parts)
    return str(Path(base_str).joinpath(*parts))


def artifact_exists(model_dir: str | Path, *parts: str) -> bool:
    """Check whether an artifact exists under model_dir (local or gs://)."""
    return exists(join_uri(model_dir, *parts))


def parent_uri(path: str | Path) -> str:
    path_str = str(path)
    if is_gcs_uri(path_str):
        return path_str.rsplit("/", 1)[0]
    return str(Path(path_str).parent)


def exists(path: str | Path) -> bool:
    path_str = str(path)
    if is_gcs_uri(path_str):
        from google.cloud import storage

        bucket_name, blob_name = parse_gcs_uri(path_str)
        return storage.Client().bucket(bucket_name).blob(blob_name).exists()
    return Path(path_str).exists()


@contextmanager
def open_binary(path: str | Path, mode: str = "rb") -> Iterator[BinaryIO]:
    path_str = str(path)
    if not is_gcs_uri(path_str):
        path_obj = Path(path_str)
        if "w" in mode:
            path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(path_obj, mode) as handle:
            yield handle
        return

    bucket_name, blob_name = parse_gcs_uri(path_str)
    from google.cloud import storage

    blob = storage.Client().bucket(bucket_name).blob(blob_name)
    if mode == "rb":
        yield io.BytesIO(blob.download_as_bytes())
        return

    if mode == "wb":
        buffer = io.BytesIO()
        try:
            yield buffer
        finally:
            buffer.seek(0)
            blob.upload_from_file(buffer, rewind=True)
        return

    raise ValueError(f"Unsupported mode for GCS: {mode}")


def write_text(path: str | Path, text: str) -> None:
    path_str = str(path)
    if is_gcs_uri(path_str):
        bucket_name, blob_name = parse_gcs_uri(path_str)
        from google.cloud import storage

        storage.Client().bucket(bucket_name).blob(blob_name).upload_from_string(
            text, content_type="text/plain"
        )
        return

    path_obj = Path(path_str)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    path_obj.write_text(text)


def write_parquet(path: str | Path, df) -> None:
    path_str = str(path)
    if is_gcs_uri(path_str):
        buffer = io.BytesIO()
        df.write_parquet(buffer)
        bucket_name, blob_name = parse_gcs_uri(path_str)
        from google.cloud import storage

        buffer.seek(0)
        storage.Client().bucket(bucket_name).blob(blob_name).upload_from_file(
            buffer, rewind=True, content_type="application/octet-stream"
        )
        return

    path_obj = Path(path_str)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path_obj)
