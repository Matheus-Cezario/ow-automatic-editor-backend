"""Blob storage. `LocalStorage` (a folder) and `S3Storage` (MinIO) behind the
same interface, chosen by the execution mode."""

from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO, Callable

from .config import Settings, get_settings


class Storage(ABC):
    @abstractmethod
    def put_file(self, key: str, path: Path) -> str: ...

    @abstractmethod
    def put_stream(self, key: str, fh: BinaryIO) -> str: ...

    @abstractmethod
    def get_file(
        self, key: str, dest: Path, *, on_bytes: Callable[[int], None] | None = None
    ) -> Path:
        """Brings the blob to `dest`. `on_bytes` receives the size of each
        chunk that arrives -- it is how the preprocessor can say how much of
        the video has downloaded, on a half-gigabyte file."""
        ...

    @abstractmethod
    def open(self, key: str) -> BinaryIO: ...

    @abstractmethod
    def exists(self, key: str) -> bool: ...

    @abstractmethod
    def size(self, key: str) -> int: ...

    @abstractmethod
    def delete(self, key: str) -> None:
        """Removes the blob. Silent when it is not there -- whoever deletes
        wants it gone, and it already is."""
        ...

    def open_range(self, key: str, start: int, length: int) -> bytes:
        """Reads a byte range. The HTML5 player asks for Range so it can seek
        into the middle of the video without downloading the whole file."""
        with self.open(key) as fh:
            fh.seek(start)
            return fh.read(length)


class LocalStorage(Storage):
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _p(self, key: str) -> Path:
        # normalises the path and prevents escaping the root
        p = (self.root / key.lstrip("/")).resolve()
        if not str(p).startswith(str(self.root.resolve())):
            raise ValueError(f"chave inválida: {key!r}")
        return p

    def put_file(self, key: str, path: Path) -> str:
        dest = self._p(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if Path(path).resolve() != dest:
            shutil.copyfile(path, dest)
        return key

    def put_stream(self, key: str, fh: BinaryIO) -> str:
        dest = self._p(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as out:
            shutil.copyfileobj(fh, out, length=1024 * 1024)
        return key

    def get_file(
        self, key: str, dest: Path, *, on_bytes: Callable[[int], None] | None = None
    ) -> Path:
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self._p(key), dest)
        if on_bytes is not None:
            on_bytes(dest.stat().st_size)
        return dest

    def open(self, key: str) -> BinaryIO:
        return open(self._p(key), "rb")

    def exists(self, key: str) -> bool:
        return self._p(key).exists()

    def size(self, key: str) -> int:
        return self._p(key).stat().st_size

    def delete(self, key: str) -> None:
        self._p(key).unlink(missing_ok=True)

    def path(self, key: str) -> Path:
        """Local-backend shortcut: avoids a copy when ffmpeg can read directly."""
        return self._p(key)

    def url(self, key: str, expires_s: int = 3600) -> str:
        """The path itself: here ffmpeg already reads straight from disk."""
        return str(self._p(key))


class S3Storage(Storage):
    def __init__(self, s: Settings):
        import boto3  # late import: only docker mode needs it

        self.bucket = s.s3_bucket
        self.client = boto3.client(
            "s3",
            endpoint_url=s.s3_endpoint,
            aws_access_key_id=s.s3_access_key,
            aws_secret_access_key=s.s3_secret_key,
        )
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except Exception:
            self.client.create_bucket(Bucket=self.bucket)

    def put_file(self, key: str, path: Path) -> str:
        self.client.upload_file(str(path), self.bucket, key)
        return key

    def put_stream(self, key: str, fh: BinaryIO) -> str:
        self.client.upload_fileobj(fh, self.bucket, key)
        return key

    def get_file(
        self, key: str, dest: Path, *, on_bytes: Callable[[int], None] | None = None
    ) -> Path:
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, key, str(dest), Callback=on_bytes)
        return dest

    def open(self, key: str) -> BinaryIO:
        obj = self.client.get_object(Bucket=self.bucket, Key=key)
        return obj["Body"]

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:
            return False

    def size(self, key: str) -> int:
        return self.client.head_object(Bucket=self.bucket, Key=key)["ContentLength"]

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=key)

    def open_range(self, key: str, start: int, length: int) -> bytes:
        end = start + length - 1
        obj = self.client.get_object(
            Bucket=self.bucket, Key=key, Range=f"bytes={start}-{end}"
        )
        return obj["Body"].read()

    def url(self, key: str, expires_s: int = 3600) -> str:
        """A temporary address ffmpeg reads on its own, via `Range`.

        Useful for *measuring* a file without downloading it: `ffprobe` pulls
        the header and stops. Downloading a whole recording just to learn its
        width would be worse than the problem.
        """
        return self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expires_s,
        )


_storage: Storage | None = None


def get_storage() -> Storage:
    global _storage
    if _storage is None:
        s = get_settings()
        _storage = S3Storage(s) if s.mode == "docker" else LocalStorage(s.blob_dir)
    return _storage


def local_copy(
    key: str,
    dest_dir: Path,
    *,
    on_progress: Callable[[float], None] | None = None,
) -> Path:
    """Ensures the blob is on local disk and returns the path (no copy when it
    already is local).

    A missing blob is an error, not an empty path: if the detector were handed
    a non-existent path it would simply detect nothing, and the job would
    finish "successfully" with no events at all. Failing here makes the error
    show up in the detector's report.

    `on_progress` receives 0..1 as the bytes arrive. It only makes sense in
    docker mode: in local mode the file is already on disk and there is nothing
    to download -- which is why the early return below does not call back.
    """
    st = get_storage()
    if not st.exists(key):
        raise FileNotFoundError(f"blob ausente no storage: {key!r}")
    if isinstance(st, LocalStorage):
        return st.path(key)
    dest = Path(dest_dir) / Path(key).name
    if on_progress is None:
        return st.get_file(key, dest)
    total = st.size(key) or 1
    downloaded = 0

    def count(n: int) -> None:
        nonlocal downloaded
        downloaded += n
        on_progress(min(1.0, downloaded / total))

    return st.get_file(key, dest, on_bytes=count)
