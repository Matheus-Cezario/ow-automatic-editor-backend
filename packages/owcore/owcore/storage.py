"""Storage de blobs. `LocalStorage` (pasta) e `S3Storage` (MinIO) atrás da
mesma interface, escolhidos pelo modo de execução."""

from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO

from .config import Settings, get_settings


class Storage(ABC):
    @abstractmethod
    def put_file(self, key: str, path: Path) -> str: ...

    @abstractmethod
    def put_stream(self, key: str, fh: BinaryIO) -> str: ...

    @abstractmethod
    def get_file(self, key: str, dest: Path) -> Path: ...

    @abstractmethod
    def open(self, key: str) -> BinaryIO: ...

    @abstractmethod
    def exists(self, key: str) -> bool: ...

    @abstractmethod
    def size(self, key: str) -> int: ...

    def open_range(self, key: str, start: int, length: int) -> bytes:
        """Le uma faixa de bytes. O player HTML5 pede Range para poder buscar
        no meio do video sem baixar o arquivo inteiro."""
        with self.open(key) as fh:
            fh.seek(start)
            return fh.read(length)


class LocalStorage(Storage):
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _p(self, key: str) -> Path:
        # normaliza e impede escapar da raiz
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

    def get_file(self, key: str, dest: Path) -> Path:
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self._p(key), dest)
        return dest

    def open(self, key: str) -> BinaryIO:
        return open(self._p(key), "rb")

    def exists(self, key: str) -> bool:
        return self._p(key).exists()

    def size(self, key: str) -> int:
        return self._p(key).stat().st_size

    def path(self, key: str) -> Path:
        """Atalho só do backend local: evita cópia quando o ffmpeg pode ler direto."""
        return self._p(key)

    def url(self, key: str, expira_s: int = 3600) -> str:
        """O próprio caminho: aqui o ffmpeg já lê direto do disco."""
        return str(self._p(key))


class S3Storage(Storage):
    def __init__(self, s: Settings):
        import boto3  # import tardio: só o modo docker precisa

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

    def get_file(self, key: str, dest: Path) -> Path:
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, key, str(dest))
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

    def open_range(self, key: str, start: int, length: int) -> bytes:
        end = start + length - 1
        obj = self.client.get_object(
            Bucket=self.bucket, Key=key, Range=f"bytes={start}-{end}"
        )
        return obj["Body"].read()

    def url(self, key: str, expira_s: int = 3600) -> str:
        """Um endereço temporário que o ffmpeg lê sozinho, por `Range`.

        Serve para *medir* um arquivo sem baixá-lo: o `ffprobe` puxa o
        cabeçalho e para. Baixar uma gravação inteira só para saber a largura
        dela seria pior do que o problema.
        """
        return self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expira_s,
        )


_storage: Storage | None = None


def get_storage() -> Storage:
    global _storage
    if _storage is None:
        s = get_settings()
        _storage = S3Storage(s) if s.mode == "docker" else LocalStorage(s.blob_dir)
    return _storage


def local_copy(key: str, dest_dir: Path) -> Path:
    """Garante o blob em disco local e devolve o caminho (sem copiar se já for local).

    Um blob ausente é erro, não um caminho vazio: se o detector recebesse um
    caminho inexistente ele simplesmente não detectaria nada, e o job
    terminaria "com sucesso" sem nenhum evento. Falhar aqui faz o erro
    aparecer no relatório do detector.
    """
    st = get_storage()
    if not st.exists(key):
        raise FileNotFoundError(f"blob ausente no storage: {key!r}")
    if isinstance(st, LocalStorage):
        return st.path(key)
    dest = Path(dest_dir) / Path(key).name
    return st.get_file(key, dest)
