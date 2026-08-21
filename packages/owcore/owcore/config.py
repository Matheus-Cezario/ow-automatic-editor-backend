"""Configuração central. Tudo vem de variáveis de ambiente com defaults que
funcionam sem nenhuma infra instalada (modo `local`)."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

#: Raiz do projeto -- de onde saem os defaults de `data/`, `config/` e
#: `templates/`. Em dev, `owcore` e instalado em modo editavel e o caminho do
#: arquivo aponta para dentro do repositorio. Dentro da imagem Docker o pacote
#: vive em `site-packages`, e ai esse calculo nao vale nada -- por isso
#: `OW_ROOT` manda quando esta definido (o Dockerfile define).
REPO_ROOT = Path(os.environ.get("OW_ROOT") or Path(__file__).resolve().parents[3])


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OW_", env_file=".env", extra="ignore")

    # ── modo de execução ────────────────────────────────────────────────────
    # "local": fila em disco + storage em pasta + SQLite. Nenhum servidor.
    # "docker": Redis Streams + MinIO + Postgres.
    mode: Literal["local", "docker"] = "local"

    data_dir: Path = REPO_ROOT / "data"
    profiles_dir: Path = REPO_ROOT / "config" / "profiles"
    templates_dir: Path = REPO_ROOT / "templates"

    #: App Flutter compilado. O frontend e um projeto irmao do backend, entao o
    #: padrao aponta para fora daqui; no container ele e montado em outro lugar
    #: e esta variavel e quem diz onde. Se a pasta nao tiver um `index.html`, o
    #: gateway simplesmente serve so a API.
    web_dir: Path = REPO_ROOT.parent / "frontend" / "build" / "web"

    # ── banco ───────────────────────────────────────────────────────────────
    database_url: str = ""  # vazio => derivado do modo

    # ── barramento ──────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"

    # ── storage ─────────────────────────────────────────────────────────────
    s3_endpoint: str = "http://localhost:9000"
    s3_bucket: str = "ow-editor"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"

    # ── binários externos ───────────────────────────────────────────────────
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"

    # ── comportamento ───────────────────────────────────────────────────────
    profile: str = "ow2_default"
    log_level: str = "INFO"
    # Segundos que o editor espera por detectores que nunca responderam antes
    # de renderizar com o que tem.
    detector_timeout_s: float = 900.0

    @property
    def resolved_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        if self.mode == "docker":
            return "postgresql+psycopg://ow:ow@postgres:5432/ow"
        db_path = self.data_dir / "ow.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{db_path.as_posix()}"

    @property
    def bus_dir(self) -> Path:
        return self.data_dir / "bus"

    @property
    def blob_dir(self) -> Path:
        return self.data_dir / "blobs"

    @property
    def work_dir(self) -> Path:
        return self.data_dir / "work"


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    for d in (s.data_dir, s.bus_dir, s.blob_dir, s.work_dir):
        d.mkdir(parents=True, exist_ok=True)
    return s
