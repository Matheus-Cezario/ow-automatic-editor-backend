"""Central configuration. Everything comes from environment variables, with
defaults that work with no infrastructure installed at all (`local` mode)."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

#: Project root -- where the defaults for `data/`, `config/` and `templates/`
#: come from. In dev, `owcore` is installed in editable mode and the file path
#: points inside the repository. Inside the Docker image the package lives in
#: `site-packages`, where that calculation is worthless -- which is why
#: `OW_ROOT` wins when it is set (the Dockerfile sets it).
REPO_ROOT = Path(os.environ.get("OW_ROOT") or Path(__file__).resolve().parents[3])


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OW_", env_file=".env", extra="ignore")

    # -- execution mode -----------------------------------------------------
    # "local": on-disk queue + folder storage + SQLite. No server at all.
    # "docker": Redis Streams + MinIO + Postgres.
    mode: Literal["local", "docker"] = "local"

    data_dir: Path = REPO_ROOT / "data"
    profiles_dir: Path = REPO_ROOT / "config" / "profiles"
    templates_dir: Path = REPO_ROOT / "templates"

    #: The compiled Flutter app. The frontend is a sibling project of the
    #: backend, so the default points outside here; in the container it is
    #: mounted elsewhere and this variable is what says where. If the folder
    #: has no `index.html`, the gateway simply serves the API alone.
    web_dir: Path = REPO_ROOT.parent / "frontend" / "build" / "web"

    # -- database -----------------------------------------------------------
    database_url: str = ""  # empty => derived from the mode
    #: Connections each process keeps open.
    #:
    #: SQLAlchemy's default (5 + 10 overflow) is sized for a web server. A
    #: worker here is single-threaded: it never uses more than one connection,
    #: and the others sat open holding a Postgres backend each -- a few MB per
    #: backend, times ten services. The gateway, on the other hand, does serve
    #: several requests at once and raises its own value via an environment
    #: variable.
    db_pool_size: int = 2
    db_max_overflow: int = 3

    # -- message bus --------------------------------------------------------
    redis_url: str = "redis://localhost:6379/0"
    #: Cap on messages kept per stream.
    #:
    #: Redis Streams do **not** delete what was delivered: `XACK` only removes
    #: the message from the group's pending list, and the entry goes on
    #: occupying Redis memory forever. Without this cap, Redis RAM grows with
    #: the number of matches already processed and never comes back.
    #:
    #: The trim is approximate (`~`), which is the cheap one: Redis trims at
    #: the node boundary, a little above the number asked for. Ten thousand
    #: messages is thousands of matches of slack -- the queue moves in seconds,
    #: and no consumer falls that far behind.
    stream_maxlen: int = 10_000
    #: How long a message already consumed by everyone stays in the on-disk
    #: queue before being swept (local mode). The slack protects a group coming
    #: up for the first time after the message was published.
    bus_retention_s: float = 3600.0

    # -- storage ------------------------------------------------------------
    s3_endpoint: str = "http://localhost:9000"
    s3_bucket: str = "ow-editor"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"

    # -- external binaries --------------------------------------------------
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"
    #: `drawtext` font. Empty looks for one on the system -- see `owcore.fonts`
    font: str = ""

    # -- behaviour ----------------------------------------------------------
    profile: str = "ow2_default"
    log_level: str = "INFO"
    # Seconds the pipeline waits for detectors that never answered before
    # closing the analysis with what it has.
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
