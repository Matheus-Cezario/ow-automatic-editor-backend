"""The system's API: upload, progress tracking and video delivery.

It is the only service exposed to the world. It processes nothing -- it stores
the file, creates the job and publishes on the bus; the rest happens in the
workers.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from owcore.config import get_settings
from owcore.db import init_db

from routers import analise, catalogo, midia, montagens, render


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="OW Editor",
    description="Overwatch 2 match editor with automatic event detection.",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # the Flutter web app runs on another port during dev
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────── routes ────────────────────────────────────


@app.get("/api/health")
def health() -> dict[str, Any]:
    s = get_settings()
    return {"ok": True, "mode": s.mode, "profile": s.profile}


for _router in (analise, render, catalogo, midia, montagens):
    app.include_router(_router.router)


# -- the compiled Flutter app, when present (single-origin deploy) ----------
#
# With the app served by the gateway itself, Flutter calls the API on a
# relative path and the same build works on any host, with no recompiling and
# no CORS setup. If nobody ran `flutter build web`, the mount simply does not
# happen and the API stays available on its own. The check is on `index.html`,
# and not on the directory: in Docker the bind mount creates the folder empty
# even when nobody compiled the app, and mounting StaticFiles on it would turn
# the site root into a 404.

_web = Path(get_settings().web_dir)
if (_web / "index.html").is_file():
    from fastapi.staticfiles import StaticFiles

    class _Revalidated(StaticFiles):
        """The app's files, with the browser told to ask before reusing them.

        Flutter names its bundles the same on every build (`main.dart.js`), and
        with no `Cache-Control` the browser caches them by heuristic: a fresh
        build kept loading the previous one. `no-cache` still lets it keep the
        file -- it only revalidates, and an unchanged file costs a 304.
        """

        async def get_response(self, path, scope):
            response = await super().get_response(path, scope)
            response.headers["Cache-Control"] = "no-cache"
            return response

    app.mount("/", _Revalidated(directory=str(_web), html=True), name="web")


# The mount comes **last** on purpose: it matches any path, and every route
# registered after it becomes unreachable -- a `POST` to one of them returns
# 405, because the file server is what answers.
