"""Loads the HUD profile (positions, colours, thresholds)."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from .config import get_settings
from .models import RoiSpec


class Profile:
    def __init__(self, data: dict[str, Any]):
        self.data = data

    @property
    def name(self) -> str:
        return self.data.get("name", "unknown")

    def roi(self, key: str) -> RoiSpec:
        raw = self.data["rois"][key]
        return RoiSpec(**raw)

    def rois(self, keys: list[str]) -> list[RoiSpec]:
        return [self.roi(k) for k in keys]

    def section(self, key: str) -> dict[str, Any]:
        return self.data.get(key, {})


@lru_cache
def load_profile(name: str | None = None) -> Profile:
    s = get_settings()
    name = name or s.profile
    path = Path(s.profiles_dir) / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"profile '{name}' não encontrado em {path}")
    return Profile(json.loads(path.read_text(encoding="utf-8")))
