from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample" / "match.mp4"
TRUTH = ROOT / "data" / "sample" / "match.truth.json"
MUSIC = ROOT / "data" / "sample" / "music.wav"
ULT_TEMPLATES = ROOT / "data" / "sample" / "ult_templates"
ABILITY_ICONS = ROOT / "data" / "sample" / "ability_icons"


def service_module(service: str, module: str = "detect") -> ModuleType:
    """Loads `services/<service>/<module>.py` by path.

    Five services have a file called `detect.py`. In production each runs in
    its own process, with its own directory on sys.path, and there is no
    ambiguity -- but in a single test process `import detect` would always pick
    the same one. Loading by path is how to name exactly what is under test.
    """
    path = ROOT / "services" / service / f"{module}.py"
    name = f"_svc_{service}_{module}"
    if name in sys.modules:
        return sys.modules[name]

    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader, f"could not load {path}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod

    # `main.py` does `from detect import ...`, relying on its own directory
    # being on sys.path -- which is how the service runs in the container. Here
    # that context is set up only while the module executes, and the short
    # name `detect` is dropped afterwards so it does not leak between services.
    svc_dir = str(ROOT / "services" / service)
    stale = sys.modules.pop("detect", None)
    sys.path.insert(0, svc_dir)
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(svc_dir)
        sys.modules.pop("detect", None)
        if stale is not None:
            sys.modules["detect"] = stale
    return mod


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Points all the infrastructure (queue, storage, database) at a temporary
    directory and resets the singletons, so a test never sees another's state."""
    monkeypatch.setenv("OW_MODE", "local")
    monkeypatch.setenv("OW_DATA_DIR", str(tmp_path))

    import owcore.bus as bus
    import owcore.config as config
    import owcore.db as db
    import owcore.profiles as profiles
    import owcore.storage as storage

    def reset() -> None:
        config.get_settings.cache_clear()
        profiles.load_profile.cache_clear()
        bus._bus = None
        storage._storage = None
        db._engine = None
        db._Session = None

    reset()
    db.init_db()
    yield config.get_settings()
    reset()


needs_sample = pytest.mark.skipif(
    not SAMPLE.exists(),
    reason="run: python tools/make_sample.py --out data/sample/match.mp4 "
           "--music data/sample/music.wav --ult-templates data/sample/ult_templates "
           "--ability-icons data/sample/ability_icons",
)


def tools_module(module: str):
    """Same idea as `service_module`, for the utilities in `tools/`."""
    path = ROOT / "tools" / f"{module}.py"
    name = f"_tool_{module}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader, f"could not load {path}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def short_sample(tmp_path_factory) -> Path:
    """A short cut of the synthetic video -- fast enough for an integration
    test to run the whole pipeline."""
    make_sample = tools_module("make_sample")
    out = tmp_path_factory.mktemp("short") / "match.mp4"
    make_sample.render(out, 12.0, "ffmpeg")
    return out
