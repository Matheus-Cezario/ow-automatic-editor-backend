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
    """Carrega `services/<service>/<module>.py` pelo caminho.

    Cinco servicos tem um arquivo chamado `detect.py`. Em producao cada um roda
    no seu processo, com o proprio diretorio no sys.path, e nao ha ambiguidade
    -- mas num unico processo de teste `import detect` pegaria sempre o mesmo.
    Carregar pelo caminho e o jeito de nomear exatamente o que se quer testar.
    """
    path = ROOT / "services" / service / f"{module}.py"
    name = f"_svc_{service}_{module}"
    if name in sys.modules:
        return sys.modules[name]

    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader, f"nao consegui carregar {path}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod

    # `main.py` faz `from detect import ...`, contando com o proprio diretorio
    # no sys.path -- que e como o servico roda no container. Aqui esse contexto
    # e montado so durante a execucao do modulo, e o nome curto `detect` e
    # descartado depois para nao vazar de um servico para outro.
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
    """Aponta toda a infra (fila, storage, banco) para um diretorio temporario
    e zera os singletons, para um teste nunca ver o estado de outro."""
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
    reason="rode: python tools/make_sample.py --out data/sample/match.mp4 "
           "--music data/sample/music.wav --ult-templates data/sample/ult_templates "
           "--ability-icons data/sample/ability_icons",
)


def tools_module(module: str):
    """Mesma ideia de `service_module`, para os utilitarios em `tools/`."""
    path = ROOT / "tools" / f"{module}.py"
    name = f"_tool_{module}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader, f"nao consegui carregar {path}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def short_sample(tmp_path_factory) -> Path:
    """Um recorte curto do video sintetico -- rapido o bastante para um teste
    de integracao rodar o pipeline inteiro."""
    make_sample = tools_module("make_sample")
    out = tmp_path_factory.mktemp("short") / "match.mp4"
    make_sample.render(out, 12.0, "ffmpeg")
    return out
