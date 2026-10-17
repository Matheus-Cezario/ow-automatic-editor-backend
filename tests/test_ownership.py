"""The table-owner check itself: it must catch a write from the wrong domain,
and stay quiet for the owner (`tests/ownership.py`)."""

from __future__ import annotations

import ownership
from conftest import ROOT

WRITE_A_PRESET = """
from owcore.db import session
from owcore.models import Preset

with session() as s:
    s.add(Preset(name="seen from here"))
"""


def _run_as(relative_path: str) -> list[str]:
    """Runs the write as if it were code from that file: the check reads the
    file name off the call stack, which is what `compile` sets."""
    ownership.violations.clear()
    exec(compile(WRITE_A_PRESET, str(ROOT / relative_path), "exec"), {})
    found = list(ownership.violations)
    ownership.violations.clear()
    return found


def test_a_write_from_another_domain_is_caught(isolated):
    found = _run_as("services/gateway/routers/catalogo.py")
    assert found == ["catalogo did insert on `presets`, which belongs to montagens"]


def test_the_owner_writes_freely(isolated):
    assert _run_as("services/gateway/routers/montagens.py") == []


def test_every_table_has_an_owner_with_code():
    from owcore.models import TABLE_OWNER, Base

    assert set(TABLE_OWNER) == set(Base.metadata.tables)
    assert set(TABLE_OWNER.values()) <= set(ownership.DOMAIN_PATHS)
    for paths in ownership.DOMAIN_PATHS.values():
        for path in paths:
            assert (ROOT / path).exists(), f"{path} no longer exists"
