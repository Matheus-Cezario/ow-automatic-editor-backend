"""Each table has one domain that writes it, and this holds the tests to it.

`TABLE_OWNER` (owcore.models.tables) names the owner. A `before_flush`
listener looks at every row about to be written, finds in the call stack the
first file that belongs to a domain, and records the write when that domain
is not the table's owner. `conftest.py` fails the test where it happened --
so every existing test checks the rule, with no test written for it.

Reading is free: the render reads the montage, the analysis lists renders.
"""

from __future__ import annotations

import sys
from pathlib import Path

from sqlalchemy import event
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.orm import Session

from owcore.models import TABLE_OWNER, Job

ROOT = Path(__file__).resolve().parents[1]

#: Where each domain's code lives. A file in none of them (owcore.db,
#: SQLAlchemy, the gateway's common.py) is skipped and the search goes on to
#: whoever called it.
DOMAIN_PATHS = {
    "analise": (
        "services/gateway/routers/analise.py",
        "services/preprocessor/",
        "services/detector_kills/",
        "services/detector_survival/",
        "services/detector_ults/",
        "services/detector_banner/",
        "services/detector_killfeed/",
        "services/planner/",
        "services/thumbs/",
        "packages/owcore/owcore/jobs.py",
        "packages/owcore/owcore/detector.py",
    ),
    "midia": (
        "services/gateway/routers/midia.py",
        "services/beats/",
    ),
    "montagens": ("services/gateway/routers/montagens.py",),
    "render": (
        "services/gateway/routers/render.py",
        "services/editor/",
        "services/previewer/",
        "packages/owcore/owcore/renders.py",
    ),
    "catalogo": ("services/gateway/routers/catalogo.py",),
}

#: Code that may write anywhere: the tests set up their own rows, and the
#: tools build sample data.
FREE_PATHS = ("tests/", "tools/")

#: Writes found so far, as text; conftest empties and checks it per test.
violations: list[str] = []


def _domain_of_caller() -> str | None:
    """The domain of the innermost domain file on the stack; None when the
    write comes from a test or a tool, or from nowhere known."""
    frame = sys._getframe(1)
    while frame is not None:
        if frame.f_code.co_filename == __file__:
            frame = frame.f_back
            continue
        try:
            rel = Path(frame.f_code.co_filename).resolve().relative_to(ROOT).as_posix()
        except ValueError:
            rel = ""
        if rel.startswith(FREE_PATHS):
            return None
        for domain, paths in DOMAIN_PATHS.items():
            if rel.startswith(paths):
                return domain
        frame = frame.f_back
    return None


def _changed_columns(obj) -> set[str]:
    state = sa_inspect(obj)
    return {
        attr.key
        for attr in state.mapper.column_attrs
        if state.attrs[attr.key].history.has_changes()
    }


def _allowed(domain: str, table: str, obj, op: str, deleting_a_job: bool) -> bool:
    # The old single draft lives in a column of the job. The montages code
    # writes it (save, delete, and moving it into the named montages); moving
    # it to a table of its own would need a data migration this project
    # does not have.
    if domain == "montagens" and table == "jobs" and op == "update":
        return _changed_columns(obj) <= {"draft", "updated_at"}
    # Deleting a match takes everything that hangs from it, in one
    # transaction, through the relationship's cascade.
    if domain == "analise" and op == "delete" and deleting_a_job:
        return True
    return False


def _before_flush(session: Session, _ctx, _instances) -> None:
    writes = (
        [("insert", o) for o in session.new]
        + [("update", o) for o in session.dirty if session.is_modified(o, include_collections=False)]
        + [("delete", o) for o in session.deleted]
    )
    if not writes:
        return
    domain = _domain_of_caller()
    if domain is None:
        return
    deleting_a_job = any(isinstance(o, Job) for o in session.deleted)
    for op, obj in writes:
        table = getattr(obj, "__tablename__", None)
        owner = TABLE_OWNER.get(table)
        if owner is None or owner == domain:
            continue
        if _allowed(domain, table, obj, op, deleting_a_job):
            continue
        violations.append(f"{domain} did {op} on `{table}`, which belongs to {owner}")


def install() -> None:
    if not event.contains(Session, "before_flush", _before_flush):
        event.listen(Session, "before_flush", _before_flush)
