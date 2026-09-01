from __future__ import annotations

import json
import time
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import JSON, create_engine, inspect, text
from sqlalchemy.exc import IntegrityError, OperationalError, ProgrammingError
from sqlalchemy.orm import Session, sessionmaker

from .config import get_settings
from .models import Base

_engine = None
_Session: sessionmaker[Session] | None = None


def engine():
    global _engine, _Session
    if _engine is None:
        s = get_settings()
        url = s.resolved_database_url
        kwargs: dict = {
            "future": True,
            "pool_pre_ping": True,
            # a process here does not need ten idle connections; each one
            # costs a Postgres backend on the other side
            "pool_size": s.db_pool_size,
            "max_overflow": s.db_max_overflow,
            # a connection idle for too long is dropped rather than kept forever
            "pool_recycle": 1800,
        }
        if url.startswith("sqlite"):
            # several workers on one file: WAL + timeout avoid "database is locked"
            kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
        _engine = create_engine(url, **kwargs)
        if url.startswith("sqlite"):
            from sqlalchemy import event

            @event.listens_for(_engine, "connect")
            def _pragmas(dbapi_conn, _rec):  # pragma: no cover - trivial
                cur = dbapi_conn.cursor()
                cur.execute("PRAGMA journal_mode=WAL")
                cur.execute("PRAGMA busy_timeout=30000")
                cur.close()

        _Session = sessionmaker(bind=_engine, expire_on_commit=False, future=True)
    return _engine


#: arbitrary but fixed key of the advisory lock that serialises schema creation
_SCHEMA_LOCK = 0x0E_D170


def init_db(retries: int = 5) -> None:
    """Creates the schema safely with several services booting at once.

    Every worker calls this on boot, and `create_all(checkfirst=True)` is not
    atomic: two processes can both see the table missing and both emit the
    CREATE, which kills one of them with a uniqueness violation. In Postgres
    the answer is a transaction-level advisory lock, which serialises the DDL;
    the retry covers the rest (SQLite, and the database being briefly down).
    """
    eng = engine()
    for attempt in range(retries):
        try:
            if eng.dialect.name == "postgresql":
                with eng.begin() as conn:
                    conn.execute(
                        text("SELECT pg_advisory_xact_lock(:key)"),
                        {"key": _SCHEMA_LOCK},
                    )
                    Base.metadata.create_all(conn)
                    _reconcile_columns(conn)
            else:
                Base.metadata.create_all(eng)
                with eng.begin() as conn:
                    _reconcile_columns(conn)
            return
        except (IntegrityError, OperationalError, ProgrammingError):
            if attempt == retries - 1:
                raise
            time.sleep(0.5 * (attempt + 1))


def _reconcile_columns(conn) -> list[str]:
    """Adds to the database the columns the model gained later.

    `create_all` creates a missing table and **ignores a table that already
    exists** -- so a new field on an old model would never reach anyone who had
    run the system before. Without this, anyone with yesterday's database saw
    the montage blow up with "column renders.timelines does not exist", and the
    only way out would be dropping the database along with the matches already
    analysed.

    The column arrives **nullable**, rather than with the model's NOT NULL:
    putting NOT NULL on a table that already has rows would require rewriting
    it (and in SQLite you simply cannot). Readers treat absence as empty
    (`r.timelines or []`), so NULL does no harm; even so the old rows are
    backfilled with the type's empty value, to keep the database from having
    two ways of saying the same thing.

    What this does **not** do: rename, drop or change the type of a column. For
    changes like that the route is still recreating the database -- this project
    has no versioned migrations, and does not need them.
    """
    inspector = inspect(conn)
    existing = set(inspector.get_table_names())
    quote = conn.dialect.identifier_preparer.quote
    added: list[str] = []

    for name, table in Base.metadata.tables.items():
        if name not in existing:
            continue  # just created by create_all, so it came complete
        in_db = {c["name"]: c for c in inspector.get_columns(name)}
        for col in table.columns:
            if col.name in in_db:
                # Column added on an earlier boot, when the backfill did not
                # yet cover this type: the rows from back then were left with
                # NULL where the model promises a value. Repairing them is
                # cheap (the UPDATE finds nothing on later runs) and keeps NULL
                # from leaking to readers -- that is how `round(job.fps, 3)`
                # took down the entire match listing.
                if in_db[col.name]["nullable"] and not col.nullable:
                    empty_value = _empty_value_for(col)
                    if empty_value is not None:
                        conn.execute(
                            text(
                                f"UPDATE {quote(name)} SET {quote(col.name)} = :v "
                                f"WHERE {quote(col.name)} IS NULL"
                            ),
                            {"v": empty_value},
                        )
                continue
            sql_type = col.type.compile(dialect=conn.dialect)
            conn.execute(
                text(f"ALTER TABLE {quote(name)} ADD COLUMN {quote(col.name)} {sql_type}")
            )
            empty_value = _empty_value_for(col)
            if empty_value is not None:
                conn.execute(
                    text(
                        f"UPDATE {quote(name)} SET {quote(col.name)} = :v "
                        f"WHERE {quote(col.name)} IS NULL"
                    ),
                    {"v": empty_value},
                )
            added.append(f"{name}.{col.name}")

    return added


def _empty_value_for(col):
    """What to put in the old rows of a newly created column.

    It comes from the default declared on the model -- `0.0` on a Float, `""`
    on a String, `[]` on a JSON -- so an old row starts reading like a new one.
    Without this it reads `NULL`, and consumers break far away from here: that
    is how `round(job.fps, 3)` took down the whole match listing after `fps`
    joined the model.

    The value comes from *executing* the default, not from recognising it:
    SQLAlchemy wraps `default=list` in a callable of its own, so comparing
    against `list` never matches.
    """
    if col.default is None:
        return None
    try:
        value = (
            col.default.arg(None) if col.default.is_callable else col.default.arg
        )
    except Exception:  # a context-dependent default: no way to guess it
        return None

    if isinstance(col.type, JSON):
        return json.dumps(value) if isinstance(value, (list, dict)) else None
    # dates and the like stay out: `utcnow` on an old row would be an invented
    # date, and NULL says "unknown" better
    if isinstance(value, bool) or isinstance(value, (int, float, str)):
        return value
    return None


@contextmanager
def session() -> Iterator[Session]:
    engine()
    assert _Session is not None
    s = _Session()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()
