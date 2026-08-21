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
        url = get_settings().resolved_database_url
        kwargs: dict = {"future": True, "pool_pre_ping": True}
        if url.startswith("sqlite"):
            # vários workers no mesmo arquivo: WAL + timeout evitam "database is locked"
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


#: chave arbitraria, mas fixa, do advisory lock que serializa a criacao do schema
_SCHEMA_LOCK = 0x0E_D170


def init_db(retries: int = 5) -> None:
    """Cria o schema de forma segura com varios servicos subindo ao mesmo tempo.

    Todo worker chama isto no boot, e `create_all(checkfirst=True)` nao e
    atomico: dois processos podem ver a tabela faltando e os dois emitirem o
    CREATE, o que faz um deles morrer com violacao de unicidade. No Postgres a
    solucao e um advisory lock a nivel de transacao, que serializa o DDL; o
    retry cobre o resto (SQLite e indisponibilidade momentanea do banco).
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
    """Acrescenta ao banco as colunas que o modelo ganhou depois.

    `create_all` cria tabela que falta e **ignora tabela que ja existe** -- entao
    um campo novo num modelo antigo nunca chegaria a quem ja rodou o sistema
    antes. Sem isto, quem tinha um banco de ontem via a montagem manual estourar
    com "column renders.timelines does not exist", e a unica saida seria apagar
    o banco e perder as partidas ja analisadas.

    A coluna entra **anulavel**, e nao com o NOT NULL do modelo: pôr NOT NULL
    numa tabela que ja tem linhas exigiria reescreve-la (e no SQLite nem da).
    Quem le trata a ausencia como vazio (`r.timelines or []`), entao NULL nao
    machuca; ainda assim as linhas antigas sao preenchidas com o vazio do tipo,
    para o banco nao ficar com dois jeitos de dizer a mesma coisa.

    O que isto **nao** faz: renomear, remover ou mudar o tipo de coluna. Para
    mudancas assim o caminho continua sendo recriar o banco -- este projeto nao
    tem (nem precisa de) migracao versionada.
    """
    inspector = inspect(conn)
    existentes = set(inspector.get_table_names())
    citar = conn.dialect.identifier_preparer.quote
    adicionadas: list[str] = []

    for nome, tabela in Base.metadata.tables.items():
        if nome not in existentes:
            continue  # acabou de ser criada por create_all, ja veio completa
        no_banco = {c["name"]: c for c in inspector.get_columns(nome)}
        for col in tabela.columns:
            if col.name in no_banco:
                # Coluna acrescentada num boot anterior, quando o backfill ainda
                # nao cobria este tipo: as linhas de entao ficaram com NULL onde
                # o modelo promete um valor. Repara-las e barato (o UPDATE nao
                # acha nada nas vezes seguintes) e evita o NULL vazar para quem
                # le -- foi assim que `round(job.fps, 3)` derrubou a listagem
                # inteira de partidas.
                if no_banco[col.name]["nullable"] and not col.nullable:
                    vazio = _valor_vazio(col)
                    if vazio is not None:
                        conn.execute(
                            text(
                                f"UPDATE {citar(nome)} SET {citar(col.name)} = :v "
                                f"WHERE {citar(col.name)} IS NULL"
                            ),
                            {"v": vazio},
                        )
                continue
            tipo = col.type.compile(dialect=conn.dialect)
            conn.execute(
                text(f"ALTER TABLE {citar(nome)} ADD COLUMN {citar(col.name)} {tipo}")
            )
            vazio = _valor_vazio(col)
            if vazio is not None:
                conn.execute(
                    text(
                        f"UPDATE {citar(nome)} SET {citar(col.name)} = :v "
                        f"WHERE {citar(col.name)} IS NULL"
                    ),
                    {"v": vazio},
                )
            adicionadas.append(f"{nome}.{col.name}")

    return adicionadas


def _valor_vazio(col):
    """O que por nas linhas antigas de uma coluna recem-criada.

    Sai do default declarado no modelo -- `0.0` num Float, `""` num String,
    `[]` num JSON --, entao uma linha antiga passa a ler igual a uma nova. Sem
    isto ela le `NULL`, e quem consome quebra em lugares distantes: foi assim
    que `round(job.fps, 3)` derrubou a listagem inteira de partidas depois de
    `fps` entrar no modelo.

    O valor sai de *executar* o default, e nao de reconhece-lo: o SQLAlchemy
    embrulha `default=list` num invocavel proprio, entao comparar com `list`
    nunca bate.
    """
    if col.default is None:
        return None
    try:
        valor = (
            col.default.arg(None) if col.default.is_callable else col.default.arg
        )
    except Exception:  # default que depende de contexto: nao da para adivinhar
        return None

    if isinstance(col.type, JSON):
        return json.dumps(valor) if isinstance(valor, (list, dict)) else None
    # datas e afins ficam de fora: `utcnow` de uma linha antiga seria uma data
    # inventada, e NULL diz melhor "nao se sabe"
    if isinstance(valor, bool) or isinstance(valor, (int, float, str)):
        return valor
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
