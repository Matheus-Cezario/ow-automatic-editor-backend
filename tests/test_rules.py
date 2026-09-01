"""Regras que cruzam o que mais de um detector viu. Funcao pura, entao da para
cobrir bem sem tocar em video.

Este arquivo ja foi tres vezes maior: cobria o motor que transformava eventos
em propostas de video -- rajadas, "sozinho contra todos", montagens por
habilidade. Essa fase nao existe mais. O que restou e a unica regra que ainda
nao cabe em detector nenhum, porque depende de olhar dois tipos de evento ao
mesmo tempo.
"""

from __future__ import annotations

from owcore.models import DetectionEvent, EventKind
from owcore.rules import derive_negated_ults


def kills(*ts: float) -> list[DetectionEvent]:
    return [DetectionEvent(kind=EventKind.KILL, t=t) for t in ts]


def test_ult_seguida_de_kill_vira_ult_anulada():
    ev = [DetectionEvent(kind=EventKind.ULT_USED, t=40.0)] + kills(41.5)
    out = derive_negated_ults(ev, 6.0)
    assert len(out) == 1
    assert out[0].t == 41.5
    assert out[0].meta["delay_s"] == 1.5


def test_ult_sem_kill_na_janela_nao_conta():
    ev = [DetectionEvent(kind=EventKind.ULT_USED, t=40.0)] + kills(50.0)
    assert derive_negated_ults(ev, 6.0) == []


def test_kill_antes_da_ult_nao_conta():
    ev = [DetectionEvent(kind=EventKind.ULT_USED, t=40.0)] + kills(39.0)
    assert derive_negated_ults(ev, 6.0) == []


def test_uma_ult_por_vez_e_a_kill_mais_proxima_fecha_a_jogada():
    """Duas ultimates anuladas na mesma partida sao dois momentos, e cada uma
    se fecha na primeira eliminacao que a segue -- nao na ultima."""
    ev = [DetectionEvent(kind=EventKind.ULT_USED, t=t) for t in (10.0, 30.0)]
    ev += kills(11.0, 12.0, 31.0)
    out = derive_negated_ults(ev, 6.0)
    assert [e.t for e in out] == [11.0, 31.0]
