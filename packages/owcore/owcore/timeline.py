"""Da linha do tempo que o usuario montou para a lista de pedacos a cortar.

Regras puras, sem ffmpeg e sem banco -- e por isso que da para testar a
montagem inteira sem abrir um video sequer.

A diferenca para `rules.py` e de quem decide. La o sistema escolhe os cortes a
partir dos eventos; aqui ele nao escolhe nada: recebe blocos ja posicionados e
so precisa dizer o que vai para o ffmpeg, em que ordem, e o que fazer com o
espaco vazio entre um bloco e o proximo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .models import MIN_CUT_S, TimelineCut

#: Buraco menor que isto nao vira preto: e menos de um quadro a 24 fps, e
#: emenda-lo custaria uma reencodagem inteira para ninguem ver a diferenca. O
#: bloco seguinte simplesmente comeca esse tanto mais cedo.
MIN_GAP_S = 0.04


@dataclass(slots=True)
class Piece:
    """Um pedaco do video final, na ordem em que sera concatenado.

    Ou e um corte da gravacao (`black=False`, com `start_s`/`end_s` na
    gravacao), ou e o preto que preenche um buraco deixado pelo usuario.
    """

    duration_s: float
    start_s: float = 0.0
    end_s: float = 0.0
    black: bool = False
    #: instante do momento que originou o corte -- so para nomear o arquivo
    source_t: float = 0.0
    kind: str = ""

    @property
    def is_cut(self) -> bool:
        return not self.black


def plan(
    cuts: Sequence[TimelineCut],
    *,
    source_duration_s: float = 0.0,
    min_gap_s: float = MIN_GAP_S,
) -> list[Piece]:
    """Os pedacos, em ordem, cobrindo o video inteiro sem furo.

    O que o usuario deixou vazio vira preto com a musica tocando por cima --
    e o que qualquer editor faz, e e o que preserva a promessa da tela: cada
    bloco cai exatamente no ponto da musica onde ele foi posto. Emendar os
    blocos para tapar o buraco seria mover todos os seguintes.

    Um corte que passa do fim da gravacao e aparado, e o que sobrar do lugar
    dele tambem vira preto -- de novo para nao empurrar quem vem depois.
    """
    ordenados = sorted(cuts, key=lambda c: c.at_s)
    pecas: list[Piece] = []
    cursor = 0.0

    for corte in ordenados:
        buraco = corte.at_s - cursor
        if buraco >= min_gap_s:
            pecas.append(Piece(duration_s=buraco, black=True))
            cursor += buraco

        duracao = corte.duration_s
        if source_duration_s > 0:
            duracao = min(duracao, max(0.0, source_duration_s - corte.start_s))

        if duracao >= MIN_CUT_S:
            pecas.append(
                Piece(
                    duration_s=duracao,
                    start_s=corte.start_s,
                    end_s=corte.start_s + duracao,
                    source_t=corte.source_t,
                    kind=corte.kind,
                )
            )

        # o que a aparacao comeu (ou o bloco inteiro, se ele caiu fora da
        # gravacao) vira preto: o proximo bloco continua no lugar marcado
        sobra = corte.duration_s - max(0.0, duracao)
        if sobra >= min_gap_s:
            pecas.append(Piece(duration_s=sobra, black=True))

        cursor = corte.until_s

    # um preto no fim nao acrescenta nada: o video acaba no ultimo corte
    while pecas and pecas[-1].black:
        pecas.pop()

    # dois pretos seguidos sao um preto so -- cada peca custa uma codificacao
    juntos: list[Piece] = []
    for peca in pecas:
        if peca.black and juntos and juntos[-1].black:
            juntos[-1].duration_s += peca.duration_s
            continue
        juntos.append(peca)
    return juntos


def total_duration_s(pecas: Sequence[Piece]) -> float:
    return sum(p.duration_s for p in pecas)


def snap(value: float, beats: Sequence[float], tolerance_s: float = 0.12) -> float:
    """Gruda um instante na batida mais proxima, se houver uma perto.

    Existe aqui, e nao so no app, porque o servidor tambem precisa da mesma
    resposta: o app gruda enquanto o usuario arrasta, e quem confere depois tem
    de chegar no mesmo numero.
    """
    if not beats:
        return value
    melhor = min(beats, key=lambda b: abs(b - value))
    return melhor if abs(melhor - value) <= tolerance_s else value
