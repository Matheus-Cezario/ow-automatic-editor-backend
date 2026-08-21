"""Onde está a fonte que o `drawtext` vai usar.

O ffmpeg não tem fonte embutida: sem um arquivo `.ttf` no caminho, todo texto
falha na hora de renderizar. Este módulo acha uma, e falha alto quando não acha
-- descobrir isso no meio de um render seria pior do que na configuração.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from .config import get_settings

#: Onde procurar, em ordem. A DejaVu vem junto com o ffmpeg na imagem Docker; as
#: outras cobrem quem roda o sistema fora dela.
CANDIDATAS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
)


@lru_cache
def padrao() -> str:
    """O caminho da fonte a usar quando ninguém escolheu uma.

    `OW_FONT` manda quando definida. Sem ela, vale a primeira das candidatas que
    existir no disco.
    """
    escolhida = get_settings().font
    if escolhida:
        return escolhida
    for caminho in CANDIDATAS:
        if Path(caminho).is_file():
            return caminho
    raise FileNotFoundError(
        "nenhuma fonte encontrada para o texto; aponte uma em OW_FONT"
    )


def existe() -> bool:
    """Dá para escrever texto nesta máquina?"""
    try:
        padrao()
    except FileNotFoundError:
        return False
    return True
