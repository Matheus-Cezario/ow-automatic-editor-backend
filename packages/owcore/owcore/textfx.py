"""O texto do editor virando `drawtext` do ffmpeg.

Mora à parte de `compose.py` por um motivo prático: escapar texto para um
filtergraph é o tipo de código em que uma barra invertida a mais ou a menos
passa despercebida na revisão e só aparece quando alguém escreve `50%` num
rótulo. Isolado, ele tem teste próprio.
"""

from __future__ import annotations

from . import fonts
from .models import TimelineClip

#: O que precisa de barra invertida antes, e por quê:
#:
#: * `\` — a própria barra, senão ela come o próximo caractere;
#: * `:` — separa uma opção da outra dentro do filtro;
#: * `'` — delimita o valor de uma opção;
#: * `%` — abre uma expansão (`%{pts}` e parentes).
#:
#: Quebra de linha vira espaço: o `drawtext` aceita várias linhas, mas o
#: filtergraph é uma linha só, e uma quebra crua ali parte o grafo em dois.
_TROCAS: tuple[tuple[str, str], ...] = (
    ("\\", "\\\\"),
    (":", "\\:"),
    ("'", "\\'"),
    ("%", "\\%"),
    ("\n", " "),
    ("\r", " "),
)


def escapar(texto: str) -> str:
    """Deixa o texto passar pelo `drawtext` sem virar sintaxe."""
    for de, para in _TROCAS:
        texto = texto.replace(de, para)
    return texto


def cadeia(clip: TimelineClip, height: int) -> str:
    """O `drawtext` deste clipe, já posicionado no quadro.

    Tamanho e contorno vêm em **fração da altura**: a mesma montagem sai igual
    em 720p e em 4K, e um corpo de 48px que fica bem num sairia minúsculo no
    outro. O contorno não é enfeite — sem ele, texto branco some em cena clara.
    """
    estilo = clip.text_style
    corpo = max(1, int(round(estilo.size * height)))
    borda = int(round(estilo.outline * corpo))
    fonte = estilo.font or fonts.padrao()

    partes = [
        f"fontfile='{escapar(fonte)}'",
        f"text='{escapar(clip.text)}'",
        f"fontsize={corpo}",
        f"fontcolor={estilo.color}",
        # o texto anda pelo quadro com o mesmo x/y de qualquer clipe: o centro
        # é 0, as bordas são -1 e 1
        f"x=(w-text_w)/2+({clip.transform.x:.4f})*(w/2)",
        f"y=(h-text_h)/2+({clip.transform.y:.4f})*(h/2)",
    ]
    if borda > 0:
        partes += [f"borderw={borda}", f"bordercolor={estilo.outline_color}"]
    return "drawtext=" + ":".join(partes)
