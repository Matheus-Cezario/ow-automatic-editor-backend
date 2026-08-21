"""Da linha do tempo em camadas para um grafo de filtros do ffmpeg.

Este arquivo **não roda ffmpeg**: ele escreve o `-filter_complex` e diz que
entradas passar. É pura montagem de texto, e por isso dá para testar o grafo
inteiro sem codificar um quadro sequer — o que importa quando um erro no grafo
derruba o render todo, e não apenas um corte.

## Por que um grafo, e não corte-e-emenda

A V1 cortava cada trecho num arquivo e concatenava. Funciona, é resistente (um
corte ruim custa só ele) e **não comporta camada**: sobrepor exige que dois
pedaços existam ao mesmo tempo, e concatenação é justamente o contrário disso.

Por isso o caminho antigo continua vivo para montagem de uma camada só, e este
aqui entra quando há camada, transformação ou som ajustado. A escolha é de
`Timeline.de_uma_camada_so`.

## A forma do grafo

Uma tela de fundo cobre o vídeo inteiro, e cada clipe é sobreposto nela na hora
certa:

    [fundo][v0] overlay(enable=...) [t0]
    [t0]   [v1] overlay(enable=...) [t1]  ...

O fundo resolve de graça o que na V1 era caso especial: buraco entre clipes é
onde nada foi sobreposto, e o que se vê ali é a própria tela de fundo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .models import ClipSource, MediaKind, Timeline, TimelineClip


@dataclass(slots=True)
class MidiaNoDisco:
    """Um item da biblioteca já baixado, com o que o grafo precisa saber dele.

    O tipo importa porque imagem não é vídeo: ela não corre no tempo, então
    entra em laço e ganha a duração que o clipe pedir.
    """

    caminho: Path
    kind: str = MediaKind.VIDEO

    @property
    def e_imagem(self) -> bool:
        return self.kind == MediaKind.IMAGE


@dataclass(slots=True)
class Entrada:
    """Um `-i` do ffmpeg, com o que vem antes dele."""

    caminho: str
    #: `-ss` antes do input: o ffmpeg pula até o keyframe mais próximo em vez de
    #: decodificar desde o começo, o que faz cada clipe custar quase nada
    seek: float | None = None
    duracao: float | None = None
    #: entradas sintéticas (cor, silêncio) vêm de `-f lavfi`
    lavfi: bool = False
    #: uma imagem é um quadro só; em laço ela vira vídeo pelo tempo que se pedir
    loop: bool = False

    def argumentos(self) -> list[str]:
        args: list[str] = []
        if self.lavfi:
            args += ["-f", "lavfi"]
        if self.loop:
            args += ["-loop", "1"]
        if self.seek is not None:
            args += ["-ss", f"{self.seek:.3f}"]
        if self.duracao is not None:
            args += ["-t", f"{self.duracao:.3f}"]
        args += ["-i", self.caminho]
        return args


@dataclass(slots=True)
class Composicao:
    entradas: list[Entrada] = field(default_factory=list)
    filtros: list[str] = field(default_factory=list)
    mapa_video: str = ""
    mapa_audio: str | None = None
    duracao_s: float = 0.0

    @property
    def filter_complex(self) -> str:
        return ";".join(self.filtros)

    def argumentos_de_entrada(self) -> list[str]:
        return [a for e in self.entradas for a in e.argumentos()]


def _posicao(clip: TimelineClip) -> tuple[str, str]:
    """Onde o clipe é sobreposto, em expressões que o ffmpeg avalia.

    `x` e `y` do transform são deslocamentos do centro normalizados pela metade
    do quadro, então a mesma montagem vale em qualquer resolução: `W` e `H` são
    a tela, `w` e `h` o clipe já escalado.
    """
    x = f"(W-w)/2+({clip.transform.x:.4f})*(W/2)"
    y = f"(H-h)/2+({clip.transform.y:.4f})*(H/2)"
    return x, y


def _cadeia_de_video(clip: TimelineClip, entrada: int, saida: str) -> str:
    """O que acontece com um clipe antes de ele encostar na tela.

    A ordem importa. A velocidade vem **antes** de tudo, porque ela muda o
    relógio do clipe: um fade de meio segundo tem de durar meio segundo no vídeo
    final, não meio segundo da fonte.
    """
    # o trim é sobre a fonte, então conta a velocidade
    passos = [f"[{entrada}:v]trim=duration={clip.fonte_consumida_s:.3f}"]
    passos.append("setpts=PTS-STARTPTS")

    if clip.speed != 1.0:
        # dividir o PTS acelera: a 2x, cada quadro vale metade do tempo
        passos.append(f"setpts=PTS/{clip.speed:.4f}")

    if not clip.color.neutra:
        passos.append(
            f"eq=brightness={clip.color.brightness:.4f}"
            f":contrast={clip.color.contrast:.4f}"
            f":saturation={clip.color.saturation:.4f}"
        )

    if clip.transform.scale != 1.0:
        passos.append(
            f"scale=iw*{clip.transform.scale:.4f}:ih*{clip.transform.scale:.4f}"
        )

    if not clip.fade.neutro:
        # sobre o relógio já acelerado: o fade dura o que dura no vídeo
        if clip.fade.in_s > 0:
            passos.append(f"fade=t=in:st=0:d={clip.fade.in_s:.3f}")
        if clip.fade.out_s > 0:
            inicio = max(0.0, clip.duration_s - clip.fade.out_s)
            passos.append(
                f"fade=t=out:st={inicio:.3f}:d={clip.fade.out_s:.3f}"
            )

    if clip.transform.opacity < 1.0:
        # o alfa só existe em rgba; sem o `format` o colorchannelmixer não tem
        # canal onde mexer
        passos.append("format=rgba")
        passos.append(f"colorchannelmixer=aa={clip.transform.opacity:.4f}")

    # só agora o clipe é posto na hora dele no vídeo final
    passos.append(f"setpts=PTS+{clip.at_s:.3f}/TB")
    return ",".join(passos) + f"[{saida}]"


def _atempo(fator: float) -> list[str]:
    """A cadeia de `atempo` para um fator qualquer.

    O filtro só aceita de 0.5 a 100 por vez; fora disso encadeiam-se dois. Sem
    isto, câmera lenta abaixo de 0.5x sairia com o áudio intacto — e o descompasso
    entre imagem e som é pior do que não ter som.
    """
    passos: list[str] = []
    resto = fator
    while resto < 0.5:
        passos.append("atempo=0.5")
        resto /= 0.5
    while resto > 100.0:
        passos.append("atempo=100")
        resto /= 100.0
    if abs(resto - 1.0) > 1e-6:
        passos.append(f"atempo={resto:.4f}")
    return passos


def _cadeia_de_audio(clip: TimelineClip, entrada: int, saida: str) -> str | None:
    """O som do clipe, atrasado até a hora em que ele entra."""
    if clip.audio.mute or clip.audio.volume <= 0:
        return None
    ms = int(round(clip.at_s * 1000))
    passos = [
        f"[{entrada}:a]atrim=duration={clip.fonte_consumida_s:.3f}",
        "asetpts=PTS-STARTPTS",
    ]
    if clip.speed != 1.0:
        passos += _atempo(clip.speed)
    if clip.audio.fade_in_s > 0:
        passos.append(f"afade=t=in:st=0:d={clip.audio.fade_in_s:.3f}")
    if clip.audio.fade_out_s > 0:
        inicio = max(0.0, clip.duration_s - clip.audio.fade_out_s)
        passos.append(
            f"afade=t=out:st={inicio:.3f}:d={clip.audio.fade_out_s:.3f}"
        )
    if clip.audio.volume != 1.0:
        passos.append(f"volume={clip.audio.volume:.4f}")
    if ms > 0:
        passos.append(f"adelay={ms}|{ms}")
    return ",".join(passos) + f"[{saida}]"


def _entrada_do_clipe(
    clip: TimelineClip,
    *,
    source: Path,
    source_duration_s: float,
    midias: dict[str, MidiaNoDisco],
) -> tuple[Entrada | None, float, bool]:
    """A entrada do ffmpeg para este clipe, a duração útil e se ele tem som.

    Devolve `None` quando o clipe cai fora da fonte — o lugar dele fica sendo a
    tela de fundo, e os clipes seguintes não saem de onde foram postos.
    """
    if clip.source is ClipSource.RECORDING:
        # o que se pede à fonte é o que a velocidade consome, não o que o clipe
        # ocupa no vídeo
        pedido = clip.fonte_consumida_s
        if source_duration_s > 0:
            pedido = min(pedido, max(0.0, source_duration_s - clip.start_s))
        if pedido <= 0:
            return None, 0.0, False
        # aparado na fonte, ele encolhe no vídeo na mesma proporção
        return (
            Entrada(caminho=str(source), seek=clip.start_s, duracao=pedido),
            pedido / clip.speed,
            True,
        )

    if clip.source is ClipSource.MEDIA:
        item = midias.get(clip.media_id or "")
        if item is None:
            raise ValueError(
                f"a midia {clip.media_id!r} nao esta na biblioteca deste job"
            )
        if item.e_imagem:
            # uma imagem não corre no tempo: ela entra em laço e dura o que o
            # clipe pedir, sem `-ss` (não há onde buscar num quadro só) e sem
            # velocidade (não há o que acelerar num quadro parado)
            return (
                Entrada(
                    caminho=str(item.caminho),
                    duracao=clip.duration_s,
                    loop=True,
                ),
                clip.duration_s,
                False,
            )
        return (
            Entrada(
                caminho=str(item.caminho),
                seek=clip.start_s,
                duracao=clip.fonte_consumida_s,
            ),
            clip.duration_s,
            True,
        )

    # cor e texto chegam nas fases que as desenham; ignorar em silêncio seria
    # pior do que não aceitar
    raise ValueError(f"fonte '{clip.source}' ainda nao e montavel")


def compor(
    timeline: Timeline,
    *,
    source: Path,
    width: int,
    height: int,
    fps: float,
    music: Path | None = None,
    music_start_s: float = 0.0,
    source_duration_s: float = 0.0,
    midias: dict[str, MidiaNoDisco] | None = None,
) -> Composicao:
    """O grafo que monta esta linha do tempo.

    As camadas entram de baixo para cima, e dentro de cada uma os clipes entram
    na ordem do tempo. Camada escondida não entra; camada muda entra sem som.

    `midias` mapeia o id de cada item da biblioteca para o arquivo dele em
    disco. Um clipe de mídia é mais uma entrada no grafo, e daí em diante passa
    pelas mesmas transformações de um trecho da gravação — é o ponto de ter um
    formato só de clipe.

    Um clipe que passa do fim da gravação é **aparado**, como na V1 — o que
    sobrar do lugar dele fica sendo a tela de fundo, e os clipes seguintes não
    saem do lugar onde foram postos.
    """
    fps = fps if fps > 0 else 30.0
    duracao = timeline.duration_s
    c = Composicao(duracao_s=duracao)

    # a tela de fundo: é ela que aparece em todo instante que ninguém cobriu
    c.entradas.append(
        Entrada(
            caminho=f"color=c=black:s={int(width)}x{int(height)}:r={fps:.3f}",
            duracao=duracao,
            lavfi=True,
        )
    )
    c.filtros.append(f"[0:v]setsar=1[fundo]")

    anterior = "fundo"
    audios: list[str] = []
    n = 0

    for camada in timeline.layers:
        if camada.hidden:
            continue
        for clip in camada.clips:
            entrada, duracao_util, tem_som = _entrada_do_clipe(
                clip,
                source=source,
                source_duration_s=source_duration_s,
                midias=midias or {},
            )
            if entrada is None:
                continue  # cai fora da fonte; o lugar dele fica de fundo

            n += 1
            c.entradas.append(entrada)
            aparado = clip.model_copy(update={"duration_s": duracao_util})

            c.filtros.append(_cadeia_de_video(aparado, n, f"v{n}"))
            x, y = _posicao(aparado)
            saida = f"t{n}"
            c.filtros.append(
                f"[{anterior}][v{n}]overlay=x={x}:y={y}:"
                f"enable='between(t,{aparado.at_s:.3f},{aparado.until_s:.3f})':"
                f"eof_action=pass[{saida}]"
            )
            anterior = saida

            if not camada.muted and tem_som:
                cadeia = _cadeia_de_audio(aparado, n, f"a{n}")
                if cadeia is not None:
                    c.filtros.append(cadeia)
                    audios.append(f"a{n}")

    if n == 0:
        raise ValueError("nenhum clipe cai dentro da gravacao")

    c.filtros.append(f"[{anterior}]trim=duration={duracao:.3f},setpts=PTS-STARTPTS[vout]")
    c.mapa_video = "[vout]"

    # Com trilha e `game_volume` em 0, ela substitui o áudio -- o que a V1 fazia.
    # Acima disso os dois se misturam, e é o que deixa o tiro aparecer por baixo
    # da música.
    if music is not None:
        c.entradas.append(Entrada(caminho=str(music), seek=music_start_s))
        indice = len(c.entradas) - 1
        volume = (
            f",volume={timeline.music_volume:.4f}"
            if timeline.music_volume != 1.0
            else ""
        )
        c.filtros.append(
            f"[{indice}:a]atrim=duration={duracao:.3f},asetpts=PTS-STARTPTS"
            f"{volume}[trilha]"
        )
        if timeline.game_volume > 0 and audios:
            jogo = "".join(f"[{a}]" for a in audios)
            c.filtros.append(
                f"{jogo}amix=inputs={len(audios)}:dropout_transition=0:"
                f"normalize=0,volume={timeline.game_volume:.4f}[jogo]"
            )
            c.filtros.append(
                "[trilha][jogo]amix=inputs=2:dropout_transition=0:"
                "normalize=0[aout]"
            )
        else:
            c.filtros.append("[trilha]anull[aout]")
        c.mapa_audio = "[aout]"
    elif len(audios) == 1:
        # misturar uma faixa só é trabalho à toa, e o `amix` ainda mexeria no
        # volume dela sem necessidade
        c.filtros.append(f"[{audios[0]}]anull[aout]")
        c.mapa_audio = "[aout]"
    elif audios:
        entrada = "".join(f"[{a}]" for a in audios)
        c.filtros.append(
            f"{entrada}amix=inputs={len(audios)}:dropout_transition=0:"
            f"normalize=0[aout]"
        )
        c.mapa_audio = "[aout]"

    return c
