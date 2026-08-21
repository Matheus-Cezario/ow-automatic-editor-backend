"""Montagem dos videos finais.

Recebe os momentos ja escolhidos (nao ve pixel nenhum ate aqui) e volta ao
video original -- em qualidade cheia -- para cortar apenas os trechos que
interessam.

Cada video pedido e independente: tem as suas opcoes, a sua musica e a sua
grade de batidas. Um trecho aproveitado num video continua disponivel para os
outros -- nada e "gasto".
"""

from __future__ import annotations

import logging
import random
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from owcore import ffmpeg, timeline as tl
from owcore.compose import MidiaNoDisco, compor
from owcore.models import BeatGrid, ClipOptions, HighlightKind, Timeline
from owcore.rules import Highlight, fit_to_window, montage_segments

log = logging.getLogger(__name__)

MONTAGE_KINDS = {
    HighlightKind.BEAT_MONTAGE,
    HighlightKind.ULT_MONTAGE,
    HighlightKind.SLEEP_MONTAGE,
    HighlightKind.STUN_MONTAGE,
}


@dataclass(slots=True)
class RenderItem:
    """Um video pedido: o que cortar, como cortar e com que musica.

    A musica e as opcoes vem por item, e nao por job: e assim que o usuario
    poe uma trilha diferente em cada video do mesmo pedido.
    """

    highlight: Highlight
    #: proposta que originou o pedido, para amarrar o clipe de volta a ela
    proposal_id: str = ""
    options: ClipOptions = field(default_factory=ClipOptions)
    music: Path | None = None
    beats: BeatGrid | None = None
    music_name: str | None = None


@dataclass(slots=True)
class TimelineItem:
    """Um video que o **usuario** montou, bloco a bloco.

    Nao tem proposta nem regra por tras: ele ja diz que trecho entra, onde na
    musica e por quanto tempo. Ao editor sobra cortar e juntar exatamente
    aquilo -- inclusive o preto dos espacos que o usuario deixou vazios.
    """

    timeline: Timeline
    title: str = "Montagem"
    music: Path | None = None
    music_name: str | None = None
    #: os itens da biblioteca que esta montagem usa, já em disco
    midias: dict = field(default_factory=dict)

    #: onde guardar e procurar a imagem já montada, por assinatura visual.
    #: `None` desliga o reaproveitamento.
    cache_dir: Path | None = None

    def cache_de(self, assinatura: str) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{assinatura}.mp4"

    def guardar_cache(self, assinatura: str, video: Path) -> None:
        alvo = self.cache_de(assinatura)
        if alvo is None or alvo == video:
            return
        alvo.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(video, alvo)


@dataclass(slots=True)
class RenderedClip:
    highlight: Highlight
    #: None quando os cortes saíram mas a montagem final falhou
    video: Path | None
    thumb: Path | None
    duration_s: float
    meta: dict
    proposal_id: str = ""
    #: zip com os cortes individuais que formaram a montagem, para quem quiser
    #: reeditar por conta própria
    segments_zip: Path | None = None


def render_all(
    source: Path,
    items: list["RenderItem | TimelineItem"],
    duration_s: float,
    out_dir: Path,
    *,
    on_progress=None,
    seed: str = "",
) -> list[RenderedClip]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[RenderedClip] = []
    # semeado pelo id do pedido: a ordem sorteada varia entre pedidos mas e
    # sempre a mesma se o mesmo pedido for reprocessado
    rng = random.Random(seed)

    for i, item in enumerate(items):
        try:
            if isinstance(item, TimelineItem):
                clip = _render_timeline(source, item, out_dir, i)
            elif item.highlight.kind in MONTAGE_KINDS:
                clip = _render_montage(source, item, duration_s, out_dir, i, rng)
            else:
                clip = _render_single(source, item, out_dir, i)
        except ffmpeg.FFmpegError:
            # um clipe problematico nao pode custar o resto da entrega
            titulo = (
                item.title if isinstance(item, TimelineItem)
                else item.highlight.title
            )
            log.exception("falha ao renderizar '%s'; sigo com os demais", titulo)
            continue
        rendered.append(clip)
        if on_progress:
            on_progress((i + 1) / max(1, len(items)))

    return rendered


def _render_single(
    source: Path, item: RenderItem, out_dir: Path, index: int
) -> RenderedClip:
    """Trecho corrido da partida. Sai sempre com o audio original -- e o barulho
    da jogada que faz a cena."""
    h = item.highlight
    dest = out_dir / f"{index:02d}_{h.kind}.mp4"
    ffmpeg.cut(source, h.start, h.end, dest, fade=0.4)
    thumb = _thumb(dest, out_dir, index)
    return RenderedClip(
        highlight=h,
        proposal_id=item.proposal_id,
        video=dest,
        thumb=thumb,
        duration_s=h.duration,
        meta={"segments": 1, "original_audio": True, **h.meta},
    )


def _render_montage(
    source: Path,
    item: RenderItem,
    duration_s: float,
    out_dir: Path,
    index: int,
    rng: random.Random,
) -> RenderedClip:
    """Um micro-clipe por momento, cada um com duracao igual a N intervalos
    entre batidas -- assim, concatenados, as trocas de cena caem na percussao.

    Sem musica escolhida a montagem continua saindo: os cortes mantem o **audio
    original** da partida e a duracao de cada um cai num tamanho fixo razoavel.
    """
    h, opts, music, beats = item.highlight, item.options, item.music, item.beats
    segments = montage_segments(
        h.beats_at, beats, opts.montage_clip_beats, duration_s
    )
    if not segments:
        raise ffmpeg.FFmpegError("montagem sem segmentos")

    music_start, target = _music_window(opts, music, beats)
    # repetir trechos so faz sentido quando o usuario delimitou a musica; sem um
    # fim escolhido, encher a musica inteira daria uma montagem de minutos
    loop = opts.montage_loop and opts.music_end_s is not None
    segments = fit_to_window(segments, target, loop=loop, rng=rng)
    if not segments:
        raise ffmpeg.FFmpegError(
            "a janela de musica escolhida e curta demais para um trecho sequer"
        )

    # com trilha por cima o audio da partida so atrapalha; sem trilha ele e
    # tudo o que o video tem, entao fica
    mudo = music is not None

    parts_dir = out_dir / f"{index:02d}_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    parts: list[Path] = []
    cortados: list[tuple[float, float]] = []
    for j, (start, end) in enumerate(segments):
        part = parts_dir / f"{j:03d}.mp4"
        try:
            # sem fade: o corte seco na batida e o efeito desejado
            ffmpeg.cut(source, start, end, part, mute=mudo)
        except ffmpeg.FFmpegError:
            log.warning("corte em %.1fs falhou; sigo com os demais", start)
            continue
        parts.append(part)
        cortados.append((start, end))

    if not parts:
        raise ffmpeg.FFmpegError("nenhum corte pode ser feito")

    # O zip sai ANTES da montagem: se a junção ou a trilha falharem, o usuário
    # ainda leva os cortes. Material feito não se joga fora por causa da etapa
    # seguinte.
    zip_path = _zip_segments(parts, cortados, out_dir / f"{index:02d}_cortes.zip")

    dest: Path | None = out_dir / f"{index:02d}_{h.kind}.mp4"
    erro_montagem: str | None = None
    try:
        joined = out_dir / f"{index:02d}_{h.kind}_raw.mp4"
        ffmpeg.concat(parts, joined, mute=mudo)
        if music is not None:
            ffmpeg.add_music(joined, music, dest, music_start=music_start)
            joined.unlink(missing_ok=True)
        else:
            joined.replace(dest)
    except ffmpeg.FFmpegError as exc:
        log.exception("montagem de '%s' falhou; entrego so os cortes", h.title)
        erro_montagem = str(exc)[:500]
        dest = None

    for p in parts:
        p.unlink(missing_ok=True)
    parts_dir.rmdir()

    segments = cortados
    thumb = _thumb(dest, out_dir, index) if dest is not None else None
    total = sum(e - s for s, e in segments)
    return RenderedClip(
        highlight=h,
        proposal_id=item.proposal_id,
        video=dest,
        thumb=thumb,
        duration_s=total,
        segments_zip=zip_path,
        meta={
            "segments": len(segments),
            "bpm": beats.bpm if beats else None,
            "beat_synced": bool(beats and beats.beats),
            "music_name": item.music_name,
            "original_audio": music is None,
            "music_start_s": round(music_start, 2) if music else None,
            "music_window_s": round(target, 2) if target else None,
            "looped": loop,
            **({"render_error": erro_montagem} if erro_montagem else {}),
            **h.meta,
        },
    )


def _render_timeline(
    source: Path, item: TimelineItem, out_dir: Path, index: int
) -> RenderedClip:
    """Monta exatamente o que o usuario desenhou na linha do tempo.

    A diferenca para `_render_montage` e que aqui nada e calculado: a duracao de
    cada corte e o ponto da musica onde ele entra vieram prontos. O unico
    trabalho de decisao e o dos buracos -- espaco que o usuario deixou vazio
    vira preto com a musica tocando, e nao um encurtamento do video, senao todo
    bloco depois dele sairia do lugar onde foi posto.
    """
    spec = item.timeline
    media = ffmpeg.probe(source)

    # Camada, transformacao, som ajustado ou midia importada nao cabem em
    # corte-e-emenda: eles exigem dois pedacos existindo ao mesmo tempo. Ai a
    # montagem vira um grafo de filtros -- mais poderoso e menos tolerante,
    # porque um erro nele derruba o render inteiro em vez de custar um corte.
    if not spec.de_uma_camada_so:
        return _render_composicao(source, item, media, out_dir, index)

    pecas = tl.plan(spec.cuts, source_duration_s=media.duration_s)
    if not any(p.is_cut for p in pecas):
        raise ffmpeg.FFmpegError(
            "nenhum dos cortes cai dentro da gravacao"
        )

    # com trilha por cima o audio da partida so atrapalha; sem trilha ele e
    # tudo o que o video tem
    mudo = item.music is not None
    taxa = 0 if mudo else media.audio_rate

    parts_dir = out_dir / f"{index:02d}_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    #: tudo o que entra no video, na ordem -- cortes e pretos
    parts: list[Path] = []
    #: so os cortes de verdade, que sao o que vai para o zip
    recortes: list[tuple[Path, tuple[float, float]]] = []

    for j, peca in enumerate(pecas):
        part = parts_dir / f"{j:03d}.mp4"
        if peca.is_cut:
            try:
                ffmpeg.cut(source, peca.start_s, peca.end_s, part, mute=mudo)
                parts.append(part)
                recortes.append((part, (peca.start_s, peca.end_s)))
                continue
            except ffmpeg.FFmpegError:
                # o corte falhou, mas o lugar dele continua reservado: virar
                # preto mantem todos os blocos seguintes no ponto da musica
                # onde o usuario os pos
                log.warning(
                    "corte em %.1fs falhou; o lugar dele fica preto", peca.start_s
                )
        try:
            ffmpeg.black_clip(
                part, peca.duration_s,
                width=media.width, height=media.height, fps=media.fps,
                audio_rate=taxa,
            )
        except ffmpeg.FFmpegError:
            log.warning("nao consegui gerar o preto de %.2fs", peca.duration_s)
            continue
        parts.append(part)

    if not recortes:
        raise ffmpeg.FFmpegError("nenhum corte pode ser feito")

    # o zip sai ANTES da montagem, como nas montagens automaticas: se juntar ou
    # por a trilha falhar, o material cortado nao se perde. So os cortes entram
    # -- o preto dos buracos nao e material de ninguem
    zip_path = _zip_segments(
        [p for p, _ in recortes],
        [trecho for _, trecho in recortes],
        out_dir / f"{index:02d}_cortes.zip",
    )

    dest: Path | None = out_dir / f"{index:02d}_custom.mp4"
    erro_montagem: str | None = None
    try:
        joined = out_dir / f"{index:02d}_custom_raw.mp4"
        ffmpeg.concat(parts, joined, mute=mudo)
        if item.music is not None:
            ffmpeg.add_music(
                joined, item.music, dest, music_start=spec.music_start_s
            )
            joined.unlink(missing_ok=True)
        else:
            joined.replace(dest)
    except ffmpeg.FFmpegError as exc:
        log.exception("montagem de '%s' falhou; entrego so os cortes", item.title)
        erro_montagem = str(exc)[:500]
        dest = None

    for p in parts:
        p.unlink(missing_ok=True)
    parts_dir.rmdir()

    # a miniatura sai do primeiro corte, e nao do primeiro segundo: quem
    # comecou a montagem com um espaco vazio teria uma capa preta
    ate_o_primeiro = 0.0
    for peca in pecas:
        if peca.is_cut:
            break
        ate_o_primeiro += peca.duration_s
    thumb = (
        _thumb(dest, out_dir, index, at=ate_o_primeiro + 0.2)
        if dest is not None else None
    )
    highlight = Highlight(
        kind=HighlightKind.CUSTOM,
        start=min(c.start_s for c in spec.cuts),
        end=max(c.end_s for c in spec.cuts),
        title=item.title or "Montagem",
        beats_at=[c.source_t for c in spec.cuts],
        meta={},
    )
    return RenderedClip(
        highlight=highlight,
        video=dest,
        thumb=thumb,
        duration_s=tl.total_duration_s(pecas),
        segments_zip=zip_path,
        meta={
            "segments": len(recortes),
            "blackfill_s": round(
                sum(p.duration_s for p in pecas if p.black), 2
            ),
            "hand_made": True,
            "music_name": item.music_name,
            "original_audio": item.music is None,
            "music_start_s": round(spec.music_start_s, 2) if item.music else None,
            **({"render_error": erro_montagem} if erro_montagem else {}),
        },
    )


def _render_composicao(
    source: Path,
    item: TimelineItem,
    media: ffmpeg.MediaInfo,
    out_dir: Path,
    index: int,
) -> RenderedClip:
    """Monta em camadas, num grafo de filtros.

    Uma tela de fundo cobre o video inteiro e cada clipe e sobreposto nela na
    hora certa. O buraco entre clipes deixa de ser caso especial: ele e
    simplesmente onde ninguem cobriu o fundo.

    Diferente do caminho de corte-e-emenda, aqui **nao ha zip de cortes**: os
    pedacos nunca chegam a existir como arquivo, e recorta-los so para o zip
    seria pagar a montagem duas vezes.
    """
    spec = item.timeline

    # Quando a trilha manda sozinha, a imagem não depende de nada do som — e aí
    # ela pode ser reaproveitada. Trocar a música e reexportar deixa de recortar
    # o vídeo inteiro de novo: monta-se o áudio por cima do que já existe.
    #
    # Com o som do jogo na mistura isso não vale: o áudio precisa dos mesmos
    # cortes que a imagem, e não há o que economizar.
    so_a_trilha = item.music is not None and spec.game_volume <= 0

    comp = compor(
        spec,
        source=source,
        width=media.width,
        height=media.height,
        fps=media.fps,
        music=None if so_a_trilha else item.music,
        music_start_s=spec.music_start_s,
        source_duration_s=media.duration_s,
        midias=item.midias,
        so_video=so_a_trilha,
    )

    dest: Path | None = out_dir / f"{index:02d}_custom.mp4"
    erro: str | None = None
    reaproveitado = False
    try:
        if so_a_trilha:
            imagem = item.cache_de(spec.assinatura_visual())
            if imagem is not None and imagem.exists():
                reaproveitado = True
            else:
                imagem = out_dir / f"{index:02d}_imagem.mp4"
                ffmpeg.compose(comp, imagem)
                item.guardar_cache(spec.assinatura_visual(), imagem)
            ffmpeg.add_music(
                imagem, item.music, dest, music_start=spec.music_start_s
            )
        else:
            ffmpeg.compose(comp, dest)
    except ffmpeg.FFmpegError as exc:
        log.exception("composicao de '%s' falhou", item.title)
        erro = str(exc)[:500]
        dest = None

    clips = spec.clips
    camadas = [l for l in spec.layers if not l.hidden]
    thumb = _thumb(dest, out_dir, index) if dest is not None else None
    return RenderedClip(
        highlight=Highlight(
            kind=HighlightKind.CUSTOM,
            start=min((c.start_s for c in clips), default=0.0),
            end=max((c.end_s for c in clips), default=0.0),
            title=item.title or "Montagem",
            beats_at=[c.source_t for c in clips],
            meta={},
        ),
        video=dest,
        thumb=thumb,
        duration_s=spec.duration_s,
        meta={
            "segments": len(clips),
            "layers": len(camadas),
            "composed": True,
            "hand_made": True,
            "reused": reaproveitado,
            "media": len({c.media_id for c in clips if c.media_id}),
            "music_name": item.music_name,
            "original_audio": item.music is None,
            "music_start_s": (
                round(spec.music_start_s, 2) if item.music else None
            ),
            **({"render_error": erro} if erro else {}),
        },
    )


def _zip_segments(
    parts: list[Path], segments: list[tuple[float, float]], dest: Path
) -> Path | None:
    """Empacota os cortes individuais para download.

    Guarda cada corte **uma vez**, em ordem cronológica: quando a montagem
    repete trechos, o mesmo material apareceria várias vezes no zip sem
    acrescentar nada a quem vai reeditar. O nome traz o instante de onde o corte
    saiu na gravação original.

    Os arquivos já estão em H.264 e não comprimem mais, então o zip é apenas um
    empacotamento (ZIP_STORED) -- recomprimir só gastaria CPU.
    """
    # Chaveado só pelo início, e ficando com a versão mais longa: o último
    # trecho da montagem costuma ser uma aparação do mesmo corte, e entregar as
    # duas versões seria entregar o mesmo material duas vezes, uma pela metade.
    unicos: dict[float, tuple[float, Path]] = {}
    for part, (start, end) in zip(parts, segments):
        chave = round(start, 2)
        atual = unicos.get(chave)
        if atual is None or (end - start) > atual[0]:
            unicos[chave] = (end - start, part)
    if not unicos:
        return None

    try:
        with zipfile.ZipFile(dest, "w", zipfile.ZIP_STORED) as zf:
            for i, (start, (_dur, part)) in enumerate(sorted(unicos.items()), start=1):
                minutos = int(start) // 60
                segundos = start - minutos * 60
                zf.write(part, f"{i:02d}_{minutos:02d}m{segundos:04.1f}s.mp4")
    except OSError:
        log.warning("nao consegui montar o zip dos cortes em %s", dest)
        return None
    return dest


def _music_window(
    options: ClipOptions, music: Path | None, beats: BeatGrid | None
) -> tuple[float, float | None]:
    """Onde a musica comeca a tocar e quanto a montagem deve durar.

    O inicio e empurrado para a primeira batida a partir do ponto escolhido pelo
    usuario: assim o primeiro corte da montagem cai no tempo, em vez de entrar
    no meio de um compasso. O fim e limitado pelo tamanho real do arquivo -- se
    alguem pedir um trecho que passa do fim da musica, entrega-se o que existe
    em vez de um video com silencio no final.
    """
    start = max(0.0, options.music_start_s)
    if beats and beats.beats:
        depois = [b for b in beats.beats if b >= start - 1e-6]
        if depois:
            start = depois[0]

    if music is None:
        return start, None

    try:
        music_duration = ffmpeg.probe(music).duration_s
    except ffmpeg.FFmpegError:
        music_duration = 0.0

    end = options.music_end_s
    if music_duration > 0:
        end = music_duration if end is None else min(end, music_duration)
    if end is None:
        return start, None
    return start, max(0.0, end - start)


def _thumb(video: Path, out_dir: Path, index: int, at: float = 0.2) -> Path | None:
    try:
        dest = out_dir / f"{index:02d}.jpg"
        return ffmpeg.thumbnail(video, dest, at=at)
    except ffmpeg.FFmpegError:
        log.warning("sem miniatura para %s", video.name)
        return None
