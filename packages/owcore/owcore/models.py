"""Modelos de domínio (pydantic) e tabelas (SQLAlchemy)."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, model_validator
from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def new_id() -> str:
    return uuid.uuid4().hex[:16]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ─────────────────────────────── enums ──────────────────────────────────────


class JobStatus(StrEnum):
    """Ciclo de vida da *análise*. Ela roda uma vez, sozinha, e termina em
    `READY`: a partir daí o job fica esperando o usuário escolher o que gerar."""

    PENDING = "pending"
    PREPROCESSING = "preprocessing"
    DETECTING = "detecting"
    READY = "ready"
    FAILED = "failed"


class RenderStatus(StrEnum):
    """Ciclo de vida de *um pedido de geração*. Um job tem quantos quiser."""

    PENDING = "pending"
    RENDERING = "rendering"
    DONE = "done"
    FAILED = "failed"


class TrackStatus(StrEnum):
    """Ciclo de vida de *um item de midia* enviado para o job.

    Uma musica sobe antes de qualquer video existir: e preciso ouvi-la no app e
    ver as batidas para poder posicionar os cortes em cima dela. Com a
    biblioteca de midia, o mesmo vale para um clipe ou uma imagem importada --
    o app precisa da miniatura e das dimensoes antes de deixar montar com eles.
    """

    PENDING = "pending"
    READY = "ready"
    FAILED = "failed"


class MediaKind(StrEnum):
    """O que o arquivo importado e.

    Decide o que a analise faz com ele: audio ganha batidas e forma de onda,
    video ganha miniatura e proxy, imagem ganha miniatura.
    """

    AUDIO = "audio"
    VIDEO = "video"
    IMAGE = "image"


class EventKind(StrEnum):
    KILL = "kill"
    DEATH = "death"
    LOW_HP = "low_hp"
    ESCAPE = "escape"
    ULT_USED = "ult_used"
    ULT_NEGATED = "ult_negated"
    #: dardo tranquilizante da Ana acertando alguem
    SLEEP = "sleep"
    #: pedrada (Accretion) do Sigma atordoando alguem
    STUN = "stun"


class HighlightKind(StrEnum):
    MULTIKILL = "multikill"
    SOLO_WIPE = "solo_wipe"
    ESCAPE = "escape"
    BEAT_MONTAGE = "beat_montage"
    ULT_MONTAGE = "ult_montage"
    SLEEP_MONTAGE = "sleep_montage"
    STUN_MONTAGE = "stun_montage"
    #: montado pelo usuario na linha do tempo -- nao sai de regra nenhuma
    CUSTOM = "custom"


#: Detectores que a análise espera antes de montar a lista de propostas.
#: Um por *região da tela*, não por habilidade: `banner` lê a faixa do rodapé e
#: distingue as habilidades pelo ícone.
DETECTORS = ("kills", "survival", "ults", "banner")


# ────────────────────────── modelos de mensagem ─────────────────────────────


class RoiSpec(BaseModel):
    """Recorte normalizado (0..1) da tela, mais o downscale aplicado."""

    name: str
    x: float
    y: float
    w: float
    h: float
    fps: float = 10.0
    width_px: int = 320
    #: se True, o recorte é a tela inteira reduzida (usado p/ detectar morte)
    fullscreen: bool = False


#: Nome da saída que vira o proxy do editor. Não é ROI de detector nenhum: é a
#: tela inteira reduzida, pendurada na mesma decodificação porque o decode do
#: vídeo pesado já está pago — pedir uma segunda passagem só para isto seria
#: gastar de novo o que o sistema mais economiza.
PROXY_ROI = "proxy"


def proxy_roi() -> RoiSpec:
    """A tela inteira, pequena e em FPS baixo: o bastante para editar.

    O corte final continua saindo da gravação original, em qualidade cheia —
    isto aqui só existe para o monitor poder buscar um instante sem arrastar
    meio giga por HTTP a cada arrasto.
    """
    return RoiSpec(
        name=PROXY_ROI,
        x=0.0,
        y=0.0,
        w=1.0,
        h=1.0,
        fps=24.0,
        width_px=640,
        fullscreen=True,
    )


class Artifact(BaseModel):
    """Referência a um blob no storage."""

    key: str
    kind: str
    meta: dict[str, Any] = Field(default_factory=dict)


class DetectionEvent(BaseModel):
    kind: EventKind
    t: float  # segundos desde o início do vídeo
    confidence: float = 1.0
    meta: dict[str, Any] = Field(default_factory=dict)


class BeatGrid(BaseModel):
    bpm: float
    beats: list[float] = Field(default_factory=list)


class JobParams(BaseModel):
    """Parâmetros da **análise**: definem o que conta como momento importante.

    Valem para o job inteiro e são aplicados quando o sistema monta a lista de
    vídeos possíveis. O que é escolhido depois, na hora de gerar cada vídeo,
    está em `ClipOptions`.
    """

    multikill_min: int = 3
    multikill_window_s: float = 10.0
    solo_wipe_min: int = 4
    solo_wipe_window_s: float = 15.0
    escape_min_events: int = 2
    escape_window_s: float = 20.0
    pre_roll_s: float = 4.0
    post_roll_s: float = 3.0
    #: uma ultimate inimiga seguida de eliminacao dentro desta janela conta
    #: como ultimate anulada
    ult_negate_window_s: float = 6.0

    make_beat_montage: bool = True
    profile: str | None = None


class ClipOptions(BaseModel):
    """Escolhas de **um** vídeo, na hora de gerar.

    Cada vídeo tem as suas: é assim que músicas diferentes convivem no mesmo
    job. Sem música escolhida, o vídeo sai com o **áudio original** da partida.
    """

    #: quantas batidas dura cada corte da montagem
    montage_clip_beats: int = 2
    #: de onde a musica comeca a tocar
    music_start_s: float = 0.0
    #: onde ela termina; None = ate o fim do arquivo
    music_end_s: float | None = None
    #: repetir trechos ate o video ter exatamente a duracao da janela de musica.
    #: Com False o video para quando acabam os momentos, sem passar dela.
    montage_loop: bool = False

    @model_validator(mode="after")
    def _janela_de_musica_coerente(self) -> "ClipOptions":
        if self.music_start_s < 0:
            raise ValueError("music_start_s nao pode ser negativo")
        if self.music_end_s is not None and self.music_end_s <= self.music_start_s:
            raise ValueError("music_end_s tem de ser maior que music_start_s")
        return self

    @property
    def music_window_s(self) -> float | None:
        """Duracao alvo do video, ou None se o usuario nao delimitou."""
        if self.music_end_s is None:
            return None
        return self.music_end_s - self.music_start_s


class Selection(BaseModel):
    """Uma proposta escolhida pelo usuário, com as opções dela."""

    proposal_id: str
    options: ClipOptions = Field(default_factory=ClipOptions)
    #: preenchido pelo gateway ao receber o upload da musica desta escolha
    music_key: str | None = None
    music_name: str | None = None


#: nada abaixo disto vale um corte: e menos de um quadro em qualquer gravacao
MIN_CUT_S = 0.05

#: Eventos que valem uma miniatura na barra lateral do editor: os que viram
#: bloco. Vida baixa e interrupcao sao o contexto da jogada, nao a jogada.
THUMB_KINDS = (
    EventKind.KILL,
    EventKind.SLEEP,
    EventKind.STUN,
    EventKind.ULT_NEGATED,
    EventKind.ESCAPE,
)


def frame_key(job_id: str, t: float) -> str:
    """Onde mora a miniatura do instante `t` daquela partida.

    Deriva do instante em vez de virar coluna no banco: quem escreve e quem le
    chegam na mesma chave sozinhos, e uma miniatura a mais nao e uma migracao a
    mais. O arredondamento em centesimos e o mesmo que a API usa para falar de
    tempo, entao o app pede exatamente o que o worker gravou.
    """
    return f"{job_id}/frames/{t:.2f}.jpg"


class TimelineCut(BaseModel):
    """Um bloco na linha do tempo: um pedaco da gravacao posto num ponto do video.

    E a unidade da montagem manual. O usuario diz *qual* trecho (`start_s` +
    `duration_s`, na gravacao) e *onde* ele entra (`at_s`, no video que vai
    sair). As duas coisas sao independentes: o mesmo momento pode aparecer
    duas vezes, em pontos diferentes da musica, com duracoes diferentes.
    """

    #: instante do momento que originou o bloco. Nao afeta o corte -- serve
    #: para o app saber de que evento este bloco veio e para nomear o arquivo
    source_t: float = 0.0
    #: onde o corte comeca na gravacao
    start_s: float
    #: quanto ele dura
    duration_s: float
    #: onde ele entra no video final; 0 e o primeiro quadro
    at_s: float
    #: kill/sleep/stun -- so rotula
    kind: str = ""

    @model_validator(mode="after")
    def _coerente(self) -> "TimelineCut":
        if self.start_s < 0:
            raise ValueError("start_s nao pode ser negativo")
        if self.at_s < 0:
            raise ValueError("at_s nao pode ser negativo")
        if self.duration_s < MIN_CUT_S:
            raise ValueError(f"um corte tem de durar ao menos {MIN_CUT_S}s")
        return self

    @property
    def end_s(self) -> float:
        """Onde o corte termina *na gravacao*."""
        return self.start_s + self.duration_s

    @property
    def until_s(self) -> float:
        """Onde o corte termina *no video*."""
        return self.at_s + self.duration_s


class ClipSource(StrEnum):
    """De onde sai a imagem de um clipe.

    Nasce discriminado para a biblioteca de midia entrar sem mexer no modelo:
    hoje o editor so produz `RECORDING` e `COLOR`, e `MEDIA` e `TEXT` chegam nas
    fases seguintes.
    """

    #: um trecho da gravacao da partida
    RECORDING = "recording"
    #: cor solida. O preto dos buracos deixa de ser um caso especial do render e
    #: passa a ser um clipe como qualquer outro
    COLOR = "color"
    #: arquivo importado pelo usuario (Fase 4)
    MEDIA = "media"
    #: texto (Fase 6)
    TEXT = "text"


class Transform(BaseModel):
    """Onde e de que tamanho o clipe aparece no quadro.

    `x` e `y` sao deslocamentos do centro, normalizados pela metade do quadro:
    -1 encosta na borda esquerda/superior, +1 na direita/inferior, 0 e o centro.
    Assim a mesma montagem vale em qualquer resolucao -- o que importa numa
    montagem e a proporcao, nao o pixel.
    """

    scale: float = 1.0
    x: float = 0.0
    y: float = 0.0
    opacity: float = 1.0

    @model_validator(mode="after")
    def _coerente(self) -> "Transform":
        if self.scale <= 0:
            raise ValueError("scale tem de ser maior que zero")
        if not 0.0 <= self.opacity <= 1.0:
            raise ValueError("opacity fica entre 0 e 1")
        return self

    @property
    def neutra(self) -> bool:
        """True quando o clipe entra do jeito que veio, sem nada por cima."""
        return (
            self.scale == 1.0
            and self.x == 0.0
            and self.y == 0.0
            and self.opacity == 1.0
        )


class ClipAudio(BaseModel):
    """O som do proprio clipe -- que nao e a trilha do video."""

    volume: float = 1.0
    mute: bool = False
    fade_in_s: float = 0.0
    fade_out_s: float = 0.0

    @model_validator(mode="after")
    def _coerente(self) -> "ClipAudio":
        if self.volume < 0:
            raise ValueError("volume nao pode ser negativo")
        if self.fade_in_s < 0 or self.fade_out_s < 0:
            raise ValueError("fade nao pode ser negativo")
        return self

    @property
    def neutro(self) -> bool:
        return (
            self.volume == 1.0
            and not self.mute
            and self.fade_in_s == 0.0
            and self.fade_out_s == 0.0
        )


class ClipColor(BaseModel):
    """Ajuste de cor do clipe.

    Os tres que resolvem quase tudo numa montagem de gameplay: gravacao escura,
    gravacao lavada, gravacao sem cor. O resto (curva, temperatura) chega quando
    fizer falta.
    """

    brightness: float = 0.0
    contrast: float = 1.0
    saturation: float = 1.0

    @model_validator(mode="after")
    def _coerente(self) -> "ClipColor":
        if not -1.0 <= self.brightness <= 1.0:
            raise ValueError("brightness fica entre -1 e 1")
        if not 0.0 <= self.contrast <= 3.0:
            raise ValueError("contrast fica entre 0 e 3")
        if not 0.0 <= self.saturation <= 3.0:
            raise ValueError("saturation fica entre 0 e 3")
        return self

    @property
    def neutra(self) -> bool:
        return (
            self.brightness == 0.0
            and self.contrast == 1.0
            and self.saturation == 1.0
        )


class ClipFade(BaseModel):
    """Entrada e saida do clipe, em segundos.

    Sao transicoes de e para o **fundo** -- que numa montagem em camadas e
    preto. A transicao *entre dois clipes* (crossfade) e outra coisa e nao cabe
    no formato de sobreposicao: ela exige os dois existindo ao mesmo tempo com
    pesos que mudam.
    """

    in_s: float = 0.0
    out_s: float = 0.0

    @model_validator(mode="after")
    def _coerente(self) -> "ClipFade":
        if self.in_s < 0 or self.out_s < 0:
            raise ValueError("fade nao pode ser negativo")
        return self

    @property
    def neutro(self) -> bool:
        return self.in_s == 0.0 and self.out_s == 0.0


class TimelineClip(BaseModel):
    """Um pedaco do video final: o que aparece, onde e por quanto tempo.

    E o `TimelineCut` da V1 com lugar para o resto: de que fonte sai, como e
    posicionado no quadro e o que acontece com o som dele. Um corte da V1 e
    exatamente um `Clip` de `RECORDING` com transform e audio neutros, e e assim
    que a migracao o le.
    """

    source: ClipSource = ClipSource.RECORDING
    #: onde ele entra no video; 0 e o primeiro quadro
    at_s: float
    duration_s: float
    #: onde comeca na fonte (gravacao ou midia). Ignorado por cor e texto
    start_s: float = 0.0
    #: instante do momento que originou o clipe -- rotulo, nao afeta o corte
    source_t: float = 0.0
    #: kill/sleep/stun, para nomear e colorir
    kind: str = ""
    #: cor solida quando `source` e COLOR. Chama-se `fill` porque `color` ja e
    #: a correcao de cor -- sao a *cor do conteudo* e o *ajuste do conteudo*,
    #: duas coisas que se encontram no mesmo clipe
    fill: str = "black"
    #: id do item da biblioteca quando `source` e MEDIA (Fase 4)
    media_id: str | None = None
    transform: Transform = Field(default_factory=Transform)
    audio: ClipAudio = Field(default_factory=ClipAudio)
    color: ClipColor = Field(default_factory=ClipColor)
    fade: ClipFade = Field(default_factory=ClipFade)

    #: Quanto mais rapido o clipe corre. 2 = dobro, 0.5 = camera lenta.
    #:
    #: Muda quanto da fonte ele consome: um clipe de 2s a 2x come 4s de
    #: gravacao. Nao muda quanto ele ocupa no video -- isso e `duration_s`, e e
    #: o que o usuario arrasta.
    speed: float = 1.0

    @model_validator(mode="after")
    def _coerente(self) -> "TimelineClip":
        if self.start_s < 0:
            raise ValueError("start_s nao pode ser negativo")
        if self.at_s < 0:
            raise ValueError("at_s nao pode ser negativo")
        if self.duration_s < MIN_CUT_S:
            raise ValueError(f"um clipe tem de durar ao menos {MIN_CUT_S}s")
        if not 0.1 <= self.speed <= 10.0:
            raise ValueError("speed fica entre 0.1 e 10")
        if self.fade.in_s + self.fade.out_s > self.duration_s + 1e-6:
            raise ValueError("os fades somados passam da duracao do clipe")
        return self

    @property
    def fonte_consumida_s(self) -> float:
        """Quanto da fonte este clipe come.

        Nao e a duracao dele no video: a 2x, dois segundos de video comem quatro
        de gravacao. E `duration_s` que o usuario arrasta na regua; isto aqui e
        consequencia.
        """
        return self.duration_s * self.speed

    @property
    def end_s(self) -> float:
        """Onde termina *na fonte* -- ja contando a velocidade."""
        return self.start_s + self.fonte_consumida_s

    @property
    def until_s(self) -> float:
        """Onde termina *no video*."""
        return self.at_s + self.duration_s

    @property
    def simples(self) -> bool:
        """Um clipe que o caminho de corte-e-emenda da V1 da conta de fazer."""
        return (
            self.source is ClipSource.RECORDING
            and self.transform.neutra
            and self.audio.neutro
            and self.color.neutra
            and self.fade.neutro
            and self.speed == 1.0
        )

    def como_corte(self) -> TimelineCut:
        """A visao V1 deste clipe, para o caminho de corte-e-emenda."""
        return TimelineCut(
            source_t=self.source_t,
            start_s=self.start_s,
            duration_s=self.duration_s,
            at_s=self.at_s,
            kind=self.kind,
        )

    @classmethod
    def de_corte(cls, cut: TimelineCut) -> "TimelineClip":
        return cls(
            source=ClipSource.RECORDING,
            at_s=cut.at_s,
            duration_s=cut.duration_s,
            start_s=cut.start_s,
            source_t=cut.source_t,
            kind=cut.kind,
        )


class Layer(BaseModel):
    """Uma camada da linha do tempo.

    Chama-se camada, e nao faixa, porque `Track` neste sistema ja e a musica que
    o usuario enviou -- dois `Track` no mesmo modelo seriam uma armadilha.

    A ordem na lista e a ordem de empilhamento: a primeira e o fundo, a ultima
    fica por cima.
    """

    name: str = ""
    muted: bool = False
    hidden: bool = False
    #: travada nao muda nada no render -- e o app que recusa editar
    locked: bool = False
    clips: list[TimelineClip] = Field(default_factory=list)

    @model_validator(mode="after")
    def _sem_sobreposicao(self) -> "Layer":
        ordenados = sorted(self.clips, key=lambda c: c.at_s)
        for anterior, seguinte in zip(ordenados, ordenados[1:]):
            if seguinte.at_s < anterior.until_s - 1e-6:
                raise ValueError(
                    f"dois clipes se sobrepoem em {seguinte.at_s:.2f}s da camada"
                )
        self.clips = ordenados
        return self

    @property
    def duration_s(self) -> float:
        return max((c.until_s for c in self.clips), default=0.0)


class MontageDraft(BaseModel):
    """A montagem **em andamento**, do jeito que ficou na tela.

    E a `Timeline` sem a exigencia de estar pronta: aceita zero cortes, porque
    um rascunho existe desde antes de o primeiro bloco entrar. Cada bloco, esse
    sim, e validado -- guardar lixo agora seria devolver lixo depois.

    Existe porque recarregar a pagina custava a montagem inteira: meia hora de
    encaixe na batida sumia num F5.
    """

    title: str = ""
    track_id: str | None = None
    music_start_s: float = 0.0
    #: formato da V1, ainda aceito enquanto o app nao manda camadas
    cuts: list[TimelineCut] = Field(default_factory=list)
    layers: list[Layer] = Field(default_factory=list)

    #: Correções do usuário à grade de batidas. Não afetam o vídeo: o corte
    #: guarda instantes absolutos, e a grade é só o imã da tela. Vêm no rascunho
    #: para não se perder num F5 -- consertar a grade duas vezes irrita mais do
    #: que consertá-la uma.
    beat_offset_s: float = 0.0
    beat_multiplier: float = 1.0
    beat_bar: int = 1


class Timeline(BaseModel):
    """Um video montado a mao: as camadas e a musica por baixo delas.

    **Le o formato da V1.** Um pedido antigo (ou um rascunho salvo antes desta
    versao) chega com `cuts` em vez de `layers`, e e convertido aqui na leitura
    -- uma camada so, de clipes de gravacao. Nao ha migracao a rodar no banco:
    o formato velho continua sendo entrada valida, e sai do outro lado no novo.
    """

    title: str = ""
    #: musica de fundo; None deixa o audio original dos clipes
    track_id: str | None = None
    #: de que ponto da musica o video comeca a tocar
    music_start_s: float = 0.0
    layers: list[Layer] = Field(default_factory=list)

    #: Volume da trilha e do som do jogo, de 0 a 2.
    #:
    #: Com `game_volume` em 0 a musica **substitui** o audio, que e o que a V1
    #: fazia. Acima disso os dois se misturam -- e o que deixa o tiro aparecer
    #: por baixo da musica.
    music_volume: float = 1.0
    game_volume: float = 0.0

    @model_validator(mode="before")
    @classmethod
    def _aceita_o_formato_da_v1(cls, dados: Any) -> Any:
        if not isinstance(dados, dict):
            return dados
        if dados.get("layers") or "cuts" not in dados:
            return dados
        cortes = dados.pop("cuts") or []
        dados["layers"] = [{"clips": [
            TimelineClip.de_corte(
                c if isinstance(c, TimelineCut) else TimelineCut(**c)
            ).model_dump()
            for c in cortes
        ]}]
        return dados

    @model_validator(mode="after")
    def _tem_o_que_montar(self) -> "Timeline":
        if self.music_start_s < 0:
            raise ValueError("music_start_s nao pode ser negativo")
        for nome, v in (("music_volume", self.music_volume),
                        ("game_volume", self.game_volume)):
            if not 0.0 <= v <= 2.0:
                raise ValueError(f"{nome} fica entre 0 e 2")
        if not any(l.clips for l in self.layers):
            raise ValueError("uma linha do tempo vazia nao vira video")
        return self

    @property
    def duration_s(self) -> float:
        """Quanto o video vai durar -- inclusive os buracos entre os clipes."""
        return max((l.duration_s for l in self.layers), default=0.0)

    @property
    def clips(self) -> list[TimelineClip]:
        """Todos os clipes, de todas as camadas, de baixo para cima."""
        return [c for l in self.layers for c in l.clips]

    @property
    def de_uma_camada_so(self) -> bool:
        """Da para montar pelo caminho de corte-e-emenda da V1?

        Uma camada, nenhum clipe com transformacao, som ajustado ou fonte que
        nao seja a gravacao. E o caso da maioria das montagens, e nele o
        caminho antigo e mais resistente: um corte que falha custa so ele,
        enquanto um erro no grafo de filtros derruba o render inteiro.
        """
        visiveis = [l for l in self.layers if not l.hidden]
        if len(visiveis) != 1:
            return False
        return all(c.simples for c in visiveis[0].clips)

    @property
    def cuts(self) -> list[TimelineCut]:
        """A visao V1 desta linha do tempo. So faz sentido com uma camada."""
        return [c.como_corte() for c in self.clips]


# ───────────────────────── mensagens do barramento ──────────────────────────

STREAM_JOBS = "ow.jobs"
STREAM_ROI = "ow.roi"
STREAM_EDIT = "ow.edit"
#: um pedido de geração recém-criado, ainda sem as batidas das músicas
STREAM_RENDER = "ow.render"
#: pedido com as batidas já analisadas, pronto para o editor cortar
STREAM_RENDER_READY = "ow.render.ready"
#: arquivo recém-enviado, esperando quem o analise (batidas e onda para áudio;
#: miniatura, dimensões e proxy para vídeo e imagem)
STREAM_MEDIA = "ow.media"
#: análise terminada: alguém que extraia uma miniatura de cada momento
STREAM_THUMBS = "ow.thumbs"


class JobCreated(BaseModel):
    job_id: str


class RoiReady(BaseModel):
    job_id: str
    detector: str
    artifacts: list[Artifact]
    duration_s: float
    params: JobParams


class EventsDetected(BaseModel):
    job_id: str
    detector: str
    events: list[DetectionEvent]
    error: str | None = None


class EditRequested(BaseModel):
    job_id: str


class RenderRequested(BaseModel):
    render_id: str


class MediaUploaded(BaseModel):
    media_id: str


class ThumbsRequested(BaseModel):
    job_id: str


# ──────────────────────────────── tabelas ───────────────────────────────────


class Base(DeclarativeBase):
    pass


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    status: Mapped[str] = mapped_column(String(24), default=JobStatus.PENDING)
    stage: Mapped[str] = mapped_column(String(64), default="")
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    video_key: Mapped[str] = mapped_column(String(255))
    video_name: Mapped[str] = mapped_column(String(255), default="")

    duration_s: Mapped[float] = mapped_column(Float, default=0.0)
    #: quadros por segundo da gravação — o editor precisa dele para o passo de
    #: um quadro fazer sentido
    fps: Mapped[float] = mapped_column(Float, default=0.0)
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    #: cópia reduzida da gravação, para o monitor do editor. Sai da mesma
    #: decodificação dos recortes, então custa quase nada
    proxy_key: Mapped[str] = mapped_column(String(255), default="")
    #: forma de onda do áudio da partida, já reduzida — é ela que mostra o tiro
    #: e a explosão na régua
    waveform: Mapped[list] = mapped_column(JSON, default=list)
    #: montagem em andamento, salva sozinha enquanto o usuario edita
    draft: Mapped[dict] = mapped_column(JSON, default=dict)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, onupdate=utcnow
    )

    events: Mapped[list["Event"]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )
    proposals: Mapped[list["Proposal"]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )
    renders: Mapped[list["Render"]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )
    clips: Mapped[list["Clip"]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )
    reports: Mapped[list["DetectorReport"]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )
    media: Mapped[list["Media"]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )


class Proposal(Base):
    """Um vídeo que o sistema *pode* gerar com os momentos que encontrou.

    Sai da análise e fica esperando: o usuário escolhe quais quer, e cada
    escolha pode ser gerada mais de uma vez, com músicas diferentes.
    """

    __tablename__ = "proposals"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"))
    kind: Mapped[str] = mapped_column(String(24))
    title: Mapped[str] = mapped_column(String(160), default="")
    start_s: Mapped[float] = mapped_column(Float, default=0.0)
    end_s: Mapped[float] = mapped_column(Float, default=0.0)
    score: Mapped[float] = mapped_column(Float, default=0.0)
    #: instantes que originam os cortes (uma montagem tem vários)
    moments: Mapped[list] = mapped_column(JSON, default=list)
    meta: Mapped[dict] = mapped_column(JSON, default=dict)

    job: Mapped[Job] = relationship(back_populates="proposals")
    clips: Mapped[list["Clip"]] = relationship(back_populates="proposal")


class Render(Base):
    """Um pedido de geração: as propostas escolhidas e as opções de cada uma."""

    __tablename__ = "renders"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"))
    status: Mapped[str] = mapped_column(String(24), default=RenderStatus.PENDING)
    stage: Mapped[str] = mapped_column(String(64), default="")
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    #: lista de `Selection` serializada -- os videos que sairam de proposta
    selections: Mapped[list] = mapped_column(JSON, default=list)
    #: lista de `Timeline` serializada -- os videos que o usuario montou
    timelines: Mapped[list] = mapped_column(JSON, default=list)
    #: grade de batidas por proposta escolhida, preenchida pelo serviço de ritmo
    beats: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, onupdate=utcnow
    )

    job: Mapped[Job] = relationship(back_populates="renders")
    clips: Mapped[list["Clip"]] = relationship(
        back_populates="render", cascade="all, delete-orphan"
    )


class Media(Base):
    """Um arquivo que o usuario trouxe: musica, clipe ou imagem.

    Comecou como "a musica do job" e virou a biblioteca de midia da partida --
    porque era a mesma coisa. Uma musica sobe, um worker a analisa e o gateway
    a entrega com `Range`: e exatamente o caminho que um clipe importado
    percorre. Generalizar custou uma coluna (`kind`) e evitou um segundo sistema
    de upload vivendo ao lado do primeiro.

    Ela e do **job**, e nao de um pedido: o mesmo arquivo serve a quantas
    montagens o usuario quiser, sem subir de novo.

    > A tabela ainda se chama `tracks`, por historia. Renomea-la exigiria migrar
    > dados por um ganho de estetica; o modelo, que e o que se le no codigo, diz
    > o que ela guarda.
    """

    __tablename__ = "tracks"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"))
    status: Mapped[str] = mapped_column(String(16), default=TrackStatus.PENDING)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    #: audio, video ou imagem. O default cobre as linhas de quando so havia
    #: musica: elas eram todas audio
    kind: Mapped[str] = mapped_column(String(16), default=MediaKind.AUDIO)
    name: Mapped[str] = mapped_column(String(255), default="")
    key: Mapped[str] = mapped_column(String(255))
    duration_s: Mapped[float] = mapped_column(Float, default=0.0)

    # ── so audio ────────────────────────────────────────────────────────────
    bpm: Mapped[float] = mapped_column(Float, default=0.0)
    #: instantes das batidas, em segundos
    beats: Mapped[list] = mapped_column(JSON, default=list)
    #: forma de onda ja reduzida a alguns milhares de picos (0..1), para o app
    #: desenhar sem baixar o audio inteiro
    peaks: Mapped[list] = mapped_column(JSON, default=list)

    # ── video e imagem ──────────────────────────────────────────────────────
    width: Mapped[int] = mapped_column(Integer, default=0)
    height: Mapped[int] = mapped_column(Integer, default=0)
    fps: Mapped[float] = mapped_column(Float, default=0.0)
    thumb_key: Mapped[str] = mapped_column(String(255), default="")
    #: copia reduzida, pelo mesmo motivo do proxy da gravacao: o monitor nao
    #: pode arrastar o arquivo cheio a cada busca
    proxy_key: Mapped[str] = mapped_column(String(255), default="")

    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    job: Mapped[Job] = relationship(back_populates="media")

    @property
    def is_audio(self) -> bool:
        return self.kind == MediaKind.AUDIO


class Event(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"))
    kind: Mapped[str] = mapped_column(String(24))
    t: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    meta: Mapped[dict] = mapped_column(JSON, default=dict)

    job: Mapped[Job] = relationship(back_populates="events")


class DetectorReport(Base):
    """Um registro por detector por job — é assim que o editor sabe que pode começar."""

    __tablename__ = "detector_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"))
    detector: Mapped[str] = mapped_column(String(32))
    ok: Mapped[int] = mapped_column(Integer, default=1)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    n_events: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    job: Mapped[Job] = relationship(back_populates="reports")


class Clip(Base):
    __tablename__ = "clips"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"))
    render_id: Mapped[str] = mapped_column(
        ForeignKey("renders.id", ondelete="CASCADE")
    )
    proposal_id: Mapped[str | None] = mapped_column(
        ForeignKey("proposals.id", ondelete="SET NULL"), nullable=True
    )
    kind: Mapped[str] = mapped_column(String(24))
    title: Mapped[str] = mapped_column(String(160), default="")
    start_s: Mapped[float] = mapped_column(Float, default=0.0)
    end_s: Mapped[float] = mapped_column(Float, default=0.0)
    score: Mapped[float] = mapped_column(Float, default=0.0)
    key: Mapped[str] = mapped_column(String(255))
    meta: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    job: Mapped[Job] = relationship(back_populates="clips")
    render: Mapped[Render] = relationship(back_populates="clips")
    proposal: Mapped["Proposal | None"] = relationship(back_populates="clips")
