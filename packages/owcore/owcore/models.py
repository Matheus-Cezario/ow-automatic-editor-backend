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
    """Ciclo de vida de *uma musica* enviada para o job.

    A musica sobe antes de qualquer video existir: e preciso ouvi-la no app e
    ver as batidas para poder posicionar os cortes em cima dela.
    """

    PENDING = "pending"
    READY = "ready"
    FAILED = "failed"


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
    cuts: list[TimelineCut] = Field(default_factory=list)

    #: Correções do usuário à grade de batidas. Não afetam o vídeo: o corte
    #: guarda instantes absolutos, e a grade é só o imã da tela. Vêm no rascunho
    #: para não se perder num F5 -- consertar a grade duas vezes irrita mais do
    #: que consertá-la uma.
    beat_offset_s: float = 0.0
    beat_multiplier: float = 1.0
    beat_bar: int = 1


class Timeline(BaseModel):
    """Um video montado a mao: os blocos e a musica por baixo deles.

    Diferente de `Selection`, aqui nao ha proposta nenhuma -- o usuario montou.
    A musica vem por `track_id` porque ela ja subiu antes: foi ouvindo ela, com
    as batidas na tela, que ele decidiu onde cada bloco cai.
    """

    title: str = ""
    #: musica de fundo; None deixa o audio original dos cortes
    track_id: str | None = None
    #: de que ponto da musica o video comeca a tocar
    music_start_s: float = 0.0
    cuts: list[TimelineCut] = Field(default_factory=list)

    @model_validator(mode="after")
    def _sem_sobreposicao(self) -> "Timeline":
        if self.music_start_s < 0:
            raise ValueError("music_start_s nao pode ser negativo")
        if not self.cuts:
            raise ValueError("uma linha do tempo vazia nao vira video")
        ordenados = sorted(self.cuts, key=lambda c: c.at_s)
        for anterior, seguinte in zip(ordenados, ordenados[1:]):
            if seguinte.at_s < anterior.until_s - 1e-6:
                raise ValueError(
                    f"dois cortes se sobrepoem em {seguinte.at_s:.2f}s do video"
                )
        self.cuts = ordenados
        return self

    @property
    def duration_s(self) -> float:
        """Quanto o video vai durar -- inclusive os buracos entre os blocos."""
        return max((c.until_s for c in self.cuts), default=0.0)


# ───────────────────────── mensagens do barramento ──────────────────────────

STREAM_JOBS = "ow.jobs"
STREAM_ROI = "ow.roi"
STREAM_EDIT = "ow.edit"
#: um pedido de geração recém-criado, ainda sem as batidas das músicas
STREAM_RENDER = "ow.render"
#: pedido com as batidas já analisadas, pronto para o editor cortar
STREAM_RENDER_READY = "ow.render.ready"
#: música recém-enviada, esperando quem extraia as batidas e a forma de onda
STREAM_TRACK = "ow.track"
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


class TrackUploaded(BaseModel):
    track_id: str


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
    tracks: Mapped[list["Track"]] = relationship(
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


class Track(Base):
    """Uma musica enviada para o job, ja ouvida pelo sistema.

    Ela sobe **antes** de qualquer video: o usuario precisa toca-la no app, ver
    as batidas e a forma de onda, e so entao decidir onde cada corte cai. Por
    isso ela e do job, e nao de um pedido -- a mesma musica serve a quantas
    montagens ele quiser, sem subir de novo.
    """

    __tablename__ = "tracks"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"))
    status: Mapped[str] = mapped_column(String(16), default=TrackStatus.PENDING)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    name: Mapped[str] = mapped_column(String(255), default="")
    key: Mapped[str] = mapped_column(String(255))
    duration_s: Mapped[float] = mapped_column(Float, default=0.0)
    bpm: Mapped[float] = mapped_column(Float, default=0.0)
    #: instantes das batidas, em segundos
    beats: Mapped[list] = mapped_column(JSON, default=list)
    #: forma de onda ja reduzida a alguns milhares de picos (0..1), para o app
    #: desenhar sem baixar o audio inteiro
    peaks: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    job: Mapped[Job] = relationship(back_populates="tracks")


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
