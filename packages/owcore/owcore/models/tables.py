"""The database: the ten tables and who reads them."""

from __future__ import annotations

from datetime import datetime

from pydantic import ValidationError
from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from .enums import JobStatus, MediaKind, RenderStatus, TrackStatus, new_id, utcnow
from .timeline import MontageDraft

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
    #: how many moments the analysis found, crossed events included. `None`
    #: until the analysis ends -- and on matches analysed before this column
    #: existed, until the gateway counts them
    n_moments: Mapped[int | None] = mapped_column(Integer, nullable=True)

    video_key: Mapped[str] = mapped_column(String(255))
    video_name: Mapped[str] = mapped_column(String(255), default="")

    duration_s: Mapped[float] = mapped_column(Float, default=0.0)
    #: frames per second of the recording -- the editor needs it for a
    #: one-frame step to make sense
    fps: Mapped[float] = mapped_column(Float, default=0.0)
    #: size of the recording. It is the export default -- and what lets the
    #: editor say whether the requested output crops the frame or leaves bars
    width: Mapped[int] = mapped_column(Integer, default=0)
    height: Mapped[int] = mapped_column(Integer, default=0)
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    #: reduced copy of the recording, for the editor's monitor. It comes out
    #: of the same decode as the crops, so it costs almost nothing
    proxy_key: Mapped[str] = mapped_column(String(255), default="")
    #: waveform of the match audio, already reduced -- it is what shows the
    #: shot and the explosion on the ruler
    waveform: Mapped[list] = mapped_column(JSON, default=list)
    #: montage in progress, saved automatically while the user edits
    draft: Mapped[dict] = mapped_column(JSON, default=dict)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, onupdate=utcnow
    )

    events: Mapped[list["Event"]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )
    renders: Mapped[list["Render"]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )
    previews: Mapped[list["Preview"]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )
    clips: Mapped[list["Clip"]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )
    reports: Mapped[list["DetectorReport"]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )
    montages: Mapped[list["Montage"]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )
    media: Mapped[list["Media"]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )


class Render(Base):
    """A render request: the montages the user sent to be rendered."""

    __tablename__ = "renders"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"))
    status: Mapped[str] = mapped_column(String(24), default=RenderStatus.PENDING)
    stage: Mapped[str] = mapped_column(String(64), default="")
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    #: serialised list of `Timeline` -- the videos the user built
    timelines: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    #: when the editor picked it up: the time left is measured from here,
    #: not from the request, so the wait in the queue does not count as work
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, onupdate=utcnow
    )

    @property
    def is_active(self) -> bool:
        return self.status in (RenderStatus.PENDING, RenderStatus.RENDERING)

    job: Mapped[Job] = relationship(back_populates="renders")
    clips: Mapped[list["Clip"]] = relationship(
        back_populates="render", cascade="all, delete-orphan"
    )


class Preview(Base):
    """An exact preview: a stretch of one montage rendered small and fast.

    The editor's monitor composes the layers in the browser, which is instant
    but approximate. This is the server's own graph -- the same
    `compose_graph` the final video goes through -- on a reduced frame, so what
    it shows is what will come out, only smaller.

    It is not a [Render]: it never shows up among the generated videos, in the
    match's zip or in its counts, and only the latest one of a match is kept.
    """

    __tablename__ = "previews"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"))
    status: Mapped[str] = mapped_column(String(24), default=RenderStatus.PENDING)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    #: one serialised `Timeline`, its export window already set
    timeline: Mapped[dict] = mapped_column(JSON, default=dict)
    #: where the stretch sits in the montage, so the app can line it up with
    #: the ruler
    from_s: Mapped[float] = mapped_column(Float, default=0.0)
    to_s: Mapped[float] = mapped_column(Float, default=0.0)
    video_key: Mapped[str] = mapped_column(String(255), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, onupdate=utcnow
    )

    job: Mapped[Job] = relationship(back_populates="previews")


class Media(Base):
    """A file the user brought in: music, clip or image.

    It began as "the job's music" and became the match's media library --
    because they were the same thing. Music is uploaded, a worker analyses it
    and the gateway serves it with `Range`: exactly the path an imported clip
    walks. Generalising cost one column (`kind`) and avoided a second upload
    system living beside the first.

    It belongs to the **job**, not to a request: the same file serves as many
    montages as the user likes, with no re-upload.

    > The table is still called `tracks`, for historical reasons. Renaming it
    > would mean migrating data for an aesthetic gain; the model, which is what
    > you read in the code, says what it holds.
    """

    __tablename__ = "tracks"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"))
    status: Mapped[str] = mapped_column(String(16), default=TrackStatus.PENDING)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    #: audio, video or image. The default covers the rows from when there was
    #: only music: they were all audio
    kind: Mapped[str] = mapped_column(String(16), default=MediaKind.AUDIO)
    name: Mapped[str] = mapped_column(String(255), default="")
    key: Mapped[str] = mapped_column(String(255))
    duration_s: Mapped[float] = mapped_column(Float, default=0.0)

    # -- audio only ---------------------------------------------------------
    bpm: Mapped[float] = mapped_column(Float, default=0.0)
    #: beat instants, in seconds
    beats: Mapped[list] = mapped_column(JSON, default=list)
    #: waveform already reduced to a few thousand peaks (0..1), so the app can
    #: draw without downloading the whole audio
    peaks: Mapped[list] = mapped_column(JSON, default=list)

    # -- video and image ----------------------------------------------------
    width: Mapped[int] = mapped_column(Integer, default=0)
    height: Mapped[int] = mapped_column(Integer, default=0)
    fps: Mapped[float] = mapped_column(Float, default=0.0)
    thumb_key: Mapped[str] = mapped_column(String(255), default="")
    #: reduced copy, for the same reason as the recording's proxy: the monitor
    #: cannot drag the full file on every seek
    proxy_key: Mapped[str] = mapped_column(String(255), default="")

    #: the sound effect this item was made from (`owcore.sfx`), empty for a
    #: file the user uploaded. It is what lets adding the same effect twice
    #: reuse the item, and the app tell an effect from a song
    sfx_id: Mapped[str] = mapped_column(String(32), default="")
    #: the sticker and colour this item was drawn from (`owcore.stickers`,
    #: "arrow:red"), empty for a picture the user uploaded -- the same job as
    #: `sfx_id`, for pictures
    sticker_id: Mapped[str] = mapped_column(String(48), default="")
    #: recorded in the editor as a voice-over, not a song: it stays out of the
    #: track picker, and its blocks go in as `VOICE_KIND`
    voice: Mapped[bool] = mapped_column(Boolean, default=False)

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
    """One record per detector per job -- how the pipeline knows it can start."""

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

class Montage(Base):
    """A named montage of a match.

    Until Phase 8 there was **one** montage per job, kept in a column of the job
    itself. That forced a choice: either the 30-second cut for Shorts or the
    long montage, never both. They are different pieces of work over the same
    material, and now each has its own name.

    The content is still a `MontageDraft` -- the same format the app already
    sent. What changed is where it lives, and the fact that there are several.
    """

    __tablename__ = "montages"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    job_id: Mapped[str] = mapped_column(
        ForeignKey("jobs.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(120), default="")
    #: the montage itself, in `MontageDraft` format
    data: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, onupdate=utcnow
    )

    job: Mapped[Job] = relationship(back_populates="montages")
    versions: Mapped[list["MontageVersion"]] = relationship(
        back_populates="montage", cascade="all, delete-orphan"
    )

    @property
    def summary(self) -> dict:
        """Enough for the montage list without downloading the whole montage.

        Whoever only wants to choose between "short vertical" and "the long one"
        does not need the clips of either.
        """
        try:
            m = MontageDraft(**(self.data or {}))
        except ValidationError:
            # a montage stored by an earlier version of the format must not
            # take down the list: it shows as empty and stays openable
            return {"n_clips": 0, "duration_s": 0.0, "has_music": False}
        clips = m.clips
        return {
            "n_clips": len(clips),
            "duration_s": round(max((c.until_s for c in clips), default=0.0), 2),
            "has_music": any(
                c.is_music
                for l in m.layers
                if l.is_audio
                for c in l.clips
            ),
        }


class MontageVersion(Base):
    """A snapshot of a montage, kept so you can go back to it.

    It is not undo -- that lives in the app and dies with the tab. This is the
    "it was good yesterday": rare, deliberate markers, taken when a video is
    generated (what came out was *this*) or when the user asks.

    Keeping a snapshot on every autosave would fill the database with identical
    states and make the list useless from sheer length.
    """

    __tablename__ = "montage_versions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    montage_id: Mapped[str] = mapped_column(
        ForeignKey("montages.id", ondelete="CASCADE"), index=True
    )
    #: why this snapshot exists: "generated the video", "before restoring"...
    label: Mapped[str] = mapped_column(String(120), default="")
    data: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    montage: Mapped[Montage] = relationship(back_populates="versions")


class Preset(Base):
    """A preset: the way of building, saved for the next match.

    It belongs to no job on purpose -- crossing from one match to another is
    precisely why it exists.
    """

    __tablename__ = "presets"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(120), default="")
    #: the recipe, in the `Recipe` format
    data: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, onupdate=utcnow
    )
