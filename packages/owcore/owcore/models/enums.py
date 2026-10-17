"""Ids, the clock and the status enums every other models module uses."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import StrEnum


def new_id() -> str:
    return uuid.uuid4().hex[:16]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------- enums -----------------------------------


class JobStatus(StrEnum):
    """Lifecycle of the *analysis*. It runs once, on its own, and ends at
    `READY`: from there the job waits for the user to open the editor."""

    PENDING = "pending"
    PREPROCESSING = "preprocessing"
    DETECTING = "detecting"
    READY = "ready"
    FAILED = "failed"


class RenderStatus(StrEnum):
    """Lifecycle of *one render request*. A job has as many as it likes."""

    PENDING = "pending"
    RENDERING = "rendering"
    DONE = "done"
    FAILED = "failed"
    #: the user stopped it; nothing it made is kept
    CANCELLED = "cancelled"


class JobStage(StrEnum):
    """Where the analysis is, finer than [JobStatus].

    A code, never a sentence: the database stores what happened and the app
    decides how to say it. Numbers do not go in here either -- how many
    moments were found is `Job.n_moments`, how far along it is `progress`.
    """

    QUEUED = "queued"
    DOWNLOADING = "downloading"
    CROPPING = "cropping"
    EXTRACTING_AUDIO = "extracting_audio"
    DETECTING = "detecting"
    #: the detectors are done and the planner is crossing their events
    PLANNING = "planning"
    READY = "ready"
    ERROR = "error"


class RenderStage(StrEnum):
    """Where a render request is. A code, like [JobStage]: how far it got is
    `progress`, and how many videos came out is read from its clips."""

    QUEUED = "queued"
    PREPARING = "preparing"
    RENDERING = "rendering"
    DONE = "done"
    #: the request brought no timeline
    NOTHING_CHOSEN = "nothing_chosen"
    #: none of the timelines has anything that can be cut
    NOTHING_TO_CUT = "nothing_to_cut"
    ERROR = "error"
    CANCELLED = "cancelled"


class TrackStatus(StrEnum):
    """Lifecycle of *one media item* uploaded to the job.

    Music arrives before any video exists: you have to hear it in the app and
    see the beats before you can place cuts on top of it. With the media
    library, the same holds for an imported clip or image -- the app needs the
    thumbnail and the dimensions before it will let you build with them.
    """

    PENDING = "pending"
    READY = "ready"
    FAILED = "failed"


class MediaKind(StrEnum):
    """What the imported file is.

    Decides what the analysis does with it: audio gets beats and a waveform,
    video gets a thumbnail and a proxy, an image gets a thumbnail.
    """

    AUDIO = "audio"
    VIDEO = "video"
    IMAGE = "image"


class EventKind(StrEnum):
    KILL = "kill"
    DEATH = "death"
    LOW_HP = "low_hp"
    ESCAPE = "escape"
    #: an ultimate was used. `meta["side"]` says whose: "self" when the footer
    #: button discharged (the player's own), "enemy" when the icon appeared in
    #: the killfeed or the audio spiked.
    ULT_USED = "ult_used"
    ULT_NEGATED = "ult_negated"
    #: critical hit -- the red X marker on the crosshair
    HEADSHOT = "headshot"
    #: someone on our team killed with an ability named in the killfeed;
    #: `meta["ability"]` carries "hero/ability"
    ABILITY_KILL = "ability_kill"
    #: Ana's sleep dart landing on someone
    SLEEP = "sleep"
    #: Sigma's Accretion rock stunning someone
    STUN = "stun"
    #: a line appearing in the killfeed, whatever it was made with.
    #: `meta["killer"]` is "player", "other" or "unknown" (the name on the plate
    #: could not be read). Not a moment: it is the evidence the planner uses to
    #: confirm the crosshair's skulls, and it is consumed there -- see
    #: `rules.unconfirmed_kills`.
    KILLFEED_LINE = "killfeed_line"


#: What a generated video is. There used to be a kind per rule -- "kill
#: streak", "solo wipe", "beat montage" -- because the system proposed
#: ready-made videos from the events. It does not any more: every video comes
#: out of the editor, and what it is, is whatever title the user gave it.
#:
#: It is still a text column in the database, so clips already generated keep
#: the old kind they had -- the app only needs to know how to draw them.
CLIP_KIND_CUSTOM = "custom"


#: Detectors the analysis waits for before considering itself finished.
#: One per *screen region*, not per ability: `banner` reads the footer strip and
#: tells the abilities apart by their icon.
DETECTORS = ("kills", "survival", "ults", "banner", "killfeed")
