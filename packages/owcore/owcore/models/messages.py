"""The bus: stream names and the messages published on them."""

from __future__ import annotations

from pydantic import BaseModel

from .analysis import Artifact, DetectionEvent, JobParams

# --------------------------- bus messages ----------------------------------

STREAM_JOBS = "ow.jobs"
STREAM_ROI = "ow.roi"
STREAM_EDIT = "ow.edit"
#: a render request, ready for the editor to cut.
#:
#: There used to be a stage between creating the request and this queue: the
#: rhythm service analysed the music of each chosen proposal and only then
#: released the editor. There are no proposals any more, and the editor's music
#: arrives already analysed through the library -- the request is born ready,
#: and the gateway publishes straight here.
STREAM_RENDER_READY = "ow.render.ready"
#: a freshly uploaded file, waiting to be analysed (beats and waveform for
#: audio; thumbnail, dimensions and proxy for video and image)
STREAM_MEDIA = "ow.media"
#: analysis finished: someone to extract a thumbnail for each moment
STREAM_THUMBS = "ow.thumbs"
#: an exact preview of a montage, waiting for the previewer. A stream of its
#: own so a preview never queues behind a full render
STREAM_PREVIEW = "ow.preview"


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


class PreviewRequested(BaseModel):
    preview_id: str


class ThumbsRequested(BaseModel):
    job_id: str
