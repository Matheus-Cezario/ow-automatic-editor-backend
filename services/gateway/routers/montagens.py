"""Montages: named montages with their versions, the presets, and the old
single draft kept for apps from before montages."""

from __future__ import annotations

from typing import Any, Sequence

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import Response
from pydantic import ValidationError
from sqlalchemy import select

from owcore.db import session
from owcore.models import (
    Job,
    MontageDraft,
    MontageVersion,
    Preset,
    Recipe,
    new_id,
    utcnow,
)
from owcore.models import Montage as MontageModel

from common import _as_aware, _iso, _montage_dict

router = APIRouter()


def _adopt_old_draft(s: Any, job: Job) -> None:
    """Brings the job's single montage into the list of named montages.

    Until Phase 8 there was a single one, in a column of the job itself. Doing
    this on read, and not in a database migration, is the same choice as the
    rest of the system: the code that reads is what knows how to convert the
    old format, and so a match untouched for months still opens.
    """
    if not job.draft or job.montages:
        return
    # through the relationship, not through `s.add`: that way `job.montages`
    # already sees it within this same request, which is where it must appear
    job.montages.append(
        MontageModel(name=job.draft.get("title") or "Montage 1", data=job.draft)
    )
    # the column is cleared so there are not two truths about the same montage
    job.draft = {}
    s.flush()


@router.put("/api/jobs/{job_id}/draft")
def save_draft(job_id: str, draft: dict = Body(...)) -> dict[str, Any]:
    """Stores the montage in progress. **V1 legacy.**

    Since Phase 8 a match has several named montages, and the app saves by the
    id of one of them. This route writes to the most recent one -- and creates
    the first one if there is none -- so an older app keeps working instead of
    silently losing work.
    """
    draft_tl = _validate_montage(draft)

    with session() as s:
        job = s.get(Job, job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        _adopt_old_draft(s, job)
        current = max(job.montages, key=lambda m: _as_aware(m.updated_at), default=None)
        if current is None:
            current = MontageModel(job_id=job_id, name="Montage 1")
            s.add(current)
        current.data = draft_tl.model_dump()
    return {"job_id": job_id, "n_cuts": len(draft_tl.clips)}


@router.delete("/api/jobs/{job_id}/draft", status_code=204)
def delete_draft(job_id: str) -> Response:
    """Throws this match's montages away and starts from scratch. **V1 legacy.**"""
    with session() as s:
        job = s.get(Job, job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        job.draft = {}
        for m in list(job.montages):
            s.delete(m)
    return Response(status_code=204)


def _free_name(existing: Sequence[Any], base: str) -> str:
    """A name that is not on the list yet.

    Repeated names in a list to pick from are as good as no names at all.
    """
    names = {m.name for m in existing}
    if base not in names:
        return base
    for i in range(2, 100):
        attempt = f"{base} {i}"
        if attempt not in names:
            return attempt
    return f"{base} {new_id()[:4]}"


def _get_montage(s: Any, job_id: str, montage_id: str) -> MontageModel:
    m = s.get(MontageModel, montage_id)
    if m is None or m.job_id != job_id:
        raise HTTPException(404, "montage not found in this match")
    return m


def _validate_montage(data: dict) -> MontageDraft:
    try:
        return MontageDraft(**(data or {}))
    except ValidationError as exc:
        raise HTTPException(422, f"invalid montage: {exc}") from exc


@router.get("/api/jobs/{job_id}/montages")
def list_montages(job_id: str) -> dict[str, Any]:
    """This match's montages, most recent first."""
    with session() as s:
        job = s.get(Job, job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        _adopt_old_draft(s, job)
        montages = sorted(job.montages, key=lambda m: _as_aware(m.updated_at), reverse=True)
        return {
            "job_id": job_id,
            "items": [_montage_dict(m, full=True) for m in montages],
        }


@router.post("/api/jobs/{job_id}/montages", status_code=201)
def create_montage(job_id: str, body: dict = Body(default={})) -> dict[str, Any]:
    """Starts a new montage, empty or from given content.

    They are different jobs over the same material -- the 30 s cut for Shorts
    and the long montage -- and until now one had to be chosen.
    """
    data = body.get("data") or {}
    _validate_montage(data)
    with session() as s:
        job = s.get(Job, job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        _adopt_old_draft(s, job)
        name = str(body.get("name") or "").strip()
        m = MontageModel(
            job_id=job_id,
            name=_free_name(job.montages, name or f"Montage {len(job.montages) + 1}"),
            data=data,
        )
        s.add(m)
        s.flush()
        return _montage_dict(m, full=True)


@router.put("/api/jobs/{job_id}/montages/{montage_id}")
def save_montage(
    job_id: str, montage_id: str, body: dict = Body(...)
) -> dict[str, Any]:
    """Stores the montage. It is what the app calls by itself while editing.

    An invalid cut is refused here: storing garbage now would mean handing
    garbage back on the next opening.
    """
    with session() as s:
        m = _get_montage(s, job_id, montage_id)
        if "data" in body:
            draft_tl = _validate_montage(body["data"])
            m.data = draft_tl.model_dump()
        if "name" in body:
            name = str(body["name"] or "").strip()
            if not name:
                raise HTTPException(422, "a montage without a name cannot be found")
            others = [o for o in m.job.montages if o.id != m.id]
            m.name = _free_name(others, name)
        s.flush()
        return _montage_dict(m)


@router.post("/api/jobs/{job_id}/montages/{montage_id}/duplicate", status_code=201)
def duplicate_montage(job_id: str, montage_id: str) -> dict[str, Any]:
    """A copy, to experiment without risking the one that is already good.

    The copy does not take the original's history: the snapshots say where
    *that* montage has been, and the copy has not been anywhere yet.
    """
    with session() as s:
        original = _get_montage(s, job_id, montage_id)
        copy = MontageModel(
            job_id=job_id,
            name=_free_name(original.job.montages, f"{original.name} (copy)"),
            data=dict(original.data or {}),
        )
        s.add(copy)
        s.flush()
        return _montage_dict(copy, full=True)


@router.delete("/api/jobs/{job_id}/montages/{montage_id}", status_code=204)
def delete_montage(job_id: str, montage_id: str) -> Response:
    with session() as s:
        s.delete(_get_montage(s, job_id, montage_id))
    return Response(status_code=204)


# ── version history ─────────────────────────────────────────────────────────

#: how many snapshots each montage keeps. Past that, the oldest goes.
#:
#: This is not undo -- that lives in the app. These are markers, and twenty
#: markers is already more history than anyone scrolls through in a list.
MAX_VERSIONS = 20


def _version_dict(v: MontageVersion, *, full: bool = False) -> dict[str, Any]:
    d: dict[str, Any] = {
        "id": v.id,
        "montage_id": v.montage_id,
        "label": v.label,
        "created_at": _iso(v.created_at),
        **MontageModel(data=v.data).summary,
    }
    if full:
        d["data"] = v.data or {}
    return d


def _take_snapshot(m: MontageModel, label: str) -> MontageVersion | None:
    """Takes a snapshot of the montage as it stands now.

    Refuses a snapshot identical to the last one: rendering the same video twice
    in a row produced no new version, and a list of identical states helps
    nobody find the "it was good yesterday".
    """
    if not m.data:
        return None
    latest = max(m.versions, key=lambda v: _as_aware(v.created_at), default=None)
    if latest is not None and latest.data == m.data:
        return None

    # the timestamp comes from here, and not from the column's `default`:
    # without it the fresh snapshot joins the list with a null `created_at` and
    # cannot even be sorted
    snapshot = MontageVersion(label=label, data=dict(m.data), created_at=utcnow())
    m.versions.append(snapshot)
    newest_first = sorted(m.versions, key=lambda v: _as_aware(v.created_at), reverse=True)
    for v in newest_first[MAX_VERSIONS:]:
        m.versions.remove(v)
    return snapshot


@router.get("/api/jobs/{job_id}/montages/{montage_id}/versions")
def list_versions(job_id: str, montage_id: str) -> dict[str, Any]:
    """This montage's snapshots, most recent first."""
    with session() as s:
        m = _get_montage(s, job_id, montage_id)
        snapshots = sorted(m.versions, key=lambda v: _as_aware(v.created_at), reverse=True)
        return {"montage_id": montage_id, "items": [_version_dict(v) for v in snapshots]}


@router.post("/api/jobs/{job_id}/montages/{montage_id}/versions", status_code=201)
def create_version(
    job_id: str, montage_id: str, body: dict = Body(default={})
) -> dict[str, Any]:
    """Marks the montage as it stands: the "it was good like this"."""
    with session() as s:
        m = _get_montage(s, job_id, montage_id)
        snapshot = _take_snapshot(m, str(body.get("label") or "marked by hand"))
        if snapshot is None:
            raise HTTPException(409, "there is nothing new to mark")
        s.flush()
        return _version_dict(snapshot, full=True)


@router.post("/api/jobs/{job_id}/montages/{montage_id}/versions/{version_id}/restore")
def restore_version(job_id: str, montage_id: str, version_id: str) -> dict[str, Any]:
    """Rolls the montage back to a snapshot.

    The current state becomes a snapshot first -- restoring never deletes work,
    it only swaps what is in front.
    """
    with session() as s:
        m = _get_montage(s, job_id, montage_id)
        snapshot = s.get(MontageVersion, version_id)
        if snapshot is None or snapshot.montage_id != montage_id:
            raise HTTPException(404, "version not found in this montage")
        _take_snapshot(m, "before restoring")
        m.data = dict(snapshot.data or {})
        s.flush()
        return _montage_dict(m, full=True)


@router.delete(
    "/api/jobs/{job_id}/montages/{montage_id}/versions/{version_id}", status_code=204
)
def delete_version(job_id: str, montage_id: str, version_id: str) -> Response:
    with session() as s:
        m = _get_montage(s, job_id, montage_id)
        snapshot = s.get(MontageVersion, version_id)
        if snapshot is None or snapshot.montage_id != montage_id:
            raise HTTPException(404, "version not found in this montage")
        m.versions.remove(snapshot)
    return Response(status_code=204)


# ── presets ─────────────────────────────────────────────────────────────────


def _preset_dict(p: Preset) -> dict[str, Any]:
    return {
        "id": p.id,
        "name": p.name,
        "data": p.data or {},
        "created_at": _iso(p.created_at),
        "updated_at": _iso(p.updated_at),
    }


def _validate_recipe(data: dict) -> Recipe:
    try:
        return Recipe(**(data or {}))
    except ValidationError as exc:
        raise HTTPException(422, f"invalid recipe: {exc}") from exc


@router.get("/api/presets")
def list_presets() -> dict[str, Any]:
    """The presets. They belong to no match -- crossing from one to another is
    why they exist."""
    with session() as s:
        items = s.execute(select(Preset).order_by(Preset.created_at)).scalars().all()
        return {"items": [_preset_dict(p) for p in items]}


@router.post("/api/presets", status_code=201)
def create_preset(body: dict = Body(...)) -> dict[str, Any]:
    name = str(body.get("name") or "").strip()
    if not name:
        raise HTTPException(422, "a preset without a name cannot be found")
    recipe = _validate_recipe(body.get("data") or {})
    with session() as s:
        existing = s.execute(select(Preset)).scalars().all()
        p = Preset(
            name=_free_name(existing, name), data=recipe.model_dump(mode="json")
        )
        s.add(p)
        s.flush()
        return _preset_dict(p)


@router.put("/api/presets/{preset_id}")
def update_preset(preset_id: str, body: dict = Body(...)) -> dict[str, Any]:
    with session() as s:
        p = s.get(Preset, preset_id)
        if p is None:
            raise HTTPException(404, "preset not found")
        if "data" in body:
            p.data = _validate_recipe(body["data"]).model_dump(mode="json")
        if "name" in body:
            name = str(body["name"] or "").strip()
            if not name:
                raise HTTPException(422, "a preset without a name cannot be found")
            others = [
                o for o in s.execute(select(Preset)).scalars().all() if o.id != p.id
            ]
            p.name = _free_name(others, name)
        s.flush()
        return _preset_dict(p)


@router.delete("/api/presets/{preset_id}", status_code=204)
def delete_preset(preset_id: str) -> Response:
    with session() as s:
        p = s.get(Preset, preset_id)
        if p is None:
            raise HTTPException(404, "preset not found")
        s.delete(p)
    return Response(status_code=204)
