# OW Editor — Project Plan

> **Where each thing lives.** In Git there are **two repositories**: this one,
> the backend's, and `ow-automatic-editor-frontend`, the Flutter app's. Paths
> in this document without a prefix (`services/`, `packages/`) are in this
> repository; those starting with `frontend/` are in the other one. The
> `docker-compose.yml` lives here, at the root.

**A video editor with automatic Overwatch 2 event detection.** It takes the
recording of a match, watches it and notes what happened; the user assembles
the video on the timeline — listening to the music in the app itself and
putting each moment at the point of it they want, for as long as they want.

> **Discontinued in Phase 11:** automatic generation. The system applied rules
> to the events and offered a list of ready-made videos to pick from, each with
> its own music. That path was removed entirely — `services/planner` stopped
> proposing, `owcore/rules.py` kept only the crossing between detectors, and
> the `proposals` table, the `selections` field and the app's generation
> screen were removed. What is left is what was always the edge: an editor
> that knows what happened in the video.

## 1. Principles

1. **Everything free / open-source.** FastAPI, Redis, S3 (RustFS), PostgreSQL,
   OpenCV, ffmpeg, librosa, Flutter. No paid service, no external API.
2. **It runs without Docker too.** Each infrastructure dependency has two
   implementations behind the same interface, chosen by an environment
   variable:

   | Resource | `local` mode (no Docker) | `docker` mode |
   |---|---|---|
   | Queue / events | disk queue (`LocalBus`) | Redis Streams |
   | Storage        | `data/` folder (`LocalStorage`) | S3 |
   | Database       | SQLite | PostgreSQL |

3. **Each microservice gets the fewest pixels possible.** The `preprocessor`
   does a *single* decode of the video and emits small crops (ROIs), at low FPS,
   one per detector. A detector never sees the whole video.
4. **Calibratable.** HUD positions, colours and thresholds live in a JSON
   *profile*, not in the code, because they change with resolution, language,
   colour-blind mode and patch.
5. **The system detects; the user assembles.** The analysis delivers the
   *instants* of each happening (kill, headshot, ability kill, dart, rock) and
   nothing more. The path is the timeline: the user listens to the music, sees
   the beats and the waveform, and puts each cut where it sounds best. No rule
   knows where the chorus of someone's music is — which is why proposals were
   removed.

   Operational corollary: **every kind of detected event must reach the
   editor's shelf.** A detector whose result does not become a block is wasted
   CPU work. The list lives in two places that must agree — `THUMB_KINDS`
   (`owcore/models.py`, what thumbnails are extracted for) and `_usefulMoments`
   (`frontend/lib/screens/timeline_screen.dart`, what the screen shows).
   Diverging gives a card without a frame, or a moment that does not exist.
6. **Analysis and rendering are separate phases.** The analysis is expensive
   and its result does not change — the match's moments are what they are.
   Choosing what becomes a video is cheap, personal and changeable. Joining the
   two would force re-analysing the video just to change the music.

## 2. Architecture

### Phase 1 — analysis (runs once, on its own)

```
                 ┌────────────┐
  Flutter  ─────▶│  gateway   │  recording upload (just the video)
  (mobile/web)   └─────┬──────┘
                       │ publishes JobCreated
                       ▼
                 ┌────────────┐   1 decode → N small crops + audio
                 │preprocessor│
                 └─────┬──────┘
     RoiReady(kills) │ (survival)   (ults)      (banner)    (killfeed)
       ┌───────────┬──┴────────┬────────────┬────────────┐
       ▼           ▼           ▼            ▼            ▼
 ┌──────────┐┌──────────┐┌──────────┐ ┌──────────┐ ┌──────────┐
 │detector_ ││detector_ ││detector_ │ │detector_ │ │detector_ │
 │kills     ││survival  ││ults      │ │banner    │ │killfeed  │
 └────┬─────┘└────┬─────┘└────┬─────┘ └────┬─────┘ └────┬─────┘
      │ events    │ events    │ events     │ events     │ events
      └───────────┴─────┬─────┴────────────┴────────────┘
                                  ▼
                          ┌───────────────┐
                          │    planner    │ waits for all, crosses what
                          │  (closes and  │ no detector sees alone,
                          │   crosses)    │ no pixels
                          └───────┬───────┘
                                  ▼
                    crossed events + job at `ready`
```

The job stops at `ready` and **waits**. No music went through here.

The `planner` used to build the list of proposals here. It no longer does;
what it still does is what nobody else can: it is the point where all the
detectors meet, and therefore the only place from which a **negated ultimate**
(enemy ultimate + a kill right after) can be seen. Those derived events are
**stored** like any other — they used to be born inside the proposal generator
and die there, and the editor never got to see them.

### Phase 1½ — the editor's library (music, clip, image)

```
  Flutter ──▶ gateway   POST /api/jobs/{id}/tracks   (just the audio file)
                │ publishes TrackUploaded
                ▼
          ┌───────────┐  duration, BPM, beats and a reduced waveform
          │   beats   │  (the same process, in a second loop)
          └─────┬─────┘
                ▼
     the music comes back ready for the app to **draw** and **play**
```

The music is uploaded **before** any video exists, and that is the inversion
the manual montage requires: you cannot decide that a cut comes in "on the
chorus's turnaround" without hearing the chorus, or fit it to the beat without
knowing where the beats are. It belongs to the job, not to a request — the same
track serves as many montages as the user wants, without uploading again.

### Phase 2 — rendering (on demand, as many times as the user wants)

```
  Flutter ──▶ gateway   POST /api/jobs/{id}/renders
  (the ruler's  │       form: timelines JSON
   montages)    │ publishes RenderRequested → ow.render.ready
                ▼
          ┌───────────┐  cuts and joins exactly what was assembled;
          │  editor   │  a gap becomes black, and with no music on the
          └─────┬─────┘  ruler the video keeps the original audio
                ▼
          final clips in storage, tied to the request
```

The request goes **straight** to the editor. There used to be a `beats` step in
between, to analyse the music of each chosen proposal; without proposals, the
only music that exists is the library's, and it was already analysed when it
was uploaded in phase 1½.

Bus: streams with *consumer groups* (Redis Streams in production, an on-disk
equivalent in local mode). Job state: a relational database via SQLAlchemy
2.0, the same model on SQLite and Postgres.

### Data model

| Table | What it stores |
|---|---|
| `jobs` | the recording, the **analysis** progress (`pending → preprocessing → detecting → ready`) and the montage in progress (`draft`) |
| `events` | what each detector found, with its instant — **including** what is only seen by crossing two detectors (`ULT_NEGATED`), stored by the `planner` |
| `tracks` | a music track uploaded for the match, with duration, BPM, beats and waveform — it is what the montage screen plays and draws |
| `renders` | a rendering **request**: the montages and the progress (`pending → rendering → done`) |
| `clips` | the rendered video, tied to the request |
| `montages` | a **named** montage of a match, with its history in `montage_versions`. A match yields more than one video |
| `presets` | the **way** of assembling, kept for the next match. It belongs to no job, on purpose |

> **Removed in Phase 11:** the `proposals` table and the `renders.selections`
> column. The schema reconciler (`owcore/db.py`) adds columns but never removes
> them — in a database that has already run the system they are still there,
> orphaned and ignored. No migration is needed.

The flow is repeatable because rendering a video uses nothing up: the match's
`events` stay where they are, and the same instant goes into as many montages
as wanted, with different music.

The montage has no table of its own: it is a list of blocks stored in the
request itself (`renders.timelines`). Each block says **what** comes in
(`start_s` + `duration_s`, in the recording) and **where** it comes in (`at_s`,
in the output video) — two independent things, and it is that independence
that lets the same moment show up twice, at different points of the music and
with different durations.

Along with the blocks goes the `export`: size, fps, quality, framing, range and
watermark. It stays **outside** the layers on purpose — the same montage
becomes a 16:9 and a 9:16 without a block moving. And it alone decides the
rendering path: a non-default output does not exist in V1's cut-and-splice,
only in the filter graph.

## 3. Detection — how each event is recognised

| Detector | ROI sent | Technique | Event emitted |
|---|---|---|---|
| `detector_kills` | ~16%×18% of the screen around the crosshair, 12 fps | HSV mask of the HUD magenta + a **shape, position and size filter**: the blob must be compact, near square, centred on the crosshair and take 5% to 14% of the region | `KILL` |
| `detector_kills` (2nd reading) | the same ROI | the **critical** hit marker is red (a normal one is white): four strokes in an X on the crosshair. The decision is by shape, not colour — the four diagonals painted and the four straight directions clear. It is that second check that stops the kill skull, also red and also on the crosshair, from becoming a headshot | `HEADSHOT` |
| `detector_survival` | strip of the health bar (bottom-left), 6 fps | reads the filled fraction by the *alternation* of the ticks; sustained low health, and health dropping to zero = interruption | `LOW_HP`, `ESCAPE`, `DEATH` |
| `detector_ults` | the footer's **ultimate button**, 5 fps | when charged, the player's ultimate is a white disc with the hero's icon and a cyan ring around it; using it turns both off. The event is the **falling edge**, and the disc's icon says whose ultimate it was | `ULT_USED` (`side="self"`) |
| `detector_ults` | killfeed (top-right corner), 5 fps | multiscale `matchTemplate` against ultimate icons provided by the user | `ULT_USED` (`side="enemy"`) |
| `detector_killfeed` | the same killfeed strip, 5 fps | finds the line by the **team colours** (a cyan plate to the left of a red one) and reads the icon in the gap between them, comparing it with `templates/abilities/`. No text, no OCR | `ABILITY_KILL` |
| `detector_banner` | the footer banner band, 4 fps | finds **every** banner in the frame (one mask per colour, because the HUD is cyan in one recording and green in another) and identifies each one **by the icon** on the left end, with a per-ability threshold. In a single frame only the winning template scores: one banner announces one ability | `SLEEP`, `STUN` |

`beats` is not in this table on purpose: it **is not a detector**. It does not
look at the match, emits no event and does not run in the analysis. It listens
to what comes into the editor's library and produces, for music, a beat grid
(`librosa.beat.beat_track`, with its own numpy estimator as a fallback).

> **What calibrating on real gameplay changed.** The first version only
> measured "how much of the region is red". Against 19 minutes of real matches
> that does not work: the game world (maps' warm lighting, enemies' red
> outlines) and the *directional damage indicator* paint the same colour almost
> all the time — half the frames went over 3% red in the region, and precision
> stayed at ~17%. What separates the skull from the rest is not colour: it is
> being a **compact, near-square blob centred on the crosshair**, whereas the
> damage indicator is a wide arc drawn on a radius above the centre. With shape
> and position in the criterion precision went up, and the missing filter was
> **size**: with the minimum at 0.4% of the region, any 20-pixel red splash
> became a kill — those splashes were most of the remaining false positives.
> The skull takes 5% to 14% of the ROI. With the four criteria, precision went
> to ~91%.
>
> Low health was inferred from the red vignette on the edges — which is the
> *damage taken* warning, present in 32% of a real match's frames, and yielded
> 122 "escapes" in 19 minutes. Today the detector reads the health bar directly
> (error ≤ 0.05 against the values on screen). Death was inferred from the
> desaturated kill cam, and found zero deaths: OW2's kill cam is not
> desaturated.
>
> Ultimates still depend on game icons, which are assets and are not in the
> repository. The alternative audio-peak path exists but **ships off**: on a
> synthetic video it gets it right, in a real match it cannot tell an ultimate
> voice line from gunfire and explosions, and there is no ground truth to
> calibrate it honestly.

## 4. What the analysis delivers

It delivers **events**, and nothing else. It does not group, score or propose.

A single crossing happens after the detectors, in
`packages/owcore/owcore/rules.py`, and the `planner` runs it when closing the
analysis:

| Derived event | Rule |
|---|---|
| `ULT_NEGATED` | an enemy `ULT_USED` followed by a `KILL` within `ult_negate_window_s` (6 s). No detector alone sees both kinds — correlation across microservices is the aggregator's job |

That is why it lives here, and not in a detector; and it is **stored as an
event**, to show up on the editor's shelf with the others.

> **The rules engine that was removed.** This file was once three times
> larger. It turned events into *highlights*: `MULTIKILL` (≥3 `KILL` in 10 s),
> `SOLO_WIPE`, `ESCAPE`, and a family of beat montages — one per event kind,
> plus one per ability. The `planner` stored all that as `proposals`, the app
> showed the list, and the user chose which to render by giving each a music
> track.
>
> It was removed entirely in Phase 11, together with `MONTAGE_KINDS`,
> `ClipOptions`, `Selection`, `montage_segments` and `fit_to_window`. The reason
> is not technical: a rule knows how to group kills in a 10-second window, and
> does not know where the music turns. It produced mediocre montages and,
> worse, worked as a **filter** between detection and the user — what became a
> proposal was what they saw. That is how headshots and ability kills, detected
> all along, never reached the editor's shelf.
>
> What holds now is the opposite: **everything detected becomes a possible
> block**, and the judgement belongs to whoever assembles.

What is left of "how to cut" lives in the editor, and is described below.

### 4.1 The montage — where everything is decided

A rule knows how to group kills in a 10-second window. It does not know that
the music turns at 47 s, or that that rock deserves an extra second of run-up.
That is what the timeline is for, and on it the system decides nothing:

| Decision | Who makes it |
|---|---|
| which moment goes in | the user, choosing from the list of instants the analysis found — by clicking, or dragging the moment to the point of the ruler where it should go |
| where it goes in the video | the user, dragging the block's body; with the **magnet** on it snaps to the nearest beat (up to 0.12 s) |
| how long it lasts | the user, dragging the edges: the right one **stretches** (grows the tail), the left one **trims** (eats the start without moving what is framed) |
| where the play falls inside the cut | by default at **70%** of the block — leaving run-up before and the impact near the end, which is where it works. Adjustable block by block, and **marked inside the block**: it is the play that aligns with the percussion, not the cut's edge |
| what shows in the empty spaces | **a black screen, with the music playing** |
| where the music comes in | the user, putting the music block where it should start. The ruler is the **video's** time: instant zero is its first frame, and the music lives within that scale |

> There used to be an automatic rule here that did not survive use: the first
> block moved the music's entry to the cursor by itself, to avoid black before
> it. The effect was the video starting with the music already at 0:02,
> throwing away its beginning — and the screen announcing that as if it had
> been asked for. Empty space at the start is a choice visible on the ruler and
> counted in the summary; retiming the music behind the user's back is not.
> (The music's entry, today, is where its block starts: the whole ruler is
> video time.)

**The music is a block, and it comes from the library.** There was a
continuous track that played under everything and could not be cut; it is gone.
Today the music comes in through the media library, like video and images, and
goes onto an **audio layer**, where it is a clip like any other: it cuts, moves,
trims and duplicates. A layer either draws or plays — never both — and the
server refuses content on the wrong kind of layer. The magnet follows the grid
of the music playing under the playhead: two tracks in one video are two
tempos.

Old montages are not migrated in the database: `track_id` and `music_start_s`
are still valid input and become, **on read**, a block that starts where the
track came in and covers the whole video.

| Decision | Who makes it |
|---|---|
| which music plays, and in which part of the video | the user, dragging from the library to the ruler or clicking to put it at the playhead |
| from which point of the music the block comes | the user, through the "music range" — and the beat grid is measured from there |
| what is heard where there is no block | the match audio, at the level of `game_volume`: silence is the absence of a block |

Stretching and trimming do not reframe the content: the cut's start only moves
when it is the left edge that moves, and by the same amount. If the picture
reframed on every pixel of the drag, it would slide under the finger. Framing
at 70% is a **creation** guess; after that, it is the framing control that
changes it, not the duration.

### 4.2 The moment thumbnails

The editor's sidebar shows each moment with a frame of the match — choosing
among thirty kills without a picture is choosing among thirty identical clocks.
The `thumbs` service extracts them, listening to the end of planning through a
*consumer group* of its own: it receives the same notice as the planner and
works in parallel, without holding the job at `ready`. If the thumbnails are
slow, or never come, everything else works the same.

There is no table or column for them: each frame's key comes from its instant
(`frame_key(job_id, t)`), so whoever writes and whoever reads get there on
their own. One more thumbnail is not one more migration.

`-ss` **before** the input makes ffmpeg jump to the nearest keyframe instead of
decoding from the start: a frame comes out in tens of milliseconds, and that is
why extracting one at a time is cheaper than a single pass over the whole
video — the opposite of what holds for the analysis crops, where the single
decode is the point.

### 4.3 The montage is not lost

Half an hour of fitting to the beat used to vanish on an F5 — the montage only
existed in the tab's memory. Now it belongs to the **job**, and there are
**several**: each in the `montages` table, with a name, because the 30 s cut
for Shorts and the long montage are different jobs over the same material. The
app saves by itself a second and a half after the last change (a whole drag
becomes a single save), and the screen recovers the most recent one when
opening.

Each montage is a `Timeline` without the requirement of being ready: it accepts
zero cuts, because it exists from before the first block goes in. Each block,
on the other hand, is validated — storing garbage now would mean handing
garbage back on the next opening.

Rendering the video does **not** delete the montage: after rendering, the
normal thing is to want to adjust and render again, and losing the work at that
point would be the same damage. What rendering does is take a **snapshot**
(`montage_versions`) — what came out was that, and it is what makes the
history useful without storing a state per autosave.

The job's `draft` column, which held the single montage, is read one last time:
on the first opening it becomes the first named montage and is emptied. The
code that reads is what knows how to convert the old format.

### 4.4 The monitor

The screen has a preview, and it **renders nothing**. It opens the original
recording (`GET /api/jobs/{id}/video`, with `Range`) and seeks inside it to the
instant the playhead asks for: if it is over a block that comes from minute 3
of the match, it is at minute 3 that the recording is positioned. Where there
is no block, a black screen — the same the server will render there.

Really rendering on every adjustment would cost a full trip through ffmpeg per
drag. Seeking inside the file that already exists is instant, and the
computation that translates "video time" into "recording instant" is the same
on both sides: `sourceAt()` in the app is the read version of the `plan()` the
server uses to cut.

The browser's video element sometimes **dies** — a half-gigabyte recording
delivered via `Range`, with dozens of seeks per second while dragging, brings
it down. The monitor used to stay black until the page was reloaded, and
reloading cost the montage. Today it is watched: on detecting the error, the
player is reopened at the same point, up to four times, and only then does the
screen offer a retry button. Seeks are also serialised — one at a time, 120 ms
apart — because overlapping seeks were exactly what brought it down.

What the monitor does **not** guarantee is frame sync with the music during
playback: they are two independent media elements, and the splice between
blocks is done by seeking. Within a block the picture runs on its own. The
exact cut is the final file's.

The black is the decision that carries the rest. Splicing the blocks to cover
the gap would save an encode and **move every following cut** — each would come
out away from where the user fitted it. The screen's promise is that a block
lands at the point of the music where it was put; empty space is a choice, not
a leftover.

For the same reason, a cut that runs past the end of the recording is
**trimmed** and what is left of its slot becomes black, instead of the video
shrinking. And when the magnet acts, it snaps by whichever side is closer to a
beat — if it is the block's end that is a hair from the percussion, the end
rules: in a montage, what is heard is the scene change.

The computations live in `owcore/timeline.py` (server) and
`frontend/lib/montage.dart` (app), apart from ffmpeg and widgets precisely
because they have right answers and can be tested on their own. The magnet
exists on both sides on purpose: the app snaps while the user drags, and
whoever checks afterwards must arrive at the same number.

## 5. REST contract

| Route | What it does |
|---|---|
| `POST /api/jobs` | multipart with `video` and `params` (JSON). Just the recording — no music |
| `GET /api/jobs/{id}` | analysis state, **events**, saved montages, library and the history of **requests** with each one's clips |
| `POST /api/jobs/{id}/tracks` | multipart with `audio`. Has the system listen to a music track: it comes back right away, `pending`, and the analysis (duration, BPM, beats, waveform) runs in the worker |
| `GET /api/tracks/{id}` | the analysed music — it is what the montage screen draws |
| `GET /api/tracks/{id}/audio` | the file itself, with `Range`, for the app's player to play and seek |
| `GET /api/jobs/{id}/video` | the original recording, with `Range` — it is what the montage screen's monitor shows |
| `GET /api/jobs/{id}/frame?t=` | that instant's frame, for the sidebar. 404 = not extracted yet |
| `POST /api/jobs/{id}/frames` | asks to extract the missing ones (new jobs already come with them) |
| `GET /api/jobs/{id}/montages` | this match's montages, from most recent to oldest |
| `POST /api/jobs/{id}/montages` | starts a montage, empty or with given content |
| `PUT /api/jobs/{id}/montages/{mid}` | stores the montage and/or renames it (the app calls it by itself) |
| `POST /api/jobs/{id}/montages/{mid}/duplicate` | a copy, without the original's history |
| `DELETE /api/jobs/{id}/montages/{mid}` | deletes the montage and its versions |
| `GET/POST /api/jobs/{id}/montages/{mid}/versions` | the history, and marking a snapshot. `409` = nothing changed since the last one |
| `POST .../versions/{vid}/restore` | goes back to a snapshot; the current one becomes a snapshot first |
| `GET/POST /api/presets`, `PUT/DELETE /api/presets/{id}` | the presets. They belong to no match |
| `PUT /api/jobs/{id}/draft` | **legacy**: writes to the most recent montage |
| `DELETE /api/jobs/{id}/draft` | **legacy**: discards the match's montages |
| `DELETE /api/tracks/{id}` | removes the music from the job; videos already rendered with it stay |
| `POST /api/jobs/{id}/renders` | `timelines` (JSON) with the montages to render. It carries no file: the music was already uploaded to the library. Requires the job at `ready` |
| `GET /api/renders/{id}` | a request's progress and clips |
| `DELETE /api/renders/{id}` | deletes the request and its videos; the saved montage stays |
| `GET /api/jobs/{id}/cuts.zip` | the whole match's package, all requests |

## 6. Frontend (Flutter, mobile-first, runs on the web)

- `/` list of jobs with live status (polling)
- `/new` choose the recording — just that: no music and no settings
- `/job/:id` timeline of events, detector report, request history, player and
  downloads. The main action is **opening the editor**
- **montage screen**: an editor layout — a shelf of moments with thumbnails in
  the sidebar, a resizable monitor on top, the music drawn (waveform + beats)
  and playing, and the blocks positioned over it — dragging the body moves,
  the edges stretch and trim, the magnet snaps to the beat. It is **the** path:
  all it takes is the analysis having found moments

> The generation screen (`generate_screen.dart`) and the music range picker
> (`music_window.dart`) were removed along with the proposals.

## 7. Build steps

Backend and frontend are separate projects, each in its own repository:

1. `packages/owcore` — config, models, db, bus, storage, ffmpeg, profiles, worker base
2. `services/gateway` — REST + upload
3. `services/preprocessor` — ROI crops + audio
4. `services/detector_*` — computer vision, one per *question about the screen*. One region can answer two (`detector_kills` reads kills and critical hits on the same crosshair) and one question can come from two regions (`detector_ults` reads the footer button and the killfeed). What does **not** exist is one service per ability: `detector_banner` and `detector_killfeed` tell each one apart by its icon
5. `services/planner` — closes phase 1: waits for all detectors, crosses what none sees alone (`ULT_NEGATED`) and stores it; puts the job at `ready` and asks for the thumbnails
6. `services/beats` — listens to what comes into the editor's library (music, clip, image); `services/editor` — cuts and joins the montage (phase 2)
7. `packages/owcore/owcore/timeline.py` — the timeline becomes a list of pieces to cut, black gaps included
8. `tools/make_sample.py` — synthetic video generator (allows testing everything without real gameplay)
9. `tests/` — unit + end-to-end on the synthetic video
10. `ow-automatic-editor-frontend` — the Flutter app
11. `docker-compose.yml` at the root of this repository — starts everything:
    builds the images from this repository and mounts the frontend's
    `build/web` into the gateway
