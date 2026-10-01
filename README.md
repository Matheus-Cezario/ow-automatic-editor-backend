# OW Editor — backend

Python microservices that take a match recording, find the key moments in it
and — when the user asks — render the videos they assembled. It runs on its
own: the frontend is optional, and the API is browsable at
<http://localhost:8000/docs>.

> All commands below assume you are **at the root of this repository**.

**The system's documentation lives here:**

| Document | What it has |
|---|---|
| [`docs/PRODUCT.md`](docs/PRODUCT.md) | the whole product: the two phases, how a video comes out, and what is verified and what is not |
| [`docs/PLAN.md`](docs/PLAN.md) | architecture, detection, manual montage and the REST contract |
| [`docs/V2.md`](docs/V2.md) | the full editor, in phases — **all done**, each with what was left out and why |

The Flutter app is a separate repository. On disk the two sit side by side —
`ow-automatic-editor-backend` and `ow-automatic-editor-frontend` — and that is
how the `docker-compose.yml` here finds the compiled app.

---

## Running

### Without Docker (just Python + ffmpeg)

```bash
python -m venv .venv && .venv/Scripts/activate    # Linux/macOS: source .venv/bin/activate
pip install -r requirements-dev.txt

python tools/dev.py
```

Starts the processes using a disk queue, folder storage and SQLite — no
external server. Each microservice is still a separate process talking over the
bus; only the implementation behind the interfaces changes.

### With Docker

The `docker-compose.yml` sits at the root of this repository:

```bash
docker compose up --build
```

- App + API → <http://localhost:8000>
- S3 console → <http://localhost:9001> (`minioadmin` / `minioadmin`)

The gateway serves the compiled Flutter app from
`../ow-automatic-editor-frontend/build/web` (another folder can be given with
`OW_FRONTEND_WEB`). Without a build the folder is empty and the gateway serves
only the API.

---

## Testing

```bash
# generates the synthetic video the accuracy tests use (once)
python tools/make_sample.py --out data/sample/match.mp4 \
    --music data/sample/music.wav --ult-templates data/sample/ult_templates \
    --ability-icons data/sample/ability_icons

pytest tests/ -q
```

About 250 tests in five layers:

| File | What it covers |
|---|---|
| `test_rules.py` | the rule that crosses two detectors' events (negated ultimates) — a pure function, without touching video |
| `test_infra.py` | bus (fan-out across groups, competition within a group), storage, vision primitives, colour tagging of the crops |
| `test_detectors.py` | each detector's accuracy against the synthetic video's ground truth: the two footer abilities not being confused, critical hits not coming out of the kill skull, the player's ultimate being marked at the instant it **is used** and not while the button is charged, and a killfeed line counting as **one** kill however long it stays on screen |
| `test_pipeline.py` | both phases going through every service — this is what checks the messages' *contracts*, plus the original audio and the cuts zip |
| `test_timeline.py` | manual montage and the V2 editor: the timeline maths (a gap becomes black, a trimmed cut does not move its neighbour), layers, effects, text and transitions checked **on the pixels** of the output mp4, export (size, fps and range via `ffprobe`; `cover` versus `contain`), several named montages with history and presets, music on the ruler (an audio layer, trimmed and positioned blocks, the silence between them) and converting the old continuous track into a block on read, plus the `Range` deliveries the app uses to play the music and show the preview |

Tests that depend on the synthetic video skip themselves if it does not exist,
saying how to generate it.

---

## Architecture

The system has **two phases**, and they do not mix.

**Phase 1 — analysis.** Runs once per recording, on its own, and ends at
`ready` with the match's timeline — what happened, and when:

```
gateway ──▶ preprocessor ──▶ detector_kills    ──┐
 (API)      (1 decode,       detector_survival   │
            N crops)         detector_ults       ├──▶ planner ──▶ events
                             detector_banner     │  (closes and   + `ready`
                             detector_killfeed ──┘   crosses)
```

**Phase 2 — editing.** The user assembles on the timeline and asks for a
render, as many times as they like:

```
gateway ──▶ editor ──▶ clips
 (API)     (cuts,
            layers,
            music)
```

The system **does not propose ready-made videos**. It used to: the planner
applied rules (kill streak, "solo wipe", beat montage) and the app offered the
list to pick from. That is gone. What the analysis delivers is the moments, and
what is done with them belongs to the editor.

A bus with *consumer groups*: the preprocessor publishes once and each detector
receives its own message. The same crop can go to two detectors — the killfeed
strip goes to `ults` and to `killfeed` — and that costs no extra decoding: the
crop is made once and the same blob is addressed to both. It is the questions
that differ. The planner only acts when every expected detector has reported —
and the transition to `ready` is claimed atomically in the database, so several
reports arriving together do not cross the events twice.

It is in that closing step that the events **no detector sees alone** are born:
a negated ultimate is an enemy ultimate followed by a kill, and only whoever has
both kinds in the same list can see it.

`thumbs` extracts one frame per moment for the editor's sidebar. It listens to
the end of the analysis in its own *consumer group* — it receives the same
notice and works in parallel, without holding the job at `ready`. Each frame's
key comes from its instant, so there is no table or column for them.

A detector that fails **does not bring the job down**: it records the error in
its report and releases the end of the analysis, which delivers what the others
found.

`beats` **is not a detector**: it does not look at the match and takes no part
in the analysis. It listens to what the user brings into the editor's library —
music, clip or image — and returns what the montage screen needs to assemble
with it: duration, BPM, beats and an envelope reduced to ~40 points per second
for music; thumbnail, dimensions and a proxy for video and image.

The music arrives before any video exists: it is by listening to it, with the
beats and the waveform on screen, that one decides where each cut falls. A video
with no music on the ruler comes out with the match's **original audio**.

### The two modes

Each infrastructure dependency has two implementations behind the same
interface, chosen by `OW_MODE`:

| | `local` (default) | `docker` |
|---|---|---|
| Queue | files in `data/bus` | Redis Streams |
| Storage | folder `data/blobs` | S3 (RustFS in the compose) |
| Database | SQLite | PostgreSQL |

### Layout

```
packages/owcore/   shared core, installed in each service
  bus.py           bus (Redis Streams | disk queue)
  storage.py       blobs (S3 | folder)
  db.py            SQLAlchemy (Postgres | SQLite)
  models.py        domain + tables + bus messages
  rules.py         rules that cross detectors (pure function)
  timeline.py      manual montage: blocks → pieces to cut (pure function)
  vision.py        computer vision primitives — includes the ability icon
                   bank and the glyph that feeds it
  ffmpeg.py        cropping, cutting, concatenation, music
  audio.py         WAV reading and waveform (music and match)
  compose.py       layered timeline -> filter graph (pure function)
  textfx.py        text -> `drawtext`, with the escaping the filtergraph needs
  fonts.py         where the font is; fails loudly when there is none
  detector.py      base of the detector microservices
  worker.py        consume loop, ack, clean shutdown
services/          one directory per microservice
config/profiles/   HUD positions and colours
templates/         reference icons — see templates/README.md
tools/             sample generator, calibration, local runner,
                   ability icon downloader
tests/
```

---

## Calibration — read before using real gameplay

The detectors look for HUD elements at positions and colours defined in
`config/profiles/ow2_default.json`. The profile in the repository was
calibrated against real Overwatch 2 gameplay at 16:9 (measured at 360p; the OW2
HUD scales with resolution, so the normalised values hold from 360p to 4K).

Even so, **language, colour-blind mode, aspect ratios other than 16:9 and game
patches change positions and colours**. If the system finds nothing in your
recording, start here.

> **A note on colour.** The preprocessor writes the crops with explicit BT.601
> colour tags, and that is not cosmetic: the detectors decide by saturation, and
> the YUV→RGB matrix changes exactly the saturation. Without those tags, the
> same file was read with saturation 231 on the host and 205 inside the
> container — and the detector found 20 kills in one place and 10 in the other,
> with the same code. `tests/test_infra.py` pins that tagging.

```bash
# 1. Are the regions in the right place? Draws the rectangles over real frames.
python tools/calibrate.py preview --video match.mp4 --at 30 90 150

# 2. Which threshold? Measures the region frame by frame and suggests numbers.
python tools/calibrate.py scan --video match.mp4 --roi kills
```

Then copy the profile, adjust it and run with `OW_PROFILE=my_profile`.

### The ability icons

Two detectors say **which** ability showed up by comparing the HUD drawing with
its official icon: the ultimates one (the footer button's disc) and the
killfeed one (the small box between the two plates). The icons are game assets
and are not shipped with the repository — download them once:

```bash
python tools/fetch_ability_icons.py
```

That is ~270 files, one per ability of each hero, in
`templates/abilities/<hero>/<ability>.png`. The list comes from Blizzard's
official heroes page (via the OverFast API), so a new hero comes in by running
the command again — there is no list written in the repository to go stale.

Without them the system **keeps working**: the player's ultimate is still
detected, just without saying whose it was; the killfeed detector stays quiet,
because a kill without knowing which ability made it is what the crosshair
detector already reports.

---

## Configuration

Everything through environment variables prefixed `OW_` (see `.env.example`):

| Variable | Default | What for |
|---|---|---|
| `OW_MODE` | `local` | `local` or `docker` |
| `OW_PROFILE` | `ow2_default` | HUD profile |
| `OW_DATA_DIR` | `./data` | uploads, crops, clips, queue |
| `OW_WEB_DIR` | `../frontend/build/web` | the compiled Flutter app, if any |
| `OW_DATABASE_URL` | derived from the mode | SQLite or Postgres |
| `OW_REDIS_URL` | `redis://localhost:6379/0` | bus in docker mode |
| `OW_S3_*` | local S3 | storage in docker mode |
| `OW_DETECTOR_TIMEOUT_S` | `900` | when the planner gives up on a silent detector |
| `OW_STREAM_MAXLEN` | `10000` | cap on messages per Redis stream. Without it Redis RAM grows with the number of matches already processed: `XACK` does not delete the entry |
| `OW_BUS_RETENTION_S` | `3600` | how long a message already finished by every group stays in the disk queue before being swept (local mode) |
| `OW_DB_POOL_SIZE` / `OW_DB_MAX_OVERFLOW` | `2` / `3` | connections per process. The default suits a single-threaded worker; the gateway raises it to `10`/`10` in the compose |

### Memory

The system has a **floor** of about 700 MB just to exist: it is eleven Python
processes, and the five detectors each load OpenCV and numpy (~80 MB of RSS
before processing anything). That is the price of splitting into microservices
— there is no leak there, and really reducing it would mean merging detectors
into a single process.

What is **not** floor, and was therefore fixed:

* the audio waveform used to be computed by loading the whole WAV into memory
  three times (~370 MB peak on a 20-min match, to produce 6000 numbers). Today
  it is read in blocks, with a fixed ceiling of ~11 MB regardless of duration;
* the match's `cuts.zip` package used to be built on disk and then read
  **whole** into the gateway's memory before going out. Today it is streamed in
  chunks;
* the Redis streams were not trimmed and the disk queue deleted nothing;
* ffmpeg's error output was accumulated without a cap, only to use 25 lines in
  the end.
