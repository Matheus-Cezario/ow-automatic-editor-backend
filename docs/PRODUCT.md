# OW Editor

> **Overview of the whole product.** In Git there are **two repositories**:
> this one is the backend's, and the Flutter app lives in
> `ow-automatic-editor-frontend`. The `docker-compose.yml` that starts
> everything lives here, at the root of this repository.

Upload the recording of an Overwatch 2 match, see what the system managed to
pick out and make the video: **assemble it yourself** — listening to the music
in the app and putting each kill, dart or rock at the point of it you want,
for as long as you want.

**100% free software**: FastAPI, Redis, S3 (RustFS), PostgreSQL, OpenCV,
ffmpeg, librosa, Flutter. No paid service, no external API, nothing that needs
an account.

---

## Two projects, two repositories

```
projects/
├── ow-automatic-editor-backend/    ← THIS repository
│   ├── docker-compose.yml              starts everything
│   ├── docs/                           this document, PLAN.md and V2.md
│   └── README.md                       how to run and test the backend
└── ow-automatic-editor-frontend/   Flutter app (mobile-first, runs on the web)
```

Each one is developed and tested on its own. The only thing linking them is
the gateway's REST contract (`/api/...`), documented automatically at
<http://localhost:8000/docs> when the backend is up — and it is that
independence that lets each have its own repository.

To run everything together, the two repositories must sit side by side in the
same folder: the compose looks for the compiled app at
`../ow-automatic-editor-frontend/build/web`.

---

## Starting everything

From the backend's root, where the compose is:

```bash
# optional: compile the app so the gateway serves it on the same origin
(cd ../ow-automatic-editor-frontend && flutter build web --dart-define=API_BASE=)

docker compose up --build
```

- App + API → <http://localhost:8000>
- S3 console → <http://localhost:9001> (`minioadmin` / `minioadmin`)

The compose builds the images from this repository and mounts the app's
`build/web` into the gateway. If you did not compile the app, the mounted
folder is empty and the gateway serves only the API — nothing breaks.

## Developing both in parallel

```bash
# terminal 1 — backend without any Docker (disk queue + SQLite)
python tools/dev.py

# terminal 2 — app with hot reload, pointing at the backend above
cd ../ow-automatic-editor-frontend && flutter run -d chrome
```

In development the app uses `API_BASE=http://localhost:8000` (the default) and
the gateway already allows CORS. In production, compiled with an empty
`API_BASE`, it calls the API on a relative path and needs no CORS.

---

## Two phases

The system is a **video editor with automatic event detection**. It renders
nothing on its own: it **analyses** and then **waits**:

```
1. you upload the recording   → the system watches and notes what happened
2. you open the editor        → the moments are on the shelf, with a frame of
                                each one
3. you assemble and render    → the video comes out exactly as assembled
   ↑__________________________________________|
   repeat as often as you like: using a moment does not use it up
```

> **What was removed.** Until Phase 11 there was a second path: the system
> applied rules to the events ("three kills within 10s make a kill streak") and
> offered a list of ready-made videos to pick from — all it took was giving
> each one a music track. That path was discontinued.
>
> It solved the wrong problem. What an Overwatch editor lacks is not someone to
> cut for them: it is an editor that **knows what happened in the video**. The
> rules made mediocre guesses about editing and, at the same time, hid moments
> the detector had found — headshots and ability kills became proposals and did
> not show up on the editor's shelf. The edge was in the detection, and it now
> goes entirely to whoever edits.

### Step 1's wait has to look like work

Analysing 11 minutes of recording takes ~8 minutes, and three quarters of that
is a single thing: cropping the HUD regions. The old bar gave that stage the
15% to 25% band — it sat on the same number for six minutes, and standing still
is how anyone reads "frozen". That was exactly the complaint that came up in
real use.

Two changes, and neither speeds anything up:

* ffmpeg is now **asked** how far it has got (`-progress`), and the video
  download counts bytes. The bar now moves all the time;
* the bar's slices became proportional to each stage's **time**, not to the
  number of stages — download 0–17%, cropping 17–90%, audio up to 94%,
  detection the rest. The numbers come from timing a real match.

Only with both can the screen say **how much is left**, and the simplest
computation there is will do: what has been done, at the speed it was done.
Measured on an 11-min match, it said "~6.5 min" and was off by 8 seconds. While
the bar lied about the shape of the work, no estimate would hold up.

### The montage, from the inside

```
a. you bring the music into the library → the system listens and returns the
   waveform and the beats
b. you put the music on the ruler and each moment wherever you want, as long as
   you want, watching on the monitor what will come out — dragging the block
   and its edges
c. the video comes out exactly like that
```

The split between the two phases exists because the two things move at
different paces. The analysis is expensive (it decodes the video, runs computer
vision) and its result does not change: the match's moments are what they are.
The choice, on the other hand, is cheap, personal and changeable — today you
want a montage with one music track, tomorrow the same montage with another.
Running the analysis again to change the music would be paying a lot for
nothing.

Two consequences follow that are worth saying out loud:

- **each video has its own music.** The track is a block on the montage's
  ruler, not a property of the match. Two montages of the same recording come
  out with different music, and the same montage changes music without
  re-analysing anything;
- **a video without music keeps the match's original audio.** It is not
  silence or some default: it is the game's sound, which is often half the fun
  of a stretch of play.

## What it finds

| Moment | How it is recognised |
|---|---|
| **Kill streak** | 3+ kills in a short window |
| **Solo wipe** | 4+ kills in a row without dying in between |
| **Narrow escape** | consecutive low-health episodes, survived |
| **Beat montage** | single kills, one per cut, on the music's beat |
| **Darts on target** | Ana's sleep dart hitting someone — its own montage, on the beat |
| **Rocks on target** | Sigma's Accretion stunning someone — its own montage, on the beat |
| **Negated ultimates** | an enemy ultimate followed by a kill within a few seconds |

> The same moment can appear in more than one video. The kill streak becomes a
> clip of its own **and** goes into the beat montage — each video is an
> independent montage, not a share of the material.

### Choosing the stretch of music

When rendering, for **each** video, you can say where the music comes in and
where it ends; the video is assembled to fit that stretch:

- **repeating stretches**: the video comes out with **exactly** the chosen
  duration, reusing moments when they run out before the music. The order is
  **shuffled**, and a moment only comes back after all of them have gone in —
  so the montage does not become the same sequence on a loop;
- **without repeating**: the video goes as far as the moments allow, **never
  going past** the chosen duration, in chronological order.

The start is fitted to the first beat from the requested point, so the first
cut falls on time instead of coming in mid-bar.

### More than one track, and music that can be cut

In a hand-made montage the music comes in through the **Library**, the same
place where video and images from outside the match come in — click to put it
at the playhead, or drag it to the point of the ruler where it should start.

On the ruler it is a **block**: it cuts, moves and trims like any video cut.
That is what lets you do what a background track never did:

- **switch music mid-video**, one track per stretch;
- **leave a stretch silent**, so the play's shot and shout come through alone;
- **put the chorus only on the turnaround**, and not from start to finish.

The magnet still works — with more than one track, it snaps to the beat **of
the music playing there**, which is the only grid that makes sense at that
point.

Montages made before this are not lost: the background track they had comes
back as a block that starts at the same point of the music and covers the
whole video. What you hear is the same; what changes is that you can now grab
its ends.

### Several montages of the same match

One match yields more than one video. The 30 s cut for Shorts and the long
montage are different jobs over the same material, and each has its own name —
you can switch between them at the top of the screen, duplicate one to
experiment without risking what is already good, and delete what did not work.

Each montage keeps a **history**. It is not undo, which only lasts while the
tab is open: these are markers. One is stored with every rendered video — what
came out was that — and you can mark one at any moment ("it was good like
this"). Going back to one of them does not delete what was in front: that also
becomes a marker first.

### Presets: the second match comes out ready

A preset does not store cuts — it stores the **way** of cutting. "Two seconds
per kill, fitted to the beat, with zoom and a counter" works for any match; a
list of cuts only works for that one.

Assemble a video the way you like it, save it as a preset, and the next match
comes out assembled in one click. What comes out is an ordinary montage: every
block still moves, trims and deletes like any other — the preset is a starting
point, not a mould you cannot leave.

The size can be in seconds or in **beats**. Beats are better: they survive a
song with a different tempo.

### Fitting the play to the beat

A cut is a stretch; the play — the kill, the dart, the rock — is an instant
inside it, and that is what needs to land on the beat. The cut starts earlier,
for run-up, so aligning by the edge leaves the impact late.

On the ruler, each block shows **where the play happens**. Put the playhead on
the beat, pick the block and align it (button on the block's panel, or the
**M** key): the play moves under the line, and the mark lights up to confirm.
If the neighbouring blocks do not allow that move, it is the stretch that
slides inside the block — the video keeps the same duration, and the screen
says what it did.

Dragging with the magnet on, the play also snaps to the beat: it competes with
the block's two edges, and whichever is closest wins.

### Writing on screen

Text goes on the ruler like any block and shows up **on the monitor**, at the
size and place it will come out. That is where you choose where it sits: drag
the line across the frame, and tap it to type. Size, colour and outline are on
the block's panel — the outline is not decoration, it is what makes white text
survive a bright scene.

A text can have **several lines**: Enter breaks one where you want it, and a
**box width** makes the words break by themselves inside it. The lines line up
on the left, the centre or the right; a **box** behind them (any of the
colours, with its opacity) and a **drop shadow** are on the same panel. The
box follows the text's entrance and exit — it fades, pops and slides with it.

**Stickers** are in the Library too: arrows, rings, a crosshair, a skull, a
crown, stars and a few more, in eight colours, each with a dark outline so it
reads over any scene. Clicking one puts it at the playhead, over the picture,
small and in the middle; from there it moves, grows, turns and fades like
any block. A picture you bring yourself (a hero icon, a logo) can be placed the
same way: **Whole** on its panel shows all of it, transparent around, instead
of filling the frame.

### Talking over it, and subtitles

The **microphone** button records a voice-over from the playhead while the
montage plays, so the words follow the picture; it lands on a **Voice** layer
exactly where the recording started. While it speaks, the music and the game
step back to the duck level and come back after.

**Subtitles** live in the text menu: write one at the playhead, import an
`.srt` or `.vtt`, or download the montage's subtitles as `.srt` for a platform
that shows its own captions.

### Choosing the output format

A montage has no format: it has cuts, layers and effects. The format is the
window you look at it through, chosen when rendering — which is why the same
work becomes a **16:9** for YouTube and a **9:16** for Shorts without any of it
changing.

You can choose size, frame rate and quality; export **just a range** (or just
what is selected, to check a splice without waiting for the whole video); and
add a **watermark** from any image in the library, in one of the four corners.

When the requested aspect ratio is not the recording's, there are two answers
and both are there: **fill**, which crops the sides — in a gameplay montage the
action is in the middle — or **fit**, which shows the whole frame and leaves
bars, for when what matters is in the corners.

Changing the music and re-exporting redoes the video: the sound is assembled
together with the picture, not on top of it. (There was a shortcut here, when
the music was a background track that could not be cut; it went away with it.)

### Taking the cuts to edit elsewhere

Each match offers a **zip with everything** — straight from the list, without
opening any video:

```
request_01/videos/01_custom.mp4               the finished videos
request_01/cuts/01_custom/01_00m43.7s.mp4     each cut, named after the
request_01/cuts/01_custom/02_01m22.2s.mp4     instant it came from
request_02/videos/01_custom.mp4               the same moment, other music
...
```

Each request has its own folder: rendering the same montage twice with
different music gives two videos, not one overwriting the other.

Each cut appears **once**, even when the montage repeated it. Inside the
player there is also the zip for just that montage, for when you only want it.

## How it works

```
analysis (once, automatic)
Flutter ──▶ gateway ──▶ preprocessor ──▶ detectors ──▶ planner ──▶ moments
             (API)      (1 decode,        (kills,       (closes and   + `ready`
                        N crops)          survival,      crosses)
                                          ults, sleep)
                                                            ⌛ waits for you

the editor's library (music, clip, image)
Flutter ──▶ gateway ──▶ beats ──▶ waveform + beats back to the app
             (API)     (listens to
                        the music)

rendering (on demand, repeatable)
Flutter ──▶ gateway ──▶ editor ──▶ clips
             (API)     (cuts,
                        layers,
                        music)
```

The key point: **the heavy video is decoded only once**. Tiny crops of the HUD
come out of that pass, at low FPS, one per detector. No detector opens the
original file — only the editor goes back to it, and only for the seconds that
matter. The music's rhythm is not part of the analysis: the music exists when
the user brings it into the library, and is listened to there. On a 640×360
recording at 30 fps, the kills detector receives a 102×64 crop at 12 fps:
**1.1% of the pixels**. The three crops added up, plus the audio, come to 21 MB
of a 100 MB video.

Architecture details: [`PLAN.md`](PLAN.md).

---

## What is verified, and what is not

Being straight about the limits.

**Calibrated against real gameplay.** The detectors were measured against 19
minutes of real matches (2 matches, 360p), not just against the tests'
synthetic video. That changed a lot:

| | before | after |
|---|---|---|
| Kill precision | ~17% | **~91%** |
| Kills in 19 min | 491 (almost all false) | 11 |
| "Escapes" in 19 min | 122 (the damage vignette firing) | 8 (later, deaths read as escapes were removed: see "The escape that ended in a death") |
| Deaths detected | 0 | 15 |

The **abilities announced in the footer** were calibrated the same way, on two
recordings, with each template trained only on the first half of its video:

| | recording | found | false |
|---|---|---|---|
| Ana's dart | 16 min of Ana | **11 of 11** | 0 |
| Sigma's rock | 11 min of Sigma | **23 of 23** (12 never seen) | 0 |

The OW2 footer stacks several banners with the same colour, shape and position
(`SAVED …`, `ORB OF HARMONY …`, `… STUNNED BY ACCRETION`); what tells them
apart is the **icon** on the left end, not the text — text changes with the
language, the icon does not. That is why it is **one detector for the whole
banner**, not one per ability: adding an ability is adding a template to the
profile.

Two things that only showed up when measuring the second recording:

- **the banner colour changes from match to match** — cyan in one, green in the
  other. Worse: the Sigma recording's bluish scenery fell in the cyan range,
  and adding both colours into a single mask glued the scenery to the green
  banner, which stopped existing as a rectangle. Searching **one colour at a
  time** recovered 5 of the 23 rocks;
- **each template separates at a different point.** The rock icon is full and
  contrasted, and matches high even with neighbouring banners; the dart's is
  made of thin strokes. A single threshold forced a choice between missing
  darts and accepting false rocks, so the threshold is **per ability** (0.85
  and 0.92).

The detector also found a rock in the *Ana* recording — checked by eye, it was
real: an allied Sigma landing an Accretion.

**The player's ultimate, critical hits and ability kills** were first
calibrated on short 2558×1438 recordings made to show exactly those HUD
elements, and then **checked on 27 minutes of full matches** (16 min of Ana and
11 min of Sigma). The two materials say different things, and it is the match
that rules: see "What the full match showed" just below.

| | material | result |
|---|---|---|
| Player's ultimate | 3 short clips + 27 min of matches | 4 uses in 27 min, with no false ones; the disc icon named hero and ability with 0.81–0.98 correlation, against 0.49 for the runner-up among 270 icons |
| Critical hit | 8 s of Ashe, scope with a magenta filter | found the red X; on a recording with kills and no headshot, zero false |
| Ability kill | 16 s of Domina + 27 min of matches | 7 of 11 kills in the 16 min of Ana, **with no false ones** and all named correctly; the 4 missed had the icon below the threshold |

### What the full match showed

The short clips passed both detectors; the full match failed them. It is the
most expensive difference this project has measured, and worth writing down:

| | short clips | full match, before | after |
|---|---|---|---|
| Player's ultimate (Ana + Sigma) | 3/3 | 14 events, 4 true | 4 events, 4 true |
| Ability kill (Ana, 11 real) | 2/2 | 27 events, 11 real lines | 7 events, 7 true |

What the short clips did not contain:

- **nobody dies in them.** On death, OW2 shows the *kill cam*: a bright disc
  with the killer's face, surrounded by a ring — in the same place as the
  ultimate button and with the same shape. In an Ana match that became a
  "Roadhog ultimate"; in a Sigma one, Hanzo and Junkrat. What separates the two
  is the clock: in 27 minutes, every short stretch (0.2–0.6 s) was false and
  every real ultimate stayed charged for 2.8 s or more;
- **the killfeed has a single, isolated line.** In a match it stacks several,
  they slide when a new one arrives, and the icon crop fails for 3 to 5
  seconds in a row in the middle of a line's life. The same kill came out two,
  three, four times;
- **almost every kill is with a regular weapon**, and the gap between the
  plates still has the `>` to match something. With the threshold at 0.55,
  `dva/light_gun` and `baptiste/exo_boots` — a weapon and a passive, which do
  not even show up in the killfeed — became kills. At 0.65 that goes away.

The lesson, written for next time: **a clip recorded to show a HUD element
shows that element and nothing else.** It has no deaths, no full killfeed, no
match around it. It serves to find the region and check the matching; it does
not serve to calibrate any threshold.

Three things that only showed up when measuring:

- **deciding headshots by colour does not work.** Ashe's scope tints the whole
  screen magenta, and then *everything* falls in red's hue range. What does not
  move with the filter is the red channel's **dominance** over the other two:
  the marker gives ~107 and the tinted scenery, ~20. And the shape decides the
  rest: the four diagonals painted with the four straight directions clear —
  the kill skull, also red and also on the crosshair, fills all eight;
- **a killfeed line stays on screen for seconds**, so its presence marks no
  instant. What marks it is it **appearing**. Counting how many lines of each
  ability are on screen almost solves it, and was the first attempt — but it
  does not distinguish "the same line vanished and came back" from "a new line
  with the same ability appeared", and those two are exactly the two cases that
  matter. Today each line is **tracked** by its horizontal edges: the inner
  ones surround the icon and are already still on the first frame; the outer
  ones are the length of the names and tell one line from another. The height
  does not count — when a new kill arrives the whole stack slides, and an
  identity tied to it would switch lines right then;
- **discarding small pieces of the drawing looked like cheap cleanup** and is
  not: half the game's icons are made of loose parts, and cutting them changes
  the framing from one frame to the next. On a real recording that turned one
  kill into four.

### The escape that ended in a death

*d1, 2026-10-07.* Some survival cuts ended with the hero dying right at the end
of the clip. Measured on the same two full matches (16 min of Ana on PC, 12 of
Sigma on PS5, both 1080p), every one of those came from the same place: **the
death was read as health coming back.**

The detector assumed that on death the HUD moves to the teammate being
spectated, so death would be a single frame at zero followed by someone else's
full bar. In these recordings it does not: the HUD stays on the player's own
card, health 0, until the respawn — and the bar becomes a **dim track with no
ticks, with the scenery showing through it.** The reader measures the ticks
against the strongest step in the strip, so the scenery became the scale and
the dead bar read as half or nearly full. The low-health stretch then "ended
in a recovery" at the very instant of the death, and became an escape whose cut
put the death at 70% of the block.

What tells the dead bar apart, and it takes both:

| | lit bar | dead bar | scoreboard (Tab) |
|---|---|---|---|
| strength of the steps | 23–34 | 3–11 | 8–11 |
| regularity of the pitch | 0.75–0.92 | 0.1–0.36 | ~0.82 |

The scoreboard darkens the bar as much as dying does, but its ticks stay
regular; a first version that looked at strength alone turned every Tab into a
death. The strength's scale is the recording's own (an upper percentile of the
whole match), and a dead bar must last 2 s — every real one lasted 4 s or more.

A death is now also **one event per stretch**: the dead bar, the killcam and the
round's end used to come out as two to five "deaths" a few seconds apart for
the same death (54 events in the two matches, against 23 now).

| | escapes | the hero died in the cut |
|---|---|---|
| before | 9 | 3 (Sigma, 0–2 s after the escape) |
| after | **6** | **0** — all six checked by eye, alive and healed |

The cut's end needed no change: a block puts the escape at 70% of its length,
so its tail is under the 4 s (`safe_after_s`) the detector already checks for a
death — as long as the death is seen.

**Known limits:**

- a headshot that *kills* may go unnoticed. The kill skull is born ~0.1 s
  after the critical marker and covers the same diagonals; at 12 fps there is
  not always a frame left between the two. The kill is still detected — what is
  lost is the "headshot" label;
- **ability kills found 7 out of every 11**, and that trade-off was
  deliberate: the icon threshold sat where precision is 100%. A shelf with one
  moment fewer is better than one that offers a cut that is not what it says:
  whoever assembles trusts the label and does not go back to check the
  recording. Two things in that number were not the threshold's fault, and
  have changed since. The icon was compared as a black-and-white cut, and at
  killfeed size compression turns thin strokes grey, under the cut: on
  synthetic thin-stroke icons at 12-16 px that comparison named **none** of
  40 frames, and the comparison in grey, at the size the icon was seen, names
  them in most frames. And one frame above the threshold named a line, so a
  line whose icon never had a good frame was lost; now every frame of the
  line votes, and the ability has to win most of them -- which is also what
  keeps a gun kill's empty gap out. Measured on a 1080p Ana match (16 min,
  153 killfeed lines checked by eye, the 270 official icons): the old version
  named 11 ability kills, repeated 5 of them and named **two gun kills** as
  weapons; the new one names the same 11, repeats 2 and names no gun kill.
  The threshold sits at 0.80 on the new scale -- the right icons scored
  0.85-0.99 and the first wrong name appears at 0.76. Eight ability kills of
  that match are still missed by both: Ana's sleep dart comes out closer to
  another hero's icon at killfeed size, and some killfeed icons (a flexed arm,
  a crossbow) look like none of the official icons at all. The old crop window
  also held the whole `>` of most gun kills; it is now erased as the rightmost
  piece of the mark;
- **on that 1080p match the player's name was not read** (3 letters, in 27% of
  frames: the card's letters are 11 px tall and break into strokes), so no
  kill there would count as the player's -- in either version. Not fixed yet;
- **a line's plates are not stable on a real match**: the scenery behind the
  translucent HUD moves their edges by tens of pixels between frames, and one
  line still splits into 2-4 tracks. That is where the remaining repeats come
  from. Not fixed yet;
- **whose kill it was used to be decided by one reading of the name.** A frame
  where two letters of the killer's name touched read as a name of another
  length and gave the player's own kill to "someone else". Every reading at the
  plate's full width now counts, and one match is enough -- other players'
  names stay at 0.21 at most against a threshold of 0.40;
- **two kills by the same player on the same victim, with the same ability and
  within ~7 s of each other, count as one.** The two lines are identical in
  everything the detector uses to recognise them. It requires the victim to
  respawn and be killed again in the same place; it is rare, and the price of
  getting it wrong the other way — repeating a kill — is much higher;
- **the ultimate requires 2 s charged.** In the practice range charging is
  instant, so a clip recorded there may yield no event. In a match that does
  not happen, and it was precisely by letting two practice-range clips rule the
  calibration that 10 false ones got through before.

What was wrong: measuring "how much of the region is red" does not distinguish
the kill skull from the game's scenery or from the directional damage
indicator. The decision now uses **shape, position and size** — the skull is a
compact blob, centred on the crosshair, taking 5% to 14% of the region. That
last filter is what stops a 20-pixel red splash from becoming a kill. And low
health came from the red vignette on the edges, which is actually the *damage
taken* warning; today the system reads the health bar directly.

### If the video does not come out, the cuts do

The stretches are cut **before** they are joined, and the zip is closed at that
point. If joining or the music fails, the clip shows up as "video not rendered
— cuts available" and the download is still there. Material already cut is not
lost because of the next step.

**Manual montage verified end to end.** The test uploads the music through the
API, runs whoever listens to it, assembles two blocks with a 1 s gap between
them, asks for the render and **opens the output mp4**: it lasts the right 4 s
(the 3 s of cuts plus 1 s of black) and has an audio track. What cannot be
tested this way is the audio player inside the app — if the music does not play
in your browser, the montage is still possible through the drawn waveform and
beats, and the screen says so instead of freezing.

**Test suite**: about 250 in the backend and 330 in the frontend. They cover
both phases, the contracts between the microservices, the manual montage (the
gap that becomes black, the trimmed cut that does not move its neighbours, the
beat magnet), resilience (a failing detector does not bring the job down; an
unreadable music file fails alone) and the detectors' accuracy against the
synthetic video's ground truth. The "no music keeps the original audio" case is
checked on the generated file: the test downloads the mp4 and checks it has an
audio track.

**What is not solved:**

- **Enemy ultimates** need game icons in `templates/ults/` — they are
  Overwatch assets and are not in the repository. Without them the detector
  emits nothing, on purpose. The alternative audio-peak path exists but ships
  off: in a real match it cannot tell an ultimate voice line from gunfire and
  explosions, and there is no ground truth to calibrate it honestly. (The
  **player's** ultimate is another story: it comes from the footer button and
  depends on no asset.)
- **Ability names in English.** The label of an ability comes from the icon's
  file, which came from Blizzard: "Orisa: Energy Javelin". That is also the
  name shown on the game's hero screen.
- **Two chained ultimates count as one.** D.Va and Dmon release the second
  ultimate right after the first; when both fall in the same window — in the
  practice range, where charging is instant — only the last becomes an event.
  In a match, where recharging takes dozens of seconds, both come out
  separately.
- **`DEATH` is not exactly "death"**. The signal is health dropping to zero or
  the HUD disappearing, which covers death, kill cam, round change and hero
  selection. For the rules that is what matters (the player's streak was
  interrupted), and the app labels it as "Interruption" instead of promising
  more than it delivers.
- **A single reference material.** The thresholds were calibrated on one
  recording. Language, colour-blind mode, screen aspect ratios other than 16:9
  and game patches may require recalibrating — `tools/calibrate.py` exists for
  that.
- **Kill recall was not measured.** I know ~94% of what it points to is a real
  kill; I do not know how many it lets through, because that would require
  manually labelling the 19 minutes of killfeed.
