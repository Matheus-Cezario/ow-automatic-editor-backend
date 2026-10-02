# Templates

PNG crops of Overwatch HUD icons, used as references for template matching.
They are **game assets**, so they are not shipped with the repository.

```
templates/
├── kills/      kill skull (optional, cropped by hand)
├── ults/       enemy ultimate icons in the killfeed (optional, by hand)
└── abilities/  official icons for ALL abilities — downloaded
```

## `abilities/` — downloaded, not cropped

```bash
python tools/fetch_ability_icons.py
```

That is ~270 files in `abilities/<hero>/<ability>.png`, one per ability of
every hero in the game. The list comes from Blizzard's official heroes page
(via the [OverFast API](https://overfast-api.tekrop.fr)), so **a new hero comes
in by running the command again** — there is no list written in the repository
to go stale with every patch.

Cropping 270 icons by hand from your own recording is not reasonable, which is
why this folder is the exception. Who uses these icons:

* the **ultimates detector**, to say which hero's ultimate the player used (the
  black drawing inside the footer button's white disc);
* the **killfeed detector**, to say which ability each kill was made with (the
  drawing in the small box between the two coloured plates).

Both compare the **mark** — the drawing's pixels, cut out from the background,
framed in a square and normalised in size. That is why the same file serves the
two ways the game draws the icon: black on a white disc (ultimate) and white on
a dark box (regular ability).

The file is stored as a **black mark on a white background**, which is what
`IconBank` expects. Blizzard's icon comes white on a transparent background —
the drawing lives in the alpha channel — and the downloader converts it.

## `kills/` and `ults/` — cropped from your recording

```bash
# generates images of the regions from your recording
python tools/calibrate.py preview --video match.mp4 --at 30 90 150
```

Open `data/calib/roi_*.png`, crop the icon tightly (no scenery margin around
it) and save it as `templates/ults/hero_name.png`. The file name becomes the
label of the detected event.

Ideal size: up to ~96px wide. Larger images are scaled down automatically.

## The system works without templates

- **Kills**, **critical hits** and **survival** use no template at all.
- The **player's ultimate** is still detected without `abilities/`: the footer
  button discharging is the event, and the icons only say whose it was.
- **Enemy ultimates** still come out of the audio peak; what is off without
  `ults/` is only recognising which ultimate showed up in the killfeed. The
  service says so in the log instead of pretending it detected something.
- **Ability kills** stay off without `abilities/` — on purpose. A kill without
  knowing what it was made with is already what the crosshair detector
  reports; repeating it here would only duplicate the event.
