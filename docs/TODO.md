# OW Editor — To-do

Open work that is noted but not started. Each item has a code, like the
roadmap's (`s` sound, `w` workflow); `d` is detection. When one is done it
moves to the doc of the area it changed (`PRODUCT.md`, `V2.md`) and leaves
this list.

## d1 — Survival detection: the hero dies right after the cut ends

*Noted 2026-10-07.*

> "Some cuts of a survival end and, right at the end of the clip, the hero
> dies."

An escape (`ESCAPE`) is cut as a survival, but the clip can end a moment
before the death, so the montage sells as a survival what was really a death.

Where to look (`services/detector_survival/detect.py`):

- an escape counts as survived when no interruption (`DEATH`) falls between
  the start of the low-health episode and `safe_after_s` (4 s) after its end.
  A death later than that window, or one that is missed, leaves the escape
  standing;
- `DEATH` is read as health dropping to zero and coming back full on the next
  frame (spectating a teammate). A death that does not show that signature —
  the bar fading out, a killcam, a reading that does not reach the
  `death_frac` threshold — never becomes an interruption;
- the cut's end is decided in the montage (`PLAN.md` §4.1), independently of
  the survival window: the clip can end before `safe_after_s` has really been
  checked, or it can include the death at its tail.

Done when, on a real match, no escape cut shows the hero dying before the clip
ends — measured by eye like the killfeed was (`PRODUCT.md`, "What the full
match showed").
