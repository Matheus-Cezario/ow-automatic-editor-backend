# Fonts

The text fonts the editor offers. Each one is listed in `catalog.json` (the
id the montage stores, the display name, the file and its licence), and its
licence sits next to the file. All come from the Google Fonts repository
(github.com/google/fonts): ten under the SIL Open Font License 1.1, Permanent
Marker under Apache 2.0 — both allow embedding and redistribution.

The gateway serves them to the app (`/api/fonts`), so the editor's monitor
draws text with the same font the render uses. The system's DejaVu faces are
offered too, from `owcore.fonts`.
