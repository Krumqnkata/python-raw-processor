# RAW Studio 3 — validation

## Automated checks

- Python 3.12, Linux, actual LibRaw/OpenCV/Pillow, Tcl/Tk 8.6 under Xvfb.
- 51 engine/batch/editor tests and 19 actual GUI tests: 70 checks in total.
- Synthetic Bayer DNG is generated and genuinely decoded, rather than mocking RAW input.
- Standalone/modular output equivalence, source synchronization and compile checks.
- Actual dark/light screenshots captured and inspected; scrollable tab backgrounds
  explicitly follow appearance changes.

## Behaviour covered

- Linear decoding, exposure/CLAHE, colour and monochrome, 16-bit output precision.
- Separate shadows/highlights, normalized brush/gradient masks, crop/rotation/reflection.
- Lens identity/correction/cancellation, matching own camera/lens profiles.
- Per-photo settings, presets preserving geometry, clipboard, Undo/Redo branches.
- Atomic projects restoring masks, ratings, inclusion, histories and selected photo.
- Exact preview dimensions, export resizing/aspect ratio, file naming validation.
- Selected EXIF fields and normalized orientation, PNG16 unchanged by metadata insertion.
- Corrupt RAW continuation, filename collisions and RAW original preservation.
- Cancel/pause boundaries, resumed exports skipping completed unchanged files.
- Watcher excludes initial files and waits for a stable copy before auto-export.
- GUI heartbeat during a held worker; Tk event handlers remain on the main thread.
- GUI neutral-patch eyedropper, brush/gradient/crop, reversed crop drag, split comparison and thumbnails.
- 100% preview accounts for display scaling; exact rendering retains full image pixels.
- Native Tk images are detached on clear/switch, preventing deleted pyimage references.
- GUI project roundtrip, real threaded watcher, paused export and journal restoration.
- Invalid export values show validation messages and preserve the selected photo.

## CI

`.github/workflows/tests.yml` runs all 51 backend/editor checks on Windows,
Linux and macOS with Python 3.12. It also runs all 19 GUI checks on Windows and
Linux, each in a separate process to avoid Tcl interpreter teardown interference.

## Scope

Real CR2/NEF/ARW/CR3 camera samples were not supplied. Camera-specific decoding,
colour intent and visual lens calibration still depend on the user's files.
Lens corrections use manual coefficients/own profiles; no factory profile database
is bundled. Profiles match camera/lens EXIF strings, not zoom/aperture calibration.
Masks are evaluated after geometry; geometry changes alter their image-relative placement.
PNG16 metadata embedding is tested without pixel requantization. NLM still uses an
8-bit denoised residual, as required by the OpenCV colour NLM API.

## Workflow and Windows bundle

- Easy mode automatic import, mode switches preserving edits, ready export profiles.
- Export requested while a preview worker is held starts after cooperative cancellation.
- Native TkDnD loads; dropped file/folder paths with spaces are scanned in a worker.
- Ctrl/Shift multi-selection, export inclusion and clickable star ratings.
- Recovery roundtrip preserves masks, history and the original project path.
- Shutdown queues the latest immutable snapshot behind an in-flight recovery write.
- Inline export validation, retry only failed sources without duplicating successful files.
- Exact numeric entry and compact 900x640 layout.
- Preview decode cache invalidates on source changes; full exports decode afresh.
- Windows workflow builds PyInstaller/ Inno Setup packages and tests portable export,
  installed export and uninstall on an actual Windows runner.
