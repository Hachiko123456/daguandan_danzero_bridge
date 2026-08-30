# Support diagnostics contract v1

This document freezes the diagnostic contract introduced by phase 1A and the
offline support-bundle schema API introduced alongside it.  Live evidence
capture and a user-facing export command are not connected in phase 1A.

## Runtime identity

Session manifests receive a `runtime_identity` object when they are first
created.  Its schema is `guandan.runtime-identity/1` and it contains a
process-stable `run_id`, explicit build status, a build identifier when one is
available, and the compatibility keys `implementation_fingerprint` and
`executable_path`.

`build_status` has three values:

- `identified`: a readable `build_manifest.json` supplied a build ID;
- `unidentified`: no build manifest was present;
- `invalid`: the build manifest was unreadable or structurally invalid.

Paths in runtime identity are logical or basename-only.  They must not expose a
Windows username or an absolute user path.

## Startup run directory

Each process uses one run directory:

```text
<diagnostics-root>/runs/<run_id>/
```

The root is selected from `DAGUANDAN_DIAGNOSTICS_ROOT`, then
`%LOCALAPPDATA%/DaguandanAssistant/diagnostics`, then a temporary-directory
fallback.  The directory can contain:

- `startup.jsonl`: `guandan.startup-event/1` lifecycle and exception records;
- `startup.log`: early startup output, including stdout/stderr when a no-console
  executable exposes neither;
- `runtime_identity.json`: the atomically written, sanitized process/build
  identity used to correlate pre-GUI failures;
- `exceptions.log`: main-thread and worker-thread uncaught tracebacks;
- `faulthandler.log`: Python/native fatal-error diagnostics;
- `doctor.json`: the default doctor result.

Startup diagnostic initialization is idempotent and fail-open.

## Doctor result

`DaguandanAssistant.exe --doctor` (or `python run.py --doctor`) emits and writes
one `guandan.doctor/1` JSON object.  `--doctor-output PATH` selects an explicit
destination instead of the run directory's default `doctor.json`.

Every element of `checks` has exactly these fields:

```json
{
  "id": "RESOURCE-TEMPLATES-JSON",
  "status": "PASS",
  "summary": "JSON resource is readable",
  "evidence": {},
  "duration_ms": 1.25
}
```

Check IDs are stable machine-readable identifiers.  Status is `PASS`, `WARN`,
or `FAIL`.  Exit codes are:

- `0`: no check has status `FAIL`;
- `2`: at least one check has status `FAIL`;
- `3`: doctor itself crashed or its requested report could not be written.

The doctor runs before DPI setup, QApplication creation, capture, recognition,
or model construction.  Missing/running WeChat is not checked and therefore is
not a failure in phase 1A.

## Phase 1A capability boundary

The doctor capability map deliberately reports the following as unavailable:

```json
{
  "window_probe": false,
  "capture_probe": false,
  "recognition_probe": false,
  "support_zip": false,
  "frames": false,
  "roi": false,
  "recognition_trace": false
}
```

The offline `guandan.support-bundle/1` exporter API exists, but it is not called
by the live runtime or CLI in phase 1A.  Its default export keeps `frames`, `roi`,
and `recognition_trace` disabled.  Those sensitive inputs require explicit
source paths and opt-in; this phase does not capture or supply them.  A later
phase may connect live collection and user-facing export without changing the
meaning of the v1 runtime and doctor schemas.

## Offline support-bundle API

`daguandan_bridge.support_bundle.export_support_bundle()` receives one
`SupportBundleSources` value and writes an atomic ZIP.  `SupportBundleSources`
uses a caller-selected `root`; every evidence path must be relative to that
root.  Absolute paths, `..`, package-external paths, symlinks, junctions, and
other reparse points are rejected.  Callers cannot choose archive entry names.

Core entries have fixed logical paths:

- `startup/startup.log`
- `runtime/runtime_identity.json`
- `doctor/doctor.json`
- `build/build_manifest.json`
- `incident/incident.json`
- `trace/recognition_trace.jsonl`
- `frames/frames_NNNN.png|jpg`
- `roi/roi_NNNN.png|jpg`
- `support_manifest.json`

`.npz`, `.ckpt`, and explicitly named environment dumps are never accepted.
Text is sanitized before it enters the ZIP: Windows/UNC paths, email addresses,
Bearer/JWT/common access tokens, key/password/secret fields, usernames, and
computer names are replaced.  A likely environment-variable mapping or
multi-line `KEY=VALUE` dump is replaced by `<OMITTED_ENVIRONMENT_DUMP>`.
Because an unquoted Windows/UNC path with spaces has no reliable terminator,
the sanitizer removes the remainder of that line rather than risk retaining a
private path suffix.  Identity matching uses Unicode-aware boundaries, so short
or Chinese account/device names are removed without changing larger words that
only contain the same characters.

Frame and ROI bytes cannot be reliably anonymized.  They require explicit
`include_frames=True` or `include_roi=True`, receive the
`sensitive-image` classification, and set
`privacy.contains_sensitive_images=true`.  Recognition traces likewise require
`include_recognition_trace=True` and are text-sanitized.

`support_manifest.json` records every payload entry's size, SHA256,
classification, capability, missing/disabled capabilities, and redaction
count.  The manifest explicitly excludes its own self-referential hash.  The
exporter writes a unique sibling temporary file, verifies the completed ZIP,
and only then atomically replaces the requested destination.  A failed export
does not replace an existing valid support bundle.

Every text/image source is statted before opening and read with a hard
`limit + 1` bound.  A cumulative payload budget is checked both before each
read and after sanitization, preventing a large frame list from being buffered
before an oversize bundle is rejected.
