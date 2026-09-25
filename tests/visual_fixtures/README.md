# Visual fixture layer


These are deterministic recognized-result sequences, not screenshots.
They deliberately stay independent of Win32, real model resources, and
recorded sessions. Each JSON file contains `frames` plus expected state/event
sequences consumed by `tests/test_visual_fixture_layers.py`.

The fixture layer tests the boundary:

```text
recognized frame -> OpeningTracker / page recovery -> trusted runtime
```

A fixture is not evidence that the visual recognizer itself can read pixels;
it verifies that recognized evidence is transferred safely into live state.
