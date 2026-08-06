# Live-session fixtures

`golden_observations.json` is a compact, synthetic semantic script for the
semi-automatic live pipeline. Tests generate fresh frames, timestamps, video,
JSONL observations, advice, and the Chinese timeline in a temporary directory;
runtime recordings and user screenshots are never committed.

The left-player action deliberately contains a false settling interval followed
by an effect hit and a second settling interval. This protects the rule that an
effect restarts dynamic settling instead of being interpreted as a stable play.
