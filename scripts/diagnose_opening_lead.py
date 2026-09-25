from __future__ import annotations

"""CLI entry point for the read-only opening lead diagnosis."""

from pathlib import Path
import sys


_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from daguandan_bridge.application.opening_lead_diagnosis import main


if __name__ == "__main__":
    raise SystemExit(main())
