from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from daguandan_bridge.application.session_workbench import inspect_sessions
from daguandan_bridge.application.unverified_batch_scan import UnverifiedBatchScanService

IDS = {
    'game_20260822_002135_9c2328',
    'game_20260829_192802_981e50',
    'game_20260821_200355_81b7ee',
    'game_20260817_002012_b523b5',
    'game_20260816_203403_00db35',
    'game_20260816_193531_f94cbf',
    'game_20260816_154444_392ea8',
    'game_20260815_001801_ba5066',
}
root = Path('data/profiles/tencent_daguandan/sessions').resolve()
profile = Path('data/profiles/tencent_daguandan').resolve()
out = Path('reports/session-corpus-validation/unverified-rescans').resolve()
desc = tuple(item for item in inspect_sessions(root) if item.session_id in IDS)
missing = IDS - {item.session_id for item in desc}
if missing:
    raise SystemExit(f'missing descriptors: {sorted(missing)}')
print('selected=', [item.session_id for item in desc], flush=True)
service = UnverifiedBatchScanService()
result = service.scan(
    desc,
    profile_root=profile,
    output_root=out,
    max_workers=2,
    on_progress=lambda sid, done, total, frame, completed, percent: print(
        f'progress {percent}% completed={completed} session={sid} frame={done}/{total}', flush=True
    ),
)
print(json.dumps({
    'output_directory': str(result.output_directory),
    'summary_path': str(result.summary_path),
    'selected': result.selected_count,
    'completed': result.completed_count,
    'failed': result.failed_count,
    'cancelled': result.cancelled,
}, ensure_ascii=False), flush=True)
