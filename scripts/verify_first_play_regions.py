"""Read-only, bounded first-play ROI verification on real local recordings.

Historical timeline/truth labels nominate candidates; they do NOT establish
that a historical video uses today's layout. Every historical sample remains
pending manual review even when the current recognizer agrees with its label.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from daguandan_bridge.annotation_service import AnnotationService
from daguandan_bridge.image_io import read_image_unicode, save_image_unicode
from daguandan_bridge.recognition_service import ScreenshotRecognitionService
from daguandan_bridge.template_service import TemplateService

SEATS = ("left", "opposite", "right", "self")
EARLY_FRAMES = (0, 3, 6, 10, 15, 20, 30)
REMOTE_LEFT = Path("C:/Users/yhx/Documents/掼蛋助手日志/2026-09-04/manual_20260904_234625/OPEN-1256916375-26465c0a/frames/standardized_000286.png")


class ReadOnlyTemplates(TemplateService):
    """Do not invoke the normal list_templates auto-registration writer."""

    def list_templates(self):
        document = json.loads(self.templates_path.read_text(encoding="utf-8"))
        return tuple(item for item in document.get("templates", []) if isinstance(item, dict))


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def declared_lead(session: Path) -> dict[str, object] | None:
    """Read at most 128 initial timeline lines, then a few small truth files."""
    timeline = session / "timeline.jsonl"
    initial = None
    if timeline.is_file():
        with timeline.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if line_number > 128:
                    break
                try:
                    event = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if not isinstance(event, dict):
                    continue
                kind = event.get("event_type")
                payload = event.get("payload") or {}
                if not isinstance(payload, dict):
                    continue
                seat = payload.get("lead_player")
                if kind not in {"initial_state_confirmed", "lead_player_confirmed"} or seat not in SEATS:
                    continue
                candidate = {
                    "expected_seat": seat, "evidence_file": str(timeline),
                    "evidence_line": line_number, "event_type": kind,
                    "monotonic_ms": event.get("monotonic_ms"),
                    "event_source": event.get("source"),
                    "evidence_kind": "historical_timeline_declaration_not_visual_truth",
                }
                if kind == "lead_player_confirmed":
                    return candidate
                initial = candidate
        if initial is not None:
            return initial
    truth_root = session / "derived" / "truth_scan_drafts"
    for truth in sorted(truth_root.glob("*/truth_log.json"), reverse=True)[:6]:
        if truth.stat().st_size > 2 * 1024 * 1024:
            continue
        try:
            document = json.loads(truth.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        seat = (document.get("initial_state") or {}).get("lead_player")
        if seat in SEATS:
            return {
                "expected_seat": seat, "evidence_file": str(truth),
                "event_type": "truth_initial_state", "monotonic_ms": None,
                "label_status": document.get("label_status"),
                "evidence_kind": "historical_truth_declaration_not_current_layout_acceptance",
            }
    return None


def event_frame_candidates(session: Path, lead: dict[str, object]) -> tuple[int, ...]:
    """A bounded fallback near the declared first turn, never full playback."""
    tick = lead.get("monotonic_ms")
    index_path = session / "video" / "frame_index.jsonl"
    if not isinstance(tick, (int, float)) or not index_path.is_file():
        return ()
    nearest = None
    with index_path.open(encoding="utf-8") as stream:
        for ordinal, line in enumerate(stream):
            if ordinal > 300:
                break
            try:
                item = json.loads(line)
                frame, stamp = int(item["frame_index"]), int(item["monotonic_ms"])
            except (ValueError, KeyError, TypeError):
                continue
            pair = (abs(stamp - tick), frame)
            if nearest is None or pair < nearest:
                nearest = pair
    if nearest is None:
        return ()
    return tuple(sorted({max(0, min(300, nearest[1] + offset)) for offset in (-2, 0, 2)}))


def roi_records(service: ScreenshotRecognitionService) -> dict[str, object]:
    regions = {item.name: item for item in service.annotation_service.list_regions()}
    result = {}
    for seat in SEATS:
        region = regions.get("first_play_" + seat)
        result[seat] = {
            "box": region.abs_box.to_list() if region else None,
            "ratio_box": list(region.ratio_box) if region else None,
            "within_1280x720": bool(region and region.abs_box.fits_within((1280, 720))),
        }
    return result


def inspect_frame(service: ScreenshotRecognitionService, image: np.ndarray, expected: str) -> dict[str, object]:
    if image.shape[:2] != (720, 1280):
        return {"recognized_seat": None, "decision": "unsupported_source_size", "source_size": [image.shape[1], image.shape[0]], "matches": {}}
    regions = {item.name: item for item in service.annotation_service.list_regions()}
    templates = service._templates()
    recognized, score, source, match = service._recognize_seat_status(
        image, regions, templates, prefix="first_play", kind="status", label="first_play",
    )
    per_seat = {}
    for seat in SEATS:
        # Diagnostic peak only. This lower search threshold is NEVER used by
        # the decision above and never marks a seat accepted or verified.
        _, peak, peak_source, peak_match = service._recognize_status(
            image, regions.get("first_play_" + seat), templates,
            label="first_play", threshold=-1.0, search_margin=(20, 20),
        )
        per_seat[seat] = {
            "diagnostic_peak": float(peak), "template_source": peak_source,
            "match_box": [peak_match.x, peak_match.y, peak_match.w, peak_match.h] if peak_match else None,
        }
    return {
        "source_size": [1280, 720], "recognized_seat": recognized,
        "accepted_score": float(score), "accepted_source": source,
        "accepted_box": [match.x, match.y, match.w, match.h] if match else None,
        "decision": "agrees_with_declared_seat" if recognized == expected else "does_not_confirm_declared_seat",
        "matches": per_seat,
    }


def candidate_rank(sample: dict[str, object]) -> tuple[bool, float]:
    expected = str(sample["expected_seat"])
    matches = sample.get("matches") or {}
    peak = (matches.get(expected) or {}).get("diagnostic_peak", 0.0)
    return sample.get("recognized_seat") == expected, float(peak)


def preview(image: np.ndarray, sample: dict[str, object], rois: dict[str, object]) -> np.ndarray:
    result = image.copy()
    expected = str(sample["expected_seat"])
    box = rois[expected]["box"]
    if box and result.shape[:2] == (720, 1280):
        x, y, width, height = box
        cv2.rectangle(result, (x, y), (x + width, y + height), (255, 160, 30), 2)
    match = sample.get("accepted_box") or ((sample.get("matches") or {}).get(expected) or {}).get("match_box")
    if match:
        x, y, width, height = match
        cv2.rectangle(result, (x, y), (x + width, y + height), (30, 230, 70), 2)
    footer = np.zeros((66, result.shape[1], 3), dtype=np.uint8)
    title = f"expected={expected} detected={sample.get('recognized_seat')} frame={sample.get('frame_index')} score={sample.get('accepted_score', 0):.3f}"
    cv2.putText(footer, title, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, .62, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(footer, str(sample.get("session_id", "remote-left")) + " | MANUAL LAYOUT REVIEW REQUIRED", (12, 50), cv2.FONT_HERSHEY_SIMPLEX, .52, (220, 220, 220), 1, cv2.LINE_AA)
    return np.vstack((result, footer))


def verify(profiles_root: Path, report_root: Path, *, profile: str = "tencent_daguandan", max_videos: int = 32, remote_left: Path | None = REMOTE_LEFT) -> dict[str, object]:
    profiles_root, report_root = profiles_root.resolve(), report_root.resolve()
    if report_root == profiles_root or report_root.is_relative_to(profiles_root):
        raise ValueError("report output must stay outside read-only profiles")
    if not 1 <= max_videos <= 64:
        raise ValueError("max_videos must be between 1 and 64")
    if (report_root / "report.json").exists():
        raise FileExistsError("existing report is preserved; choose a fresh report-root")
    profile_root = profiles_root / profile
    inputs = [profile_root / "regions_config.json", profile_root / "templates_config.json"]
    before = {str(path): file_hash(path) for path in inputs}
    service = ScreenshotRecognitionService(AnnotationService(profiles_root, profile), ReadOnlyTemplates(profiles_root, profile))
    rois = roi_records(service)
    scans, selected, selected_images = [], {}, {}
    videos = sorted((profile_root / "sessions").glob("*/video/game.avi"), reverse=True)[:max_videos]
    decoded = 0
    for video in videos:
        session = video.parent.parent
        lead = declared_lead(session)
        if lead is None:
            scans.append({"session_id": session.name, "video": str(video), "status": "no_declared_lead", "frames": []})
            continue
        expected = str(lead["expected_seat"])
        capture = cv2.VideoCapture(str(video))
        frame_results = []
        best, best_image = None, None
        try:
            if not capture.isOpened():
                scans.append({"session_id": session.name, "video": str(video), "status": "video_open_failed", "frames": []})
                continue
            frames = list(EARLY_FRAMES)
            for frame_index in frames:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, image = capture.read()
                if not ok or image is None:
                    continue
                decoded += 1
                sample = {**lead, **inspect_frame(service, image, expected), "session_id": session.name, "video": str(video), "frame_index": frame_index,
                          "acceptance": "pending_manual_layout_review", "modern_layout_verified": False}
                frame_results.append(sample)
                if best is None or candidate_rank(sample) > candidate_rank(best):
                    best, best_image = sample, image
            if best is None or best.get("recognized_seat") != expected:
                for frame_index in event_frame_candidates(session, lead):
                    if frame_index in frames:
                        continue
                    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    ok, image = capture.read()
                    if not ok or image is None:
                        continue
                    decoded += 1
                    sample = {**lead, **inspect_frame(service, image, expected), "session_id": session.name, "video": str(video), "frame_index": frame_index,
                              "acceptance": "pending_manual_layout_review", "modern_layout_verified": False}
                    frame_results.append(sample)
                    if best is None or candidate_rank(sample) > candidate_rank(best):
                        best, best_image = sample, image
        finally:
            capture.release()
        scans.append({"session_id": session.name, "video": str(video), "status": "sampled", "expected_seat": expected,
                      "frames": [{"frame_index": item["frame_index"], "recognized_seat": item["recognized_seat"], "accepted_score": item.get("accepted_score"), "expected_peak": ((item.get("matches") or {}).get(expected) or {}).get("diagnostic_peak")} for item in frame_results]})
        if best is not None and (expected not in selected or candidate_rank(best) > candidate_rank(selected[expected])):
            selected[expected], selected_images[expected] = best, best_image
    if remote_left is not None and remote_left.is_file():
        image = read_image_unicode(remote_left)
        sample = {**inspect_frame(service, image, "left"), "expected_seat": "left", "image": str(remote_left.resolve()), "image_sha256": file_hash(remote_left),
                  "frame_index": 286, "session_id": "remote_20260904_opening", "evidence_kind": "previously_human_inspected_current_remote_frame",
                  "acceptance": "verified_remote_left_frame" if service.recognize_lead_player(image) == "left" else "remote_left_failed", "modern_layout_verified": True}
        selected["left"], selected_images["left"] = sample, image
    report_root.mkdir(parents=True, exist_ok=True)
    for seat, sample in selected.items():
        output = report_root / f"{seat}_candidate.jpg"
        if output.exists():
            raise FileExistsError(f"existing preview is preserved: {output}")
        save_image_unicode(output, preview(selected_images[seat], sample, rois))
        sample["preview"] = str(output)
    unchanged = all(file_hash(path) == before[str(path)] for path in inputs)
    report = {
        "schema": "guandan.first-play-region-verification/1", "created_at": datetime.now().astimezone().isoformat(),
        "profile": profile, "regions": rois, "input_sha256": before, "source_inputs_unchanged": unchanged,
        "acceptance": "pending_manual_historical_layout_review", "all_four_modern_seats_accepted": False,
        "bounds": {"max_videos": max_videos, "early_frames": list(EARLY_FRAMES), "fallback_max_frame": 300, "max_fallback_frames_per_video": 3},
        "videos_considered": len(videos), "frames_decoded": decoded,
        "selected": {seat: selected.get(seat, {"acceptance": "no_sample", "expected_seat": seat}) for seat in SEATS}, "scans": scans,
        "review_instruction": "Open up to four previews; verify the visible first-play marker and its seat against current Tencent layout. Timeline/truth labels alone are not visual ground truth.",
    }
    (report_root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not unchanged:
        raise RuntimeError("input config changed during verification; report is not an acceptance result")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles-root", type=Path, default=PROJECT_ROOT / "data" / "profiles")
    parser.add_argument("--profile", default="tencent_daguandan")
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--max-videos", type=int, default=32)
    parser.add_argument("--remote-left", type=Path, default=REMOTE_LEFT)
    args = parser.parse_args()
    report = verify(args.profiles_root, args.report_root, profile=args.profile, max_videos=args.max_videos, remote_left=args.remote_left)
    print(json.dumps({"report": str(args.report_root.resolve() / "report.json"), "videos": report["videos_considered"], "decoded": report["frames_decoded"],
                      "selected": {seat: {"recognized_seat": item.get("recognized_seat"), "frame_index": item.get("frame_index"), "acceptance": item["acceptance"]} for seat, item in report["selected"].items()}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
