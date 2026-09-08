"""Bounded, presentation-free projection of live-v2 engine evidence."""
from __future__ import annotations

import json
import re

from ..live_v2.results import EngineUpdate


MAX_ITEMS = 8
MAX_CARDS = 27
MAX_DIAGNOSTICS = 8
MAX_TEXT = 96
MAX_PAYLOAD_BYTES = 16_000
MAX_INTEGER = (1 << 63) - 1
_CODE = re.compile(r"[^A-Za-z0-9_.-]+")
_TOKEN = re.compile(r"[^A-Za-z0-9_.?*-]+")


def project_engine_update(update: EngineUpdate) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": "guandan.live-v2.engine-update/2",
        "session_id": _token(update.version.session_id),
        "capture_generation": _integer(update.version.capture_generation),
        "state_revision": _integer(update.version.state_revision),
        "update_sequence": _integer(update.version.update_sequence),
        "turn_index": _integer(update.version.turn_index),
        "reason": _code(update.reason.value),
        "captured_ms": _integer(update.captured_ms),
        "processing_ms": _integer(update.processing_ms),
        "observations": [_observation(item) for item in update.observations[:MAX_ITEMS]],
        "candidates": [_candidate(item) for item in update.candidates[:MAX_ITEMS]],
        "confirmed_actions": [
            _confirmed(item) for item in update.confirmed_actions[:MAX_ITEMS]
        ],
        "gap": _gap(update.gap),
        "opportunity_status": (
            _code(update.advice_opportunity.status.value)
            if update.advice_opportunity else ""
        ),
        "truncated": {
            "observations": max(0, len(update.observations) - MAX_ITEMS),
            "candidates": max(0, len(update.candidates) - MAX_ITEMS),
            "confirmed_actions": max(0, len(update.confirmed_actions) - MAX_ITEMS),
        },
    }
    _fit_payload(payload)
    return payload


def _observation(item) -> dict[str, object]:
    return {
        "seat": _code(item.seat.value), "kind": _code(item.kind.value),
        "cards": _cards(item.cards), "confidence": round(float(item.confidence), 4),
        "reason": _code(item.reason.value),
        "frame_sequence": _integer(item.frame.frame_sequence),
        "captured_ms": _integer(item.frame.captured_ms),
        "diagnostic_codes": _diagnostic_codes(item.diagnostics),
    }


def _candidate(item) -> dict[str, object]:
    return {
        "candidate_id": _token(item.candidate_id),
        "seat": _code(item.seat.value), "kind": _code(item.kind.value),
        "cards": _cards(item.cards), "confidence": round(float(item.confidence), 4),
        "reason": _code(item.reason.value), "action_epoch": _integer(item.action_epoch),
        "first_frame": _frame(item.first_frame), "last_frame": _frame(item.last_frame),
        "evidence_count": _integer(len(item.all_evidence_ids)),
        "diagnostic_codes": _diagnostic_codes(item.diagnostics),
    }


def _confirmed(item) -> dict[str, object]:
    return {
        "action_id": _token(item.action_id), "seat": _code(item.seat.value),
        "kind": _code(item.kind.value), "cards": _cards(item.cards),
        "confidence": round(float(item.source_candidate.confidence), 4),
        "reason": _code(item.reason.value), "action_epoch": _integer(item.action_epoch),
        "first_frame": _frame(item.first_frame), "last_frame": _frame(item.last_frame),
        "evidence_count": _integer(len(item.source_candidate.all_evidence_ids)),
    }


def _gap(gap) -> dict[str, object] | None:
    if gap is None:
        return None
    return {
        "phase": _code(gap.phase.value), "reason": _code(gap.reason.value),
        "expected_seats": [_code(seat.value) for seat in gap.expected_seats[:4]],
        "evidence_count": _integer(len(gap.evidence_ids)),
    }


def _frame(frame) -> dict[str, int]:
    return {
        "sequence": _integer(frame.frame_sequence),
        "captured_ms": _integer(frame.captured_ms),
    }


def _cards(cards) -> list[str]:
    return [_token(card, 16) for card in tuple(cards)[:MAX_CARDS]]


def _diagnostic_codes(values) -> list[str]:
    codes: list[str] = []
    for raw in tuple(values)[:MAX_DIAGNOSTICS]:
        text = _text(raw)
        if "=" in text:
            head = text.split("=", 1)[0]
            code = _code(head, 48)
        elif _is_path(text):
            code = "path"
        else:
            head = text.split(":", 1)[0]
            code = _code(head, 48)
        if code and code not in codes:
            codes.append(code)
    return codes


def _text(value: object, limit: int = MAX_TEXT) -> str:
    text = str(value).replace("\r", " ").replace("\n", " ")
    return text.encode("utf-8")[:limit].decode("utf-8", errors="ignore")


def _is_path(value: str) -> bool:
    # These fields are identifiers, cards, or diagnostic codes; none has a
    # legitimate directory separator. Treat any separator as a path marker so
    # even an embedded or malformed absolute path cannot reveal local names.
    return "/" in value or "\\" in value


def _token(value: object, limit: int = MAX_TEXT) -> str:
    text = _text(value, limit * 4)
    if _is_path(text):
        return "path"
    token = _TOKEN.sub("_", text).strip("_")
    return _text(token, limit) or "redacted"


def _code(value: object, limit: int = 48) -> str:
    text = _text(value, limit * 4)
    if _is_path(text):
        return "path"
    return _text(_CODE.sub("_", text).strip("_"), limit) or "unknown"


def _integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return min(MAX_INTEGER, max(0, value))


def _payload_size(payload: dict[str, object]) -> int:
    return len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))


def _fit_payload(payload: dict[str, object]) -> None:
    """Drop only trailing duplicate detail until the hard JSON budget fits."""

    collections = ("observations", "candidates", "confirmed_actions")
    while _payload_size(payload) > MAX_PAYLOAD_BYTES:
        removable = [
            name for name in collections
            if isinstance(payload.get(name), list) and len(payload[name]) > 1
        ]
        if removable:
            name = max(removable, key=lambda key: len(payload[key]))
            payload[name].pop()
            payload["truncated"][name] += 1
            continue
        changed = False
        for name in collections:
            for item in payload.get(name, []):
                for field in ("diagnostic_codes", "cards"):
                    values = item.get(field)
                    if isinstance(values, list) and len(values) > 1:
                        values.pop()
                        changed = True
        if not changed:
            raise ValueError("live-v2 diagnostic projection exceeds its hard byte budget")


__all__ = ["MAX_ITEMS", "MAX_PAYLOAD_BYTES", "MAX_TEXT", "project_engine_update"]
