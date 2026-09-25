from __future__ import annotations

"""Evidence-bound, copy-friendly presentation for one-frame recognition reports.

This module deliberately stays a presentation adapter.  It never infers a game
state from an empty field, never lowers a recognizer threshold, and never turns
a low-scoring candidate into a recognized card or action.
"""

from dataclasses import dataclass, field
import json
from typing import Any, Mapping, Sequence


_STATUS_TEXT = {"PASS": "已就绪", "WAIT": "等待处理", "FAIL": "需要处理"}
_CODE_TEXT = {
    "READY": ("开局已就绪", "当前已经满足建立对局的基本条件", "可以继续等待推荐"),
    "READY_WAITING_FIRST_ACTION": ("已进入牌桌，等待自己首出", "起手牌已经确认，正在等待自己第一次出牌", "等待自己首出；首出后继续识别出牌"),
    "LISTENING": ("正在监听", "程序已经连接牌桌，正在等待有效开局", "保持牌桌窗口可见"),
    "LOBBY": ("等待进入牌桌", "当前画面还不是可以建立对局的牌桌画面", "进入牌桌后重新检测"),
    "DEAL_IN_PROGRESS": ("当前对局已经开始", "没有捕获到完整的开局证据", "等待下一局重新发牌"),
    "MID_GAME_HAND_COUNT": ("手牌数量不足", "当前可能是进行中的牌局，不能直接建立新对局", "等待下一局重新发牌"),
    "HAND_UNSTABLE": ("手牌识别不稳定", "连续两次识别结果还不一致", "保持牌桌清晰，等待下一次确认"),
    "OPENING_UNRESOLVED": ("开局证据未确认", "级牌、手牌或首发信息还没有全部确认", "保持牌桌清晰，等待开局证据稳定"),
    "WINDOW_NOT_FOUND": ("没有找到牌桌窗口", "程序没有找到目标大掼蛋窗口", "打开大掼蛋后重新刷新窗口列表"),
    "MULTIPLE_WINDOWS": ("找到多个可能的窗口", "程序无法确定应该监听哪一个窗口", "关闭重复窗口或手动选择目标窗口"),
    "WINDOW_MINIMIZED": ("牌桌窗口已最小化", "最小化窗口不能提供可靠的牌桌画面", "恢复大掼蛋窗口后重新检测"),
    "CAPTURE_FAILED": ("画面捕获失败", "程序没有取得可用于识别的有效画面", "确认窗口可见后重新测试截图"),
    "ROI_FATAL": ("识别区域需要调整", "关键识别区域存在冲突或超出画面", "打开区域配置重新调整识别区域"),
    "WORKER_FAULT": ("后台识别失败", "识别线程没有正常完成工作", "打开完整助手后重新连接牌桌"),
}

_FIELD_LABELS = {
    "round_level": "级牌",
    "wild_rank": "百变牌",
    "my_hand": "我的手牌",
    "hand_count": "手牌数量",
    "expected_hand_count": "应有手牌数量",
    "current_player": "当前行动者",
    "lead_player": "首出玩家",
    "buttons": "按钮状态",
    "events": "各家动作",
}
_TRACE_FIELD_ALIASES = {
    "level_rank": {"round_level", "wild_rank"},
    "round_level": {"round_level", "wild_rank"},
    "wild_rank": {"round_level", "wild_rank"},
    "my_hand": {"my_hand"},
    "current_player": {"current_player", "active_player"},
    "lead_player": {"lead_player", "first_play", "opening_signal"},
    "buttons": {"buttons", "button_actions", "game_end_controls"},
    "events": {"events", "my_play", "left_play", "opposite_play", "right_play"},
}


def _mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            converted = to_dict()
            if isinstance(converted, Mapping):
                return converted
        except Exception:
            pass
    values = getattr(value, "__dict__", None)
    return values if isinstance(values, Mapping) else {}


def _get(value: object, key: str, default: object = None) -> object:
    mapped = _mapping(value)
    if key in mapped:
        return mapped[key]
    return getattr(value, key, default)


def _plain(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(item) for item in value]
    if hasattr(value, "value") and isinstance(getattr(value, "value"), (str, int, float)):
        return getattr(value, "value")
    values = _mapping(value)
    if values:
        return {str(key): _plain(item) for key, item in values.items() if not str(key).startswith("_")}
    return str(value)


def _list(value: object) -> list[object]:
    if value is None or isinstance(value, (str, bytes, Mapping)):
        return []
    if isinstance(value, Sequence):
        return list(value)
    try:
        return list(value)  # type: ignore[arg-type]
    except TypeError:
        return []


def _number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _empty(value: object) -> bool:
    return value is None or value == "" or value == () or value == []


def _display(value: object, *, empty: str = "未识别/证据不足") -> str:
    if _empty(value):
        return empty
    if isinstance(value, (list, tuple)):
        if not value:
            return empty
        return ", ".join(_display(item, empty="") for item in value)
    if isinstance(value, Mapping):
        return json.dumps(_plain(value), ensure_ascii=False, sort_keys=True)
    return str(_plain(value))


def _result_from(report: Mapping[str, object] | None, result: object | None) -> object:
    if result is not None:
        return result
    if report is None:
        return {}
    recognition = _mapping(report.get("recognition"))
    return recognition.get("result", {})


def _trace_from(report: Mapping[str, object] | None, trace: object | None) -> Mapping[str, object]:
    if trace is not None:
        return _mapping(trace)
    if report is None:
        return {}
    return _mapping(_mapping(report.get("recognition")).get("trace"))


def _readiness_from(report: Mapping[str, object] | None, readiness: object | None) -> Mapping[str, object]:
    if readiness is not None:
        return _mapping(readiness)
    if report is None:
        return {}
    inputs = _mapping(report.get("opening_readiness_inputs"))
    return _mapping(inputs.get("readiness"))


def _gate_from(report: Mapping[str, object] | None, gate: object | None) -> Mapping[str, object]:
    if gate is not None:
        return _mapping(gate)
    if report is None:
        return {}
    return _mapping(_mapping(report.get("opening_readiness_inputs")).get("gate"))


def _trace_candidates(trace: Mapping[str, object]) -> list[dict[str, object]]:
    raw = trace.get("candidates", ())
    return [dict(_mapping(item)) for item in _list(raw) if _mapping(item)]


def _candidate_matches(field: str, candidate: Mapping[str, object]) -> bool:
    raw_field = str(candidate.get("field") or "")
    aliases = _TRACE_FIELD_ALIASES.get(field, {field})
    return raw_field in aliases or str(candidate.get("label") or "") in aliases


def _lead_evidence(result: object) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for item in _list(_get(result, "lead_evidence", ())):
        value = _mapping(item)
        if not value:
            continue
        first = _number(value.get("first_play_score")) or 0.0
        cards = _number(value.get("card_action_score")) or 0.0
        timer = _number(value.get("timer_score")) or 0.0
        output.append({
            "candidate_seat": value.get("candidate_seat", value.get("seat")),
            "first_play_score": first,
            "card_action_score": cards,
            "timer_score": timer,
            "status": value.get("status", "pending"),
            "rejection_reason": value.get("rejection_reason"),
            "candidate_score": max(first, cards, timer),
        })
    return output


def _threshold_analysis(trace: Mapping[str, object], field_confidences: Mapping[str, object]) -> dict[str, object]:
    candidates: list[dict[str, object]] = []
    findings: list[dict[str, object]] = []
    for raw in _trace_candidates(trace):
        score = _number(raw.get("score"))
        threshold = _number(raw.get("threshold"))
        accepted = raw.get("accepted")
        rejection = raw.get("rejection_reason")
        rejection_text = str(rejection or "").lower()
        classification = "observed"
        if "conflict" in rejection_text or "ambiguous" in rejection_text:
            classification = "candidate_conflict"
            findings.append({
                "code": "CANDIDATE_CONFLICT",
                "category": "conflict",
                "field": raw.get("field"),
                "label": raw.get("label"),
                "message": f"候选 {raw.get('label') or raw.get('field') or '未命名'} 存在候选冲突，未擅自选择",
                "evidence": {"score": score, "threshold": threshold, "rejection_reason": rejection},
            })
        elif rejection_text in {"below_threshold", "insufficient_evidence", "low_score"} or (
            score is not None and threshold is not None and score < threshold
        ):
            classification = "below_threshold"
            if score is not None and threshold is not None:
                findings.append({
                    "code": "BELOW_THRESHOLD",
                    "category": "threshold",
                    "field": raw.get("field"),
                    "label": raw.get("label"),
                    "message": f"候选 {raw.get('label') or raw.get('field') or '未命名'} 得分 {score:.3f} 低于阈值 {threshold:.3f}，未作为识别结果采用",
                    "evidence": {"score": score, "threshold": threshold, "rejection_reason": rejection},
                })
        elif accepted is True and (raw.get("low_confidence") is True or str(raw.get("confidence_status") or "").lower() == "low"):
            classification = "accepted_low_confidence"
            findings.append({
                "code": "ACCEPTED_LOW_CONFIDENCE",
                "category": "threshold",
                "field": raw.get("field"),
                "label": raw.get("label"),
                "message": f"候选 {raw.get('label') or raw.get('field') or '未命名'} 虽通过，但 trace 明确标记为低置信度",
                "evidence": {"score": score, "threshold": threshold, "confidence_status": raw.get("confidence_status")},
            })
        candidates.append({
            "field": raw.get("field"),
            "label": raw.get("label"),
            "region": raw.get("field"),
            "source": raw.get("source"),
            "kind": raw.get("kind"),
            "score": score,
            "threshold": threshold,
            "accepted": accepted,
            "rejection_reason": rejection,
            "classification": classification,
            "match_box": raw.get("match_box"),
            "roi_box": raw.get("roi_box"),
        })

    for field_name, confidence_value in field_confidences.items():
        confidence = _number(confidence_value)
        if confidence is None:
            continue
        field_candidates = [item for item in candidates if _candidate_matches(str(field_name), item)]
        thresholds = [item["threshold"] for item in field_candidates if isinstance(item.get("threshold"), (int, float))]
        if thresholds and confidence < min(thresholds):
            findings.append({
                "code": "FIELD_CONFIDENCE_BELOW_THRESHOLD",
                "category": "threshold",
                "field": field_name,
                "message": f"字段 {field_name} 的聚合置信度 {confidence:.3f} 低于 trace 阈值 {min(thresholds):.3f}",
                "evidence": {"confidence": confidence, "threshold": min(thresholds)},
            })
    return {"candidates": candidates, "findings": findings, "threshold_policy": trace.get("threshold_policy")}


def _field_evidence(result: object, trace: Mapping[str, object], lead: list[dict[str, object]]) -> list[dict[str, object]]:
    confidences = _mapping(_get(result, "field_confidences", {}))
    sources = _mapping(_get(result, "sources", {}))
    unresolved = {str(item) for item in _list(_get(result, "unresolved_fields", ())) }
    output: list[dict[str, object]] = []
    fields = ("round_level", "wild_rank", "my_hand", "hand_count", "expected_hand_count", "current_player", "lead_player", "buttons", "events")
    values = {
        "round_level": _get(result, "round_level"),
        "wild_rank": _get(result, "wild_rank"),
        "my_hand": list(_get(result, "my_hand", ()) or ()),
        "hand_count": len(list(_get(result, "my_hand", ()) or ())),
        "expected_hand_count": 27,
        "current_player": _get(result, "current_player"),
        "lead_player": _get(result, "lead_player"),
        "buttons": list(_get(result, "buttons", ()) or ()),
        "events": list(_get(result, "events", ()) or ()),
    }
    trace_items = _trace_candidates(trace)
    for field_name in fields:
        value = values[field_name]
        confidence = _number(confidences.get(field_name))
        source = sources.get(field_name)
        candidates = [item for item in trace_items if _candidate_matches(field_name, item)]
        conflict = any("conflict" in str(item.get("rejection_reason") or "").lower() for item in candidates)
        if field_name == "lead_player":
            conflict = conflict or any(item.get("status") == "conflict" for item in lead)
        if field_name == "expected_hand_count":
            status = "reference"
            source = "game_rule"
        elif conflict:
            status = "conflict"
        elif field_name in unresolved or _empty(value):
            status = "unresolved"
        elif confidence is not None and candidates:
            thresholds = [item.get("threshold") for item in candidates if isinstance(item.get("threshold"), (int, float))]
            status = "low_confidence" if thresholds and confidence < min(thresholds) else "confirmed"
        elif field_name in confidences and confidence is not None and confidence < 0.5:
            status = "low_confidence"
        else:
            status = "confirmed" if not _empty(value) else "unresolved"
        output.append({
            "field": field_name,
            "label": _FIELD_LABELS[field_name],
            "value": _plain(value),
            "status": status,
            "confidence": confidence,
            "source": source,
            "unresolved": field_name in unresolved,
            "candidate_count": len(candidates),
            "evidence": candidates,
        })
    return output


def _action_evidence(result: object) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for event in _list(_get(result, "events", ())):
        cards = list(_get(event, "cards", ()) or ())
        is_pass = bool(_get(event, "is_pass", False))
        recognized_play = bool(is_pass or cards)
        item: dict[str, object] = {
            "player": _get(event, "player"),
            "cards": _plain(cards),
            "is_pass": is_pass,
            "confidence": _get(event, "confidence"),
            "source": _get(event, "source"),
            "recognized_play": recognized_play,
            "status": "过牌" if is_pass else "已识别出牌" if cards else "未识别（没有牌面证据）",
        }
        for key in ("play_type", "card_type", "annotations", "diagnostics", "post_hand", "post_hand_confidence", "suit_options"):
            value = _get(event, key, None)
            if value is not None:
                item[key] = _plain(value)
        output.append(item)
    return output


def _roi_issue_is_opening_blocking(issue: object) -> bool:
    """Return whether one ROI issue should block opening/readiness.

    ROI validation also reports action-only geometry warnings.  Those warnings
    remain useful evidence for post-opening action recognition, but they must
    not be promoted to the opening-level ``ROI_INVALID`` blocker.  The
    ``action_only`` is the explicit exception; the code/severity fallback keeps
    older persisted reports compatible.  Issue-level ``opening_blocking`` is
    intentionally not trusted to hide other structural failures.
    """

    item = _mapping(issue)
    if not item:
        return True
    if item.get("action_only") is True:
        return False
    code = str(item.get("code") or item.get("type") or "").casefold()
    severity = str(item.get("severity") or "").casefold()
    return not (code in {"roi.critical_play_overlap", "critical_play_overlap"} and severity == "warning")


def _roi_opening_issues(roi_validation: Mapping[str, object]) -> list[object]:
    issues = _list(roi_validation.get("issues"))
    return [issue for issue in issues if _roi_issue_is_opening_blocking(issue)]


def _blockers(
    *,
    report: Mapping[str, object] | None,
    result: object,
    trace_analysis: Mapping[str, object],
    readiness: Mapping[str, object],
    gate: Mapping[str, object],
    roi_validation: Mapping[str, object],
    errors: object,
    lead: list[dict[str, object]],
) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []

    def add(category: str, code: str, message: str, evidence: object, recommendation: str) -> None:
        blockers.append({"category": category, "code": code, "message": message, "evidence": _plain(evidence), "recommendation": recommendation})

    error_items = _list(errors)
    if not error_items and report is not None:
        error_items = _list(report.get("errors"))
    for error in error_items:
        error_map = _mapping(error)
        add("window_or_screenshot", str(error_map.get("code") or "CAPTURE_FAILED"), str(error_map.get("message") or "截图或窗口证据不可用"), error_map, "先确认大掼蛋窗口可见且截图有效，再重新诊断。")

    roi_status = str(roi_validation.get("status") or "").lower()
    all_roi_issues = _list(roi_validation.get("issues"))
    roi_issues = _roi_opening_issues(roi_validation)
    roi_status_blocks = roi_status not in {"", "pass", "ok", "valid"}
    # A legacy report may say ``fail`` while carrying only the known
    # action-only overlap warning.  Do not turn that warning into ROI_INVALID.
    if all_roi_issues and not roi_issues:
        roi_status_blocks = False
    if roi_issues or roi_status_blocks:
        add("roi", "ROI_INVALID", "识别区域校验未通过或包含问题，不能可靠解释识别结果。", {"status": roi_validation.get("status"), "issues": roi_issues}, "根据列出的区域问题修正 ROI 后重新诊断。")

    for finding in _list(trace_analysis.get("findings")):
        finding_map = _mapping(finding)
        category = str(finding_map.get("category") or "threshold")
        if category == "conflict":
            add("candidate_conflict", str(finding_map.get("code") or "CANDIDATE_CONFLICT"), str(finding_map.get("message") or "候选存在冲突，未擅自选择"), finding_map.get("evidence", finding_map), "检查截图清晰度、ROI 和模板；不要直接降低阈值。")
        else:
            add("threshold", str(finding_map.get("code") or "THRESHOLD"), str(finding_map.get("message") or "候选未达到识别阈值"), finding_map.get("evidence", finding_map), "先检查截图、ROI 和模板资源；不要为了通过而降低生产阈值。")

    hand = list(_get(result, "my_hand", ()) or ())
    if hand and len(hand) != 27:
        add("hand_count", "HAND_COUNT_MISMATCH", f"只从截图证据中识别到 {len(hand)} 张手牌，期望数量为 27 张。", {"hand_count": len(hand), "expected_hand_count": 27}, "检查手牌 ROI、遮挡、清晰度和模板匹配结果；当前不能把缺失牌补出来。")
    diagnostics = [str(item) for item in _list(_get(result, "diagnostics", ()))]
    for diagnostic in diagnostics:
        lowered = diagnostic.lower()
        if any(token in lowered for token in ("template", "模板")):
            add("template_resource", "TEMPLATE_OR_RECOGNITION", diagnostic, {"diagnostic": diagnostic}, "检查对应模板文件、模板版本和模板匹配分数。")
        elif any(token in lowered for token in ("roi", "区域")):
            add("roi", "ROI_DIAGNOSTIC", diagnostic, {"diagnostic": diagnostic}, "检查对应识别区域的位置、尺寸和窗口缩放。")
        elif any(token in lowered for token in ("遮挡", "模糊", "blur", "occlu", "contrast", "对比")):
            add("occlusion_or_blur", "IMAGE_QUALITY", diagnostic, {"diagnostic": diagnostic}, "确认牌桌没有被遮挡，窗口尺寸和 DPI 与标定一致。")
        elif any(token in lowered for token in ("冲突", "重复", "物理")):
            add("hand_or_state_conflict", "RECOGNITION_CONFLICT", diagnostic, {"diagnostic": diagnostic}, "保留冲突证据并重新采集清晰截图，不能强行选择结果。")

    if any(item.get("status") == "conflict" for item in lead):
        add("opening_evidence_conflict", "LEAD_EVIDENCE_CONFLICT", "首出候选之间存在冲突，当前不能确定首出玩家。", lead, "查看每个候选的三类分数和拒绝原因，确认首出标记或动作证据。")
    elif "lead_player" in {str(item) for item in _list(_get(result, "unresolved_fields", ())) }:
        add("opening_evidence_missing", "LEAD_EVIDENCE_MISSING", "截图没有提供足够的首出玩家证据。", {"lead_evidence": lead, "gate": gate}, "补充包含首出标记、计时器或首出牌面证据的清晰截图。")

    unresolved = {str(item) for item in _list(_get(result, "unresolved_fields", ())) }
    if "events" in unresolved:
        add("action_evidence_missing", "ACTION_EVIDENCE_MISSING", "动作识别被标记为未解决；没有把缺失证据当作没有出牌。", {"unresolved_fields": sorted(unresolved)}, "检查各家出牌 ROI 和对应模板，并使用下一张截图复核。")

    reason = str(readiness.get("primary_reason") or gate.get("reason") or "")
    if reason and reason not in {"READY", "LISTENING"} and not blockers:
        title = _CODE_TEXT.get(reason, (reason, "开局门控尚未通过", "查看门控详情和字段证据"))[0]
        add("opening_gate", reason, title, {"readiness": readiness, "gate": gate}, _CODE_TEXT.get(reason, ("", "", "查看门控详情和字段证据"))[2])
    return blockers


def _detailed_text(payload: Mapping[str, object]) -> str:
    summary = _mapping(payload.get("summary"))
    lines = ["详细识别报告", "", "【牌局摘要】"]
    for key in ("round_level", "wild_rank", "hand_count", "expected_hand_count", "current_player", "lead_player", "buttons"):
        lines.append(f"{_FIELD_LABELS.get(key, key)}：{_display(summary.get(key))}")
    hand = summary.get("my_hand")
    lines.append(f"我的手牌：{_display(hand)}")
    readiness = _mapping(summary.get("readiness"))
    gate = _mapping(summary.get("gate"))
    lines.append(f"开局 readiness：{_display(readiness.get('status'))} / {_display(readiness.get('primary_reason'))}")
    lines.append(f"开局 gate：{_display(gate.get('reason') or gate.get('status'))}")

    lines.extend(["", "【各家动作】"])
    actions = _list(payload.get("actions"))
    if not actions:
        lines.append("没有识别到动作事件；这不代表没有出牌，只表示当前帧没有足够动作证据。")
    else:
        for action in actions:
            item = _mapping(action)
            cards = _display(item.get("cards"))
            detail = f"{_display(item.get('player'))}：{item.get('status', '未识别')}"
            if item.get("is_pass"):
                detail += "（过牌证据）"
            elif item.get("recognized_play"):
                detail += f"，牌面：{cards}"
            else:
                detail += "，没有足够牌面证据"
            detail += f"，置信度：{_display(item.get('confidence'))}，来源：{_display(item.get('source'))}"
            lines.append(detail)

    lines.extend(["", "【首出证据】"])
    evidence = _list(payload.get("lead_evidence"))
    if not evidence:
        lines.append("没有首出候选证据，不能推断首出玩家。")
    else:
        for item in evidence:
            value = _mapping(item)
            lines.append(
                f"{_display(value.get('candidate_seat'))}：状态 {_display(value.get('status'))}，"
                f"首出标记 {_display(value.get('first_play_score'))}，牌面动作 {_display(value.get('card_action_score'))}，"
                f"计时器 {_display(value.get('timer_score'))}，候选分 {_display(value.get('candidate_score'))}，"
                f"原因 {_display(value.get('rejection_reason'), empty='无')}"
            )

    lines.extend(["", "【字段证据】"])
    for item in _list(payload.get("field_evidence")):
        value = _mapping(item)
        lines.append(
            f"{_display(value.get('label'))}：{_display(value.get('value'))}；状态 {_display(value.get('status'))}；"
            f"置信度 {_display(value.get('confidence'))}；来源 {_display(value.get('source'))}"
        )

    lines.extend(["", "【阈值与候选分析】"])
    findings = _list(_mapping(payload.get("threshold_analysis")).get("findings"))
    if not findings:
        lines.append("当前 trace 没有提供额外的阈值拒绝或候选冲突证据。")
    else:
        for item in findings:
            value = _mapping(item)
            lines.append(f"- {_display(value.get('message'))}")

    lines.extend(["", "【阻塞原因和建议】"])
    blockers = _list(payload.get("blockers"))
    if not blockers:
        lines.append("当前没有额外的阻塞原因证据。")
    else:
        for item in blockers:
            value = _mapping(item)
            lines.append(f"- {_display(value.get('message'))} 建议：{_display(value.get('recommendation'))}")
    return "\n".join(lines)


def build_detailed_diagnostic(
    report: Mapping[str, object] | None = None,
    *,
    result: object | None = None,
    trace: object | None = None,
    readiness: object | None = None,
    gate: object | None = None,
    roi_validation: object | None = None,
    capture: object | None = None,
    errors: object | None = None,
) -> dict[str, object]:
    """Build the evidence-only detailed report shared by all frame sources."""

    result_value = _result_from(report, result)
    trace_value = _trace_from(report, trace)
    readiness_value = _readiness_from(report, readiness)
    gate_value = _gate_from(report, gate)
    roi_value = _mapping(roi_validation if roi_validation is not None else (report or {}).get("roi_validation", {}))
    result_hand = list(_get(result_value, "my_hand", ()) or ())
    field_confidences = _mapping(_get(result_value, "field_confidences", {}))
    lead = _lead_evidence(result_value)
    threshold = _threshold_analysis(trace_value, field_confidences)
    fields = _field_evidence(result_value, trace_value, lead)
    summary = {
        "round_level": _plain(_get(result_value, "round_level")),
        "wild_rank": _plain(_get(result_value, "wild_rank")),
        "my_hand": _plain(result_hand),
        "hand_count": len(result_hand),
        "expected_hand_count": 27,
        "current_player": _plain(_get(result_value, "current_player")),
        "lead_player": _plain(_get(result_value, "lead_player")),
        "buttons": _plain(list(_get(result_value, "buttons", ()) or ())),
        "readiness": _plain(readiness_value),
        "gate": _plain(gate_value),
    }
    payload: dict[str, object] = {
        "schema": "guandan.detailed-diagnostic/v1",
        "evidence_only": True,
        "summary": summary,
        "actions": _action_evidence(result_value),
        "lead_evidence": lead,
        "field_evidence": fields,
        "threshold_analysis": threshold,
        "recognition_diagnostics": _plain(list(_get(result_value, "diagnostics", ()) or ())),
        "roi_validation": _plain(roi_value),
        "capture": _plain(capture),
        "errors": _plain(_list(errors)),
    }
    payload["blockers"] = _blockers(
        report=report,
        result=result_value,
        trace_analysis=threshold,
        readiness=readiness_value,
        gate=gate_value,
        roi_validation=roi_value,
        errors=errors,
        lead=lead,
    )
    payload["recommendations"] = [item["recommendation"] for item in _list(payload["blockers"]) if _mapping(item).get("recommendation")]
    payload["user_text"] = _detailed_text(payload)
    return payload


# Explicit alias for callers that prefer the word "report".
build_detailed_recognition_report = build_detailed_diagnostic


@dataclass(frozen=True)
class UserDiagnosticView:
    status: str
    severity: str
    title: str
    summary: str
    next_action: str
    groups: tuple[dict[str, object], ...]
    error_copy: str
    summary_copy: str
    technical_copy: str
    detailed_report: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "状态": self.status,
            "严重程度": self.severity,
            "标题": self.title,
            "说明": self.summary,
            "下一步操作": self.next_action,
            "分组": [dict(group) for group in self.groups],
            "可复制错误说明": self.error_copy,
            "可复制诊断摘要": self.summary_copy,
            "技术详情": self.technical_copy,
            "详细识别报告": self.detailed_report,
            "详细识别文本": str(self.detailed_report.get("user_text") or ""),
        }


def _readiness(report: Mapping[str, object]) -> Mapping[str, object]:
    return _readiness_from(report, None)


def _group(name: str, status: str, details: Mapping[str, object]) -> dict[str, object]:
    return {"名称": name, "状态": status, "详情": [{"字段": str(key), "内容": str(value)} for key, value in details.items()]}


def build_user_view(report: Mapping[str, object]) -> UserDiagnosticView:
    readiness = _readiness(report)
    code = str(readiness.get("primary_reason") or readiness.get("error_code") or "").upper()
    status_code = str(readiness.get("status") or "").upper()
    if not code:
        errors = report.get("errors")
        if isinstance(errors, list) and errors:
            first = errors[0] if isinstance(errors[0], Mapping) else {}
            code = str(first.get("code") or "CAPTURE_FAILED").upper()
            status_code = "FAIL"
        else:
            code = "LISTENING"
            status_code = "WAIT"
    title, default_summary, default_action = _CODE_TEXT.get(code, ("诊断完成", "已完成一次窗口和牌局检查", "查看诊断结果并决定下一步操作"))
    status = _STATUS_TEXT.get(status_code, "等待处理")
    severity = "正常" if status_code == "PASS" else "阻断" if status_code == "FAIL" else "等待"
    summary = str(readiness.get("message") or default_summary)
    next_action = str(readiness.get("suggested_action") or default_action)

    window = report.get("window") if isinstance(report.get("window"), Mapping) else {}
    capture = report.get("capture") if isinstance(report.get("capture"), Mapping) else {}
    roi = report.get("roi_validation") if isinstance(report.get("roi_validation"), Mapping) else {}
    recognition = report.get("recognition") if isinstance(report.get("recognition"), Mapping) else {}
    result = recognition.get("result") if isinstance(recognition.get("result"), Mapping) else {}
    client = window.get("client_rect") if isinstance(window.get("client_rect"), Mapping) else {}
    standard = capture.get("standardization") if isinstance(capture.get("standardization"), Mapping) else {}
    hand = result.get("my_hand") if isinstance(result.get("my_hand"), list) else []
    detailed = report.get("detailed_diagnostic")
    if not isinstance(detailed, Mapping):
        detailed = build_detailed_diagnostic(report)
    detailed_copy = dict(_plain(detailed)) if isinstance(_plain(detailed), Mapping) else {}

    roi_issues = _roi_opening_issues(roi)
    roi_all_issues = _list(roi.get("issues"))
    roi_action_only = bool(roi_all_issues) and not roi_issues
    roi_status = str(roi.get("status", "")).lower()
    roi_normal = bool(roi) and (
        roi_status in {"pass", "ok", "valid"}
        or roi_action_only
    )

    groups = (
        _group("目标窗口", "正常" if window and not window.get("iconic") else "需要处理", {
            "窗口标题": window.get("title", "未取得"),
            "所属程序": window.get("process_name", "未取得"),
            "窗口状态": "已最小化" if window.get("iconic") else "可见",
            "客户区大小": f"{client.get('width', '?')} × {client.get('height', '?')}",
            "显示缩放": f"{window.get('dpi', '?')}",
        }),
        _group("截图捕获", "正常" if capture else "等待处理", {
            "捕获方式": capture.get("backend", "尚未测试"),
            "原始尺寸": str(capture.get("size", "尚未测试")),
            "标准尺寸": str(standard.get("standardized_size", "尚未标准化")),
        }),
        _group("识别区域", "正常" if roi_normal else "需要处理" if roi else "等待处理", {
            "校验状态": roi.get("status", "尚未校验"),
            "问题数量": len(roi.get("issues", [])) if isinstance(roi.get("issues"), list) else 0,
        }),
        _group("牌局识别", "正常" if result else "等待处理", {
            "级牌": result.get("round_level", "尚未识别"),
            "手牌数量": f"{len(hand)} 张",
            "当前行动者": result.get("current_player", "尚未识别"),
        }),
        _group("开局判断", status, {"状态": status, "主要原因": title, "说明": summary}),
    )

    detailed_text = str(detailed_copy.get("user_text") or "")
    error_copy = f"问题：{title}\n\n说明：{summary}\n\n下一步：{next_action}"
    summary_copy = "\n".join([
        "大掼蛋窗口诊断摘要",
        f"总体状态：{status}",
        f"主要问题：{title}",
        f"说明：{summary}",
        f"下一步操作：{next_action}",
        f"手牌数量：{len(hand)} 张",
        f"报告状态码：{code}",
        "",
        detailed_text,
    ])
    technical_copy = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    return UserDiagnosticView(status, severity, title, summary, next_action, groups, error_copy, summary_copy, technical_copy, detailed_copy)


__all__ = [
    "UserDiagnosticView",
    "build_detailed_diagnostic",
    "build_detailed_recognition_report",
    "build_user_view",
]
