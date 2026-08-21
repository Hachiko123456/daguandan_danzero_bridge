from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
import json
from math import prod

from ..danzero.rules import (
    action_for_cards,
    actions_for_cards,
    logical_action_label,
    wildcard_substitutions,
)
from ..danzero.state import GuanDanState, PlayEvent


@dataclass(frozen=True)
class ActionSemanticVariants:
    """Bounded temporary states for unresolved historical action semantics."""

    states: tuple[GuanDanState, ...]
    source_history_indices: tuple[int, ...] = ()
    candidate_counts: tuple[tuple[int, int], ...] = ()
    total_variant_count: int = 1
    branch_points: tuple[dict[str, object], ...] = ()
    error: str = ""

    @property
    def is_uncertain(self) -> bool:
        return bool(self.source_history_indices)

    def to_diagnostic(self) -> dict[str, object]:
        return {
            "semantic_uncertain": self.is_uncertain,
            "source_history_indices": list(self.source_history_indices),
            "candidate_counts": {
                str(index): count for index, count in self.candidate_counts
            },
            "total_variant_count": self.total_variant_count,
            "branch_points": list(self.branch_points),
            "error": self.error,
        }


def state_variants_for_action_semantics(
    state: GuanDanState,
    *,
    limit: int = 32,
) -> ActionSemanticVariants:
    """Expand unresolved semantic declarations without mutating canonical state."""

    max_variants = max(1, int(limit))
    resolved_history = list(state.play_history)
    branch_points: list[tuple[int, PlayEvent, tuple[dict[str, object], ...]]] = []
    branch_audits: list[dict[str, object]] = []

    for event_index, event in enumerate(state.play_history):
        if event.is_pass:
            continue
        metadata = dict(event.action_metadata or {})
        selected = metadata.get("selected_interpretation")
        unresolved = not isinstance(selected, dict) and (
            str(metadata.get("selection_source", "")).strip().casefold()
            == "unresolved"
            or bool(
                metadata.get(
                    "interpretation_ambiguous",
                    metadata.get("ambiguity", False),
                )
            )
        )
        contains_wildcard = f"{state.wild_rank}H" in event.cards
        if not unresolved and not (contains_wildcard and not isinstance(selected, dict)):
            continue

        # A level-heart card is the wildcard. Its physical suit can be known
        # while its declared rank/type is still missing from older histories.
        # GuanDan's convention is deterministic here: choose the strongest
        # legal declaration instead of asking the advisor to guess later.
        if contains_wildcard and not isinstance(selected, dict):
            try:
                candidates = _derived_candidates(event.cards, state.wild_rank)
                strongest = _strongest_candidate(event.cards, state.wild_rank)
            except Exception as exc:
                return _semantic_error(
                    state,
                    event_index,
                    (),
                    f"规则引擎恢复最大逢人配语义时出错：{type(exc).__name__}: {exc}",
                    branch_audits,
                )
            if strongest is None:
                return _semantic_error(
                    state,
                    event_index,
                    candidates,
                    "没有记录候选，规则引擎也无法从实体牌恢复任何合法解释",
                    branch_audits,
                )

            metadata.update(
                {
                    "interpretation_ambiguous": len(candidates) > 1,
                    "candidate_interpretations": [dict(item) for item in candidates],
                    "selected_interpretation": strongest,
                    "selection_source": "rules_strongest_wildcard",
                }
            )
            resolved_history[event_index] = replace(
                event,
                action_metadata=metadata,
            )
            branch_audits.append(
                {
                    "source_history_index": event_index + 1,
                    "physical_cards": list(event.cards),
                    "level": state.round_level,
                    "wild_rank": state.wild_rank,
                    "candidate_source": "rules_strongest_wildcard",
                    "candidate_count": len(candidates),
                    "candidate_interpretations": [dict(item) for item in candidates],
                    "selected_interpretation": dict(strongest),
                }
            )
            continue

        candidates = _metadata_candidates(metadata)
        candidate_source = "recorded_candidates"
        if not candidates:
            try:
                candidates = _derived_candidates(event.cards, state.wild_rank)
            except Exception as exc:
                return _semantic_error(
                    state,
                    event_index,
                    (),
                    f"规则引擎恢复候选时出错：{type(exc).__name__}: {exc}",
                    branch_audits,
                )
            candidate_source = "rules_inferred_candidates"

        if not candidates:
            return _semantic_error(
                state,
                event_index,
                (),
                "没有记录候选，规则引擎也无法从实体牌恢复任何合法解释",
                branch_audits,
            )
        if len(candidates) == 1:
            # A single candidate is safe for the adapter to infer directly.
            continue

        audit = {
            "source_history_index": event_index + 1,
            "physical_cards": list(event.cards),
            "level": state.round_level,
            "wild_rank": state.wild_rank,
            "candidate_source": candidate_source,
            "candidate_count": len(candidates),
            "candidate_interpretations": list(candidates),
        }
        branch_audits.append(audit)
        branch_points.append((event_index, event, candidates))

    trick_count = len(state.trick_plays)
    resolved_state = replace(
        state,
        play_history=resolved_history,
        trick_plays=(
            resolved_history[-trick_count:] if trick_count else []
        ),
    )

    if not branch_points:
        return ActionSemanticVariants(
            states=(resolved_state,),
            branch_points=tuple(branch_audits),
        )

    total_variants = prod(len(candidates) for _, _, candidates in branch_points)
    source_indices = tuple(index + 1 for index, _, _ in branch_points)
    candidate_counts = tuple(
        (index + 1, len(candidates)) for index, _, candidates in branch_points
    )
    if total_variants > max_variants:
        counts_text = "、".join(
            f"第 {index} 条有 {count} 种"
            for index, count in candidate_counts
        )
        return ActionSemanticVariants(
            states=(),
            source_history_indices=source_indices,
            candidate_counts=candidate_counts,
            total_variant_count=total_variants,
            branch_points=tuple(branch_audits),
            error=(
                f"历史动作语义分支共 {total_variants} 个，超过安全上限 "
                f"{max_variants}（{counts_text}）；未执行不完整的模型评估"
            ),
        )

    states: list[GuanDanState] = []
    candidate_groups = [candidates for _, _, candidates in branch_points]
    for branch_index, choices in enumerate(product(*candidate_groups), start=1):
        history = list(resolved_history)
        for (event_index, event, candidates), selected in zip(branch_points, choices):
            metadata = dict(event.action_metadata or {})
            metadata.update(
                {
                    "interpretation_ambiguous": True,
                    "candidate_interpretations": [dict(item) for item in candidates],
                    "selected_interpretation": dict(selected),
                    "selection_source": "candidate_branch",
                    "semantic_branch": {
                        "branch_index": branch_index,
                        "source_history_index": event_index + 1,
                        "candidate_index": candidates.index(selected),
                        "candidate_count": len(candidates),
                    },
                }
            )
            history[event_index] = replace(event, action_metadata=metadata)
        trick = history[-trick_count:] if trick_count else []
        states.append(replace(state, play_history=history, trick_plays=trick))

    return ActionSemanticVariants(
        states=tuple(states),
        source_history_indices=source_indices,
        candidate_counts=candidate_counts,
        total_variant_count=total_variants,
        branch_points=tuple(branch_audits),
    )


def _metadata_candidates(
    metadata: dict[str, object],
) -> tuple[dict[str, object], ...]:
    raw = metadata.get("candidate_interpretations", ())
    if not isinstance(raw, (list, tuple)):
        return ()
    candidates = [dict(item) for item in raw if isinstance(item, dict)]
    return _deduplicate_candidates(candidates)


def _derived_candidates(
    cards: tuple[str, ...],
    wild_rank: str,
) -> tuple[dict[str, object], ...]:
    candidates: list[dict[str, object]] = []
    for action in actions_for_cards(cards, wild_rank):
        candidates.append(
            {
                "move_type": str(action[0]),
                "key": str(action[1]),
                "logical_label": logical_action_label(action, wild_rank),
                "wildcard_assignments": [
                    {"physical_card": card, "as_rank": rank}
                    for card, rank in wildcard_substitutions(action, wild_rank)
                ],
            }
        )
    return _deduplicate_candidates(candidates)


def _strongest_candidate(
    cards: tuple[str, ...],
    wild_rank: str,
) -> dict[str, object] | None:
    action = action_for_cards(cards, wild_rank)
    if action is None:
        return None
    return {
        "move_type": str(action[0]),
        "key": str(action[1]),
        "logical_label": logical_action_label(action, wild_rank),
        "wildcard_assignments": [
            {"physical_card": card, "as_rank": rank}
            for card, rank in wildcard_substitutions(action, wild_rank)
        ],
    }


def _deduplicate_candidates(
    candidates: list[dict[str, object]],
) -> tuple[dict[str, object], ...]:
    unique: dict[str, dict[str, object]] = {}
    for candidate in candidates:
        key = json.dumps(candidate, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        unique.setdefault(key, candidate)
    return tuple(unique.values())


def _semantic_error(
    state: GuanDanState,
    event_index: int,
    candidates: tuple[dict[str, object], ...],
    reason: str,
    branch_audits: list[dict[str, object]],
) -> ActionSemanticVariants:
    event = state.play_history[event_index]
    audit = {
        "source_history_index": event_index + 1,
        "physical_cards": list(event.cards),
        "level": state.round_level,
        "wild_rank": state.wild_rank,
        "candidate_count": len(candidates),
        "candidate_interpretations": list(candidates),
        "reason": reason,
    }
    return ActionSemanticVariants(
        states=(),
        source_history_indices=(event_index + 1,),
        candidate_counts=((event_index + 1, len(candidates)),),
        total_variant_count=0,
        branch_points=tuple((*branch_audits, audit)),
        error=(
            f"第 {event_index + 1} 条历史动作语义无法恢复：实体牌 "
            f"{' '.join(event.cards)}，级牌/逢人配点数 {state.wild_rank}；{reason}"
        ),
    )


__all__ = ["ActionSemanticVariants", "state_variants_for_action_semantics"]
