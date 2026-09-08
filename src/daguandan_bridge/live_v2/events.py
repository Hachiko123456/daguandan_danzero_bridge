"""Compatibility exports for live-v2 action and result contracts."""

from .action_semantics import ActionInterpretation, ActionSemantics
from .candidates import (
    ActionCandidate,
    ActionKind,
    CandidateReason,
    ConfirmationReason,
    ConfirmedAction,
    EvidenceOrigin,
)
from .corrections import ConfirmedCorrection, CorrectionCommand, CorrectionReason
from .results import (
    AdviceOpportunity,
    CommitReason,
    CommitResult,
    EngineUpdate,
    EngineUpdateReason,
    EvidenceDropReason,
    GapPhase,
    GapReason,
    GapState,
    OpportunityReason,
    OpportunityStatus,
    ProjectionReason,
    ProjectionResult,
    ScheduledItemKind,
    SchedulingDropReason,
)

__all__ = [
    "ActionCandidate",
    "ActionInterpretation",
    "ActionKind",
    "ActionSemantics",
    "AdviceOpportunity",
    "CandidateReason",
    "CommitReason",
    "CommitResult",
    "ConfirmationReason",
    "ConfirmedAction",
    "ConfirmedCorrection",
    "CorrectionCommand",
    "CorrectionReason",
    "EngineUpdate",
    "EngineUpdateReason",
    "EvidenceOrigin",
    "EvidenceDropReason",
    "GapPhase",
    "GapReason",
    "GapState",
    "OpportunityReason",
    "OpportunityStatus",
    "ProjectionReason",
    "ProjectionResult",
    "ScheduledItemKind",
    "SchedulingDropReason",
]
