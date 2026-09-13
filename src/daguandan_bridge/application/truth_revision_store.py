"""Versioned storage for manually reviewed TruthLogs.

The session directory is intentionally treated as an opaque directory.  This
module never derives a profile or template path from its parents and it never
touches the source video, frame index, or timeline.  Only the TruthLog itself
and the store's own ``truth_revisions`` directory are written.

TruthLog files contain both semantic labels (initial state, actions and
outcome) and review metadata (evidence frames, provenance and label status).
Derived tests can declare whether they depend on semantic labels.  When a
semantic revision is saved, those results are automatically reported as
``stale``; evidence-only edits keep semantic results fresh while still
preserving the exact revision that produced them.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ..domain.truth import LabelStatus, normalize_label_status
from ..live.truth_log import (
    TruthLog,
    load_truth_log,
    truth_log_from_dict,
    validate_truth_log_card_inventory,
)
from .replay_turn_draft import (
    validate_truth_log_with_live_reducer,
    validate_turn_actor_chain,
)
from ..storage import atomic_write_json


MANIFEST_SCHEMA = "guandan.truth-revisions/1"
MANIFEST_FILENAME = "truth_revision_manifest.json"
REVISION_DIRECTORY = "truth_revisions"
_REVISION_PATTERN = re.compile(r"^revision-(\d{6})\.json$")


class TruthRevisionError(RuntimeError):
    """Base exception raised by :class:`TruthRevisionStore`."""


class RevisionConflictError(TruthRevisionError):
    """Raised when a caller tries to save over a newer revision."""


class RevisionNotFoundError(TruthRevisionError):
    """Raised when a requested revision does not exist."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def truth_log_sha256(log: TruthLog) -> str:
    """Return the hash of the canonical persisted TruthLog representation."""

    return _sha256(truth_log_from_dict(log.to_dict()).to_dict())


def _semantic_dict(log: TruthLog) -> dict[str, object]:
    """Project a TruthLog to fields that affect replay and recommendations.

    Review evidence, provenance and label status are deliberately excluded.
    ``source_video`` and ``frame_index_path`` identify the immutable source,
    but changing a relative path in a moved session must not invalidate the
    semantic labels.
    """

    return {
        "source_session_id": log.source_session_id,
        "initial_state": {
            "round_level": log.initial_state.round_level,
            "lead_player": log.initial_state.lead_player,
            "my_hand": list(log.initial_state.my_hand),
            "seat_hand_sizes": dict(log.initial_state.seat_hand_sizes),
        },
        "turns": [
            {
                "index": turn.index,
                "trick_id": turn.trick_id,
                "actor": turn.actor,
                "is_pass": turn.is_pass,
                "cards": list(turn.cards),
                "move_semantics": turn.move_semantics,
            }
            for turn in log.turns
        ],
        "outcome": log.outcome.to_dict(),
    }


def truth_log_semantic_sha256(log: TruthLog) -> str:
    """Return a stable hash for fields used by semantic replay/evaluation."""

    return _sha256(_semantic_dict(log))


def _diff_values(before: object, after: object, prefix: str = "") -> list[str]:
    """Return human-readable leaf paths that differ between JSON values."""

    if isinstance(before, dict) and isinstance(after, dict):
        result: list[str] = []
        for key in sorted(set(before) | set(after)):
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in before or key not in after:
                result.append(child)
            else:
                result.extend(_diff_values(before[key], after[key], child))
        return result
    if isinstance(before, list) and isinstance(after, list):
        result = []
        for index in range(max(len(before), len(after))):
            child = f"{prefix}[{index}]"
            if index >= len(before) or index >= len(after):
                result.append(child)
            else:
                result.extend(_diff_values(before[index], after[index], child))
        return result
    return [] if before == after else [prefix or "value"]


def truth_log_changed_fields(before: TruthLog | None, after: TruthLog) -> tuple[str, ...]:
    """Compute changed TruthLog paths for revision review and audit reports."""

    if before is None:
        return ("initial_state", "turns", "outcome")
    return tuple(_diff_values(before.to_dict(), after.to_dict()))


@dataclass(frozen=True)
class TruthRevision:
    revision_id: str
    session_id: str
    parent_revision_id: str | None
    truth_sha256: str
    semantic_sha256: str
    label_status: LabelStatus
    changed_fields: tuple[str, ...] = ()
    created_at: str = ""
    author: str = ""
    truth_path: str = "truth_log.json"

    def __post_init__(self) -> None:
        object.__setattr__(self, "label_status", normalize_label_status(self.label_status))

    def to_dict(self) -> dict[str, object]:
        return {
            "revision_id": self.revision_id,
            "session_id": self.session_id,
            "parent_revision_id": self.parent_revision_id,
            "truth_sha256": self.truth_sha256,
            "semantic_sha256": self.semantic_sha256,
            "label_status": self.label_status,
            "changed_fields": list(self.changed_fields),
            "created_at": self.created_at,
            "author": self.author,
            "truth_path": self.truth_path,
        }

    @classmethod
    def from_dict(cls, raw: object) -> "TruthRevision":
        if not isinstance(raw, dict):
            raise TruthRevisionError("revision metadata must be an object")
        changed = raw.get("changed_fields", ())
        if not isinstance(changed, (list, tuple)):
            raise TruthRevisionError("revision changed_fields must be an array")
        return cls(
            revision_id=str(raw.get("revision_id", "")),
            session_id=str(raw.get("session_id", "")),
            parent_revision_id=(
                str(raw["parent_revision_id"])
                if raw.get("parent_revision_id") is not None
                else None
            ),
            truth_sha256=str(raw.get("truth_sha256", "")),
            semantic_sha256=str(raw.get("semantic_sha256", "")),
            label_status=normalize_label_status(raw.get("label_status", "draft")),
            changed_fields=tuple(str(item) for item in changed),
            created_at=str(raw.get("created_at", "")),
            author=str(raw.get("author", "")),
            truth_path=str(raw.get("truth_path", "truth_log.json")),
        )


@dataclass(frozen=True)
class DerivedResult:
    result_id: str
    path: str
    based_on_revision_id: str
    based_on_truth_sha256: str
    based_on_semantic_sha256: str
    semantic_dependency: bool = True
    status: str = "fresh"
    stale_reason: str = ""
    registered_at: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "result_id": self.result_id,
            "path": self.path,
            "based_on_revision_id": self.based_on_revision_id,
            "based_on_truth_sha256": self.based_on_truth_sha256,
            "based_on_semantic_sha256": self.based_on_semantic_sha256,
            "semantic_dependency": self.semantic_dependency,
            "status": self.status,
            "stale_reason": self.stale_reason,
            "registered_at": self.registered_at,
        }

    @classmethod
    def from_dict(cls, raw: object) -> "DerivedResult":
        if not isinstance(raw, dict):
            raise TruthRevisionError("derived result metadata must be an object")
        status = str(raw.get("status", "fresh"))
        if status not in {"fresh", "stale"}:
            raise TruthRevisionError(f"invalid derived result status: {status}")
        return cls(
            result_id=str(raw.get("result_id", "")),
            path=str(raw.get("path", "")),
            based_on_revision_id=str(raw.get("based_on_revision_id", "")),
            based_on_truth_sha256=str(raw.get("based_on_truth_sha256", "")),
            based_on_semantic_sha256=str(raw.get("based_on_semantic_sha256", "")),
            semantic_dependency=bool(raw.get("semantic_dependency", True)),
            status=status,
            stale_reason=str(raw.get("stale_reason", "")),
            registered_at=str(raw.get("registered_at", "")),
        )


@dataclass(frozen=True)
class TruthRevisionManifest:
    session_id: str
    current_revision_id: str | None
    current_truth_sha256: str | None
    current_semantic_sha256: str | None
    revisions: tuple[TruthRevision, ...] = ()
    derived_results: tuple[DerivedResult, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": MANIFEST_SCHEMA,
            "session_id": self.session_id,
            "current_revision_id": self.current_revision_id,
            "current_truth_sha256": self.current_truth_sha256,
            "current_semantic_sha256": self.current_semantic_sha256,
            "revisions": [item.to_dict() for item in self.revisions],
            "derived_results": [item.to_dict() for item in self.derived_results],
        }

    @classmethod
    def from_dict(cls, raw: object) -> "TruthRevisionManifest":
        if not isinstance(raw, dict) or raw.get("schema") != MANIFEST_SCHEMA:
            raise TruthRevisionError("unsupported TruthLog revision manifest")
        revisions = raw.get("revisions", ())
        results = raw.get("derived_results", ())
        if not isinstance(revisions, list) or not isinstance(results, list):
            raise TruthRevisionError("revision manifest arrays are invalid")
        return cls(
            session_id=str(raw.get("session_id", "")),
            current_revision_id=(
                str(raw["current_revision_id"])
                if raw.get("current_revision_id") is not None
                else None
            ),
            current_truth_sha256=(
                str(raw["current_truth_sha256"])
                if raw.get("current_truth_sha256") is not None
                else None
            ),
            current_semantic_sha256=(
                str(raw["current_semantic_sha256"])
                if raw.get("current_semantic_sha256") is not None
                else None
            ),
            revisions=tuple(TruthRevision.from_dict(item) for item in revisions),
            derived_results=tuple(DerivedResult.from_dict(item) for item in results),
        )


class TruthRevisionStore:
    """Persist TruthLog revisions for one arbitrarily located session."""

    def __init__(
        self,
        session_path: Path,
        *,
        truth_filename: str = "truth_log.json",
        revision_directory: str = REVISION_DIRECTORY,
        session_id: str | None = None,
    ) -> None:
        session = Path(session_path).expanduser().resolve()
        if not session.exists() or not session.is_dir():
            raise ValueError(f"session_path must be an existing directory: {session}")
        if not truth_filename or Path(truth_filename).name != truth_filename:
            raise ValueError("truth_filename must be a file name, not a path")
        if not revision_directory or Path(revision_directory).name != revision_directory:
            raise ValueError("revision_directory must be a directory name")
        self.session_path = session
        self.truth_path = session / truth_filename
        self.revision_path = session / revision_directory
        self.manifest_path = session / MANIFEST_FILENAME
        self._explicit_session_id = str(session_id).strip() if session_id else None

    def current(self) -> TruthLog | None:
        if not self.truth_path.exists():
            return None
        return load_truth_log(self.truth_path, session_id=self._session_id())

    def manifest(self) -> TruthRevisionManifest:
        if self.manifest_path.exists():
            raw = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            manifest = TruthRevisionManifest.from_dict(raw)
            if manifest.session_id and manifest.session_id != self._session_id():
                raise TruthRevisionError("revision manifest belongs to another session")
            return manifest
        current = self.current()
        if current is None:
            return TruthRevisionManifest(self._session_id(), None, None, None)
        # A pre-versioning TruthLog is exposed as a virtual baseline.  The
        # baseline is materialized on the first save, preserving compatibility
        # without rewriting an existing session during a read-only inspection.
        baseline = self._baseline_revision(current)
        return TruthRevisionManifest(
            self._session_id(),
            baseline.revision_id,
            baseline.truth_sha256,
            baseline.semantic_sha256,
            (baseline,),
        )

    def save_draft(
        self,
        log: TruthLog,
        *,
        changed_fields: Iterable[str] | None = None,
        author: str = "",
        expected_parent_revision_id: str | None = None,
    ) -> TruthRevision:
        return self._save(
            replace(log, label_status="draft"),
            changed_fields=changed_fields,
            author=author,
            expected_parent_revision_id=expected_parent_revision_id,
        )

    def publish(
        self,
        log: TruthLog,
        *,
        changed_fields: Iterable[str] | None = None,
        author: str = "",
        expected_parent_revision_id: str | None = None,
    ) -> TruthRevision:
        # Publishing is the storage-level verified gate.  The GUI performs the
        # same checks earlier so it can offer an explicit draft-only fallback,
        # but non-GUI callers must not be able to bypass the production state
        # machine before a log is marked verified.
        validate_turn_actor_chain(log)
        validate_truth_log_with_live_reducer(log)
        return self._save(
            replace(log, label_status="verified"),
            changed_fields=changed_fields,
            author=author,
            expected_parent_revision_id=expected_parent_revision_id,
        )

    def verify(
        self,
        log: TruthLog,
        *,
        changed_fields: Iterable[str] | None = None,
        author: str = "",
        expected_parent_revision_id: str | None = None,
    ) -> TruthRevision:
        """Explicit human-review alias for :meth:`publish`."""

        return self.publish(
            log,
            changed_fields=changed_fields,
            author=author,
            expected_parent_revision_id=expected_parent_revision_id,
        )

    def load_revision(self, revision_id: str) -> TruthLog:
        for revision in self.manifest().revisions:
            if revision.revision_id == revision_id:
                path = self.revision_path / f"{revision.revision_id}.json"
                if not path.exists():
                    # The first revision of a pre-versioning session is a
                    # virtual baseline until the first write materializes it.
                    if revision_id == "revision-000001" and revision.created_at == "":
                        current = self.current()
                        if current is not None and truth_log_sha256(current) == revision.truth_sha256:
                            return current
                    raise RevisionNotFoundError(f"revision file is missing: {revision_id}")
                log = load_truth_log(path, session_id=self._session_id())
                if truth_log_sha256(log) != revision.truth_sha256:
                    raise TruthRevisionError(f"revision hash mismatch: {revision_id}")
                return log
        raise RevisionNotFoundError(revision_id)

    def register_derived_result(
        self,
        result_id: str,
        path: Path | str,
        *,
        based_on_revision_id: str | None = None,
        semantic_dependency: bool = True,
    ) -> DerivedResult:
        if not str(result_id).strip():
            raise ValueError("result_id cannot be empty")
        manifest = self.manifest()
        revision_id = based_on_revision_id or manifest.current_revision_id
        if revision_id is None:
            raise TruthRevisionError("cannot register a result before saving a TruthLog")
        revision = next(
            (item for item in manifest.revisions if item.revision_id == revision_id), None
        )
        if revision is None:
            raise RevisionNotFoundError(revision_id)
        result = DerivedResult(
            result_id=str(result_id),
            path=str(Path(path)),
            based_on_revision_id=revision.revision_id,
            based_on_truth_sha256=revision.truth_sha256,
            based_on_semantic_sha256=revision.semantic_sha256,
            semantic_dependency=bool(semantic_dependency),
            registered_at=_now(),
        )
        results = [item for item in manifest.derived_results if item.result_id != result.result_id]
        results.append(result)
        self._write_manifest(replace(manifest, derived_results=tuple(results)))
        return result

    def derived_results(self) -> tuple[DerivedResult, ...]:
        manifest = self.manifest()
        current_semantic = manifest.current_semantic_sha256
        current_truth = manifest.current_truth_sha256
        refreshed: list[DerivedResult] = []
        changed = False
        for result in manifest.derived_results:
            stale = (
                result.semantic_dependency
                and current_semantic is not None
                and result.based_on_semantic_sha256 != current_semantic
            ) or (
                not result.semantic_dependency
                and current_truth is not None
                and result.based_on_truth_sha256 != current_truth
            )
            status = "stale" if stale else "fresh"
            reason = "truth_semantics_changed" if stale and result.semantic_dependency else (
                "truth_revision_changed" if stale else ""
            )
            if result.status != status or result.stale_reason != reason:
                result = replace(result, status=status, stale_reason=reason)
                changed = True
            refreshed.append(result)
        if changed and self.manifest_path.exists():
            self._write_manifest(replace(manifest, derived_results=tuple(refreshed)))
        return tuple(refreshed)

    def _save(
        self,
        log: TruthLog,
        *,
        changed_fields: Iterable[str] | None,
        author: str,
        expected_parent_revision_id: str | None,
    ) -> TruthRevision:
        # Physical-card limits are hard invariants for both verified logs and
        # forced drafts.  A draft may preserve an actor-chain error, but it
        # must never persist an impossible double-deck inventory.
        validate_truth_log_card_inventory(log)
        previous = self.current()
        manifest = self.manifest()
        # A copied session can live under any directory name.  When no
        # version metadata exists yet, the TruthLog's own source ID becomes
        # the identity; subsequent saves are checked against the manifest.
        if previous is None and not self.manifest_path.exists():
            manifest = replace(manifest, session_id=log.source_session_id)
        if log.source_session_id != manifest.session_id:
            raise ValueError(
                "TruthLog source_session_id does not match the selected session"
            )
        parent_id = manifest.current_revision_id
        if expected_parent_revision_id is not None and expected_parent_revision_id != parent_id:
            raise RevisionConflictError(
                f"expected parent {expected_parent_revision_id}, current is {parent_id}"
            )
        virtual_baseline = next(
            (
                item
                for item in manifest.revisions
                if item.revision_id == manifest.current_revision_id
                and item.created_at == ""
            ),
            None,
        )
        if (
            previous is not None
            and virtual_baseline is not None
            and not (self.revision_path / f"{virtual_baseline.revision_id}.json").exists()
        ):
            # A legacy truth_log.json predates revision tracking.  Freeze it
            # before the canonical file is replaced so revision 1 remains an
            # auditable snapshot rather than a virtual record.
            self._materialize_bootstrap(previous, manifest)
            manifest = self.manifest()
            parent_id = manifest.current_revision_id
        truth_hash = truth_log_sha256(log)
        semantic_hash = truth_log_semantic_sha256(log)
        current_revision = next(
            (
                item
                for item in manifest.revisions
                if item.revision_id == manifest.current_revision_id
            ),
            None,
        )
        # Saving the same canonical content and label status is a no-op.  This
        # prevents the GUI's repeated Save click from producing meaningless
        # copies while still allowing draft -> verified to create a revision.
        if (
            current_revision is not None
            and current_revision.truth_sha256 == truth_hash
            and current_revision.label_status == log.label_status
        ):
            if not self.manifest_path.exists():
                self._materialize_bootstrap(previous, manifest)
                manifest = self.manifest()
                current_revision = next(
                    item
                    for item in manifest.revisions
                    if item.revision_id == manifest.current_revision_id
                )
            return current_revision
        revision_number = self._next_revision_number(manifest)
        revision_id = f"revision-{revision_number:06d}"
        fields = tuple(changed_fields) if changed_fields is not None else truth_log_changed_fields(previous, log)
        revision = TruthRevision(
            revision_id=revision_id,
            session_id=self._session_id(),
            parent_revision_id=parent_id,
            truth_sha256=truth_hash,
            semantic_sha256=semantic_hash,
            label_status=log.label_status,
            changed_fields=tuple(dict.fromkeys(str(item) for item in fields)),
            created_at=_now(),
            author=str(author),
            truth_path=self.truth_path.name,
        )
        self.revision_path.mkdir(parents=True, exist_ok=True)
        revision_file = self.revision_path / f"{revision_id}.json"
        atomic_write_json(revision_file, log.to_dict())
        atomic_write_json(self.truth_path, log.to_dict())
        revisions = [item for item in manifest.revisions if item.revision_id != revision_id]
        revisions.append(revision)
        next_manifest = replace(
            manifest,
            session_id=self._session_id(),
            current_revision_id=revision.revision_id,
            current_truth_sha256=revision.truth_sha256,
            current_semantic_sha256=revision.semantic_sha256,
            revisions=tuple(revisions),
        )
        self._write_manifest(next_manifest)
        # Evaluate lazily through the public method so stale status is persisted
        # consistently and evidence-only edits retain semantic results.
        self.derived_results()
        return revision

    def _write_manifest(self, manifest: TruthRevisionManifest) -> None:
        atomic_write_json(self.manifest_path, manifest.to_dict())

    def _materialize_bootstrap(
        self,
        log: TruthLog | None,
        manifest: TruthRevisionManifest,
    ) -> None:
        """Persist a legacy current TruthLog as revision 1 on first save."""

        if log is None or manifest.current_revision_id is None:
            return
        self.revision_path.mkdir(parents=True, exist_ok=True)
        revision_file = self.revision_path / f"{manifest.current_revision_id}.json"
        if not revision_file.exists():
            atomic_write_json(revision_file, log.to_dict())
        self._write_manifest(manifest)

    def _baseline_revision(self, log: TruthLog) -> TruthRevision:
        return TruthRevision(
            revision_id="revision-000001",
            session_id=self._session_id(),
            parent_revision_id=None,
            truth_sha256=truth_log_sha256(log),
            semantic_sha256=truth_log_semantic_sha256(log),
            label_status=log.label_status,
            changed_fields=("bootstrap",),
            created_at="",
            author="bootstrap",
            truth_path=self.truth_path.name,
        )

    def _next_revision_number(self, manifest: TruthRevisionManifest) -> int:
        numbers = []
        for revision in manifest.revisions:
            match = _REVISION_PATTERN.match(f"{revision.revision_id}.json")
            if match:
                numbers.append(int(match.group(1)))
        return max(numbers, default=0) + 1

    def _session_id(self) -> str:
        if self._explicit_session_id:
            return self._explicit_session_id
        if self.manifest_path.exists():
            try:
                raw = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                value = str(raw.get("session_id", "")).strip()
                if value:
                    return value
            except (OSError, json.JSONDecodeError, AttributeError):
                pass
        if self.truth_path.exists():
            try:
                return load_truth_log(self.truth_path).source_session_id
            except Exception:
                pass
        return self.session_path.name


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


__all__ = [
    "DerivedResult",
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA",
    "REVISION_DIRECTORY",
    "RevisionConflictError",
    "RevisionNotFoundError",
    "TruthRevision",
    "TruthRevisionError",
    "TruthRevisionManifest",
    "TruthRevisionStore",
    "save_truth_log_versioned",
    "truth_log_changed_fields",
    "truth_log_semantic_sha256",
    "truth_log_sha256",
]


def save_truth_log_versioned(
    path: Path,
    log: TruthLog,
    *,
    publish: bool = False,
    changed_fields: Iterable[str] | None = None,
    author: str = "",
    expected_parent_revision_id: str | None = None,
) -> TruthRevision:
    """Compatibility adapter for the legacy ``save_truth_log(path, log)`` call.

    Existing callers can replace one import/call without learning the store's
    directory layout.  ``path.parent`` is the selected session directory and
    no parent-directory convention is required.
    """

    store = TruthRevisionStore(Path(path).parent, truth_filename=Path(path).name)
    if publish:
        return store.publish(
            log,
            changed_fields=changed_fields,
            author=author,
            expected_parent_revision_id=expected_parent_revision_id,
        )
    return store.save_draft(
        log,
        changed_fields=changed_fields,
        author=author,
        expected_parent_revision_id=expected_parent_revision_id,
    )


__all__.append("save_truth_log_versioned")
