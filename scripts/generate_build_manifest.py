"""Create and verify portable release build manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from daguandan_bridge.build_manifest import (  # noqa: E402
    BUILD_MANIFEST_FILENAME,
    BuildManifestError,
    collect_release_build_inputs,
    collect_source_identity,
    verify_build_manifest,
    write_build_manifest,
    write_release_record,
)
from daguandan_bridge.storage import atomic_write_json  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="write a bundle build manifest")
    create.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    create.add_argument("--bundle-root", type=Path, required=True)
    create.add_argument("--output", type=Path)
    create.add_argument("--executable-name", default="DaguandanAssistant.exe")
    create.add_argument("--profile-name", default="tencent_daguandan")
    create.add_argument("--source-identity", type=Path)
    create.add_argument("--release-input-audit", type=Path)
    create.add_argument("--native-audit", type=Path)

    source = subparsers.add_parser(
        "source-identity",
        help="capture Git source identity before the sanitized build environment",
    )
    source.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    source.add_argument("--output", type=Path, required=True)

    verify = subparsers.add_parser("verify", help="verify files against a manifest")
    verify.add_argument("--bundle-root", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument(
        "--strict",
        action="store_true",
        help="also reject files that were not present when the manifest was created",
    )

    release = subparsers.add_parser(
        "release-record",
        help="write archive SHA256 and a machine-readable release record",
    )
    release.add_argument("--manifest", type=Path, required=True)
    release.add_argument("--archive", type=Path, required=True)
    release.add_argument("--record", type=Path, required=True)
    release.add_argument("--checksum", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "create":
            bundle_root = args.bundle_root.resolve()
            output = (
                args.output.resolve()
                if args.output is not None
                else bundle_root / BUILD_MANIFEST_FILENAME
            )
            if (args.release_input_audit is None) != (args.native_audit is None):
                raise BuildManifestError(
                    "release-input-audit and native-audit must be supplied together"
                )
            source_identity = (
                _read_json_object(args.source_identity)
                if args.source_identity is not None
                else None
            )
            build_inputs = (
                collect_release_build_inputs(
                    args.project_root,
                    bundle_root,
                    release_input_audit=args.release_input_audit,
                    native_audit=args.native_audit,
                )
                if args.release_input_audit is not None
                else None
            )
            document = write_build_manifest(
                args.project_root,
                bundle_root,
                output,
                executable_name=args.executable_name,
                profile_name=args.profile_name,
                source_identity=source_identity,
                build_inputs=build_inputs,
            )
            print(
                json.dumps(
                    {
                        "ok": True,
                        "build_id": document["build_id"],
                        "manifest": str(output),
                    },
                    ensure_ascii=False,
                )
            )
            return 0

        if args.command == "source-identity":
            identity = collect_source_identity(args.project_root)
            atomic_write_json(args.output, identity)
            print(
                json.dumps(
                    {
                        "ok": True,
                        "dirty": identity["dirty"],
                        "output": str(args.output.resolve()),
                    },
                    ensure_ascii=False,
                )
            )
            return 0

        if args.command == "verify":
            result = verify_build_manifest(
                args.bundle_root,
                args.manifest,
                strict=args.strict,
            )
            print(json.dumps(result.to_dict(), ensure_ascii=False))
            return 0 if result.ok else 1

        if args.command == "release-record":
            record = write_release_record(
                args.manifest,
                args.archive,
                args.record,
                args.checksum,
            )
            print(
                json.dumps(
                    {
                        "ok": True,
                        "release_id": record["release_id"],
                        "record": str(args.record.resolve()),
                        "checksum": str(args.checksum.resolve()),
                    },
                    ensure_ascii=False,
                )
            )
            return 0
    except (BuildManifestError, OSError, ValueError) as exc:
        print(f"build manifest error: {exc}", file=sys.stderr)
        return 2
    raise AssertionError(f"unhandled command: {args.command}")


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BuildManifestError(f"JSON input is unreadable: {path.name}") from exc
    if not isinstance(value, dict):
        raise BuildManifestError(f"JSON input must be an object: {path.name}")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
