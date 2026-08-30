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
    verify_build_manifest,
    write_build_manifest,
    write_release_record,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="write a bundle build manifest")
    create.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    create.add_argument("--bundle-root", type=Path, required=True)
    create.add_argument("--output", type=Path)
    create.add_argument("--executable-name", default="DaguandanAssistant.exe")
    create.add_argument("--profile-name", default="tencent_daguandan")

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
            document = write_build_manifest(
                args.project_root,
                bundle_root,
                output,
                executable_name=args.executable_name,
                profile_name=args.profile_name,
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


if __name__ == "__main__":
    raise SystemExit(main())
