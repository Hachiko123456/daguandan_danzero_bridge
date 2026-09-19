from __future__ import annotations

import argparse
from pathlib import Path
import os
import tempfile
import zipfile


def normalize(path: Path) -> None:
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temp = Path(temp_name)
    try:
        with zipfile.ZipFile(path, "r") as source:
            entries = [(name, source.read(name)) for name in sorted(source.namelist())]
        with zipfile.ZipFile(
            temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as target:
            for name, data in entries:
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.create_version = 20
                info.extract_version = 20
                info.flag_bits = 0x800
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                target.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", type=Path, required=True)
    args = parser.parse_args()
    normalize(args.wheel.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
