#!/usr/bin/env python3
"""Rebuild the exact XCB Native-UQ800 package with B16+B12 import prewarm."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import zipfile


SOURCE_ARCHIVE_SHA256 = (
    "b77cc5e00f3137cb9861f55e2662da31286b071b2c8dee79499f0332a86f400c"
)
SOURCE_ARCHIVE_BYTES = 159_425_103
CANDIDATE_ARCHIVE_SHA256 = (
    "1b682db6e9a46df902641897c651a647e11dbecac5822096e2911aaba268cf29"
)
CANDIDATE_ARCHIVE_BYTES = 159_425_134
SOURCE_WRAPPER_SHA256 = (
    "8436505f084b2cebc85f907c2f0e9454112575d583b96eb09e5b7dc9a1b9e150"
)
CANDIDATE_WRAPPER_SHA256 = (
    "1b92b0e61a90599fe95ec0c24e2bb19e2a43888dbb3afad05b812b1a0493761a"
)
EXPECTED_MEMBERS = (
    "submission.py",
    "models/__init__.py",
    "models/unet2d.py",
    "models/ensemble_unet2d.py",
    "ensemble_unet2d.pth",
)
SOURCE_MEMBER_SHA256 = {
    "submission.py": SOURCE_WRAPPER_SHA256,
    "models/__init__.py": (
        "ca44e67d82b6d00c74b8043fd918b91fb17e186b133e45361155e3a949ca3187"
    ),
    "models/unet2d.py": (
        "5e9ba7438bb9cc29612e4a32f26799bef112a15beec64172b1fb751c8f434233"
    ),
    "models/ensemble_unet2d.py": (
        "14ac5c2dbaa63592a831b01314d0bae2d2e343f6803c4a2e799cdecae994ae58"
    ),
    "ensemble_unet2d.pth": (
        "c9bc37120a8b8a0da930092fa5fa9047827b3820c8a9d84d5d5ff0dd8655fa81"
    ),
}
MAX_ARCHIVE_BYTES = 256 * 1024**2


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def patch_wrapper(source: bytes) -> bytes:
    doc_needle = (
        b'"""Track 1 v9 shared-backbone ensemble with residual-quantile bounds."""'
    )
    doc_replacement = b'"""Track 1 B210 with shape-complete A800 cuDNN autotuning."""'
    batch_needle = b"_BATCH_SIZE = 16\n_VALIDATE_OUTPUTS = False"
    batch_replacement = (
        b"_BATCH_SIZE = 16\n_FINAL_BATCH_SIZE = 12\n_VALIDATE_OUTPUTS = False"
    )
    prewarm_needle = (
        b"    dummy = np.zeros((_BATCH_SIZE, 20, 32, 64, 3), dtype=np.float32)\n"
        b"    predict(dummy, metadata=None)\n"
    )
    prewarm_replacement = (
        b"    for batch_size in (_BATCH_SIZE, _FINAL_BATCH_SIZE):\n"
        b"        dummy = np.zeros((batch_size, 20, 32, 64, 3), dtype=np.float32)\n"
        b"        predict(dummy, metadata=None)\n"
    )
    if (
        source.count(doc_needle) != 1
        or source.count(batch_needle) != 1
        or source.count(prewarm_needle) != 1
    ):
        raise ValueError("source wrapper does not match the frozen XCB contract")
    result = source.replace(doc_needle, doc_replacement)
    result = result.replace(batch_needle, batch_replacement)
    result = result.replace(prewarm_needle, prewarm_replacement)
    if b"torch.backends.cudnn.benchmark = True" in result:
        raise ValueError("PW12 must not enable cuDNN benchmark")
    if sha256_bytes(result) != CANDIDATE_WRAPPER_SHA256:
        raise ValueError("patched wrapper identity differs")
    return result


def build(source: Path, output: Path) -> dict[str, object]:
    source = source.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if (
        source.stat().st_size != SOURCE_ARCHIVE_BYTES
        or sha256(source) != SOURCE_ARCHIVE_SHA256
    ):
        raise ValueError("source is not the exact frozen XCB Native-UQ800 archive")

    with zipfile.ZipFile(source) as archive:
        infos = archive.infolist()
        names = tuple(info.filename for info in infos)
        if names != EXPECTED_MEMBERS or len(names) != len(set(names)):
            raise ValueError("source archive members differ")
        if archive.testzip() is not None:
            raise ValueError("source archive CRC failed")
        payloads = {info.filename: archive.read(info) for info in infos}
    observed = {name: sha256_bytes(payload) for name, payload in payloads.items()}
    if observed != SOURCE_MEMBER_SHA256:
        raise ValueError("source member ledger differs")
    wrapper = patch_wrapper(payloads["submission.py"])

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".building")
    if temporary.exists():
        raise FileExistsError(f"stale temporary archive exists: {temporary}")
    try:
        with zipfile.ZipFile(temporary, "w", allowZip64=True) as target:
            for info in infos:
                target.writestr(
                    copy.copy(info),
                    (
                        wrapper
                        if info.filename == "submission.py"
                        else payloads[info.filename]
                    ),
                )
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()

    if (
        output.stat().st_size != CANDIDATE_ARCHIVE_BYTES
        or output.stat().st_size > MAX_ARCHIVE_BYTES
        or sha256(output) != CANDIDATE_ARCHIVE_SHA256
    ):
        raise RuntimeError("rebuilt candidate identity differs")
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("rebuilt candidate CRC failed")
        built = {info.filename: archive.read(info) for info in archive.infolist()}
    unchanged = {
        name: built[name] == payloads[name]
        for name in EXPECTED_MEMBERS
        if name != "submission.py"
    }
    if not all(unchanged.values()) or built["submission.py"] != wrapper:
        raise RuntimeError("build changed more than submission.py")
    return {
        "schema": "track1_xcb_pw12_build_v1",
        "decision": "PASS_EXACT_XCB_PW12_BUILD",
        "source": {
            "path": str(source),
            "bytes": source.stat().st_size,
            "sha256": SOURCE_ARCHIVE_SHA256,
        },
        "candidate": {
            "path": str(output),
            "bytes": output.stat().st_size,
            "sha256": CANDIDATE_ARCHIVE_SHA256,
        },
        "wrapper_sha256": CANDIDATE_WRAPPER_SHA256,
        "non_wrapper_members_bitwise_identical": unchanged,
        "mechanism": "deterministic B16+B12 import prewarm; no model or output change",
        "automatic_submission_authorized": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.report.exists():
        raise FileExistsError(f"refusing to overwrite {args.report}")
    result = build(args.source, args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
