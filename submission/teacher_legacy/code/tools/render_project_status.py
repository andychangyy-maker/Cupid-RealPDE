from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path


METRIC_KEYS = ("rel_l2", "tke", "mvpe", "time", "sps")
TRACK_KEYS = ("track1", "track2")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
README_START = "<!-- PROJECT_STATUS:START -->"
README_END = "<!-- PROJECT_STATUS:END -->"
RESULTS_START = "<!-- CURRENT_RESULTS:START -->"
RESULTS_END = "<!-- CURRENT_RESULTS:END -->"
DISPLAY_METRICS = (
    ("Rel-L2", "rel_l2"),
    ("TKE", "tke"),
    ("MVPE", "mvpe"),
    ("Time", "time"),
    ("SPS", "sps"),
)


class StatusError(ValueError):
    pass


def _require_keys(mapping: object, keys: tuple[str, ...], context: str) -> None:
    if not isinstance(mapping, dict):
        raise StatusError(f"{context} must be an object")
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise StatusError(f"{context} missing required keys: {', '.join(missing)}")


def _score_number(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StatusError(f"{context} must be a numeric value")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 100.0:
        raise StatusError(f"{context} must be finite and within 0..100")
    return number


def metric_mean(track: dict) -> float:
    metrics = track["metrics"]
    _require_keys(metrics, METRIC_KEYS, "metrics")
    return sum(
        _score_number(metrics[key], f"metrics.{key}") for key in METRIC_KEYS
    ) / len(METRIC_KEYS)


def validate_status(data: object) -> None:
    _require_keys(
        data,
        ("schema_version", "milestone", "code", "autodl", "assets", "tracks"),
        "status",
    )
    assert isinstance(data, dict)
    if data["schema_version"] != 1:
        raise StatusError("schema_version must be 1")
    _require_keys(
        data["milestone"],
        ("id", "name", "note", "date", "optimization_status"),
        "milestone",
    )
    milestone = data["milestone"]
    assert isinstance(milestone, dict)
    optimization_status = milestone["optimization_status"]
    if not isinstance(optimization_status, str) or not optimization_status.strip():
        raise StatusError("milestone.optimization_status must be a non-empty string")
    _require_keys(data["autodl"], ("gpu",), "autodl")
    _require_keys(data["tracks"], TRACK_KEYS, "tracks")
    tracks = data["tracks"]
    assert isinstance(tracks, dict)

    for track_name in TRACK_KEYS:
        track = tracks[track_name]
        _require_keys(
            track,
            (
                "method",
                "kit",
                "phase",
                "submission_id",
                "submitted_at",
                "metric_source",
                "final_score",
                "metrics",
                "local_validation",
                "artifact",
                "checkpoint",
            ),
            track_name,
        )
        assert isinstance(track, dict)
        if track["metric_source"] != "codabench":
            raise StatusError(f"{track_name} metric_source must be codabench")
        _require_keys(track["metrics"], METRIC_KEYS, f"{track_name}.metrics")
        metrics = track["metrics"]
        assert isinstance(metrics, dict)
        for metric_name in METRIC_KEYS:
            _score_number(metrics[metric_name], f"{track_name}.{metric_name}")
        final_score = _score_number(track["final_score"], f"{track_name}.final_score")
        final_score_rule = track.get("final_score_rule", "arithmetic_mean")
        if final_score_rule not in ("arithmetic_mean", "leaderboard_unpublished"):
            raise StatusError(
                f"{track_name}.final_score_rule must be arithmetic_mean or "
                "leaderboard_unpublished"
            )
        if (final_score_rule == "arithmetic_mean"
                and abs(metric_mean(track) - final_score) > 1e-6):
            raise StatusError(
                f"{track_name}.final_score does not equal the five-metric mean"
            )
        for identity_name in ("artifact", "checkpoint"):
            identity = track[identity_name]
            _require_keys(identity, ("filename", "sha256"), f"{track_name}.{identity_name}")
            assert isinstance(identity, dict)
            sha256 = identity["sha256"]
            if not isinstance(sha256, str) or not SHA256_RE.fullmatch(sha256):
                raise StatusError(
                    f"{track_name}.{identity_name}.sha256 must be 64 lowercase hex chars"
                )
        artifact = track["artifact"]
        assert isinstance(artifact, dict)
        artifact_size = artifact.get("size_bytes")
        if not isinstance(artifact_size, int) or isinstance(artifact_size, bool) or artifact_size <= 0:
            raise StatusError(f"{track_name}.artifact.size_bytes must be positive")


def load_status(path: Path) -> dict:
    def reject_constant(constant: str) -> None:
        raise StatusError(f"invalid JSON constant: {constant}")

    try:
        data = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=reject_constant
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise StatusError(f"cannot read status file {path}: {exc}") from exc
    validate_status(data)
    assert isinstance(data, dict)
    return data


def replace_marked_section(text: str, start: str, end: str, body: str) -> str:
    if text.count(start) != 1 or text.count(end) != 1:
        raise StatusError(f"expected exactly one marker pair: {start} / {end}")
    start_index = text.index(start) + len(start)
    end_index = text.index(end)
    if start_index >= end_index:
        raise StatusError(f"marker order is invalid: {start} / {end}")
    return text[:start_index] + "\n" + body.rstrip() + "\n" + text[end_index:]


def _score(value: float) -> str:
    return f"{float(value):.6f}"


def _svg_text(x: int, y: int, text: str, size: int, weight: int = 400) -> str:
    safe = html.escape(str(text))
    return (
        f'<text x="{x}" y="{y}" font-family="Inter,Segoe UI,Arial,sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="#0f172a">{safe}</text>'
    )


def render_scorecard_svg(data: dict) -> str:
    t1 = data["tracks"]["track1"]
    t2 = data["tracks"]["track2"]
    milestone = data["milestone"]
    gpu = data["autodl"]["gpu"]
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="660" '
        'viewBox="0 0 1200 660" role="img" '
        'aria-labelledby="scorecard-title scorecard-desc">',
        f'<title id="scorecard-title">RealPDE {milestone["id"]} {html.escape(milestone["name"])} scores</title>',
        '<desc id="scorecard-desc">'
        f'Track 1: {_score(t1["final_score"])}; '
        f'Track 2: {_score(t2["final_score"])}; '
        'five metrics compared.</desc>',
        '<rect width="1200" height="660" rx="24" fill="#f8fafc"/>',
        _svg_text(48, 38, f'{milestone["id"]} \u00b7 {milestone["name"]} \u00b7 {milestone["date"]}', 22, 700),
        _svg_text(48, 64, milestone["note"], 16, 500),
        _svg_text(48, 88, f"GPU: {gpu}", 16, 600),
        '<rect x="48" y="108" width="532" height="154" rx="18" fill="#eff6ff" stroke="#2563eb"/>',
        '<rect x="620" y="108" width="532" height="154" rx="18" fill="#f5f3ff" stroke="#7c3aed"/>',
        _svg_text(80, 148, f'Track 1 \u00b7 {t1["method"]}', 20, 700),
        _svg_text(652, 148, f'Track 2 \u00b7 {t2["method"]}', 20, 700),
        _svg_text(80, 174, t1["phase"], 16, 600),
        _svg_text(652, 174, t2["phase"], 16, 600),
        _svg_text(80, 232, _score(t1["final_score"]), 56, 800),
        _svg_text(652, 232, _score(t2["final_score"]), 56, 800),
    ]
    y = 300
    bar_x = 250
    bar_width = 760
    for label, key in DISPLAY_METRICS:
        left = float(t1["metrics"][key])
        right = float(t2["metrics"][key])
        parts.extend(
            (
                _svg_text(48, y + 20, label, 20, 700),
                f'<rect x="{bar_x}" y="{y}" width="{bar_width}" height="12" rx="6" fill="#e2e8f0"/>',
                f'<rect x="{bar_x}" y="{y}" width="{bar_width * left / 100:.2f}" height="12" rx="6" fill="#2563eb"/>',
                f'<rect x="{bar_x}" y="{y + 20}" width="{bar_width}" height="12" rx="6" fill="#e2e8f0"/>',
                f'<rect x="{bar_x}" y="{y + 20}" width="{bar_width * right / 100:.2f}" height="12" rx="6" fill="#7c3aed"/>',
                _svg_text(1030, y + 11, _score(left), 16, 700),
                _svg_text(1030, y + 31, _score(right), 16, 700),
            )
        )
        y += 62
    parts.extend(
        (
            '<circle cx="48" cy="634" r="7" fill="#2563eb"/>',
            _svg_text(64, 640, "Track 1", 17, 600),
            '<circle cx="160" cy="634" r="7" fill="#7c3aed"/>',
            _svg_text(176, 640, "Track 2", 17, 600),
            _svg_text(650, 640, f'{milestone["note"]} \u00b7 Scores are not directly comparable across tracks', 14, 500),
            "</svg>",
        )
    )
    return "\n".join(parts) + "\n"


def render_readme_status(data: dict) -> str:
    t1 = data["tracks"]["track1"]
    t2 = data["tracks"]["track2"]
    milestone = data["milestone"]
    gpu = data["autodl"]["gpu"].replace("NVIDIA GeForce ", "")
    return "\n".join(
        (
            "## \U0001f6a6 \u5f53\u524d\u91cc\u7a0b\u7891",
            "",
            "> [!IMPORTANT]",
            f"> **{milestone['id']}\uff1a{milestone['name']} \u5df2\u5efa\u7acb\u3002**",
            f"> {milestone['note']}",
            "",
            f"<kbd>\u2705 \u53cc\u8d5b\u9053\u5df2\u63d0\u4ea4</kbd> <kbd>\U0001f5a5\ufe0f {gpu} \u5df2\u9a8c\u8bc1</kbd> "
            f"<kbd>\U0001f9ea {milestone['optimization_status']}</kbd>",
            "",
            f"![RealPDE {milestone['id']} {milestone['name']} \u6210\u7ee9\u5361](docs/scorecard.svg)",
            "",
            "| \u8d5b\u9053 | \u5f53\u524d\u65b9\u6cd5 | \u5b98\u65b9\u5206\u6570 | Submission ID |",
            "|:---|:---|---:|---:|",
            f"| **Track 1** | {t1['method']} | **{_score(t1['final_score'])}** | `{t1['submission_id']}` |",
            f"| **Track 2** | {t2['method']} | **{_score(t2['final_score'])}** | `{t2['submission_id']}` |",
            "",
            f"\u6210\u7ee9\u6765\u6e90\uff1aCodaBench {t1['phase']} / {t2['phase']}\uff0c\u8bb0\u5f55\u65e5\u671f\uff1a{milestone['date']}\u3002",
            "\u4e24\u6761\u8d5b\u9053\u7684\u4efb\u52a1\u534f\u8bae\u4e0d\u540c\uff0c\u8fd9\u91cc\u7684\u5e76\u6392\u5c55\u793a\u7528\u4e8e\u6c47\u62a5\u8fdb\u5ea6\uff0c\u4e0d\u80fd\u76f4\u63a5\u6a2a\u5411\u6bd4\u8f83\u5206\u6570\u9ad8\u4f4e\u3002",
            "",
            "[v9 \u5b9e\u9a8c\u603b\u770b\u677f\uff08HTML\uff09](docs/TRACK1_V9_EXPERIMENT_DASHBOARD.html) \u00b7 "
            "[\u5b8c\u6574\u6210\u7ee9\u4e0e\u590d\u73b0\u8bc1\u636e](docs/RESULTS.md) \u00b7 "
            "[\u4e09\u6a21\u578b\u5fae\u8c03\u4e0e\u5206\u5dee\u8bca\u65ad](docs/FINETUNE_REPRODUCTION.md) \u00b7 "
            "[FNO TKE loss \u4e0e\u63d0\u4ea4\u5305](docs/TKE_LOSS_EXPERIMENTS.md) \u00b7 "
            "[\u7ed3\u6784\u5316 SPS \u533a\u95f4\u5b9e\u9a8c](docs/STRUCTURED_SPS_EXPERIMENTS.md) \u00b7 "
            "[\u52a8\u6001 SPS \u533a\u95f4\u5b9e\u9a8c](docs/DYNAMIC_SPS_EXPERIMENTS.md) \u00b7 "
            "[\u56fa\u5b9a SPS \u6821\u51c6\u4e0e\u5b98\u7f51\u590d\u6838](docs/SPS_CALIBRATION.md) \u00b7 "
            "[AutoDL\u590d\u73b0\u624b\u518c](docs/AUTODL_BASELINE_RUNBOOK.md) \u00b7 "
            "[\u65b0\u961f\u5458\u914d\u7f6e\u4e0e\u5927\u6587\u4ef6\u83b7\u53d6](docs/TEAM_SHARING_GUIDE.md)",
        )
    )


def render_results_summary(data: dict) -> str:
    t1 = data["tracks"]["track1"]
    t2 = data["tracks"]["track2"]
    rows = (
        ("Rel-L2", "rel_l2"),
        ("TKE", "tke"),
        ("MVPE", "mvpe"),
        ("Time", "time"),
        ("SPS", "sps"),
        ("Final", None),
    )
    table = ["| \u6307\u6807 | Track 1 | Track 2 |", "|:---|---:|---:|"]
    for label, key in rows:
        left = t1["final_score"] if key is None else t1["metrics"][key]
        right = t2["final_score"] if key is None else t2["metrics"][key]
        table.append(f"| {label} | {_score(left)} | {_score(right)} |")
    return "\n".join(
        (
            "## \u5f53\u524d\u6b63\u5f0f\u6210\u7ee9",
            "",
            f"- \u8bc4\u6d4b\u6765\u6e90\uff1aCodaBench\uff0cTrack 1 `{t1['phase']}`\uff1bTrack 2 `{t2['phase']}`",
            f"- \u8bb0\u5f55\u65e5\u671f\uff1a{data['milestone']['date']}",
            f"- Track 1 Submission ID\uff1a`{t1['submission_id']}`",
            f"- Track 2 Submission ID\uff1a`{t2['submission_id']}`",
            "",
            *table,
            "",
            "### \u63d0\u4ea4\u5305\u4e0e checkpoint \u8eab\u4efd",
            "",
            "| \u8d5b\u9053 | \u65b9\u6cd5 | Phase | ZIP | ZIP SHA-256 | Checkpoint | Checkpoint SHA-256 |",
            "|:---|:---|:---|:---|:---|:---|:---|",
            f"| Track 1 | {t1['method']} | {t1['phase']} | `{t1['artifact']['filename']}` | `{t1['artifact']['sha256']}` | `{t1['checkpoint']['filename']}` | `{t1['checkpoint']['sha256']}` |",
            f"| Track 2 | {t2['method']} | {t2['phase']} | `{t2['artifact']['filename']}` | `{t2['artifact']['sha256']}` | `{t2['checkpoint']['filename']}` | `{t2['checkpoint']['sha256']}` |",
            "",
            "\u8fd9\u4e9b\u662f Development \u9636\u6bb5\u9690\u85cf\u6807\u7b7e\u4e0a\u7684\u5b98\u65b9\u63d0\u4ea4\u6210\u7ee9\uff1b\u672c\u5730 example \u548c CPU local evaluator \u5206\u6570\u4e0d\u5728\u6b64\u8868\u4e2d\u3002",
            "Track 1 \u65b0\u9636\u6bb5\u7684 Final \u7ec4\u5408\u89c4\u5219\u672a\u516c\u5e03\uff0c\u56e0\u6b64\u5b98\u7f51 Final \u4e0d\u8981\u7528\u4e94\u9879\u7b97\u672f\u5e73\u5747\u91cd\u7b97\u3002",
        )
    )


def build_outputs(root: Path, data: dict) -> dict[Path, str]:
    readme_path = root / "README.md"
    results_path = root / "docs" / "RESULTS.md"
    svg_path = root / "docs" / "scorecard.svg"
    readme = readme_path.read_text(encoding="utf-8")
    results = results_path.read_text(encoding="utf-8")
    return {
        readme_path: replace_marked_section(
            readme, README_START, README_END, render_readme_status(data)
        ),
        results_path: replace_marked_section(
            results, RESULTS_START, RESULTS_END, render_results_summary(data)
        ),
        svg_path: render_scorecard_svg(data),
    }


def stale_outputs(outputs: dict[Path, str]) -> list[Path]:
    stale = []
    for path, expected in outputs.items():
        actual = path.read_text(encoding="utf-8") if path.exists() else None
        if actual != expected:
            stale.append(path)
    return stale


def _temporary_path(path: Path, suffix: str) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=suffix, dir=path.parent
    )
    os.close(descriptor)
    return Path(name)


def write_outputs(outputs: dict[Path, str]) -> None:
    temporary: dict[Path, Path] = {}
    backups: dict[Path, Path | None] = {}
    replaced: list[Path] = []
    clean_backups = True
    try:
        for path, content in outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = _temporary_path(path, ".tmp")
            temp_path.write_text(content, encoding="utf-8", newline="\n")
            temporary[path] = temp_path
        for path in outputs:
            if path.exists():
                backup_path = _temporary_path(path, ".bak")
                shutil.copyfile(path, backup_path)
                backups[path] = backup_path
            else:
                backups[path] = None
        for path, temp_path in temporary.items():
            temp_path.replace(path)
            replaced.append(path)
    except OSError:
        rollback_error: OSError | None = None
        for path in reversed(replaced):
            backup_path = backups[path]
            try:
                if backup_path is None:
                    path.unlink()
                else:
                    backup_path.replace(path)
            except OSError as exc:
                rollback_error = exc
                clean_backups = False
        if rollback_error is not None:
            raise StatusError(f"replacement failed and rollback failed: {rollback_error}")
        raise
    finally:
        for temp_path in temporary.values():
            if temp_path.exists():
                temp_path.unlink()
        if clean_backups:
            for backup_path in backups.values():
                if backup_path is not None and backup_path.exists():
                    backup_path.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render the RealPDE project status")
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    try:
        data = load_status(root / "docs" / "status.json")
        outputs = build_outputs(root, data)
        stale = stale_outputs(outputs)
        if args.check:
            if stale:
                print("Outdated generated files:", file=sys.stderr)
                for path in stale:
                    print(f"  {path.relative_to(root)}", file=sys.stderr)
                return 1
            return 0
        write_outputs(outputs)
        return 0
    except (StatusError, OSError) as exc:
        print(f"status render failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
