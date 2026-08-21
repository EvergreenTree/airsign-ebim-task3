#!/usr/bin/env python3
"""Print the official-score table for a set of acceptance run directories.

Usage:
    python3 scripts/summarize_runs.py RUN_DIR [RUN_DIR ...]

Each RUN_DIR must contain ``summary.json``. The table reports the four stage
scores, the total, the highest fully completed stage, the recovery ratio, and
the completion time, followed by the arithmetic mean over every directory
given. Every run passed on the command line is included in the mean, so a
reported mean cannot silently omit a weak run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


FIELDS = ("stage1", "stage2", "stage3", "stage4", "total")


def deferred_scopes(run_dir: Path) -> list[str]:
    """Recover the deferred-scope list from the terminal episode event.

    ``summary.json`` holds live telemetry only; the policy records which scopes
    it gave up on in the ``episode_complete``/``episode_failed`` event.
    """
    episode = run_dir / "episode.jsonl"
    if not episode.is_file():
        return []
    deferred: list[str] = []
    with episode.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") in {"episode_complete", "episode_failed"}:
                details = event.get("details") or {}
                deferred = list(details.get("deferred_failures") or [])
    return deferred


def load(run_dir: Path) -> dict:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    scores = summary.get("stage_scores") or {}
    return {
        "run": run_dir.name,
        "seed": summary.get("seed"),
        "lifecycle": summary.get("lifecycle"),
        "highest_completed_stage": summary.get("highest_completed_stage"),
        "recovery_ratio": summary.get("recovery_ratio") or 0.0,
        "simulated_seconds": summary.get("simulated_seconds") or 0.0,
        "wall_seconds": summary.get("wall_seconds") or 0.0,
        "deferred": summary.get("deferred_failures") or deferred_scopes(run_dir),
        "failure_reason": summary.get("failure_reason"),
        **{name: float(scores.get(name) or 0.0) for name in FIELDS},
    }


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    rows = [load(Path(item)) for item in argv[1:]]
    header = (
        f"{'run':<28} {'seed':>4} {'S1':>5} {'S2':>5} {'S3':>5} {'S4':>5} "
        f"{'total':>6} {'high':>5} {'recov':>6} {'sim_s':>8} {'lifecycle':<10}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['run']:<28} {row['seed']!s:>4} {row['stage1']:>5.2f} "
            f"{row['stage2']:>5.2f} {row['stage3']:>5.2f} {row['stage4']:>5.2f} "
            f"{row['total']:>6.2f} {row['highest_completed_stage']!s:>5} "
            f"{row['recovery_ratio']:>6.2f} {row['simulated_seconds']:>8.1f} "
            f"{row['lifecycle']:<10}"
        )
    print("-" * len(header))
    count = len(rows)
    means = {name: sum(row[name] for row in rows) / count for name in FIELDS}
    print(
        f"{'mean of ' + str(count) + ' run(s)':<28} {'':>4} {means['stage1']:>5.2f} "
        f"{means['stage2']:>5.2f} {means['stage3']:>5.2f} {means['stage4']:>5.2f} "
        f"{means['total']:>6.2f}"
    )
    for row in rows:
        if row["failure_reason"]:
            print(f"\n{row['run']}: FAILED — {row['failure_reason']}")
        if row["deferred"]:
            print(f"\n{row['run']} deferred scopes:")
            for item in row["deferred"]:
                print(f"  - {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
