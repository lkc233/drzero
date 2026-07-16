#!/usr/bin/env python3
"""Summarize VERL per-step timing metrics from a training log."""

from __future__ import annotations

import argparse
import re
import statistics
from pathlib import Path


ANSI = re.compile(r"\x1b\[[0-9;]*[mK]")
METRIC = re.compile(r"(timing_s/[a-z_]+):([0-9.eE+-]+)")
GENERATION_METRIC = re.compile(r"(generation/[a-z0-9_]+):([0-9.eE+-]+)")


def parse_rows(path: Path) -> list[dict[str, float]]:
    rows = []
    for raw_line in path.read_text(errors="replace").splitlines():
        line = ANSI.sub("", raw_line)
        if "timing_s/step:" not in line:
            continue
        metrics = {key: float(value) for key, value in METRIC.findall(line)}
        metrics.update({key: float(value) for key, value in GENERATION_METRIC.findall(line)})
        step = re.search(r"step:(\d+)", line)
        if step:
            metrics["step"] = float(step.group(1))
            rows.append(metrics)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--last", type=int, default=0, help="only analyze the last N steps")
    args = parser.parse_args()
    rows = parse_rows(args.log)
    if args.last:
        rows = rows[-args.last :]
    if not rows:
        raise SystemExit(f"no completed timing rows found in {args.log}")

    components = [
        ("Generation", "timing_s/gen"),
        ("Reward", "timing_s/reward"),
        ("Old log-prob", "timing_s/old_log_prob"),
        ("Actor update", "timing_s/update_actor"),
        ("Advantage", "timing_s/adv"),
    ]
    step_total = sum(row["timing_s/step"] for row in rows)
    print(f"steps: {int(rows[0]['step'])}-{int(rows[-1]['step'])} ({len(rows)} rows)")
    print("module\tmean_s\tmedian_s\tmin_s\tmax_s\tshare")
    for label, key in components:
        values = [row[key] for row in rows]
        print(
            f"{label}\t{statistics.mean(values):.2f}\t{statistics.median(values):.2f}"
            f"\t{min(values):.2f}\t{max(values):.2f}\t{sum(values) / step_total:.1%}"
        )
    step_values = [row["timing_s/step"] for row in rows]
    print(
        f"Total step\t{statistics.mean(step_values):.2f}\t{statistics.median(step_values):.2f}"
        f"\t{min(step_values):.2f}\t{max(step_values):.2f}\t100.0%"
    )
    split_keys = [
        ("Model latency share", "generation/model_latency_share"),
        ("Retriever latency share", "generation/retriever_latency_share"),
        ("Tool/framework overhead share", "generation/overhead_latency_share"),
        ("Retriever requests per step", "generation/retriever_request_count"),
        ("Retriever request P95 seconds", "generation/retriever_request_seconds_p95"),
        ("Retriever request P99 seconds", "generation/retriever_request_seconds_p99"),
    ]
    available = [(label, key) for label, key in split_keys if all(key in row for row in rows)]
    if available:
        print("\ngeneration/retrieval split (mean across selected steps)")
        for label, key in available:
            print(f"{label}\t{statistics.mean(row[key] for row in rows):.4f}")


if __name__ == "__main__":
    main()
