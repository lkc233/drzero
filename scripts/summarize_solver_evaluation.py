#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path


def markdown_cell(value: object) -> str:
    return str(value).replace("|", r"\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.jsonl.read_text().splitlines() if line.strip()]
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row.get("data_source", "unknown"))].append(row)

    lines = [
        "# Solver Test Results",
        "",
        f"Source: `{args.jsonl}`",
        "",
        "| Dataset | Samples | Correct | Accuracy | Mean score |",
        "|---|---:|---:|---:|---:|",
    ]
    for dataset, items in sorted(grouped.items()):
        scores = [float(item.get("score", 0.0)) for item in items]
        correct = sum(score > 0 for score in scores)
        accuracy = correct / len(scores) if scores else 0.0
        mean_score = sum(scores) / len(scores) if scores else 0.0
        lines.append(
            f"| {markdown_cell(dataset)} | {len(items)} | {correct} | "
            f"{accuracy:.4f} | {mean_score:.4f} |"
        )

    scores = [float(row.get("score", 0.0)) for row in rows]
    correct = sum(score > 0 for score in scores)
    accuracy = correct / len(scores) if scores else 0.0
    mean_score = sum(scores) / len(scores) if scores else 0.0
    lines.append(
        f"| **Overall** | **{len(rows)}** | **{correct}** | "
        f"**{accuracy:.4f}** | **{mean_score:.4f}** |"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
