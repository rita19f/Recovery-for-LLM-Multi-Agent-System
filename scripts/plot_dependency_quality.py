#!/usr/bin/env python3
"""Plot plan-level dependency precision/recall from reviewed run summaries."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHODS = (
    ("declared", "LLM-declared"),
    ("structured", "Structured"),
    ("conservative", "Conservative"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/summaries/run_level.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/summaries"),
    )
    return parser.parse_args()


def pooled_scores(rows: list[dict[str, str]], prefix: str) -> tuple[float, float]:
    counts = {
        key: sum(int(float(row.get(f"{prefix}_{key}") or 0)) for row in rows)
        for key in ("tp", "fp", "fn", "tn")
    }
    precision_denominator = counts["tp"] + counts["fp"]
    recall_denominator = counts["tp"] + counts["fn"]
    precision = counts["tp"] / precision_denominator if precision_denominator else 0.0
    recall = counts["tp"] / recall_denominator if recall_denominator else 0.0
    return precision, recall


def main() -> None:
    args = parse_args()
    with args.input.open(newline="", encoding="utf-8") as stream:
        rows = [
            row
            for row in csv.DictReader(stream)
            if row.get("mode") == "full-restart" and row.get("run_status") == "ok"
        ]
    if not rows:
        raise SystemExit("no completed full-restart rows found")

    scores = [pooled_scores(rows, prefix) for prefix, _ in METHODS]
    precision = [score[0] for score in scores]
    recall = [score[1] for score in scores]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 7.2,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )

    fig, ax = plt.subplots(figsize=(3.22, 2.22))
    x = np.arange(len(METHODS))
    width = 0.34
    bars_precision = ax.bar(
        x - width / 2,
        precision,
        width,
        label="Precision",
        color="#0072B2",
        edgecolor="white",
        linewidth=0.5,
    )
    bars_recall = ax.bar(
        x + width / 2,
        recall,
        width,
        label="Recall",
        color="#E69F00",
        edgecolor="#6F4F00",
        linewidth=0.45,
        hatch="//",
    )
    ax.set_ylabel("Plan-level score")
    ax.set_ylim(0, 1.12)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in METHODS])
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.75)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.03))

    for bars in (bars_precision, bars_recall):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.025,
                f"{bar.get_height():.2f}",
                ha="center",
                va="bottom",
                fontsize=6.8,
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(args.output_dir / f"dependency_precision_recall.{suffix}")
    plt.close(fig)


if __name__ == "__main__":
    main()
