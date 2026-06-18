#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

LABEL_FONTSIZE = 18
TICK_FONTSIZE = 15
LEGEND_FONTSIZE = 15


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def floats(rows: list[dict[str, str]], name: str) -> list[float]:
    return [float(row[name]) for row in rows]


def ints(rows: list[dict[str, str]], name: str) -> list[int]:
    return [int(row[name]) for row in rows]


def plot_turning_summary(output_dir: Path, map_dir: Path | None = None) -> None:
    rows = read_tsv(output_dir / "turning_fraction_summary.tsv")
    rows.sort(key=lambda row: int(row["entry_index"]))

    x = ints(rows, "entry_index")
    true = floats(rows, "true_turning_fraction")
    mean = floats(rows, "mean")
    q05 = floats(rows, "q05")
    q25 = floats(rows, "q25")
    q75 = floats(rows, "q75")
    q95 = floats(rows, "q95")

    map_values = None
    if map_dir is not None:
        map_rows = read_tsv(map_dir / "lbfgs_turning_fraction_map.tsv")
        map_rows.sort(key=lambda row: int(row["entry_index"]))
        map_values = floats(map_rows, "map_turning_fraction")

    fig, ax = plt.subplots(figsize=(13.5, 6.0))
    ax.vlines(x, q05, q95, color="#6296c8", alpha=0.45, linewidth=2.4, label="90% interval")
    ax.vlines(x, q25, q75, color="#1f5e99", alpha=0.65, linewidth=4.5, label="50% interval")
    ax.scatter(x, mean, s=32, color="#0d3b66", label="Mean", zorder=3)
    if map_values is not None:
        ax.scatter(
            x,
            map_values,
            s=13,
            color="#c62828",
            marker="s",
            edgecolors="#ffffff",
            linewidths=0.25,
            label="MAP",
            zorder=4,
        )
    ax.scatter(x, true, s=34, color="#111111", marker="x", linewidths=1.8, label="Truth", zorder=5)
    ax.set_xlabel("Target turning fraction", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Turning fraction", fontsize=LABEL_FONTSIZE)
    ax.set_ylim(-0.04, 1.04)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.grid(True, axis="y", color="#d8d8d8", linewidth=0.7)
    ax.legend(loc="upper right", frameon=True, fontsize=LEGEND_FONTSIZE)
    fig.tight_layout()
    fig.savefig(output_dir / "turning_fraction_recovery.png", dpi=220)
    plt.close(fig)


def plot_prediction_summary(output_dir: Path) -> None:
    rows = read_tsv(output_dir / "prediction_summary.tsv")
    rows.sort(key=lambda row: int(row["observation_index"]))

    x = ints(rows, "observation_index")
    true = floats(rows, "true_density")
    observed = floats(rows, "observed_density")
    mean = floats(rows, "mean")
    q05 = floats(rows, "q05")
    q25 = floats(rows, "q25")
    q75 = floats(rows, "q75")
    q95 = floats(rows, "q95")

    fig, ax = plt.subplots(figsize=(13.5, 5.5))
    ax.fill_between(x, q05, q95, color="#8bb8df", alpha=0.35, linewidth=0, label="90% interval")
    ax.fill_between(x, q25, q75, color="#2f79b7", alpha=0.35, linewidth=0, label="50% interval")
    ax.plot(x, mean, color="#0d3b66", linewidth=1.8, label="Mean")
    ax.scatter(x, true, s=15, color="#202020", marker="x", linewidths=1.1, label="Truth")
    ax.scatter(x, observed, s=12, color="#8a8a8a", alpha=0.65, label="Noisy observations")
    ax.set_xlabel("Sensor observation")
    ax.set_ylabel("Density")
    ax.set_ylim(-0.04, 1.04)
    ax.grid(True, axis="y", color="#d8d8d8", linewidth=0.7)
    ax.legend(loc="upper right", frameon=True, fontsize=11)
    fig.tight_layout()
    fig.savefig(output_dir / "sensor_prediction_recovery.png", dpi=220)
    plt.close(fig)


def plot_sensor_scatter(output_dir: Path) -> None:
    rows = read_tsv(output_dir / "prediction_summary.tsv")
    true = floats(rows, "true_density")
    mean = floats(rows, "mean")
    observed = floats(rows, "observed_density")

    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ax.scatter(true, observed, s=20, color="#9a9a9a", alpha=0.60, label="Noisy observations")
    ax.scatter(true, mean, s=24, color="#0d3b66", alpha=0.85, label="Posterior mean")
    ax.plot([0, 1], [0, 1], color="#202020", linewidth=1.0)
    ax.set_xlabel("True density")
    ax.set_ylabel("Recovered density")
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.04, 1.04)
    ax.grid(True, color="#dddddd", linewidth=0.7)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    fig.tight_layout()
    fig.savefig(output_dir / "sensor_prediction_scatter.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--map-dir", type=Path, default=None)
    args = parser.parse_args()

    plot_turning_summary(args.output_dir, args.map_dir)
    plot_prediction_summary(args.output_dir)
    plot_sensor_scatter(args.output_dir)


if __name__ == "__main__":
    main()
