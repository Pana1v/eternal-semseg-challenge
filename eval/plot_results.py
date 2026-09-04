#!/usr/bin/env python3
"""Renders the four figures a scored run is worth looking at: per-class IoU
with the mean marked, 3D mIoU by range bin, a reliability diagram from the ECE
bins, and a grouped comparison of every method scored into the results
directory.

Usage:
    python eval/plot_results.py --summary results/score_bl_paint_TS/summary.json \\
        [--results results]

score.py calls main(argv) in process after every scoring run. Reads
summary.json and nothing else, so it needs no label file, no submission and no
import from the rest of the repo: a figure can always be regenerated from a
scored result directory alone.

The per-run figures land next to their summary.json; the cross-method
comparison lands in the results directory itself, next to report.html, because
it describes the directory rather than any one run.
"""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Colourblind-safe pair, used for the two modalities everywhere so a reader
# learns the mapping once: blue is 2D, vermillion is 3D.
COLOR_2D = "#0072b2"
COLOR_3D = "#d55e00"
COLOR_REFERENCE = "#666666"

FIG_WIDE = (9, 4.5)
FIG_TALL = (7, 6)
DPI = 120

BAR_WIDTH = 0.4

# Metrics the cross-method chart compares, as (column label, path into the
# summary). Four is the most a grouped bar chart stays readable at.
COMPARISON_METRICS = (
    ("mIoU 2D", ("metrics_2d", "miou")),
    ("mIoU 3D", ("metrics_3d", "miou")),
    ("boundary mIoU 2D", ("metrics_2d", "boundary_miou")),
    ("consistency", ("consistency", "consistency")),
)

PER_CLASS_FILENAME = "per_class_iou.png"
BY_RANGE_FILENAME = "miou_by_range.png"
RELIABILITY_FILENAME = "reliability.png"
COMPARISON_FILENAME = "method_comparison.png"


def _save(fig, out_path: str) -> None:
    """The one place tight_layout, savefig and close appear. Four figures with
    four copies of the triplet is how one of them ends up leaking a figure
    handle in a sweep that renders hundreds."""
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def _nan_array(values) -> np.ndarray:
    """JSON has no NaN, so an undefined IoU arrives as None. Back to nan here,
    where matplotlib draws no bar at all, which is the honest rendering: a zero
    height bar would read as a class the method got completely wrong."""
    return np.array([np.nan if v is None else float(v) for v in values], dtype=np.float64)


def plot_per_class_iou(summary: dict, out_dir: str) -> None:
    """Per-class IoU in both modalities, with each modality's mean drawn as a
    line. Section 6.1 says never report only the mean, and the point of the
    line is to show how little the mean says: one tall bar and one absent class
    can produce the same mean as nine mediocre ones.
    """
    names = list(summary["class_names"])
    iou_2d = _nan_array([summary["metrics_2d"]["iou_per_class"][n] for n in names])
    iou_3d = _nan_array([summary["metrics_3d"]["iou_per_class"][n] for n in names])

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=FIG_WIDE)
    ax.bar(x - BAR_WIDTH / 2, iou_2d, BAR_WIDTH, color=COLOR_2D, label="2D")
    ax.bar(x + BAR_WIDTH / 2, iou_3d, BAR_WIDTH, color=COLOR_3D, label="3D")

    for value, color, label in ((summary["metrics_2d"]["miou"], COLOR_2D, "mIoU 2D"),
                                (summary["metrics_3d"]["miou"], COLOR_3D, "mIoU 3D")):
        if value is None:
            continue
        ax.axhline(value, color=color, linestyle="--", linewidth=1,
                   label=f"{label} {value:.3f}")

    # A class absent from both the ground truth and the prediction has an
    # undefined IoU, not a zero. Saying so on the axis stops an empty slot
    # being read as a total failure.
    for index in np.where(np.isnan(iou_2d) & np.isnan(iou_3d))[0]:
        ax.text(index, 0.02, "undefined", rotation=90, fontsize=7,
                ha="center", va="bottom", color=COLOR_REFERENCE)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("IoU")
    ax.set_title(f"Per-class IoU, {summary['method']} on split {summary['split']}")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, os.path.join(out_dir, PER_CLASS_FILENAME))


def plot_miou_by_range(summary: dict, out_dir: str) -> None:
    """3D mIoU per range bin, with the pooled value as a line and the scored
    point count on each bar.

    The counts are on the plot because a bin can be almost empty for the
    classes that matter, and a bar drawn from a handful of points looks exactly
    like a bar drawn from millions.
    """
    bins = list(summary["parameters"]["range_bin_names"])
    by_range = summary["metrics_3d"]["miou_by_range"]
    values = _nan_array([by_range[name]["miou"] for name in bins])
    counts = [by_range[name]["n_scored"] for name in bins]

    x = np.arange(len(bins))
    fig, ax = plt.subplots(figsize=FIG_WIDE)
    ax.bar(x, values, BAR_WIDTH * 1.5, color=COLOR_3D)

    pooled = summary["metrics_3d"]["miou"]
    if pooled is not None:
        ax.axhline(pooled, color=COLOR_REFERENCE, linestyle="--", linewidth=1,
                   label=f"pooled 3D mIoU {pooled:.3f}")
        ax.legend(fontsize=8)

    for index, (value, count) in enumerate(zip(values, counts)):
        height = 0.0 if np.isnan(value) else value
        ax.text(index, height + 0.02, f"n={count}", ha="center", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(bins)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("range from the lidar origin")
    ax.set_ylabel("3D mIoU")
    ax.set_title(f"3D mIoU by range, {summary['method']}")
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, os.path.join(out_dir, BY_RANGE_FILENAME))


def _reliability_points(bins: dict):
    """(mean confidence, accuracy, share of elements) over the filled bins.

    An empty bin has no accuracy, so it is dropped rather than plotted at zero,
    which would draw a curve through a bin the method never used.
    """
    counts = np.asarray(bins["counts"], dtype=np.float64)
    conf_sum = np.asarray(bins["conf_sum"], dtype=np.float64)
    correct = np.asarray(bins["correct"], dtype=np.float64)

    total = counts.sum()
    if total == 0:
        return None

    filled = counts > 0
    return (conf_sum[filled] / counts[filled],
            correct[filled] / counts[filled],
            counts[filled] / total)


def plot_reliability(summary: dict, out_dir: str) -> None:
    """Reliability diagram plus the bin populations underneath it.

    ECE is a single number and cannot be read back into a shape: a head that is
    overconfident everywhere and one that is badly wrong in a single crowded
    bin can post the same value. The population panel is not decoration, it is
    what says whether a deviation is a real one or three pixels.
    """
    series = []
    for modality, color in (("2d", COLOR_2D), ("3d", COLOR_3D)):
        points = _reliability_points(summary["calibration"][modality])
        if points is None:
            continue
        error = summary[f"metrics_{modality}"]["ece"]
        series.append((modality.upper(), color, points, error))

    # No confidences anywhere means the method declined to report any, which is
    # allowed (ECE is then skipped), so there is nothing to draw.
    if not series:
        return

    fig, (ax, ax_pop) = plt.subplots(2, 1, figsize=FIG_TALL, sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1]})
    ax.plot([0, 1], [0, 1], color=COLOR_REFERENCE, linestyle=":", linewidth=1,
            label="perfect calibration")

    for label, color, (conf, acc, share), error in series:
        suffix = "n/a" if error is None else f"{error:.4f}"
        ax.plot(conf, acc, marker="o", markersize=4, color=color,
                label=f"{label}, ECE {suffix}")
        ax_pop.plot(conf, share, marker="o", markersize=3, color=color, label=label)

    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("accuracy")
    ax.set_title(f"Reliability, {summary['method']} "
                 f"({summary['calibration']['num_bins']} bins)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    ax_pop.set_xlim(0.0, 1.0)
    ax_pop.set_xlabel("mean predicted confidence in the bin")
    ax_pop.set_ylabel("share of elements")
    ax_pop.grid(True, alpha=0.3)
    _save(fig, os.path.join(out_dir, RELIABILITY_FILENAME))


def _dig(summary: dict, path):
    value = summary
    for key in path:
        value = value[key]
    return value


def load_methods(results_dir: str) -> dict:
    """{method: summary} over every scored run under results_dir, later run
    wins, keyed exactly as score.py's report keys them so the chart and the
    report cannot disagree about which runs exist."""
    methods = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*", "summary.json"))):
        with open(path) as f:
            summary = json.load(f)
        method = summary.get("method") or os.path.basename(os.path.dirname(path))
        methods[method] = summary

    return methods


def plot_method_comparison(results_dir: str) -> None:
    """Every method scored into results_dir, grouped by metric.

    Grouped by metric rather than by method so the ablation reads off the plot
    directly: the camera-only, lidar-only and fused bars sit next to each other
    within one metric, which is the comparison section 6.2 actually asks for.

    No published reference line is drawn here. The published PTv3 figure is in
    the folded 8-class space and these bars are 9-class, so a line across the
    chart would invite exactly the comparison docs/ONTOLOGY.md warns against.
    """
    methods = load_methods(results_dir)
    if not methods:
        return

    labels = [label for label, _ in COMPARISON_METRICS]
    x = np.arange(len(labels))
    width = min(BAR_WIDTH * 2 / len(methods), BAR_WIDTH)

    fig, ax = plt.subplots(figsize=FIG_WIDE)
    for index, (method, summary) in enumerate(sorted(methods.items())):
        values = _nan_array([_dig(summary, path) for _, path in COMPARISON_METRICS])
        offset = (index - (len(methods) - 1) / 2) * width
        ax.bar(x + offset, values, width, label=method)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("value")
    ax.set_title(f"Methods scored into {os.path.basename(os.path.abspath(results_dir))}")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, os.path.join(results_dir, COMPARISON_FILENAME))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True,
                        help="summary.json of the run to plot")
    parser.add_argument("--results",
                        help="results root holding every scored run "
                             "(default: the parent of the run directory)")
    args = parser.parse_args(argv)

    with open(args.summary) as f:
        summary = json.load(f)

    out_dir = os.path.dirname(os.path.abspath(args.summary))
    results_dir = args.results or os.path.dirname(out_dir)

    plot_per_class_iou(summary, out_dir)
    plot_miou_by_range(summary, out_dir)
    plot_reliability(summary, out_dir)
    plot_method_comparison(results_dir)

    print(f"wrote plots to {out_dir}")


if __name__ == "__main__":
    main()
