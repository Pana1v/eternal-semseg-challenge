#!/usr/bin/env python3
"""Official scorer (spec ch. 7). Reads one submission, writes
results/<split>_<method>_<timestamp>/{summary.json,per_class.csv} and
regenerates results/report.html over every method scored into --out-dir so
far, not just this one.

Usage:
    python eval/score.py --submission submission.json --split score \\
        --method bl_paint --out-dir results [--no-plots]

**There is no --gt flag, and that is deliberate.** The gloc scorer this file
otherwise mirrors takes ground truth on the command line; here the ground
truth is already folded into the submission's confusion matrices by
eval/io_formats.py::Accumulator. That is what keeps the scored artefact
kilobytes instead of gigabytes: a whole split of label images and clouds
reduces to a handful of 9x9 integer matrices, from which every number in
section 6.1 is recoverable. A reader looking for --gt is looking for a flag
that would have nothing left to read.

Every metric lives in eval/metrics.py; this file is I/O, aggregation and
presentation only. The two exceptions are stated where they appear: the
chance floor, which is a reporting reference rather than a section 6.1
metric, and the HTML report, which is small enough here that a separate
module would only add an import.
"""

import argparse
import csv
import glob
import html
import json
import math
import os
import platform
import sys
from datetime import datetime

import numpy as np

# Documented as `python eval/score.py` (docs/CHALLENGE.md), which puts eval/ on
# sys.path and not the repo root, so `import eval.metrics` cannot resolve
# without this. Harmless under `python -m eval.score`, where the root is
# already there.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.io_formats import (CONSISTENCY_FIELDS, load_compute_meta,  # noqa: E402
                             load_submission)
from eval.metrics import (BOUNDARY_WIDTH_PX, ECE_BINS, RANGE_BIN_NAMES,  # noqa: E402
                          all_acc, consistency, coverage, ece, fwiou,
                          iou_per_class, macc, miou)
from semseg.labels import CLASS_NAMES, FOLDED_CLASS_NAMES, fold_sky  # noqa: E402
from semseg.types import NUM_CLASSES  # noqa: E402

# plotting is optional: matplotlib is in the runtime image but a bare scoring
# environment may not have it, and a missing plot must not fail a run
try:
    from eval import plot_results
except ImportError:
    plot_results = None

DEFAULT_OUT_DIR = "results"
DEFAULT_SPLIT = "score"

REPORT_FILENAME = "report.html"
SUMMARY_FILENAME = "summary.json"
PER_CLASS_FILENAME = "per_class.csv"

JSON_INDENT = 2
CSV_DIGITS = 6

# The 8-class view GOOSE's own 3D config uses. Named once because both the
# folded numbers and the published reference are quoted in it, and pairing a
# 9-class mIoU with an 8-class reference is the apples-to-oranges error
# docs/CHALLENGE.md warns about.
FOLDED_LABEL_SPACE = "goose8 (sky folded into other)"

# Published PTv3 numbers, quoted for scale and nothing else. This repo did not
# measure them and cannot: they come from a trained transformer on the full
# GOOSE val split. The summary.json field name carries the disclaimer as well,
# because a bare "reference" key sitting next to our own numbers is exactly how
# a quoted figure ends up cited as a measurement.
PUBLISHED_REFERENCE_FIELD = "published_reference_not_measured_here"
PUBLISHED_REFERENCE = {
    "measured_by_this_repo": False,
    "source": "GOOSE devkit README",
    "model": "PTv3",
    "split": "GOOSE 3D val, full split",
    "label_space": FOLDED_LABEL_SPACE,
    "miou": 0.8096,
    "macc": 0.8576,
    "all_acc": 0.9197,
    "iou_per_class": {
        "other": 0.9686,
        "artificial_structures": 0.8773,
        "artificial_ground": 0.7097,
        "natural_ground": 0.8220,
        "obstacle": 0.4554,
        "vehicle": 0.8954,
        "vegetation": 0.9179,
        "human": 0.8302,
    },
}

# The three classes where a mistake hurts somebody. mIoU gives each of the nine
# one ninth of the weight while the data gives `human` three hundredths of a
# percent (docs/CHALLENGE.md), so the report marks these columns rather than
# leaving a reader to find them.
SAFETY_CLASSES = ("obstacle", "vehicle", "human")

CSS = """
body{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;font-size:14px;
  line-height:1.5;color:#111;background:#fff;margin:24px;max-width:1400px}
h1{font-size:20px;margin:0 0 4px}
h2{font-size:15px;margin:28px 0 8px;border-bottom:1px solid #999;padding-bottom:3px}
p.note{color:#444;font-size:13px;margin:6px 0 0;max-width:90ch}
table{border-collapse:collapse;margin-top:6px;font-size:13px}
th,td{border:1px solid #999;padding:4px 8px;text-align:right;white-space:nowrap}
th{background:#eee;font-weight:600;text-align:right}
th.safety{background:#e6dcc6}
th:first-child,td:first-child{text-align:left}
td.name,td.num{font-family:ui-monospace,Menlo,Consolas,monospace;
  font-variant-numeric:tabular-nums}
tr.reference td{background:#f4f4f4;font-style:italic}
img{max-width:100%;border:1px solid #999;display:block;margin-top:6px}
.scroll{overflow-x:auto}
footer{margin-top:28px;border-top:1px solid #999;padding-top:8px;
  font-size:12px;color:#444}
"""


def chance_floor_miou(conf) -> float:
    """Expected mIoU of a predictor that guesses uniformly at random, the floor
    a real method has to beat.

    Analytic, from the submission's own ground-truth row marginals, which is
    the neatest argument that dropping --gt costs nothing: for class c with
    support fraction p_c and C equiprobable guesses the expected intersection
    is p_c / C and the expected union is p_c + 1/C - p_c/C, so
    IoU_c = p_c / (C * p_c + 1 - p_c). A ratio of expected counts, not a Monte
    Carlo draw and not the scored bl_prior run.

    Averaged over the classes the ground truth actually contains, the same
    rule iou_per_class applies to a class with no support: an absent class has
    no floor to beat.

    Lives here rather than in metrics.py because it is a reporting reference
    and not one of the section 6.1 metrics, and because it reads the matrix as
    a class-frequency table rather than as a scoring result.
    """
    conf = np.asarray(conf, dtype=np.float64)
    total = conf.sum()
    if total == 0:
        return float("nan")

    share = conf.sum(axis=1) / total
    present = share[share > 0]
    num_classes = conf.shape[0]

    return float((present / (num_classes * present + 1.0 - present)).mean())


def _jsonable(value):
    """nan to None. Undefined is not a number, bare NaN is not valid strict
    JSON, and `value or None` would turn a genuine IoU of 0.0 (the model
    invented a class that is not in the scene, a real error) into undefined.
    """
    if value is None:
        return None

    value = float(value)
    return None if math.isnan(value) else value


def _per_class(iou, names) -> dict:
    return {name: _jsonable(iou[index]) for index, name in enumerate(names)}


def metrics_2d(sub: dict) -> dict:
    """Every 2D number section 6.1 asks for, per-class IoU in full included."""
    conf = sub["conf_2d"]
    boundary = sub["conf_2d_boundary"]
    calib = sub["ece_2d"]

    return {
        "miou": _jsonable(miou(conf)),
        "fwiou": _jsonable(fwiou(conf)),
        "macc": _jsonable(macc(conf)),
        "all_acc": _jsonable(all_acc(conf)),
        # Section 6.1: never report only the mean. A headline mIoU with an
        # excellent `vegetation` and a zero `human` is not a good result.
        "iou_per_class": _per_class(iou_per_class(conf), CLASS_NAMES),
        "boundary_miou": _jsonable(miou(boundary)),
        "ece": _jsonable(ece(calib["counts"], calib["conf_sum"], calib["correct"])),
        "n_scored": int(conf.sum()),
        "n_scored_boundary": int(boundary.sum()),
    }


def metrics_3d(sub: dict) -> dict:
    """The 3D counterpart, plus the two stratifications that only exist in 3D:
    range bins (density falls off as 1/r^2, section 6.1) and the frustum split
    (section 6.2 asks how much of a fusion gain comes from points the camera
    never saw).
    """
    conf = sub["conf_3d"]
    calib = sub["ece_3d"]
    folded = fold_sky(conf)

    by_range = {}
    for name in RANGE_BIN_NAMES:
        binned = sub["conf_3d_by_range"][name]
        # The scored count travels with the mIoU because some bins are nearly
        # empty for the classes that matter: 858 `human` points inside 5 m over
        # the whole split (docs/CHALLENGE.md), so that bin's number rests on
        # almost nothing and has to say so.
        by_range[name] = {"miou": _jsonable(miou(binned)), "n_scored": int(binned.sum())}

    return {
        "miou": _jsonable(miou(conf)),
        "fwiou": _jsonable(fwiou(conf)),
        "macc": _jsonable(macc(conf)),
        "all_acc": _jsonable(all_acc(conf)),
        "iou_per_class": _per_class(iou_per_class(conf), CLASS_NAMES),
        "ece": _jsonable(ece(calib["counts"], calib["conf_sum"], calib["correct"])),
        "n_scored": int(conf.sum()),
        "miou_by_range": by_range,
        "miou_in_frustum": _jsonable(miou(sub["conf_3d_in_frustum"])),
        "miou_out_frustum": _jsonable(miou(sub["conf_3d_out_frustum"])),
        "n_scored_in_frustum": int(sub["conf_3d_in_frustum"].sum()),
        "n_scored_out_frustum": int(sub["conf_3d_out_frustum"].sum()),
        # Kept next to the published reference below, and nowhere near the
        # 9-class mIoU above, since the reference is quoted in this space.
        "folded_8class": {
            "label_space": FOLDED_LABEL_SPACE,
            "miou": _jsonable(miou(folded)),
            "iou_per_class": _per_class(iou_per_class(folded), FOLDED_CLASS_NAMES),
        },
    }


def consistency_block(sub: dict) -> dict:
    """Cross-modal consistency with its coverage, never one without the other:
    assumption A2 says most points have no pixel at all, so 0.95 over 3 percent
    of the cloud and 0.95 over 80 percent are different results.
    """
    counts = sub["consistency"]
    block = {
        "consistency": _jsonable(consistency(counts["matched"], counts["scorable"])),
        "coverage": _jsonable(coverage(counts["scorable"], counts["total_points"])),
    }
    block.update({name: int(counts[name]) for name in CONSISTENCY_FIELDS})

    return block


def calibration_block(sub: dict) -> dict:
    """The accumulated ECE bins, copied through verbatim.

    The single ECE number cannot be unpacked back into a reliability diagram,
    so the bins travel with it and plot_results reads only summary.json.
    """
    block = {"num_bins": ECE_BINS}
    for modality, field in (("2d", "ece_2d"), ("3d", "ece_3d")):
        block[modality] = {key: [float(v) for v in values]
                           for key, values in sub[field].items()}

    return block


def write_summary_json(out_path: str, sub: dict, args, m2: dict, m3: dict,
                        floor: dict, compute) -> None:
    summary = {
        "generated_at": datetime.now().isoformat(),
        "method": args.method,
        "split": args.split,
        "label_space": sub["label_space"],
        "num_classes": NUM_CLASSES,
        "n_frames": sub["n_frames"],
        "class_names": list(CLASS_NAMES),
        "metrics_2d": m2,
        "metrics_3d": m3,
        "consistency": consistency_block(sub),
        "calibration": calibration_block(sub),
        "chance_floor": floor,
        "margin_over_chance": {
            "miou_2d": _margin(m2["miou"], floor["miou_2d"]),
            "miou_3d": _margin(m3["miou"], floor["miou_3d"]),
        },
        # Reported alongside the score, never folded into it: the figure is
        # self-declared by the submitting method, so ranking on it would be
        # trivially gameable.
        "compute": compute,
        # PUBLISHED, not measured by this repo. The field name says so, this
        # comment says so, and the printed line and the report say so.
        PUBLISHED_REFERENCE_FIELD: PUBLISHED_REFERENCE,
        "parameters": {
            "num_classes": NUM_CLASSES,
            "range_bin_names": list(RANGE_BIN_NAMES),
            "ece_bins": ECE_BINS,
            "boundary_width_px": BOUNDARY_WIDTH_PX,
        },
        "run": {
            "submission": os.path.abspath(args.submission),
            "method": args.method,
            "split": args.split,
            "machine": {"cpu": platform.processor() or platform.machine(),
                        "platform": platform.platform()},
        },
    }

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=JSON_INDENT)
        f.write("\n")


def _margin(value, floor):
    if value is None or floor is None:
        return None

    return value - floor


def _csv_num(value) -> str:
    """Empty cell for an undefined IoU, as gloc's stats.csv does for a nan
    error: a spreadsheet reading 0.0 there would average it in."""
    value = float(value)
    if math.isnan(value):
        return ""

    return f"{value:.{CSV_DIGITS}f}"


def write_per_class_csv(out_path: str, sub: dict) -> None:
    """One row per class: 2D IoU, 3D IoU, and 3D IoU in each range bin.

    The mean is in summary.json. This file is the section 6.1 requirement that
    the per-class numbers exist somewhere a reader can sort and diff.
    """
    iou_2d = iou_per_class(sub["conf_2d"])
    iou_3d = iou_per_class(sub["conf_3d"])
    by_range = {name: iou_per_class(sub["conf_3d_by_range"][name])
                for name in RANGE_BIN_NAMES}

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "class_name", "iou_2d", "iou_3d"]
                        + [f"iou_3d_{name}" for name in RANGE_BIN_NAMES])

        for class_id, name in enumerate(CLASS_NAMES):
            writer.writerow([class_id, name,
                             _csv_num(iou_2d[class_id]), _csv_num(iou_3d[class_id])]
                            + [_csv_num(by_range[b][class_id]) for b in RANGE_BIN_NAMES])


def _fmt(value, digits=4, dash="n/a") -> str:
    if value is None:
        return dash

    value = float(value)
    if math.isnan(value):
        return dash

    return f"{value:.{digits}f}"


def print_compute(compute) -> None:
    """Compute cost is an independent KPI: reported next to the score, never
    part of it. A method that wins on accuracy while costing 100x the compute
    should be visibly doing so, not silently penalized.
    """
    if compute is None:
        print("compute: n/a (no <submission>.meta.json sidecar)")
        return

    parts = []
    if compute["runtime_sec_total"] is not None:
        parts.append(f"{compute['runtime_sec_total']:.1f}s total")
    if compute["runtime_sec_per_frame"] is not None:
        parts.append(f"{compute['runtime_sec_per_frame']:.3f}s/frame")
    if compute["peak_rss_mb"] is not None:
        parts.append(f"{compute['peak_rss_mb']:.0f} MB peak RSS")
    print("compute (independent KPI): " + (", ".join(parts) if parts else "n/a"))


def print_headline(sub: dict, m2: dict, m3: dict, floor: dict, cons: dict) -> None:
    print(f"scored {sub['n_frames']} frames")
    print(f"headline mIoU: 2D {_fmt(m2['miou'])}  3D {_fmt(m3['miou'])}")
    print(f"chance floor (uniform over {NUM_CLASSES} classes): "
          f"2D {_fmt(floor['miou_2d'])}  3D {_fmt(floor['miou_3d'])}  "
          f"(margin 2D {_fmt(_margin(m2['miou'], floor['miou_2d'])):>7}, "
          f"3D {_fmt(_margin(m3['miou'], floor['miou_3d'])):>7})")
    print(f"fwIoU: 2D {_fmt(m2['fwiou'])}  3D {_fmt(m3['fwiou'])}   "
          f"mAcc: 2D {_fmt(m2['macc'])}  3D {_fmt(m3['macc'])}   "
          f"allAcc: 2D {_fmt(m2['all_acc'])}  3D {_fmt(m3['all_acc'])}")
    print(f"boundary mIoU 2D: {_fmt(m2['boundary_miou'])}")

    ranges = "  ".join(f"{name} {_fmt(m3['miou_by_range'][name]['miou'], 3)}"
                       for name in RANGE_BIN_NAMES)
    print(f"3D mIoU by range: {ranges}")
    print(f"3D mIoU by frustum: in {_fmt(m3['miou_in_frustum'])}  "
          f"out {_fmt(m3['miou_out_frustum'])}")
    print(f"ECE: 2D {_fmt(m2['ece'])}  3D {_fmt(m3['ece'])}")
    print(f"consistency: {_fmt(cons['consistency'])} over "
          f"{_fmt(cons['coverage'])} coverage "
          f"({cons['scorable']} of {cons['total_points']} points scorable)")

    reference = PUBLISHED_REFERENCE
    print(f"folded 8-class 3D mIoU: {_fmt(m3['folded_8class']['miou'])}  "
          f"(published {reference['model']} reference {reference['miou']:.4f} from the "
          f"{reference['source']}, NOT measured here)")


def write_plots(summary_path: str, out_dir: str) -> None:
    """Renders the plots into the result directory alongside summary.json, plus
    the cross-method comparison into --out-dir itself."""
    if plot_results is None:
        print("skipping plots: matplotlib not installed", file=sys.stderr)
        return

    plot_results.main(["--summary", summary_path, "--results", out_dir])


def load_runs(out_dir: str) -> dict:
    """One entry per method scored into out_dir, keyed by method so a rescored
    method replaces its own older row rather than appearing twice."""
    runs = {}
    for path in sorted(glob.glob(os.path.join(out_dir, "*", SUMMARY_FILENAME))):
        with open(path) as f:
            summary = json.load(f)

        run_dir = os.path.dirname(path)
        method = summary.get("method") or os.path.basename(run_dir)
        runs[method] = {"summary": summary, "dir": run_dir}   # later run wins

    return runs


def _table(head, body) -> str:
    ths = []
    for name in head:
        shade = ' class="safety"' if name in SAFETY_CLASSES else ""
        ths.append(f"<th{shade}>{html.escape(name)}</th>")

    return (f'<div class="scroll"><table><thead><tr>{"".join(ths)}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def _cells(values, digits=4) -> str:
    return "".join(f'<td class="num">{_fmt(v, digits, dash="-")}</td>' for v in values)


def _reference_row(label: str, values: dict, head) -> str:
    """A reference row addressed by column name rather than by position, so a
    column added to the head keeps the sparse rows aligned instead of silently
    shifting their numbers one cell to the left."""
    cells = "".join(f'<td class="num">{_fmt(values.get(name), 4, dash="-")}</td>'
                    for name in head[1:])
    return f'<tr class="reference"><td class="name">{html.escape(label)}</td>{cells}</tr>'


def summary_table(runs: dict) -> tuple:
    """One row per method, with the chance floor as a reference row."""
    head = ["method", "frames", "mIoU 2D", "mIoU 3D", "fwIoU 2D", "fwIoU 3D",
            "boundary mIoU 2D", "ECE 2D", "ECE 3D", "consistency", "coverage",
            "3D mIoU folded", "sec/frame", "peak RSS (MB)"]
    body = []

    ranked = sorted(runs.items(),
                    key=lambda kv: -(kv[1]["summary"]["metrics_3d"]["miou"] or 0.0))
    for method, run in ranked:
        summary = run["summary"]
        m2, m3 = summary["metrics_2d"], summary["metrics_3d"]
        cons = summary["consistency"]
        compute = summary.get("compute") or {}

        body.append(
            f'<tr><td class="name">{html.escape(method)}</td>'
            f'<td class="num">{summary["n_frames"]}</td>'
            + _cells([m2["miou"], m3["miou"], m2["fwiou"], m3["fwiou"],
                      m2["boundary_miou"], m2["ece"], m3["ece"],
                      cons["consistency"], cons["coverage"],
                      m3["folded_8class"]["miou"]])
            + f'<td class="num">{_fmt(compute.get("runtime_sec_per_frame"), 3, dash="-")}</td>'
            f'<td class="num">{_fmt(compute.get("peak_rss_mb"), 0, dash="-")}</td></tr>')

    any_summary = next(iter(runs.values()))["summary"]

    floor = any_summary["chance_floor"]
    body.append(_reference_row("chance floor",
                                {"mIoU 2D": floor["miou_2d"], "mIoU 3D": floor["miou_3d"]},
                                head))

    # The published figure goes in the folded column and nowhere else: it was
    # measured in the 8-class space, and dropping it under mIoU 3D would invite
    # a comparison against a 9-class number.
    reference = any_summary[PUBLISHED_REFERENCE_FIELD]
    body.append(_reference_row(f'{reference["model"]} (published, not measured here)',
                                {"3D mIoU folded": reference["miou"]}, head))

    return head, body


def per_class_table(runs: dict, modality: str) -> tuple:
    """Method x class IoU. Section 6.1 forbids reporting only the mean, so the
    report leads with the means and then shows all nine of them."""
    field = f"metrics_{modality}"
    head = ["method"] + list(CLASS_NAMES)
    body = []

    for method, run in sorted(runs.items()):
        per_class = run["summary"][field]["iou_per_class"]
        body.append(f'<tr><td class="name">{html.escape(method)}</td>'
                    + _cells([per_class.get(name) for name in CLASS_NAMES], 3) + '</tr>')

    if modality == "3d":
        reference = next(iter(runs.values()))["summary"][PUBLISHED_REFERENCE_FIELD]
        # The published row is in the folded 8-class space, so `sky` has no
        # cell to fill and is left empty rather than zero.
        body.append(
            f'<tr class="reference"><td class="name">'
            f'{html.escape(reference["model"])} (published, folded 8-class)</td>'
            + _cells([reference["iou_per_class"].get(name) for name in CLASS_NAMES], 3)
            + '</tr>')

    return head, body


def range_table(runs: dict) -> tuple:
    head = ["method"] + [f"3D mIoU {name}" for name in RANGE_BIN_NAMES] \
        + ["in frustum", "out of frustum"]
    body = []

    for method, run in sorted(runs.items()):
        m3 = run["summary"]["metrics_3d"]
        values = [m3["miou_by_range"][name]["miou"] for name in RANGE_BIN_NAMES]
        values += [m3["miou_in_frustum"], m3["miou_out_frustum"]]
        body.append(f'<tr><td class="name">{html.escape(method)}</td>'
                    + _cells(values, 3) + '</tr>')

    return head, body


def plots_section(runs: dict, out_dir: str) -> str:
    """Links every plot already on disk, so a --no-plots run degrades to the
    tables instead of to broken images."""
    paths = sorted(glob.glob(os.path.join(out_dir, "*.png")))
    for run in runs.values():
        paths += sorted(glob.glob(os.path.join(run["dir"], "*.png")))

    if not paths:
        return ""

    images = "".join(
        f'<p class="note">{html.escape(os.path.relpath(path, out_dir))}</p>'
        f'<img src="{html.escape(os.path.relpath(path, out_dir))}">'
        for path in paths)
    return f"<h2>Plots</h2>{images}"


def render_report(runs: dict, out_dir: str, title: str) -> str:
    any_summary = next(iter(runs.values()))["summary"]
    n_frames = any_summary["n_frames"]
    generated = any_summary.get("generated_at", "")

    facts = (f'<p class="note">Split {html.escape(str(any_summary["split"]))} &middot; '
             f'{n_frames} frames &middot; '
             f'{len(runs)} method{"s" if len(runs) != 1 else ""} &middot; '
             f'label space {html.escape(str(any_summary["label_space"]))} &middot; '
             f'{ECE_BINS} calibration bins &middot; '
             f'boundary band {BOUNDARY_WIDTH_PX} px &middot; '
             f'generated {html.escape(generated[:19])}</p>')

    sections = [
        f'<h2>Methods</h2>{_table(*summary_table(runs))}'
        f'<p class="note">The chance floor is the expected mIoU of a uniform guess, '
        f'computed from the ground-truth class frequencies alone, so it is one row for '
        f'the split rather than one per method. Compute is reported beside the score and '
        f'never folded into it. The last reference row is PUBLISHED by the GOOSE authors '
        f'and was not measured by this repo; it is quoted in the folded 8-class space, '
        f'which is why it sits in the folded column and not under mIoU 3D.</p>',

        f'<h2>Per-class IoU, 3D</h2>{_table(*per_class_table(runs, "3d"))}'
        f'<p class="note">Never read the mean alone (section 6.1). The shaded columns are '
        f'the classes where a mistake hurts somebody, and they carry a few hundredths of a '
        f'percent of the data while mIoU hands each of them one ninth of the weight. An '
        f'empty cell is an undefined IoU, a class absent from both the ground truth and '
        f'the prediction, which is not the same as zero.</p>',

        f'<h2>Per-class IoU, 2D</h2>{_table(*per_class_table(runs, "2d"))}'
        f'<p class="note">`sky` is around 29 percent of pixels and 0 percent of points, '
        f'so the two modalities are not scoring the same scene even on the same frame.</p>',

        f'<h2>3D IoU by range and by frustum</h2>{_table(*range_table(runs))}'
        f'<p class="note">Point density falls off as 1/r^2, so one pooled 3D mIoU hides a '
        f'model failing at 30 m. The frustum columns answer how much of a fusion gain came '
        f'from points the camera never saw (section 6.2).</p>',
    ]

    plots = plots_section(runs, out_dir)
    if plots:
        sections.append(plots)

    return f"""<title>{html.escape(title)}</title>
<style>{CSS}</style>
<h1>{html.escape(title)}</h1>
{facts}
{"".join(sections)}
<footer>Generated by eval/score.py from every summary.json under this
directory. Ground truth is folded into the submission matrices, so the scorer
never reads a label file.</footer>"""


def write_report(out_dir: str, title: str):
    """Regenerated after every scoring run, so the report always covers every
    method scored into out_dir so far and not just the one just scored."""
    runs = load_runs(out_dir)
    if not runs:
        return None

    path = os.path.join(out_dir, REPORT_FILENAME)
    with open(path, "w") as f:
        f.write(render_report(runs, out_dir, title))

    return path


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--method")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--no-plots", action="store_true", help="skip rendering plots")
    args = parser.parse_args(argv)

    # No --gt: see the module docstring. The matrices already carry it.
    sub = load_submission(args.submission)

    if args.method is None:
        args.method = sub.get("method") or os.path.splitext(
            os.path.basename(args.submission))[0]

    # A submission records the split it was produced on. Scoring a fit-split
    # submission and filing it under `score` is how a number fitted on the data
    # it was scored on ends up in a table, so say it out loud.
    if sub.get("split") is not None and sub["split"] != args.split:
        print(f"warning: submission was written on split '{sub['split']}' "
              f"but --split says '{args.split}'", file=sys.stderr)

    m2 = metrics_2d(sub)
    m3 = metrics_3d(sub)
    floor = {
        "predictor": f"uniform over {NUM_CLASSES} classes",
        "derivation": "analytic, from the submission's own ground-truth class frequencies",
        "miou_2d": _jsonable(chance_floor_miou(sub["conf_2d"])),
        "miou_3d": _jsonable(chance_floor_miou(sub["conf_3d"])),
    }
    compute = load_compute_meta(args.submission)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(args.out_dir, f"{args.split}_{args.method}_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)

    per_class_path = os.path.join(result_dir, PER_CLASS_FILENAME)
    summary_path = os.path.join(result_dir, SUMMARY_FILENAME)
    write_per_class_csv(per_class_path, sub)
    write_summary_json(summary_path, sub, args, m2, m3, floor, compute)

    print_headline(sub, m2, m3, floor, consistency_block(sub))
    print_compute(compute)
    print(f"wrote {per_class_path}")
    print(f"wrote {summary_path}")

    if not args.no_plots:
        write_plots(summary_path, args.out_dir)

    report_path = write_report(args.out_dir, f"Semseg Eval - split {args.split}")
    if report_path:
        print(f"wrote {report_path}")

    return result_dir


if __name__ == "__main__":
    main()
