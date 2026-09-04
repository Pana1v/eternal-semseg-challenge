#!/usr/bin/env python3
"""Renders every scored run in a results directory into one self-contained
HTML report: the leaderboard, per-class IoU in full for both modalities, the
range-stratified 3D breakdown, the in-frustum versus out-of-frustum split, the
robustness sweeps with their crossover verdicts, and whatever plots are on
disk.

score.py invokes this after each scoring run, so the report always covers every
method scored into that directory so far and not just the one just scored. It
can also be run directly:

    python eval/report.py --results <dir> --out report.html

Reads only summary.json, per_class.csv and the sweep CSVs, so it never needs a
label file, a cloud or an image. Standard library only, for the same reason:
this file has to render in a bare Python, and importing eval.metrics for a
handful of names would drag cv2 in behind it.

The summary.json fields it reads, which is its contract with score.py:

    run.method, run.split, generated_at, n_frames
    metrics_2d.{miou, fwiou, boundary_miou, ece, iou_per_class}
    metrics_3d.{miou, fwiou, ece, iou_per_class, folded_miou,
                miou_by_range, miou_in_frustum, miou_out_frustum}
    consistency.{consistency, coverage}
    compute.runtime_sec_per_frame
    reference          the published PTv3 block

Each is also accepted under one gloc-shaped alias (`overall.miou_3d` and so
on). Nothing here raises on an unrecognised layout, but a run whose 3D mIoU
cannot be found warns on stderr naming what was looked for: a leaderboard of
dashes still renders, and a rendered page reads as finished, which is a worse
failure than a crash.
"""

import argparse
import csv
import glob
import html
import json
import os
import sys

# The chance floor (spec ch. 10). Shown, but never sorted in with the measured
# rows: it is the number every other row has to beat, not a competitor.
FLOOR_METHOD = "bl_prior"

# Marked in the per-class table. Section 8 of the problem statement words the
# list as person, obstacle and structure; goose9 calls person `human`, and a
# vehicle collision is the same class of failure as a person collision, so
# these three carry the mark. Assumption A5 is why the mark exists at all: a 2
# percent gain on terrain is worth less than a 2 percent gain on person, and
# nine equal-looking numbers in a table hide exactly that.
SAFETY_CLASSES = ("human", "vehicle", "obstacle")
SAFETY_MARK = "*"

# PTv3 on GOOSE 3D val, quoted from the GOOSE devkit README (spec ch. 1).
# PUBLISHED, and NOT measured by this repo. It is also in the FOLDED 8-class
# space, where sky is merged into `other`, so it is not the same quantity as a
# goose9 3D mIoU. Both facts are stated in the row itself and the row is
# appended rather than sorted in, because a reference that sorts into a
# leaderboard reads as a result.
PTV3_LABEL = "PTv3: published, NOT measured here"
PTV3_SOURCE = "GOOSE devkit README, PTv3 on GOOSE 3D val, 8-class folded space"
PTV3_MIOU_3D = 0.8096
PTV3_IOU_PER_CLASS = {
    "other": 0.9686,
    "artificial_structures": 0.8773,
    "artificial_ground": 0.7097,
    "natural_ground": 0.8220,
    "obstacle": 0.4554,
    "vehicle": 0.8954,
    "vegetation": 0.9179,
    "human": 0.8302,
}

# Bin order for the range table, restated from metrics.RANGE_BIN_NAMES rather
# than imported. A bin a summary carries that is not listed here is appended in
# file order rather than dropped, so a future bin edit cannot silently lose a
# column.
RANGE_BIN_ORDER = ("0-5m", "5-15m", "15-30m", "30m+")

# Where each leaderboard number lives in summary.json. One primary path plus at
# most one alias: a longer candidate list would make a schema mismatch quieter
# without making it less likely, and the stderr warning is what actually
# catches it.
METRIC_PATHS = {
    "miou_2d": ("metrics_2d.miou", "overall.miou_2d"),
    "miou_3d": ("metrics_3d.miou", "overall.miou_3d"),
    "fwiou_3d": ("metrics_3d.fwiou", "overall.fwiou_3d"),
    "folded_miou_3d": ("metrics_3d.folded_miou", "metrics_3d_folded.miou"),
    "boundary_miou_2d": ("metrics_2d.boundary_miou", "overall.boundary_miou_2d"),
    "ece_2d": ("metrics_2d.ece", "overall.ece_2d"),
    "ece_3d": ("metrics_3d.ece", "overall.ece_3d"),
    "consistency": ("consistency.consistency", "overall.consistency"),
    "coverage": ("consistency.coverage", "overall.coverage"),
    "miou_in_frustum": ("metrics_3d.miou_in_frustum", "overall.miou_3d_in_frustum"),
    "miou_out_frustum": ("metrics_3d.miou_out_frustum", "overall.miou_3d_out_frustum"),
    "sec_per_frame": ("compute.runtime_sec_per_frame", "compute.runtime_sec_per_scenario"),
}

# The number the leaderboard is ordered by. A run that cannot supply it is the
# one case worth warning about.
SORT_METRIC = "miou_3d"

# Per-class IoU is read from per_class.csv when it is there, because that file
# carries the class NAMES and the summary's lists are positional.
PER_CLASS_FILE = "per_class.csv"
CLASS_COLUMNS = ("class", "class_name", "name", "class_id")
IOU_2D_COLUMNS = ("iou_2d", "iou2d")
IOU_3D_COLUMNS = ("iou_3d", "iou3d")

# A sweep CSV is recognised by its own header carrying a `sweep` column, not by
# where it sits: spec ch. 8 writes them to results/sweeps/ but a run that put
# them somewhere else is still worth rendering. Column names are taken from the
# header as read, since the orchestrated column list carries `baseline` and
# spec ch. 8 omits it.
SWEEP_MARKER_COLUMN = "sweep"

# Companion-JSON shapes for the crossover verdict. It is the headline sweep
# deliverable (section 6.3), so a verdict that cannot be parsed is reported as
# missing rather than left out: silence collapses into "no crossover", which is
# a completely different finding.
CROSSOVER_KEYS = ("crossover", "crossovers", "crossover_magnitude")
REASON_KEYS = ("reason", "message", "verdict")
MAGNITUDE_KEYS = ("magnitude", "value", "crossover_magnitude")
VERDICT_FIELDS = ("magnitude", "value", "unit", "axis", "baseline", "crossed",
                  "reason", "message", "verdict")

# The published reference is quoted at the precision it was published at, so a
# reader can match the row against the devkit README character for character.
PUBLISHED_DIGITS = 4

PLOT_SUFFIX = ".png"

DEFAULT_TITLE = "Semseg Eval Report"


def _dig(payload, path):
    """Value at a dotted path, or None when any step of it is missing."""
    node = payload
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def _pick(payload, *paths):
    """First present value among `paths`. Absence is None, and a real 0.0 is
    kept, which is why this tests for None rather than for truthiness."""
    for path in paths:
        value = _dig(payload, path)
        if value is not None:
            return value
    return None


def fnum(value, digits=3, dash="n/a"):
    """Format a metric, or `dash` when it is absent or undefined.

    nan is checked explicitly: an undefined IoU is a real outcome (a class
    present in neither the ground truth nor the prediction, see
    metrics.iou_per_class) and rendering it as the string "nan" would read as
    a bug in the scorer rather than as missing data.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return dash

    if number != number:
        return dash

    return f"{number:.{digits}f}"


def _column(row, candidates):
    """The row's value under the first candidate column present, or None."""
    for name in candidates:
        if name in row:
            return row[name]
    return None


def _read_per_class(path):
    """[(class name, iou_2d, iou_3d)] from a per_class.csv, in file order."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    parsed = []
    for index, row in enumerate(rows):
        name = _column(row, CLASS_COLUMNS)
        parsed.append((str(name) if name is not None else f"class {index}",
                       _column(row, IOU_2D_COLUMNS),
                       _column(row, IOU_3D_COLUMNS)))
    return parsed


def _per_class_from_summary(summary):
    """Fallback for a run with no per_class.csv.

    The summary's iou_per_class lists are positional, so the names only appear
    when the summary also carries them. Numbered placeholders are used instead
    of guessing at the ontology: report.py does not own the label space.
    """
    iou_2d = _pick(summary, "metrics_2d.iou_per_class", "overall.iou_per_class_2d") or []
    iou_3d = _pick(summary, "metrics_3d.iou_per_class", "overall.iou_per_class_3d") or []
    names = _pick(summary, "class_names", "label_space_class_names")

    count = max(len(iou_2d), len(iou_3d))
    parsed = []
    for index in range(count):
        name = names[index] if names and index < len(names) else f"class {index}"
        parsed.append((str(name),
                       iou_2d[index] if index < len(iou_2d) else None,
                       iou_3d[index] if index < len(iou_3d) else None))
    return parsed


def _warn_schema(method, summary_path, summary):
    """One stderr line when the sort metric is missing.

    A leaderboard of dashes renders happily and looks finished, so a schema
    mismatch between score.py and this file has to announce itself somewhere.
    Naming both the paths tried and the keys actually present makes the
    reconcile a one-line diff.
    """
    paths = ", ".join(METRIC_PATHS[SORT_METRIC])
    keys = ", ".join(sorted(summary)) if isinstance(summary, dict) else "not an object"
    print(f"report: {method}: no {SORT_METRIC} in {summary_path}; looked for {paths}; "
          f"top-level keys are [{keys}]", file=sys.stderr)


def load_runs(results_dir: str):
    """{method: run} for every scored run under `results_dir`.

    Run directories are named <split>_<method>_<timestamp>, so the sorted glob
    visits them oldest first and the newest scoring of a method wins.
    """
    runs = {}
    pattern = os.path.join(results_dir, "*", "summary.json")

    for summary_path in sorted(glob.glob(pattern)):
        with open(summary_path) as f:
            summary = json.load(f)

        run_dir = os.path.dirname(summary_path)
        method = _pick(summary, "run.method", "method") or os.path.basename(run_dir)

        per_class_path = os.path.join(run_dir, PER_CLASS_FILE)
        per_class = (_read_per_class(per_class_path)
                     if os.path.exists(per_class_path) else _per_class_from_summary(summary))

        metrics = {name: _pick(summary, *paths) for name, paths in METRIC_PATHS.items()}
        if metrics[SORT_METRIC] is None:
            _warn_schema(method, summary_path, summary)

        by_range = _pick(summary, "metrics_3d.miou_by_range", "overall.miou_3d_by_range") or {}

        runs[str(method)] = {
            "summary": summary,
            "metrics": metrics,
            "by_range": by_range,
            "per_class": per_class,
        }

    return runs


CSS = """
body{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;font-size:14px;
  line-height:1.5;color:#111;background:#fff;margin:24px;max-width:1400px}
h1{font-size:20px;margin:0 0 4px}
h2{font-size:15px;margin:28px 0 8px;border-bottom:1px solid #999;padding-bottom:3px}
h3{font-size:13px;margin:18px 0 4px;font-family:ui-monospace,Menlo,Consolas,monospace}
p.note{color:#444;font-size:13px;margin:6px 0 0;max-width:90ch}
p.verdict{font-size:13px;margin:6px 0 0;max-width:90ch;border-left:3px solid #999;
  padding-left:8px;color:#111}
table{border-collapse:collapse;margin-top:6px;font-size:13px}
th,td{border:1px solid #999;padding:4px 8px;text-align:right;white-space:nowrap}
th{background:#eee;font-weight:600;text-align:right}
th:first-child,td:first-child{text-align:left}
td.name,td.num{font-family:ui-monospace,Menlo,Consolas,monospace;
  font-variant-numeric:tabular-nums}
td.desc{white-space:normal;text-align:left;font-size:12px;color:#333;max-width:60ch}
tr.reference td{background:#f0f0f0;font-style:italic;color:#333}
tr.floor td{background:#f7f7f7;color:#333}
td.aux{color:#555}
span.aux{color:#555;font-size:11px}
span.safety{color:#a00;font-weight:600}
.scroll{overflow-x:auto}
figure{margin:14px 0 0}
figure img{max-width:100%;border:1px solid #999;display:block}
figcaption{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;
  color:#444;margin-top:3px}
footer{margin-top:28px;border-top:1px solid #999;padding-top:8px;
  font-size:12px;color:#444}
"""


def table_html(head, body):
    if not head:
        return ""

    ths = "".join(f"<th>{html.escape(h)}</th>" for h in head)
    return (f'<div class="scroll"><table><thead><tr>{ths}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def rank_key(item):
    """Sort by 3D mIoU descending, unknown and nan last, method name breaking
    ties.

    Methods with no 3D mIoU are an expected input, not an edge case: spec 13.3
    says the fused arm and cross-modal consistency must be SKIPPED on the
    public GOOSE release for want of an extrinsic. A bare `-value` key would
    raise on None and order nondeterministically on nan.
    """
    method, run = item
    value = run["metrics"][SORT_METRIC]
    known = value is not None and float(value) == float(value)
    return (0 if known else 1, -float(value) if known else 0.0, method)


def ranked_methods(runs):
    """(measured, floor) method names. The chance floor is held out so it can
    be rendered below the measured rows instead of competing with them."""
    order = [m for m, _ in sorted(runs.items(), key=rank_key)]
    measured = [m for m in order if m != FLOOR_METHOD]
    floor = [m for m in order if m == FLOOR_METHOD]
    return measured, floor


def _consistency_cell(metrics):
    """Consistency and its coverage in one cell.

    Assumption A2 says most points have no pixel at all, so the bare ratio is
    unreadable: 0.95 over 3 percent of the cloud and 0.95 over 80 percent are
    different results and the ratio alone cannot tell them apart.
    """
    return (f'<td class="num">{fnum(metrics["consistency"])} '
            f'<span class="aux">cov {fnum(metrics["coverage"], 3)}</span></td>')


def _miou_3d_cell(metrics):
    """9-class 3D mIoU, with the folded 8-class view beside it when the scorer
    supplied one. The folded number is the only one comparable to the published
    PTv3 row below (spec ch. 1)."""
    folded = metrics["folded_miou_3d"]
    aux = "" if folded is None else f' <span class="aux">8c {fnum(folded)}</span>'
    return f'<td class="num">{fnum(metrics["miou_3d"])}{aux}</td>'


LEADERBOARD_HEAD = ["method", "2D mIoU", "3D mIoU", "3D fwIoU", "boundary 2D mIoU",
                    "consistency (coverage)", "ECE 2D / 3D", "sec/frame"]


def _leaderboard_row(method, run, row_class=""):
    metrics = run["metrics"]
    klass = f' class="{row_class}"' if row_class else ""
    return (f"<tr{klass}><td class=\"name\">{html.escape(method)}</td>"
            f'<td class="num">{fnum(metrics["miou_2d"])}</td>'
            f'{_miou_3d_cell(metrics)}'
            f'<td class="num">{fnum(metrics["fwiou_3d"])}</td>'
            f'<td class="num">{fnum(metrics["boundary_miou_2d"])}</td>'
            f'{_consistency_cell(metrics)}'
            f'<td class="num">{fnum(metrics["ece_2d"])} / {fnum(metrics["ece_3d"])}</td>'
            f'<td class="num">{fnum(metrics["sec_per_frame"], 2)}</td></tr>')


def _reference_row(runs):
    """The published PTv3 row, appended and visually set apart.

    Its mIoU is taken from the scorer's own reference block when it carries
    one, so the page cannot disagree with the artefact it describes, and falls
    back to the module constant so the row renders either way.
    """
    published = None
    for run in runs.values():
        published = _pick(run["summary"], "reference.miou_3d",
                          "reference.miou", "reference_published.miou_3d")
        if published is not None:
            break

    miou = PTV3_MIOU_3D if published is None else published
    return (f'<tr class="reference"><td class="name">{html.escape(PTV3_LABEL)}</td>'
            f'<td class="num">n/a</td>'
            f'<td class="num">{fnum(miou, PUBLISHED_DIGITS)} '
            f'<span class="aux">8c</span></td>'
            f'<td class="num">n/a</td><td class="num">n/a</td>'
            f'<td class="num">n/a</td><td class="num">n/a</td>'
            f'<td class="num">n/a</td></tr>')


def leaderboard(runs):
    measured, floor = ranked_methods(runs)

    body = [_leaderboard_row(m, runs[m]) for m in measured]
    body += [_leaderboard_row(m, runs[m], row_class="floor") for m in floor]
    body.append(_reference_row(runs))

    return LEADERBOARD_HEAD, body


def class_order(runs, methods):
    """Class names in the order the runs list them, unioned across methods so
    a method that scored fewer classes still lines up with the others."""
    order = []
    for method in methods:
        for name, _iou_2d, _iou_3d in runs[method]["per_class"]:
            if name not in order:
                order.append(name)
    return order


def _class_label(name):
    mark = f' <span class="safety">{SAFETY_MARK}</span>' if name in SAFETY_CLASSES else ""
    return f"{html.escape(name)}{mark}"


def per_class_table(runs):
    """Rows are classes, columns are methods, each cell 2D IoU / 3D IoU.

    In full, both modalities, because section 6.1 says never report only the
    mean. The published reference gets its own column and stops at 8 classes,
    since the folded space has no sky.
    """
    measured, floor = ranked_methods(runs)
    methods = measured + floor
    names = class_order(runs, methods)
    if not names:
        return None, None

    head = ["class"] + [f"{m}: 2D / 3D" for m in methods] + ["PTv3 3D (published)"]
    body = []
    for name in names:
        cells = [f'<td class="name">{_class_label(name)}</td>']
        for method in methods:
            found = [(a, b) for n, a, b in runs[method]["per_class"] if n == name]
            iou_2d, iou_3d = found[0] if found else (None, None)
            cells.append(f'<td class="num">{fnum(iou_2d)} / {fnum(iou_3d)}</td>')

        published = PTV3_IOU_PER_CLASS.get(name)
        cells.append(f'<td class="num aux">{fnum(published, PUBLISHED_DIGITS)}</td>')
        body.append("<tr>" + "".join(cells) + "</tr>")

    return head, body


def range_table(runs):
    """Range-stratified 3D mIoU, one row per method.

    Point density falls off as 1/r^2, so a pooled 3D mIoU hides a model that
    is failing past 30 m while the near field carries the average
    (section 6.1).
    """
    measured, floor = ranked_methods(runs)
    methods = measured + floor

    present = {name for run in runs.values() for name in run["by_range"]}
    if not present:
        return None, None

    bins = [b for b in RANGE_BIN_ORDER if b in present]
    bins += sorted(b for b in present if b not in RANGE_BIN_ORDER)

    head = ["method"] + bins
    body = []
    for method in methods:
        cells = [f'<td class="name">{html.escape(method)}</td>']
        cells += [f'<td class="num">{fnum(runs[method]["by_range"].get(b))}</td>' for b in bins]
        body.append("<tr>" + "".join(cells) + "</tr>")

    return head, body


def frustum_table(runs):
    """In-frustum against out-of-frustum 3D mIoU, with the gap.

    This is the section 6.2 question, and the gap column is the answer to it:
    how much of a reported gain comes from points the camera never saw. A
    fused arm whose out-of-frustum column matches the lidar-only arm's has
    bought its gain entirely inside the frustum.
    """
    measured, floor = ranked_methods(runs)
    methods = measured + floor

    head = ["method", "3D mIoU in frustum", "3D mIoU out of frustum", "in minus out"]
    body = []
    for method in methods:
        metrics = runs[method]["metrics"]
        inside, outside = metrics["miou_in_frustum"], metrics["miou_out_frustum"]

        gap = None
        if inside is not None and outside is not None:
            gap = float(inside) - float(outside)

        body.append(f'<tr><td class="name">{html.escape(method)}</td>'
                    f'<td class="num">{fnum(inside)}</td>'
                    f'<td class="num">{fnum(outside)}</td>'
                    f'<td class="num">{fnum(gap)}</td></tr>')

    return head, body


def find_sweep_csvs(results_dir: str):
    """Every CSV under `results_dir` whose header declares a `sweep` column.

    Identified by content rather than by path, so a sweep written outside
    results/sweeps/ is still found, and per_class.csv is never mistaken for
    one.
    """
    found = []
    pattern = os.path.join(results_dir, "**", "*.csv")

    for path in sorted(glob.glob(pattern, recursive=True)):
        with open(path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)

        if header and SWEEP_MARKER_COLUMN in [h.strip() for h in header]:
            found.append(path)

    return found


def _load_companion(csv_path: str):
    """The sweep's companion JSON, holding the crossover verdict (spec ch. 8)."""
    stem = os.path.splitext(csv_path)[0]
    for candidate in (stem + ".json", stem + "_crossover.json"):
        if os.path.exists(candidate):
            with open(candidate) as f:
                return candidate, json.load(f)

    return None, None


def _is_verdict(node):
    return isinstance(node, dict) and any(field in node for field in VERDICT_FIELDS)


def crossover_entries(payload):
    """[(label, verdict dict)] from a companion JSON, whatever nesting it used:
    a single verdict at the top level, or one per baseline under a map."""
    if not isinstance(payload, dict):
        return []

    node = _pick(payload, *CROSSOVER_KEYS)
    if node is None:
        # A crossover that did not happen is still a verdict, carried as a
        # reason string with no magnitude beside it.
        reason = _pick(payload, *REASON_KEYS)
        return [(None, {"reason": reason})] if reason else []

    if _is_verdict(node):
        return [(None, node)]

    if isinstance(node, dict):
        return sorted(((str(k), v if isinstance(v, dict) else {}) for k, v in node.items()),
                      key=lambda kv: kv[0])

    return [(None, {"magnitude": node, "unit": payload.get("unit")})]


def verdict_sentence(label, verdict, fallback_reason=None):
    """One sentence a reader can quote, per sweep and per baseline.

    Section 6.3 calls the crossover magnitude the calibration accuracy the
    robot must sustain in production, so it is stated as prose and not left to
    be read off the CSV.
    """
    who = f"{label}: " if label else ""

    magnitude = _pick(verdict, *MAGNITUDE_KEYS)
    if magnitude is not None:
        # Built up conditionally rather than formatted with empty holes: a
        # companion JSON that omits the unit or the axis must still read as a
        # sentence, and a magnitude with no unit is a real gap worth seeing.
        unit = verdict.get("unit")
        axis = verdict.get("axis")
        amount = f"{fnum(magnitude)} {unit}" if unit else fnum(magnitude)
        about = f" about {axis}" if axis else ""
        return f"{who}fusion drops below lidar-only at {amount}{about}."

    reason = _pick(verdict, *REASON_KEYS) or fallback_reason
    if reason:
        return f"{who}no crossover: {reason}"

    # Never collapse a missing verdict into "no crossover". "Fusion never fell
    # below within the swept range" and "fusion was already below at the
    # smallest perturbation" are opposite findings, and an absent verdict
    # distinguishes neither.
    return f"{who}crossover verdict present but unreadable, so no finding can be quoted."


def sweep_sections(results_dir: str):
    """One block per sweep CSV: the crossover verdict, then the tidy rows."""
    blocks = []
    for csv_path in find_sweep_csvs(results_dir):
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)

        head, body_rows = rows[0], rows[1:]
        body = ["<tr>" + "".join(f'<td class="num">{html.escape(cell)}</td>' for cell in row)
                + "</tr>" for row in body_rows]

        companion_path, payload = _load_companion(csv_path)
        entries = crossover_entries(payload)
        fallback = _pick(payload or {}, *REASON_KEYS)

        if entries:
            sentences = [verdict_sentence(label, verdict or {}, fallback)
                         for label, verdict in entries]
        else:
            where = os.path.basename(companion_path or csv_path)
            sentences = [f"No crossover verdict found next to {where}. That is not the "
                         f"same finding as no crossover."]

        verdicts = "".join(f'<p class="verdict">{html.escape(s)}</p>' for s in sentences)
        name = os.path.relpath(csv_path, results_dir)
        blocks.append(f"<h3>{html.escape(name)}</h3>{verdicts}{table_html(head, body)}")

    return blocks


def find_plots(results_dir: str, out_path: str):
    """[(caption, href)] for every PNG under `results_dir`.

    Hrefs are relative to the report, not to the results directory, because
    --out can point outside it.
    """
    base = os.path.dirname(os.path.abspath(out_path)) or "."
    pattern = os.path.join(results_dir, "**", "*" + PLOT_SUFFIX)

    plots = []
    for path in sorted(glob.glob(pattern, recursive=True)):
        href = os.path.relpath(os.path.abspath(path), base)
        plots.append((os.path.relpath(path, results_dir), href))

    return plots


def plot_section(plots):
    if not plots:
        return ""

    figures = "".join(
        f'<figure><img src="{html.escape(href)}" alt="{html.escape(caption)}">'
        f"<figcaption>{html.escape(caption)}</figcaption></figure>"
        for caption, href in plots)
    return f"<h2>Plots</h2>{figures}"


def _facts(runs):
    summary = next(iter(runs.values()))["summary"]
    split = _pick(summary, "run.split", "split") or "?"
    n_frames = _pick(summary, "n_frames", "run.n_frames")
    generated = str(_pick(summary, "generated_at", "run.generated_at") or "")

    plural = "s" if len(runs) != 1 else ""
    return (f'<p class="note">split {html.escape(str(split))} &middot; '
            f'{html.escape(str(n_frames)) if n_frames is not None else "?"} frames &middot; '
            f'{len(runs)} method{plural} &middot; label space goose9, 9 classes &middot; '
            f'generated {html.escape(generated[:19])}</p>')


def render(runs, results_dir, title, plots=None):
    sections = [
        f"<h2>Leaderboard</h2>{table_html(*leaderboard(runs))}",
        f'<p class="note">Sorted by 3D mIoU over the goose9 9-class space. '
        f'Consistency is the fraction of scorable points whose 3D label matches the 2D label '
        f'of the pixel they project into, and its coverage is the fraction of points that had '
        f'a visible pixel at all: assumption A2 says most points have no pixel, so the ratio '
        f'alone is unreadable. Consistency and every sweep need the camera-to-lidar '
        f'extrinsic, which the public GOOSE release does not ship (spec 13.3), so those cells '
        f'are fixture-only and a blank one is a skipped measurement rather than a broken '
        f'method. sec/frame comes from the self-declared compute sidecar and is reported '
        f'beside the score, never folded into it. The {html.escape(FLOOR_METHOD)} row is the '
        f'chance floor and the PTv3 row is PUBLISHED, not measured here: it is quoted from '
        f'the {html.escape(PTV3_SOURCE)}, so it is not directly comparable to the 9-class '
        f'column, only to the folded 8-class figure shown beside it.</p>',
    ]

    head, body = per_class_table(runs)
    if head:
        sections.append(f"<h2>Per-class IoU</h2>{table_html(head, body)}"
                        f'<p class="note">In full and for both modalities, because section 6.1 '
                        f'says never report only the mean. '
                        f'<span class="safety">{SAFETY_MARK}</span> marks the safety-relevant '
                        f'classes: assumption A5 is that a 2 percent mIoU gain on terrain is '
                        f'worth less than a 2 percent gain on person, so a mean that moved on '
                        f'vegetation and a mean that moved on human are not the same result. '
                        f'n/a is an undefined IoU, a class present in neither the ground truth '
                        f'nor the prediction, which is not the same as 0.</p>')

    head, body = range_table(runs)
    if head:
        sections.append(f"<h2>Range-stratified 3D mIoU</h2>{table_html(head, body)}"
                        f'<p class="note">Lidar point density falls off as 1/r^2, so a pooled '
                        f'3D mIoU hides a model that is failing at 30 m while the near field '
                        f'carries the average. Range is measured from the lidar origin, never '
                        f'camera depth.</p>')

    sections.append(f"<h2>In frustum versus out of frustum</h2>{table_html(*frustum_table(runs))}"
                    f'<p class="note">Section 6.2 asks how much of a reported fusion gain comes '
                    f'from points the camera never saw. The gap column is the answer: an arm '
                    f'whose out-of-frustum column matches the lidar-only arm bought its whole '
                    f'gain inside the frustum, and a 90 degree horizontal field of view holds '
                    f'at most 23.2 percent of a real GOOSE cloud.</p>')

    blocks = sweep_sections(results_dir)
    if blocks:
        sections.append(f'<h2>Robustness sweeps</h2>{"".join(blocks)}'
                        f'<p class="note">Section 6.3 wants these run before architecture work, '
                        f'not after, and calls the crossover magnitude the calibration accuracy '
                        f'the robot must sustain in production. Rotations are swept per axis '
                        f'because a rotation about the optical axis is nearly harmless while '
                        f'the same magnitude about pitch is a lateral shift proportional to '
                        f'range.</p>')

    sections.append(plot_section(plots or []))

    return f"""<title>{html.escape(title)}</title>
<style>{CSS}</style>
<h1>{html.escape(title)}</h1>
{_facts(runs)}
{"".join(sections)}
<footer>Generated by eval/report.py from summary.json, per_class.csv and the sweep CSVs.
No ground truth, cloud or image is read.</footer>"""


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True,
                        help="directory holding scored run subdirectories")
    parser.add_argument("--out", help="output HTML (default: <results>/report.html)")
    parser.add_argument("--title", default=DEFAULT_TITLE)
    args = parser.parse_args(argv)

    runs = load_runs(args.results)
    if not runs:
        print(f"report: no scored runs found under {args.results}")
        return 1

    out = args.out or os.path.join(args.results, "report.html")
    plots = find_plots(args.results, out)

    with open(out, "w") as f:
        f.write(render(runs, args.results, args.title, plots))

    plural = "s" if len(runs) != 1 else ""
    print(f"wrote {out} ({len(runs)} method{plural})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
