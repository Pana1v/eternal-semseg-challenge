"""Tests for eval/report.py.

The report has one piece of real logic, the leaderboard ordering, and one real
obligation, that every number it was handed actually reaches the page. Both are
tested by perturbation: a metric is moved by a small amount and the rendered
HTML has to move with it. Asserting that the output contains "<table>" would
pass against a file that dropped every value on the floor.

report.py reads only summary.json, per_class.csv and the sweep CSVs, so nothing
here needs a stub: the fixtures below write those three files directly.
"""

import json
import os
import re

import pytest

from eval.report import (
    FLOOR_METHOD, METRIC_PATHS, PTV3_LABEL, PTV3_MIOU_3D, SAFETY_CLASSES,
    SAFETY_MARK, SORT_METRIC, crossover_entries, fnum, load_runs, main,
    rank_key, verdict_sentence,
)

# goose9, spec ch. 1. Test data only: report.py deliberately does not own the
# label space and reads these names out of per_class.csv.
CLASS_NAMES = ("other", "artificial_structures", "artificial_ground", "natural_ground",
               "obstacle", "vehicle", "vegetation", "human", "sky")

# One value per METRIC_PATHS key, so a test can override exactly one of them
# and know that nothing else moved.
DEFAULT_METRICS = {
    "miou_2d": 0.412,
    "miou_3d": 0.601,
    "fwiou_3d": 0.734,
    "folded_miou_3d": 0.658,
    "boundary_miou_2d": 0.287,
    "ece_2d": 0.061,
    "ece_3d": 0.093,
    "consistency": 0.812,
    "coverage": 0.290,
    "miou_in_frustum": 0.655,
    "miou_out_frustum": 0.573,
    "sec_per_frame": 1.24,
}

RANGE_MIOU = {"0-5m": 0.55, "5-15m": 0.62, "15-30m": 0.48, "30m+": 0.31}

# Large enough to survive three-digit formatting, small enough that it is a
# perturbation and not a different scenario.
DELTA = 0.05


def summary_payload(method, split="score", metrics=None, with_compute=True):
    values = dict(DEFAULT_METRICS)
    values.update(metrics or {})

    payload = {
        "generated_at": "2026-09-04T18:00:00",
        "n_frames": 12,
        "run": {"method": method, "split": split},
        "metrics_2d": {
            "miou": values["miou_2d"],
            "fwiou": 0.501,
            "boundary_miou": values["boundary_miou_2d"],
            "ece": values["ece_2d"],
            "iou_per_class": [0.4] * len(CLASS_NAMES),
        },
        "metrics_3d": {
            "miou": values["miou_3d"],
            "fwiou": values["fwiou_3d"],
            "folded_miou": values["folded_miou_3d"],
            "ece": values["ece_3d"],
            "iou_per_class": [0.6] * len(CLASS_NAMES),
            "miou_by_range": dict(RANGE_MIOU),
            "miou_in_frustum": values["miou_in_frustum"],
            "miou_out_frustum": values["miou_out_frustum"],
        },
        "consistency": {"consistency": values["consistency"], "coverage": values["coverage"]},
        "reference": {"miou_3d": PTV3_MIOU_3D},
    }

    if with_compute:
        payload["compute"] = {"runtime_sec_per_frame": values["sec_per_frame"],
                              "peak_rss_mb": 900.0}

    return payload


def write_run(results_dir, method, metrics=None, split="score", stamp="20260904_180000",
              with_compute=True, with_per_class=True, payload=None):
    """One scored run directory, named the way score.py names them."""
    run_dir = os.path.join(results_dir, f"{split}_{method}_{stamp}")
    os.makedirs(run_dir, exist_ok=True)

    body = payload if payload is not None else summary_payload(
        method, split=split, metrics=metrics, with_compute=with_compute)
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(body, f)

    if with_per_class:
        lines = ["class,iou_2d,iou_3d"]
        for index, name in enumerate(CLASS_NAMES):
            lines.append(f"{name},{0.30 + 0.01 * index:.3f},{0.50 + 0.01 * index:.3f}")
        with open(os.path.join(run_dir, "per_class.csv"), "w") as f:
            f.write("\n".join(lines) + "\n")

    return run_dir


def render_to(results_dir, out_name="report.html"):
    """Run the CLI and hand back the rendered HTML."""
    out = os.path.join(str(results_dir), out_name)
    assert main(["--results", str(results_dir), "--out", out]) == 0

    with open(out) as f:
        return f.read()


def row_order(page, methods):
    """Where each method's leaderboard row starts, in page order."""
    return sorted(methods, key=lambda m: page.index(f">{m}<"))


def test_single_method_renders(tmp_path):
    """One method, no sweeps, no plots: the minimum a scoring run produces."""
    write_run(tmp_path, "bl_paint")
    page = render_to(tmp_path)

    assert "bl_paint" in page
    assert "0.601" in page               # the 3D mIoU it was handed
    assert "Robustness sweeps" not in page
    assert "<img" not in page
    assert "<title>" in page


def test_empty_results_dir_reports_and_fails(tmp_path, capsys):
    assert main(["--results", str(tmp_path)]) == 1
    assert "no scored runs" in capsys.readouterr().out


def test_sorted_by_3d_miou(tmp_path):
    write_run(tmp_path, "bl_cam2d", metrics={"miou_3d": 0.40})
    write_run(tmp_path, "bl_paint", metrics={"miou_3d": 0.60})

    page = render_to(tmp_path)
    assert row_order(page, ["bl_paint", "bl_cam2d"]) == ["bl_paint", "bl_cam2d"]


def test_order_flips_when_3d_miou_overtakes(tmp_path):
    """The perturbation case for the one ordering decision this file makes."""
    write_run(tmp_path, "bl_cam2d", metrics={"miou_3d": 0.40})
    write_run(tmp_path, "bl_paint", metrics={"miou_3d": 0.60})
    before = render_to(tmp_path, "before.html")

    write_run(tmp_path, "bl_cam2d", metrics={"miou_3d": 0.70}, stamp="20260904_190000")
    after = render_to(tmp_path, "after.html")

    assert row_order(before, ["bl_paint", "bl_cam2d"]) == ["bl_paint", "bl_cam2d"]
    assert row_order(after, ["bl_paint", "bl_cam2d"]) == ["bl_cam2d", "bl_paint"]


@pytest.mark.parametrize("field", sorted(METRIC_PATHS))
def test_every_metric_reaches_the_page(tmp_path, field):
    """Move one metric, the page must move. A metric that never reaches the
    HTML would pass every structural assertion in this file."""
    baseline_dir = tmp_path / "baseline"
    moved_dir = tmp_path / "moved"

    write_run(baseline_dir, "bl_paint")
    write_run(moved_dir, "bl_paint", metrics={field: DEFAULT_METRICS[field] + DELTA})

    before = render_to(baseline_dir)
    after = render_to(moved_dir)
    assert before != after, f"{field} does not reach the report"


def test_newest_run_of_a_method_wins(tmp_path):
    write_run(tmp_path, "bl_paint", metrics={"miou_3d": 0.10}, stamp="20260904_180000")
    write_run(tmp_path, "bl_paint", metrics={"miou_3d": 0.90}, stamp="20260904_190000")

    runs = load_runs(str(tmp_path))
    assert len(runs) == 1
    assert runs["bl_paint"]["metrics"]["miou_3d"] == pytest.approx(0.90)


def leaderboard_row(page, method):
    """The one leaderboard row for `method`, tags stripped to its cells."""
    start = page.index(f">{method}<")
    row = page[start:page.index("</tr>", start)]
    return [cell for cell in re.sub(r"<[^>]+>", "|", row).split("|") if cell.strip()]


def test_missing_compute_sidecar_shows_na(tmp_path):
    """Compute is a self-declared optional KPI (spec 6.4): its absence must
    never fail a scoring run. Asserted on the sec/frame cell itself, since the
    published reference row carries n/a in every column regardless."""
    write_run(tmp_path, "bl_geom3d", with_compute=False)
    write_run(tmp_path, "bl_paint")
    page = render_to(tmp_path)

    assert leaderboard_row(page, "bl_geom3d")[-1] == "n/a"
    assert leaderboard_row(page, "bl_paint")[-1] == "1.24"


def test_unrecognised_schema_renders_and_warns(tmp_path, capsys):
    """A leaderboard of dashes looks finished, so a schema mismatch between
    score.py and report.py has to announce itself."""
    write_run(tmp_path, "bl_paint", payload={"run": {"method": "bl_paint"},
                                             "totally_different": {"miou": 0.5}},
              with_per_class=False)
    page = render_to(tmp_path)
    errors = capsys.readouterr().err

    assert "bl_paint" in page
    assert "bl_paint" in errors
    assert SORT_METRIC in errors
    assert "totally_different" in errors


def test_absent_3d_miou_sorts_last(tmp_path):
    """Expected input, not an edge case: spec 13.3 skips the fused arm on the
    public GOOSE release for want of an extrinsic."""
    write_run(tmp_path, "bl_none", metrics={"miou_3d": None})
    write_run(tmp_path, "bl_nan", metrics={"miou_3d": float("nan")})
    write_run(tmp_path, "bl_real", metrics={"miou_3d": 0.30})

    page = render_to(tmp_path)
    assert row_order(page, ["bl_real", "bl_none", "bl_nan"])[0] == "bl_real"


def test_rank_key_is_stable_on_ties():
    runs = {"bl_b": {"metrics": {SORT_METRIC: 0.5}},
            "bl_a": {"metrics": {SORT_METRIC: 0.5}}}
    order = [m for m, _ in sorted(runs.items(), key=rank_key)]
    assert order == ["bl_a", "bl_b"]


def test_floor_row_never_outranks_a_measured_row(tmp_path):
    """bl_prior is the number every other row has to beat, so even a floor
    that scored higher stays below the measured rows."""
    write_run(tmp_path, FLOOR_METHOD, metrics={"miou_3d": 0.99})
    write_run(tmp_path, "bl_paint", metrics={"miou_3d": 0.30})

    page = render_to(tmp_path)
    assert page.index(">bl_paint<") < page.index(f">{FLOOR_METHOD}<")


def test_reference_row_is_last_and_labelled_published(tmp_path):
    write_run(tmp_path, "bl_paint", metrics={"miou_3d": 0.30})
    page = render_to(tmp_path)

    assert PTV3_LABEL in page
    assert "NOT measured here" in page
    assert page.index(">bl_paint<") < page.index(PTV3_LABEL)
    assert "8-class folded" in page

    # quoted at the precision it was published at, not rounded to the report's
    # usual three digits
    assert f"{PTV3_MIOU_3D:.4f}" in page


def test_safety_classes_marked(tmp_path):
    write_run(tmp_path, "bl_paint")
    page = render_to(tmp_path)

    for name in SAFETY_CLASSES:
        marked = f'{name} <span class="safety">{SAFETY_MARK}</span>'
        assert marked in page, f"{name} is not marked safety relevant"

    assert 'vegetation <span class="safety">' not in page


def test_per_class_falls_back_to_summary(tmp_path):
    """No per_class.csv, so the names are unavailable and positional
    placeholders stand in rather than a guessed ontology."""
    write_run(tmp_path, "bl_paint", with_per_class=False)
    page = render_to(tmp_path)

    assert "class 0" in page
    assert "0.600" in page      # the summary's positional 3D IoU


def test_range_table_keeps_an_unknown_bin(tmp_path):
    payload = summary_payload("bl_paint")
    payload["metrics_3d"]["miou_by_range"]["50m+"] = 0.11
    write_run(tmp_path, "bl_paint", payload=payload)

    page = render_to(tmp_path)
    assert "50m+" in page
    assert "0.110" in page


def test_frustum_gap_is_in_minus_out(tmp_path):
    write_run(tmp_path, "bl_paint", metrics={"miou_in_frustum": 0.70,
                                             "miou_out_frustum": 0.50})
    page = render_to(tmp_path)
    assert "0.200" in page


def _write_sweep(root, name, rows, companion=None):
    sweep_dir = os.path.join(str(root), "sweeps")
    os.makedirs(sweep_dir, exist_ok=True)

    csv_path = os.path.join(sweep_dir, name + ".csv")
    with open(csv_path, "w") as f:
        f.write("sweep,baseline,axis,magnitude,unit,miou_3d\n")
        for row in rows:
            f.write(",".join(str(c) for c in row) + "\n")

    if companion is not None:
        with open(os.path.join(sweep_dir, name + ".json"), "w") as f:
            json.dump(companion, f)

    return csv_path


def test_sweep_table_and_crossover_sentence(tmp_path):
    write_run(tmp_path, "bl_paint")
    _write_sweep(tmp_path, "decalib",
                 [("decalib", "bl_paint", "pitch", 0.5, "deg", 0.58),
                  ("decalib", "bl_paint", "pitch", 1.0, "deg", 0.41)],
                 companion={"crossover": {"magnitude": 0.62, "unit": "deg", "axis": "pitch"}})

    page = render_to(tmp_path)
    assert "Robustness sweeps" in page
    assert "decalib.csv" in page
    assert "0.620 deg" in page
    assert "about pitch" in page
    assert "drops below lidar-only" in page


def test_sweep_reason_only_verdict_is_quoted_verbatim(tmp_path):
    """The two no-crossover findings are opposite and must not both render as
    "no crossover" with nothing else said (spec ch. 8)."""
    write_run(tmp_path, "bl_paint")
    reason = "fusion never falls below lidar-only within 2.0 deg"
    _write_sweep(tmp_path, "decalib", [("decalib", "bl_paint", "roll", 2.0, "deg", 0.55)],
                 companion={"crossover": None, "reason": reason})

    page = render_to(tmp_path)
    assert reason in page


def test_sweep_per_baseline_verdicts(tmp_path):
    write_run(tmp_path, "bl_paint")
    _write_sweep(tmp_path, "time_offset", [("time_offset", "bl_paint", "", 50, "ms", 0.5)],
                 companion={"crossovers": {
                     "bl_paint": {"magnitude": 38.0, "unit": "ms"},
                     "bl_cam2d": {"reason": "already below lidar-only at 0 ms"}}})

    page = render_to(tmp_path)
    assert "bl_paint: fusion drops below lidar-only at 38.000 ms" in page
    assert "bl_cam2d: no crossover: already below lidar-only at 0 ms" in page


def test_sweep_without_companion_json_says_the_verdict_is_missing(tmp_path):
    write_run(tmp_path, "bl_paint")
    _write_sweep(tmp_path, "dropout", [("dropout", "bl_paint", "", 0.5, "frac", 0.44)])

    page = render_to(tmp_path)
    assert "No crossover verdict found next to dropout.csv" in page
    assert "not the same finding as no crossover" in page


def test_per_class_csv_is_not_mistaken_for_a_sweep(tmp_path):
    write_run(tmp_path, "bl_paint")
    page = render_to(tmp_path)
    assert "Robustness sweeps" not in page


def test_plots_embedded_by_relative_path(tmp_path):
    write_run(tmp_path, "bl_paint")
    plots_dir = tmp_path / "plots"
    plots_dir.mkdir()
    (plots_dir / "per_class_iou.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    page = render_to(tmp_path)
    assert '<img src="plots/per_class_iou.png"' in page
    assert "per_class_iou.png" in page


def test_plot_href_is_relative_to_out_not_results(tmp_path):
    """--out can point outside the results directory, so the href has to be
    anchored on the report rather than on --results."""
    results = tmp_path / "results"
    results.mkdir()
    write_run(results, "bl_paint")
    (results / "sweep_decalib.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    out = tmp_path / "elsewhere" / "report.html"
    out.parent.mkdir()
    assert main(["--results", str(results), "--out", str(out)]) == 0

    page = out.read_text()
    assert 'src="../results/sweep_decalib.png"' in page


def test_title_is_used(tmp_path):
    write_run(tmp_path, "bl_paint")
    out = tmp_path / "report.html"
    assert main(["--results", str(tmp_path), "--out", str(out), "--title", "Fixture run"]) == 0

    page = out.read_text()
    assert "<title>Fixture run</title>" in page
    assert "<h1>Fixture run</h1>" in page


def test_default_out_is_inside_results(tmp_path):
    write_run(tmp_path, "bl_paint")
    assert main(["--results", str(tmp_path)]) == 0
    assert (tmp_path / "report.html").exists()


def test_fnum_renders_absent_and_undefined_alike():
    assert fnum(0.5) == "0.500"
    assert fnum(0.5, digits=1) == "0.5"
    assert fnum(None) == "n/a"
    assert fnum(float("nan")) == "n/a"
    assert fnum("not a number") == "n/a"

    # a real zero is a measurement, not missing data
    assert fnum(0.0) == "0.000"


def test_fnum_moves_with_its_input():
    assert fnum(0.500) != fnum(0.501)


def test_crossover_entries_handles_every_shape():
    assert crossover_entries(None) == []
    assert crossover_entries({}) == []

    single = crossover_entries({"crossover": {"magnitude": 1.0, "unit": "deg"}})
    assert single == [(None, {"magnitude": 1.0, "unit": "deg"})]

    per_baseline = crossover_entries({"crossovers": {"b": {"magnitude": 2.0},
                                                     "a": {"magnitude": 1.0}}})
    assert [label for label, _ in per_baseline] == ["a", "b"]

    bare = crossover_entries({"crossover": 0.75, "unit": "deg"})
    assert bare == [(None, {"magnitude": 0.75, "unit": "deg"})]

    reason_only = crossover_entries({"reason": "no crossing"})
    assert reason_only == [(None, {"reason": "no crossing"})]


def test_verdict_sentence_distinguishes_the_findings():
    crossed = verdict_sentence(None, {"magnitude": 0.5, "unit": "deg"})
    never = verdict_sentence(None, {"reason": "fusion never falls below within 2.0 deg"})
    already = verdict_sentence(None, {"reason": "fusion is already below at 0.1 deg"})
    unreadable = verdict_sentence(None, {})

    assert "0.500 deg" in crossed
    assert never != already
    assert "never falls below" in never
    assert "already below" in already
    assert "unreadable" in unreadable
