#!/usr/bin/env python3
"""Renders the candidate-facing problem statement PDF.

Kept as a script rather than a hand-made document because every number in it
is measured, and a measured number goes stale. Re-run this after a scoring
run and the baseline table updates itself; nothing in the PDF is typed in by
hand except the prose.

    python tools/make_problem_statement.py --out semseg-problem-statement.pdf \\
        [--results results] [--goose-root /path/to/goose]

The design follows the GLoc challenge statement so the two read as a set:
A4, one accent green, Manrope for text and Courier for anything a candidate
has to type. Palette values were sampled from that document rather than
guessed.
"""

import argparse
import csv
import glob
import json
import os

from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

# Palette sampled from eternal-gloc-problem-statement.pdf, not invented, so the
# two documents sit side by side without a visible seam.
INK = HexColor("#0e1512")
ACCENT = HexColor("#1f6b45")
MUTED = HexColor("#6b7671")
FADED = HexColor("#99a09d")
BAND = HexColor("#f3f6f4")
RULE = HexColor("#dadddb")

PAGE_W, PAGE_H = A4
MARGIN = 46.0
CONTENT_W = PAGE_W - 2 * MARGIN

# Tight leading is what lets a dense statement stay short enough to read in
# one sitting. The reference document runs at roughly the same ratio.
SIZE_TITLE = 19.0
SIZE_LEAD = 8.0
SIZE_HEAD = 8.6
SIZE_BODY = 7.8
SIZE_MONO = 7.0
SIZE_META = 7.6
LEAD_BODY = 10.2
LEAD_MONO = 9.6
GAP_SECTION = 7.0
GAP_PARA = 3.2

FONT_DIRS = (
    "/home/pan-navigator/workspaces/references/isaac-sim-dashboard/control-app/app/static/fonts",
    "/home/pan-navigator/.local/share/fonts",
    "/usr/share/fonts/truetype/manrope",
)
FONT_FILES = {
    "Manrope": "Manrope-Regular.ttf",
    "Manrope-Md": "Manrope-Medium.ttf",
    "Manrope-Sb": "Manrope-SemiBold.ttf",
    "Manrope-Bd": "Manrope-Bold.ttf",
}

# Fallbacks keep the script runnable on a machine without the brand font. The
# document then looks wrong but still builds, which beats failing a release.
FALLBACK = {
    "Manrope": "Helvetica",
    "Manrope-Md": "Helvetica",
    "Manrope-Sb": "Helvetica-Bold",
    "Manrope-Bd": "Helvetica-Bold",
}

# The one frame the whole statement is written around. Choosing a single
# concrete case rather than surveying the dataset is deliberate: a candidate
# who understands this frame understands the task.
CASE_SEQUENCE = "2023-05-17_neubiberg_sunny"
CASE_FRAME = "2023-05-17_neubiberg_sunny__0419_1684329877834694088"

# Camera field of view used for the frustum-coverage estimate. This is an
# upper bound on what any forward camera could see, computed from the cloud's
# own azimuths, so it needs no calibration and cannot be mistaken for one.
ESTIMATE_HFOV_DEG = 90.0

# Published elsewhere, not measured here. Carried so a candidate knows what
# good looks like on this data.
PTV3_REFERENCE_MIOU = 0.8096


def register_fonts():
    """Returns True when the brand font was found, False when falling back."""
    for name, filename in FONT_FILES.items():
        path = next((os.path.join(d, filename) for d in FONT_DIRS
                     if os.path.exists(os.path.join(d, filename))), None)
        if path is None:
            return False
        pdfmetrics.registerFont(TTFont(name, path))

    return True


class Flow:
    """A single-column text flow with automatic page breaks.

    Written by hand rather than with platypus because the layout is one column
    of short blocks and a couple of hairline tables. Platypus would add a frame
    and style abstraction for no gain here.
    """

    def __init__(self, c, fonts_ok):
        self.c = c
        self.fonts = FONT_FILES if fonts_ok else FALLBACK
        self.y = PAGE_H
        self.page = 0
        self._start_page()

    def _f(self, key):
        return FALLBACK[key] if self.fonts is FALLBACK else key

    def _start_page(self):
        self.page += 1
        self.y = PAGE_H - 39.0
        self._header()

    def _header(self):
        c = self.c
        band_h = 23.0
        c.setFillColor(BAND)
        c.rect(MARGIN, self.y - band_h, CONTENT_W, band_h, stroke=0, fill=1)

        c.setFillColor(ACCENT)
        c.rect(MARGIN, self.y - band_h, 2.2, band_h, stroke=0, fill=1)

        c.setFillColor(INK)
        c.setFont(self._f("Manrope-Bd"), 9.4)
        c.drawString(MARGIN + 14, self.y - band_h + 7.6, "eternal.ag")

        c.setFillColor(MUTED)
        c.setFont(self._f("Manrope"), 7.6)
        c.drawRightString(PAGE_W - MARGIN - 10, self.y - band_h + 7.6,
                          "Track A  ·  Problem Statement")

        self.y -= band_h + 13.0

    def space(self, pts):
        self.y -= pts

    def need(self, pts):
        if self.y - pts < MARGIN:
            self.c.showPage()
            self._start_page()

    def title(self, text):
        self.c.setFillColor(INK)
        self.c.setFont(self._f("Manrope-Bd"), SIZE_TITLE)
        self.c.drawString(MARGIN, self.y - SIZE_TITLE, text)
        self.y -= SIZE_TITLE + 4.0

    def heading(self, text):
        self.need(26.0)
        self.space(GAP_SECTION)
        self.c.setFillColor(ACCENT)
        self.c.setFont(self._f("Manrope-Bd"), SIZE_HEAD)
        self.c.drawString(MARGIN, self.y - SIZE_HEAD, text)
        self.y -= SIZE_HEAD + 3.4

    def para(self, text, size=SIZE_BODY, color=INK, indent=0.0, bold_lead=None):
        """bold_lead is a leading clause set in semibold, the pattern the
        reference document uses to put the point of a paragraph first."""
        words = text.split()
        avail = CONTENT_W - indent
        x = MARGIN + indent
        line, lead_done = [], bold_lead is None

        def width(chunk, font, sz):
            return pdfmetrics.stringWidth(chunk, font, sz)

        regular = self._f("Manrope")
        semibold = self._f("Manrope-Sb")

        if bold_lead is not None:
            self.need(LEAD_BODY)
            self.c.setFillColor(INK)
            self.c.setFont(semibold, size)
            self.c.drawString(x, self.y - size, bold_lead)
            offset = width(bold_lead + " ", semibold, size)
        else:
            offset = 0.0

        first_line_avail = avail - offset
        cursor_avail = first_line_avail

        for word in words:
            trial = " ".join(line + [word])
            if width(trial, regular, size) <= cursor_avail or not line:
                line.append(word)
                continue

            self.need(LEAD_BODY)
            self.c.setFillColor(color)
            self.c.setFont(regular, size)
            self.c.drawString(x + (offset if cursor_avail == first_line_avail else 0.0),
                              self.y - size, " ".join(line))
            self.y -= LEAD_BODY
            line = [word]
            cursor_avail = avail
            offset = 0.0

        if line:
            self.need(LEAD_BODY)
            self.c.setFillColor(color)
            self.c.setFont(regular, size)
            self.c.drawString(x + (offset if cursor_avail == first_line_avail else 0.0),
                              self.y - size, " ".join(line))
            self.y -= LEAD_BODY

    def numbered(self, index, bold_lead, text):
        self.need(LEAD_BODY)
        self.c.setFillColor(INK)
        self.c.setFont(self._f("Manrope-Bd"), SIZE_BODY + 0.6)
        self.c.drawString(MARGIN, self.y - SIZE_BODY, f"{index}.")
        self.para(text, indent=14.0, bold_lead=bold_lead)

    def bullet(self, text, bold_lead=None):
        self.need(LEAD_BODY)
        self.c.setFillColor(ACCENT)
        self.c.setFont(self._f("Manrope-Bd"), SIZE_BODY)
        self.c.drawString(MARGIN + 2, self.y - SIZE_BODY, "•")
        self.para(text, indent=12.0, bold_lead=bold_lead)

    def mono(self, lines):
        self.need(LEAD_MONO * len(lines) + 4)
        self.space(2.0)
        self.c.setFillColor(INK)
        self.c.setFont("Courier", SIZE_MONO)
        for line in lines:
            self.c.drawString(MARGIN + 8, self.y - SIZE_MONO, line)
            self.y -= LEAD_MONO
        self.space(2.0)

    def table(self, headers, rows, widths, faded_rows=()):
        """Hairline table: a rule under the header and one under the body, the
        way the reference document does it. No grid, no fills."""
        self.need(LEAD_BODY * (len(rows) + 3))
        self.space(2.0)
        xs, x = [], MARGIN
        for w in widths:
            xs.append(x)
            x += w

        self.c.setFillColor(MUTED)
        self.c.setFont(self._f("Manrope"), SIZE_BODY)
        for i, head in enumerate(headers):
            if i == 0:
                self.c.drawString(xs[i], self.y - SIZE_BODY, head)
            else:
                self.c.drawRightString(xs[i] + widths[i], self.y - SIZE_BODY, head)
        self.y -= LEAD_BODY + 1.0

        self.c.setStrokeColor(RULE)
        self.c.setLineWidth(0.6)
        self.c.line(MARGIN, self.y + 0.6, MARGIN + sum(widths), self.y + 0.6)
        self.y -= 4.0

        for r, row in enumerate(rows):
            self.need(LEAD_BODY)
            self.c.setFillColor(FADED if r in faded_rows else INK)
            self.c.setFont(self._f("Manrope"), SIZE_BODY)
            for i, cell in enumerate(row):
                if i == 0:
                    self.c.drawString(xs[i], self.y - SIZE_BODY, cell)
                else:
                    self.c.drawRightString(xs[i] + widths[i], self.y - SIZE_BODY, cell)
            self.y -= LEAD_BODY

        self.c.setStrokeColor(RULE)
        self.c.line(MARGIN, self.y + 0.8, MARGIN + sum(widths), self.y + 0.8)
        self.y -= 6.0

    def footer(self):
        self.c.setFillColor(FADED)
        self.c.setFont(self._f("Manrope"), 6.6)
        self.c.drawRightString(PAGE_W - MARGIN, MARGIN - 14,
                               f"eternal.ag  SemSeg Challenge  ·  page {self.page}")


def load_case_stats(goose_root):
    """Per-class share of pixels and of points for the one frame the statement
    is built around. Returns None when the dataset is not on this machine, and
    the caller then omits the table rather than printing numbers nobody
    measured.
    """
    import numpy as np
    from PIL import Image

    mapping = os.path.join(goose_root, "challenge_label_mapping.csv")
    img_dir = os.path.join(goose_root, "raw_2d", "labels", "val", CASE_SEQUENCE)
    lidar = os.path.join(goose_root, "raw_3d", "lidar", "val", CASE_SEQUENCE,
                         CASE_FRAME + "_vls128.bin")
    label3d = os.path.join(goose_root, "raw_3d", "labels", "val", CASE_SEQUENCE,
                           CASE_FRAME + "_goose.label")
    label2d = os.path.join(img_dir, CASE_FRAME + "_labelids.png")

    for path in (mapping, lidar, label3d, label2d):
        if not os.path.exists(path):
            return None

    lut = np.full(256, 255, np.uint8)
    names = {}
    with open(mapping) as f:
        for row in csv.DictReader(f):
            # The upstream header carries a typo, challege_category_id. Accept
            # both spellings rather than editing the published file.
            cid = row.get("challege_category_id", row.get("challenge_category_id"))
            lut[int(row["label_key"])] = int(cid)
            names[int(cid)] = row["challenge_category_name"]

    labels_2d = lut[np.array(Image.open(label2d))]
    fine_3d = (np.fromfile(label3d, dtype=np.uint32) & 0xFFFF).astype(np.uint8)
    labels_3d = lut[fine_3d]
    points = np.fromfile(lidar, dtype=np.float32).reshape(-1, 4)

    azimuth = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
    in_wedge = float((np.abs(azimuth) <= ESTIMATE_HFOV_DEG / 2).mean())

    rows = []
    for cid in range(len(names)):
        share_2d = 100.0 * float((labels_2d == cid).mean())
        share_3d = 100.0 * float((labels_3d == cid).mean())
        rows.append((names[cid], share_2d, share_3d))

    return {
        "rows": rows,
        "n_points": int(len(points)),
        "image_shape": tuple(labels_2d.shape),
        "wedge_fraction": in_wedge,
    }


def load_baselines(results_dir):
    """Measured baseline scores, newest run per (method, split).

    Returns [] when nothing has been scored yet, and the table is then omitted.
    A problem statement that quotes an unmeasured baseline is worse than one
    that quotes none.

    Reads the scorer's own summary.json layout rather than a guess at it:
    metrics_2d / metrics_3d each carry miou, and consistency carries the ratio
    beside its coverage because neither is readable alone.
    """
    found = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*", "summary.json"))):
        with open(path) as f:
            summary = json.load(f)

        key = (summary.get("method"), summary.get("split"))
        if key[0] is None:
            continue

        found[key] = summary

    rows = []
    for (method, split), summary in sorted(found.items()):
        compute = summary.get("compute") or {}
        consistency = summary.get("consistency") or {}
        rows.append({
            "method": method,
            "split": split,
            "miou_2d": (summary.get("metrics_2d") or {}).get("miou"),
            "miou_3d": (summary.get("metrics_3d") or {}).get("miou"),
            "consistency": consistency.get("consistency"),
            "coverage": consistency.get("coverage"),
            "sec_per_frame": compute.get("runtime_sec_per_frame"),
        })

    return rows


def fmt(value, digits=3, dash="-"):
    if value is None:
        return dash
    if isinstance(value, float) and value != value:
        return dash
    return f"{value:.{digits}f}"


def build(out_path, results_dir, goose_root):
    fonts_ok = register_fonts()
    case = load_case_stats(goose_root)
    baselines = load_baselines(results_dir)

    c = canvas.Canvas(out_path, pagesize=A4)
    c.setTitle("eternal.ag SemSeg Challenge - Track A Problem Statement")
    c.setAuthor("eternal.ag")
    c.setSubject("Image and lidar semantic segmentation challenge")

    f = Flow(c, fonts_ok)

    f.title("eternal.ag SemSeg Challenge")
    f.para(
        "Track A. Label every pixel of one camera image and every point of one lidar sweep of "
        "the same scene, and make the two labellings agree. Fork the repo, improve a baseline "
        "or write your own method.",
        size=SIZE_LEAD, color=MUTED,
    )

    f.heading("What we are asking for, in order")
    f.numbered(1, "Get the highest score.",
               "Both modalities count, and so does whether they agree with each other.")
    f.numbered(2, "Keep compute down.",
               "Latency and memory are graded, not just a tie-break. A model that scores three "
               "extra points of mIoU and costs 200 ms is a worse result than one that scores "
               "one extra point at 30 ms.")
    f.numbered(3, "Explain your method.",
               "What you tried, why, what did not work, what you would do next.")

    f.heading("The problem")
    f.para(
        "A robot that only knows where things are cannot decide what to do about them. A "
        "costmap treats a hanging leaf, a suspended irrigation pipe, a person and a concrete "
        "pillar as the same occupied cell. Semantics is what separates driving through it from "
        "stopping.")
    f.para(
        "The camera and the lidar fail in opposite ways. The camera has colour and texture and "
        "no range. The lidar has range and no appearance, and its angular resolution falls off "
        "as one over distance. Fusing them should beat either one alone. Your job is to show "
        "that it does, and to say when it stops being true.")

    f.heading("One typical frame")
    if case is None:
        f.para(
            "The statement is written around a single frame of the GOOSE off-road validation "
            "split, human-labelled in both modalities. Download the split and re-run "
            "tools/make_problem_statement.py to fill in its measured class balance here.")
    else:
        h, w = case["image_shape"]
        f.para(
            f"Everything in this section is measured from {CASE_FRAME.split('__')[1]} of "
            f"{CASE_SEQUENCE}, one frame of the GOOSE off-road validation split: a "
            f"{w} by {h} windshield image and a {case['n_points']:,} point 128-beam sweep of "
            "the same instant, each labelled by hand.")

        rows = [(name, f"{p2:.1f}%", f"{p3:.1f}%") for name, p2, p3 in case["rows"]]
        f.table(["class", "share of pixels", "share of points"], rows,
                [188.0, 120.0, 120.0])

        f.para("Read that table before you design anything. Three things in it set the task.")
        f.bullet(
            "is 29 percent of the image and zero percent of the sweep. The two label spaces "
            "are not the same space, and whatever you do about that is a decision you have to "
            "state, not a detail.",
            bold_lead="Sky")
        f.bullet(
            "is 70 percent of the points and 14 percent of the pixels. The two sensors do not "
            "even agree on what the scene is mostly made of, so a model that averages them "
            "will inherit the disagreement.",
            bold_lead="Vegetation")
        f.bullet(
            "does not appear in this frame at all. Its IoU here is undefined, not zero. Score "
            "an absent class as zero and your headline number stops meaning anything.",
            bold_lead="Human")
        f.para(
            f"One more thing the table cannot show. The sweep covers 360 degrees and the camera "
            f"is a forward frustum. Taking the azimuths of these points, a "
            f"{ESTIMATE_HFOV_DEG:.0f} degree horizontal field of view puts at most "
            f"{100 * case['wedge_fraction']:.0f} percent of them in front of the camera at all. "
            "Most of your points have no image evidence. What your model does with those is "
            "most of the problem.")

    f.heading("Why this matters to us")
    f.para(
        "Our robots work in greenhouses. A gutter, a support post, a hanging vine and a person "
        "crouched between two rows all read as occupied to a lidar, and the right response to "
        "each is different. Geometry alone cannot tell them apart, and the crop changes shape "
        "every week, so a rule written in March is wrong by June. We need a perception stack "
        "that says what a thing is, on the robot, inside the navigation budget.")

    f.heading("What you get")
    f.para(
        "A time-synchronised camera and 3D lidar observing the same scene, camera intrinsics, "
        "human labels in both modalities over a shared 9-class set, and a synthetic scene "
        "generator whose calibration is exact by construction. Four baselines ship with the "
        "repo: a chance floor, a lidar-only arm, a camera-only arm, and a naive fusion arm that "
        "paints each point with its pixel colour. All four share one classifier and differ only "
        "in which features they get, so comparing them measures fusion rather than "
        "architecture.")
    f.para(
        "the public GOOSE release ships no camera-to-lidar extrinsic and no poses with its "
        "annotated split. That was checked, not assumed: neither zip holds a calibration file, "
        "and the published transform tree carries frame topology with no numbers in it. So "
        "anything that depends on projecting one sensor into the other is measured on the "
        "synthetic scene, and the repo says which source every number came from. Supply an "
        "extrinsic with --calib and the real data opens up.",
        bold_lead="One honest limitation:")

    f.heading("What you submit")
    f.para(
        "Predictions are label maps and point arrays, far too large to commit, so what you "
        "submit is the accumulated statistics they imply. Your method writes its labels to a "
        "run directory and emits one small JSON:")
    f.mono([
        "{ \"submission_version\": 1, \"method\": \"...\", \"label_space\": \"goose9\",",
        "  \"conf_2d\": [[..9x9..]], \"conf_2d_boundary\": [[..]],",
        "  \"conf_3d\": [[..]], \"conf_3d_by_range\": { \"0-5m\": [[..]], ... },",
        "  \"conf_3d_in_frustum\": [[..]], \"conf_3d_out_frustum\": [[..]],",
        "  \"ece_2d\": {...}, \"ece_3d\": {...},",
        "  \"consistency\": { \"matched\": N, \"scorable\": N,",
        "                    \"in_frustum\": N, \"total_points\": N } }",
    ])
    f.para(
        "Every IoU we report is recoverable from those matrices, so the scored artefact stays "
        "in the tens of kilobytes and continuous integration can grade it. The range-binned and "
        "frustum-split matrices are not optional: a single global number hides where a model "
        "actually fails.")

    f.heading("How it is scored")
    f.para(
        "Mean IoU in each modality, with every per-class IoU reported in full, never the mean "
        "alone. Classes absent from the split are excluded from the mean rather than counted as "
        "zero. Alongside those:")
    f.mono([
        "boundary IoU (2D)       IoU within 3 px of a ground-truth class edge",
        "range-stratified IoU    3D, binned 0-5, 5-15, 15-30, over 30 m",
        "cross-modal consistency fraction of visible points whose 3D label",
        "                        matches the 2D label of their own pixel,",
        "                        always reported with its coverage",
        "calibration error (ECE) a head feeding a costmap must be trustworthy",
    ])
    f.para(
        "Consistency needs no ground truth, which means you can compute it on unlabelled robot "
        "logs. That is the point of having it.")

    f.heading("The number that decides the design")
    f.para(
        "Fusion works through the projection from lidar frame to image, so it is only as good "
        "as the calibration and the time sync. Perturb the extrinsic by a fraction of a degree "
        "and image features attach to the wrong points; the network learns to distrust the "
        "camera and you have paid for a lidar-only model with extra latency.")
    f.para(
        "So the harness sweeps it. Rotation from 0.1 to 2 degrees, per axis, because the axes are "
        "not interchangeable: measured on the fixture, yaw crosses over at 0.86 degrees, pitch "
        "at 1.19, and roll not within the 2 degrees swept. Translation from 1 to 10 cm. Time offset from 0 "
        "to 200 ms at three speeds. The headline output is the crossover: the perturbation at "
        "which the fused arm falls below the lidar-only arm. That single number is the "
        "calibration accuracy the robot has to hold in production. Run the sweep before you "
        "design an architecture, not after.")

    if baselines:
        f.heading("Baselines to beat")
        rows = [(b["method"], b["split"], fmt(b["miou_2d"]), fmt(b["miou_3d"]),
                 fmt(b["consistency"]), fmt(b["sec_per_frame"], 2)) for b in baselines]
        faded = {i for i, b in enumerate(baselines) if b["method"].startswith("bl_prior")}
        f.table(["baseline", "split", "mIoU 2D", "mIoU 3D", "consistency", "sec/frame"], rows,
                [126.0, 62.0, 74.0, 74.0, 84.0, 82.0], faded_rows=faded)
        f.para(
            "A consistency of 1.000 is not an achievement here. Each shipped arm has a "
            "single head and derives the other modality from it by projection, so it "
            "cannot disagree with itself. bl_prior, which predicts the two "
            "independently, is the only informative row. On the goose split the "
            "projection-dependent columns read as dashes because the public release "
            "ships no extrinsic and nothing here invents one.")
        f.para(
            f"For scale, a PTv3 model trained on this data reports {PTV3_REFERENCE_MIOU:.4f} 3D "
            "mIoU on the full validation split. That figure is published by the dataset "
            "authors, not measured here, and it is 3D only.")

    f.heading("Rules")
    f.para(
        "Everything runs inside the provided image. No internet while scoring. Report per-class "
        "numbers, not just means. Pretrained weights are fine if you name them and they run "
        "offline. Fit only on the fit split; the score split is assigned by a stable hash and "
        "moving it counts as tuning on the test set. If you claim a fusion gain, show the "
        "single-modality arms it beats and the perturbation at which the gain disappears.")

    f.heading("Grading and effort")
    f.para(
        "40 percent score. 30 percent write-up of two pages or fewer, covering what you tried "
        "and why. 20 percent code quality. 10 percent experimental hygiene, meaning the "
        "ablations are present and the defaults are sane. Plan on two to four days. A strong "
        "entry clearly beats one baseline and explains one thing we did not already know.")

    f.footer()
    c.save()
    return out_path


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="semseg-problem-statement.pdf")
    parser.add_argument("--results", default="results",
                        help="scoring output dir; the baseline table is omitted if empty")
    parser.add_argument("--goose-root", default="/home/pan-navigator/datasets/goose",
                        help="GOOSE root; the typical-frame table is omitted if absent")
    args = parser.parse_args(argv)

    path = build(args.out, args.results, args.goose_root)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
