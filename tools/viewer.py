#!/usr/bin/env python3
"""Frame viewer, the 5-minute path's first stop: four panels whose only job is
to make a calibration error visible.

Kept as a tool and not a test because the argument it makes is a visual one.
Problem statement section 2 says the consistency constraint is the whole
problem and that everything hard about the task lives in the projection
operator `pi(p) = K . T_cam_lidar . p`. A number in a CSV does not communicate
that. Two figures side by side, one at the nominal extrinsic and one two
degrees off in pitch, do: the same cloud, the same image, the same code, and
a band of red along every class boundary in the second one.

    python tools/viewer.py --dataset fixture --root <path> --frame 000000 \\
        [--decalib-pitch-deg 2.0] [--out figure.png]

Panels:
    1. the RGB image
    2. the ground truth 2D label map, colourised with semseg.labels.CLASS_COLORS
    3. the lidar points projected onto that image and coloured by their OWN 3D
       ground truth class. Under a correct extrinsic each point lands on a
       pixel of its own class, so panel 3 reads as a faithful stencil of panel
       2. Under --decalib-pitch-deg it visibly does not.
    4. the agreement mask: green where the projected point's 3D class matches
       the pixel's 2D class, red where it does not, with the agreement
       fraction and its coverage in the title.

Panel 4 is the decalibration sweep of problem statement section 6.3 reduced to
one frame, and it is measured by the same `eval.metrics.consistency` the scorer
uses, on ground truth in both modalities rather than on a prediction. Ground
truth against ground truth means every red point is the projection operator
being wrong and nothing else: no model, no fit, no baseline to blame.

Both projection dependent panels need `T_cam_lidar`, and real GOOSE val ships
none at all (interface spec 13.3). So `--dataset goose` works when `--calib`
supplies one and otherwise exits with the adapter's own error naming the three
places it looked. It does not fall back to a plausible default, here or
anywhere else in this repo: a viewer that drew a fabricated extrinsic would be
drawing a fiction, and this is the one file whose whole purpose is to show what
a wrong extrinsic looks like.
"""

import argparse
import sys
from dataclasses import dataclass

import numpy as np

from baselines.common.runner import build_dataset
from eval.metrics import consistency, coverage
from semseg.labels import CLASS_COLORS, CLASS_NAMES
from semseg.projection import perturb_extrinsic, project, zbuffer
from semseg.types import NUM_CLASSES, UNLABELED

# Pitch is the axis the CLI exposes because it is the one that hurts. A
# rotation about the optical axis (roll) barely moves an on-axis point, while
# pitch is a vertical image shift whose lateral world error grows linearly with
# range: problem statement section 3 layer 3 item 1 puts 0.5 degrees at 10 m at
# about 9 cm. Translation is left at zero so one flag means one thing.
DECALIB_AXIS = "pitch"
DECALIB_TRANS_M = 0.0
DEFAULT_DECALIB_PITCH_DEG = 0.0

# Matches the fixture's own default seed, so `--dataset fixture` with no --root
# generates the same world the rest of the harness scored.
DEFAULT_SEED = 0

# Figure geometry. Four panels of one image, so a wide canvas: real GOOSE
# frames are 2048x1000 and the fixture is 320x240, and both have to stay
# readable without a per-dataset special case.
FIGURE_SIZE_IN = (13.0, 8.0)
FIGURE_DPI = 110
PANEL_ROWS = 2
PANEL_COLS = 2

# The projected-point panels draw on a desaturated, brightened copy of the
# image. At full contrast a nine-colour scatter reads as noise over the photo
# it is drawn on, and the point of panels 3 and 4 is the points.
BACKDROP_FADE = 0.62

# Marker areas in points squared. Mismatches are drawn larger and last, on
# purpose: 17 percent red among 83 percent green is a band along every class
# boundary when the red is on top and speckle when it is underneath, and
# "obvious at a glance" is the requirement panel 4 has to meet.
POINT_MARKER_PT2 = 5.0
AGREE_MARKER_PT2 = 5.0
DISAGREE_MARKER_SCALE = 2.4
DISAGREE_ZORDER = 3

AGREE_COLOR = "#2e7d32"
DISAGREE_COLOR = "#d32f2f"

TITLE_FONT_PT = 10
LEGEND_FONT_PT = 7

# One strip under the whole figure rather than a box inside panel 2. The class
# colours decode panels 2 and 3 both, and an in-panel legend sits on top of the
# image corner where this scene keeps a human and the wall.
LEGEND_COLUMNS = NUM_CLASSES

# How many frame ids an unresolvable --frame prints back. A GOOSE split holds
# 961 of them and a full dump is not a diagnostic.
MAX_REPORTED_FRAMES = 8

EXIT_NO_CALIBRATION = 2


@dataclass
class Agreement:
    """Cross-modal agreement of the two ground truths on one frame, plus the
    arrays panels 3 and 4 draw.

    A value object rather than a bare tuple because the four counts are only
    interpretable together. `fraction` alone is unreadable: interface spec
    section 5 requires consistency to be reported next to its coverage,
    because assumption A2 of the problem statement says most lidar points have
    no pixel at all, and 0.98 over 29 percent of the cloud and 0.98 over all
    of it are different results.

    The plot arrays ride along so the figure costs one projection pass instead
    of three.
    """
    matched: int
    scorable: int
    in_frustum: int
    total_points: int
    fraction: float
    coverage: float
    point_uv: np.ndarray        # (V, 2) int32 pixels of the visible points
    point_class: np.ndarray     # (V,) uint8 their own 3D ground truth class
    scorable_uv: np.ndarray     # (S, 2) int32 pixels of the scorable points
    scorable_match: np.ndarray  # (S,) bool, True where the two classes agree


def agreement(frame, T_cam_lidar) -> Agreement:
    """-> Agreement for `frame` projected through `T_cam_lidar`.

    Deliberately the same chain as `eval.io_formats.Accumulator`: project,
    z-buffer, then require the point to own its pixel. The z-buffer is not
    decoration. A point behind the wall still projects onto the wall's pixel,
    so without it panel 4 would paint occluded points red for a disagreement
    that is an artefact of the projection rather than a calibration error, and
    the panel would look equally bad at every perturbation.

    Both ground truths are required. This tool compares ground truth against
    ground truth, so a frame missing either one has nothing to show and says
    so instead of drawing an empty panel.
    """
    if frame.labels_2d_gt is None or frame.labels_3d_gt is None:
        raise ValueError(
            f"frame {frame.frame_id}: the viewer compares the 2D and 3D ground truths "
            f"against each other and this frame is missing one of them"
        )

    height, width = frame.image.shape[:2]
    uv, depth, in_frustum = project(frame.points, frame.K, T_cam_lidar, width, height)
    _owner, visible = zbuffer(uv, depth, in_frustum, width, height)

    # absolute point indices throughout, so nothing downstream has to remember
    # whether a mask is indexed into the cloud or into the in-frustum subset
    inside = np.flatnonzero(in_frustum)
    label_3d = frame.labels_3d_gt[inside]
    label_2d = frame.labels_2d_gt[uv[inside, 1], uv[inside, 0]]

    scorable = visible[inside] & (label_3d != UNLABELED) & (label_2d != UNLABELED)
    picked = inside[scorable]
    match = label_3d[scorable] == label_2d[scorable]

    n_matched = int(match.sum())
    n_scorable = int(scorable.sum())
    n_total = int(frame.points.shape[0])

    return Agreement(
        matched=n_matched,
        scorable=n_scorable,
        in_frustum=int(in_frustum.sum()),
        total_points=n_total,
        fraction=consistency(n_matched, n_scorable),
        coverage=coverage(n_scorable, n_total),
        point_uv=uv[visible],
        point_class=frame.labels_3d_gt[visible],
        scorable_uv=uv[picked],
        scorable_match=match,
    )


def decalibrate(T_cam_lidar, pitch_deg: float) -> np.ndarray:
    """-> a copy of `T_cam_lidar` rotated about the camera x axis by
    `pitch_deg`, or the matrix unchanged at zero.

    Routed through `semseg.projection.perturb_extrinsic` rather than composing
    a rotation here, so the figure and eval/sweep.py perturb the extrinsic by
    the same code in the same frame. A second implementation would eventually
    disagree by a sign, and a sign error in this file would look like a
    finding.
    """
    if pitch_deg == 0.0:
        return np.asarray(T_cam_lidar, dtype=np.float64)

    return perturb_extrinsic(T_cam_lidar, DECALIB_AXIS, pitch_deg, DECALIB_TRANS_M)


def colorise(labels) -> np.ndarray:
    """goose9 label map -> (H, W, 3) uint8 RGB using the devkit colours.

    UNLABELED, and any other id outside the ontology, comes out black. Same
    choice as `semseg.labels.save_label_png`: an unlabelled region has to look
    absent, and painting it with class 0 `other` would make it look scored.
    """
    labels = np.asarray(labels)
    rgb = np.zeros(labels.shape + (3,), dtype=np.uint8)

    known = labels < NUM_CLASSES
    rgb[known] = CLASS_COLORS[labels[known]]

    return rgb


def backdrop(image) -> np.ndarray:
    """-> a desaturated, brightened copy of `image` for the scatter panels."""
    gray = np.asarray(image, dtype=np.float64).mean(axis=2, keepdims=True)
    faded = gray + (255.0 - gray) * BACKDROP_FADE

    return np.repeat(faded, 3, axis=2).astype(np.uint8)


def resolve_frame_id(frame_ids, requested: str) -> str:
    """-> the one frame id `requested` names.

    Frame ids are not interchangeable between datasets: the fixture numbers
    them `fixture_0000` and GOOSE carries the sequence and the sensor
    timestamp, `<sequence>__<index>_<stamp>`. Nobody types the second kind, so
    an exact id, a unique substring of one, and a plain index all resolve.

    A unique substring beats the index reading, because on GOOSE a short digit
    string is far more likely to be part of a frame id than an ordinal. An
    ambiguous non-numeric substring raises rather than taking the first match.
    Whatever resolves, main() prints the id it settled on, so no run is left
    guessing which frame it just looked at.
    """
    frame_ids = list(frame_ids)
    if requested in frame_ids:
        return requested

    matches = [frame_id for frame_id in frame_ids if requested in frame_id]
    if len(matches) == 1:
        return matches[0]

    if requested.isdigit() and int(requested) < len(frame_ids):
        return frame_ids[int(requested)]

    if matches:
        raise ValueError(
            f"{requested!r} matches {len(matches)} frames, e.g. "
            f"{matches[:MAX_REPORTED_FRAMES]}. Pass a full frame id."
        )

    raise ValueError(
        f"no frame matching {requested!r} in {len(frame_ids)} frames, which start "
        f"{frame_ids[:MAX_REPORTED_FRAMES]}"
    )


def select_backend(out_path) -> None:
    """Pick Agg when the figure is being saved rather than shown.

    Called before the first pyplot import, which is why every pyplot import in
    this file sits inside a function. A saved figure has to work with no
    display at all: run_all.sh renders one inside the runtime container, which
    has no X socket, and an interactive backend there fails at import.
    """
    import matplotlib

    if out_path:
        matplotlib.use("Agg")


def _extrinsic_label(pitch_deg: float) -> str:
    if pitch_deg == 0.0:
        return "nominal extrinsic"

    return f"pitch decalibrated {pitch_deg:+.2f} deg"


def _agreement_title(active: Agreement, nominal: Agreement) -> str:
    """The panel 4 title, which is the one piece of text in the figure that has
    to stand on its own. The colour convention is stated rather than assumed,
    and the delta against the nominal extrinsic is spelled out because it is
    the whole argument: reading it off two figures opened at different times is
    exactly how the argument gets lost."""
    text = ("4. green where the 3D class matches the pixel, red where it does not"
            f"\nagreement {active.fraction:.3f}   coverage {active.coverage:.3f} "
            f"({active.scorable} of {active.total_points} points)")

    if nominal is None:
        return text

    # its own line, not appended: the title is left aligned to the panel, so a
    # longer line runs off the right edge of the figure and is clipped
    return text + (f"\nnominal {nominal.fraction:.3f}, "
                   f"delta {active.fraction - nominal.fraction:+.3f}")


def _legend_handles(frame):
    """One patch per goose9 class actually present in either ground truth. A
    full nine-class legend on a frame holding four of them wastes the reader's
    attention on classes that are not on screen, and on a real GOOSE frame
    `human` is often one of the absent ones (interface spec 13.5)."""
    from matplotlib.patches import Patch

    present = np.union1d(np.unique(frame.labels_2d_gt), np.unique(frame.labels_3d_gt))
    present = present[present < NUM_CLASSES]

    return [Patch(facecolor=CLASS_COLORS[c] / 255.0, edgecolor="none", label=CLASS_NAMES[c])
            for c in present]


def _bare_axes(ax, image, title: str) -> None:
    """Every panel is an image with a title and no ticks: pixel indices carry
    no information here and four sets of them crowd out the titles."""
    height, width = image.shape[:2]

    ax.imshow(image)
    ax.set_title(title, fontsize=TITLE_FONT_PT, loc="left")
    ax.set_xticks([])
    ax.set_yticks([])

    # after imshow and before the scatter, then re-applied by the caller: a
    # scatter marker centred on an edge pixel would otherwise autoscale the
    # axes outwards and shift the image away from its panel
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)


def render(frame, active: Agreement, nominal: Agreement = None,
           decalib_pitch_deg: float = 0.0):
    """-> a 2x2 matplotlib Figure over one frame.

    `active` is the agreement at the extrinsic being drawn and `nominal` is the
    one at the unperturbed extrinsic, or None when they are the same thing.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(PANEL_ROWS, PANEL_COLS, figsize=FIGURE_SIZE_IN,
                             constrained_layout=True)
    fig.suptitle(f"{frame.frame_id}   {_extrinsic_label(decalib_pitch_deg)}",
                 fontsize=TITLE_FONT_PT + 2)

    faded = backdrop(frame.image)

    _bare_axes(axes[0][0], frame.image, "1. RGB image")

    _bare_axes(axes[0][1], colorise(frame.labels_2d_gt), "2. ground truth 2D labels, goose9")

    _bare_axes(axes[1][0], faded, "3. lidar points, coloured by their own 3D class")
    axes[1][0].scatter(active.point_uv[:, 0], active.point_uv[:, 1],
                       s=POINT_MARKER_PT2,
                       c=CLASS_COLORS[active.point_class] / 255.0,
                       marker=".", linewidths=0.0)

    _bare_axes(axes[1][1], faded, _agreement_title(active, nominal))
    agree = active.scorable_match
    axes[1][1].scatter(active.scorable_uv[agree, 0], active.scorable_uv[agree, 1],
                       s=AGREE_MARKER_PT2, c=AGREE_COLOR, marker=".", linewidths=0.0)
    axes[1][1].scatter(active.scorable_uv[~agree, 0], active.scorable_uv[~agree, 1],
                       s=AGREE_MARKER_PT2 * DISAGREE_MARKER_SCALE, c=DISAGREE_COLOR,
                       marker=".", linewidths=0.0, zorder=DISAGREE_ZORDER)

    for ax in axes.ravel():
        ax.set_xlim(-0.5, frame.image.shape[1] - 0.5)
        ax.set_ylim(frame.image.shape[0] - 0.5, -0.5)

    fig.legend(handles=_legend_handles(frame), loc="outside lower center",
               ncol=LEGEND_COLUMNS, fontsize=LEGEND_FONT_PT, frameon=False)

    return fig


def show_or_save(fig, out_path=None) -> None:
    """Save to `out_path`, or open a window when there is none."""
    import matplotlib.pyplot as plt

    if not out_path:
        plt.show()
        return

    fig.savefig(out_path, dpi=FIGURE_DPI)
    plt.close(fig)
    print(f"wrote {out_path}")


def _extrinsic_or_none(dataset, frame_id: str):
    """-> T_cam_lidar, or None after printing why there is not one.

    GooseDataset.extrinsic raises CalibrationUnavailable, a RuntimeError,
    because GOOSE's val zips carry no calibration anywhere and the numbers live
    in the GOOSE-DB bags' /tf_static (interface spec 13.3). The exception is
    caught by its base class so this module never imports the GOOSE adapter,
    which would drag its dependencies into a fixture-only run.

    Recovering is not on the table. The adapter's message already names every
    path it searched and the --calib flag that fixes it, so the honest thing is
    to print it and stop.
    """
    try:
        return dataset.extrinsic(frame_id)
    except RuntimeError as error:
        print(f"cannot draw panels 3 and 4: {error}", file=sys.stderr)
        return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True, choices=("fixture", "goose"),
                        help="goose reads the real val split; fixture generates a known world")
    parser.add_argument("--root", help="dataset root; the fixture generates one when omitted")
    parser.add_argument("--frame", required=True,
                        help="a frame id, a unique substring of one, or an index")
    parser.add_argument("--decalib-pitch-deg", type=float, default=DEFAULT_DECALIB_PITCH_DEG,
                        help="rotate the extrinsic about the camera x axis before projecting, "
                             "which is what a real calibration residual looks like")
    parser.add_argument("--out", help="save the figure here instead of opening a window")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--calib", help="4x4 T_cam_lidar as .npy or .json. REQUIRED on goose: "
                                        "the val zips ship no calibration at all")
    args = parser.parse_args(argv)

    select_backend(args.out)

    # the same resolver every bl_*/run.py uses, so --dataset, --root, --seed
    # and --calib mean exactly here what they mean in a scoring run
    dataset = build_dataset(args)

    frame_id = resolve_frame_id(dataset.frame_ids(), args.frame)

    # the extrinsic first, then the frame: a GOOSE frame is a 100k point cloud
    # and a 2048x1000 image, and there is no reason to read either one when
    # the projection the whole figure is about cannot be done at all
    T_nominal = _extrinsic_or_none(dataset, frame_id)
    if T_nominal is None:
        return EXIT_NO_CALIBRATION

    frame = dataset.load(frame_id)
    nominal = agreement(frame, T_nominal)
    decalib_deg = args.decalib_pitch_deg

    if decalib_deg == 0.0:
        active, reference = nominal, None
    else:
        active, reference = agreement(frame, decalibrate(T_nominal, decalib_deg)), nominal

    print(f"{frame_id}: {_extrinsic_label(decalib_deg)}")
    print(f"agreement {active.fraction:.4f} over {active.scorable} scorable points, "
          f"coverage {active.coverage:.4f} of {active.total_points}")
    if reference is not None:
        print(f"nominal agreement {reference.fraction:.4f}, "
              f"delta {active.fraction - reference.fraction:+.4f}")

    show_or_save(render(frame, active, reference, decalib_deg), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
