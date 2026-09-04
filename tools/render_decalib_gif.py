#!/usr/bin/env python3
"""Renders the repo's one animated figure: cross-modal agreement collapsing as
the extrinsic is rotated away from truth.

Why an animation and not another table. The claim this repo is built to test is
that a fusion gain rests on the projection operator being nearly exact, and that
the gain disappears well before the miscalibration becomes obvious to a person
looking at the data. A static pair of figures shows two points on that curve. An
animation shows the whole thing, including the part that matters most: how
little rotation it takes before a visible fraction of the cloud is landing on
the wrong class.

Both ground truths, no model. Every red point is the projection operator being
wrong and nothing else, so there is no fit, no baseline and no seed to blame.
That is the same choice tools/viewer.py makes, and this file reuses its
`agreement` and `decalibrate` so the two cannot drift apart.

    python tools/render_decalib_gif.py --out docs/images/decalib.gif \\
        [--dataset fixture] [--root <path>] [--frame fixture_0000] \\
        [--max-deg 2.0] [--steps 24] [--calib <path>]

Reproduces from a clean checkout with no dataset: with no --root it generates a
fixture frame in memory, the same property that makes run_all.sh useful.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")   # set before pyplot, this file never opens a window

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semseg.labels import CLASS_COLORS, CLASS_NAMES
from semseg.types import NUM_CLASSES
from tools.viewer import agreement, backdrop, decalibrate

# The sweep runs out to MAX_DEG and back so the loop reads as one motion rather
# than snapping from worst to best at the wrap.
DEFAULT_MAX_DEG = 2.0
DEFAULT_STEPS = 14
DEFAULT_FPS = 12
HOLD_FRAMES = 4         # pause at each end, so a reader can register both states

FIGURE_SIZE_IN = (12.0, 5.4)
FIGURE_DPI = 66
POINT_SIZE = 1.1
DISAGREE_SIZE = 3.2
BACKDROP_ALPHA = 1.0

AGREE_COLOR = "#2e7d32"
DISAGREE_COLOR = "#d32f2f"
INK = "#0e1512"
ACCENT = "#1f6b45"
MUTED = "#6b7671"

BAR_HEIGHT = 0.026
BAR_BOTTOM = 0.055
TITLE_PT = 11
READOUT_PT = 13
CAPTION_PT = 8.5


def sweep_angles(max_deg: float, steps: int):
    """-> the pitch angles of every animation frame, out and back with a hold
    at each end. Returned as a list so len() is the frame count FuncAnimation
    needs."""
    out = np.linspace(0.0, max_deg, steps)
    back = out[::-1][1:-1]

    return ([0.0] * HOLD_FRAMES + list(out) +
            [max_deg] * HOLD_FRAMES + list(back))


def build_dataset(args):
    """-> (dataset, frame, T_cam_lidar). Kept here rather than shared with
    viewer.py's resolver because this tool takes one frame and needs no
    substring matching."""
    if args.dataset == "fixture":
        from semseg.datasets.fixture import FixtureDataset
        dataset = (FixtureDataset(root=args.root) if args.root
                   else FixtureDataset(n_frames=1, seed=args.seed))
    else:
        from semseg.datasets.goose import GooseDataset
        dataset = GooseDataset(args.root, calib=args.calib)

    frame_ids = dataset.frame_ids()
    frame_id = args.frame if args.frame in frame_ids else frame_ids[0]

    return dataset, dataset.load(frame_id), dataset.extrinsic(frame_id)


def draw_static(ax_left, ax_right, image):
    """Paints the two backdrops once. The scatters are the only thing that
    changes per animation frame, so redrawing the image every frame would cost
    the whole render time for no visible difference."""
    faded = backdrop(image)
    for ax in (ax_left, ax_right):
        ax.imshow(faded, alpha=BACKDROP_ALPHA)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)


def class_legend(fig, frame):
    """A single row of class swatches for the classes actually in this frame.
    Legending all nine when the frame holds six would invite a reader to look
    for the missing three."""
    present = sorted(set(np.unique(frame.labels_3d_gt).tolist()) & set(range(NUM_CLASSES)))
    handles = [plt.Line2D([], [], marker="o", linestyle="none", markersize=5,
                          color=CLASS_COLORS[c] / 255.0, label=CLASS_NAMES[c])
               for c in present]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, fontsize=CAPTION_PT, bbox_to_anchor=(0.5, 0.0),
               handletextpad=0.35, columnspacing=1.1, labelcolor=INK)


def render(frame, T_nominal, angles, out_path, fps):
    fig = plt.figure(figsize=FIGURE_SIZE_IN, dpi=FIGURE_DPI)
    fig.patch.set_facecolor("white")
    ax_left = fig.add_axes([0.015, 0.15, 0.475, 0.72])
    ax_right = fig.add_axes([0.510, 0.15, 0.475, 0.72])
    draw_static(ax_left, ax_right, frame.image)

    nominal = agreement(frame, T_nominal)
    class_legend(fig, frame)

    left_scatter = ax_left.scatter([], [], s=POINT_SIZE, marker=".", linewidths=0)
    agree_scatter = ax_right.scatter([], [], s=POINT_SIZE, marker=".",
                                     color=AGREE_COLOR, linewidths=0)
    # drawn second and larger: the disagreements are the subject of the panel
    # and at one pixel a point they are invisible against the agreeing mass
    bad_scatter = ax_right.scatter([], [], s=DISAGREE_SIZE, marker=".",
                                   color=DISAGREE_COLOR, linewidths=0, zorder=3)

    ax_left.set_title("lidar points, coloured by their own 3D ground truth class",
                      fontsize=TITLE_PT, color=INK, pad=6)
    right_title = ax_right.set_title("", fontsize=TITLE_PT, color=INK, pad=6)

    readout = fig.text(0.015, 0.945, "", fontsize=READOUT_PT, color=INK,
                       ha="left", va="center", fontweight="bold")
    caption = fig.text(0.985, 0.945, "", fontsize=CAPTION_PT, color=MUTED,
                       ha="right", va="center")

    bar_bg = fig.add_axes([0.015, BAR_BOTTOM, 0.97, BAR_HEIGHT])
    bar_bg.set_xlim(0, max(angles)); bar_bg.set_ylim(0, 1)
    bar_bg.set_yticks([]); bar_bg.set_facecolor("#f3f6f4")
    for spine in bar_bg.spines.values():
        spine.set_visible(False)
    bar_bg.tick_params(labelsize=CAPTION_PT - 0.5, colors=MUTED, length=2)
    bar_bg.set_xlabel("")
    bar_fill = bar_bg.barh([0.5], [0.0], height=1.0, color=ACCENT)[0]

    max_deg = max(angles)

    def update(i):
        deg = angles[i]
        state = agreement(frame, decalibrate(T_nominal, deg))

        left_scatter.set_offsets(state.point_uv)
        left_scatter.set_color(CLASS_COLORS[state.point_class] / 255.0)

        good = state.scorable_uv[state.scorable_match]
        bad = state.scorable_uv[~state.scorable_match]
        agree_scatter.set_offsets(good if len(good) else np.empty((0, 2)))
        bad_scatter.set_offsets(bad if len(bad) else np.empty((0, 2)))

        lost = nominal.fraction - state.fraction
        right_title.set_text(
            f"agreement {state.fraction:.3f} over {state.coverage:.0%} of the cloud"
            + (f"   ({lost:+.3f} against nominal)" if deg else "   (nominal)")
        )
        readout.set_text(f"pitch decalibration {deg:.2f}°")
        caption.set_text(
            f"{len(bad):,} of {state.scorable:,} scorable points now land on the wrong class"
        )
        bar_fill.set_width(deg)
        bar_fill.set_color(ACCENT if deg < max_deg / 2 else DISAGREE_COLOR)

        return left_scatter, agree_scatter, bad_scatter, bar_fill

    anim = FuncAnimation(fig, update, frames=len(angles), blit=False)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    # Palette-quantise on save. A 24-bit GIF of a scatter plot is mostly
    # near-duplicate anti-aliasing shades, so reducing to a fixed palette costs
    # nothing visible and roughly thirds the file. README media that nobody
    # waits for is README media nobody sees.
    anim.save(out_path, writer=PillowWriter(fps=fps),
              savefig_kwargs={"facecolor": "white"})
    _quantise_gif(out_path)
    plt.close(fig)

    return nominal


GIF_PALETTE_COLORS = 96


def _quantise_gif(path):
    """Rewrite the GIF with a reduced palette, in place."""
    from PIL import Image, ImageSequence

    with Image.open(path) as src:
        frames = [f.convert("RGB").quantize(colors=GIF_PALETTE_COLORS, method=Image.MEDIANCUT)
                  for f in ImageSequence.Iterator(src)]
        duration = src.info.get("duration", 80)
        loop = src.info.get("loop", 0)

    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=duration, loop=loop, optimize=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="fixture", choices=("fixture", "goose"))
    parser.add_argument("--root", help="dataset root; the fixture generates one when omitted")
    parser.add_argument("--frame", default="", help="frame id; the first frame when omitted")
    parser.add_argument("--calib", help="calibration path, required for --dataset goose")
    parser.add_argument("--out", default="docs/images/decalib.gif")
    parser.add_argument("--max-deg", type=float, default=DEFAULT_MAX_DEG)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    _dataset, frame, T_nominal = build_dataset(args)
    angles = sweep_angles(args.max_deg, args.steps)
    nominal = render(frame, T_nominal, angles, args.out, args.fps)

    print(f"wrote {args.out}  ({len(angles)} frames, 0 to {args.max_deg} deg pitch)")
    print(f"nominal agreement {nominal.fraction:.4f} over "
          f"{nominal.scorable:,} scorable points ({nominal.coverage:.1%} coverage)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
