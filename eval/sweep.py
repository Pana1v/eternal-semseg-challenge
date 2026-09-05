#!/usr/bin/env python3
"""The robustness sweep harness: decalibration, time offset, modality dropout
and the deskew ablation, behind one CLI.

Usage:
    python eval/sweep.py --baseline bl_paint --baseline bl_geom3d \\
        --dataset fixture --root /fixture/frames --split score \\
        --sweep decalib --rot-deg 0.5 2.0 --trans-cm 2 10 \\
        --limit 2 --out-dir results/sweeps

Why this file is custom code at all. Problem statement section 5 says to use
existing tooling and not to build a harness, and then names the three
exceptions: "the cross-modal consistency metric, the decalibration/time-offset
sweep harness, and the deployment path". This is the second of the three. No
model zoo perturbs an extrinsic between inference calls, so there is nothing
to reuse.

Why it runs first. Section 6.3 lists "fusion gain does not survive realistic
decalibration" as a risk "detected early by running Section 6.3 BEFORE
architecture work, not after". The sweeps are not a validation step at the end
of a project, they are the measurement that decides whether fusion is worth
architecting.

THE HEADLINE OUTPUT is `crossover`. Section 6.3, on the decalibration sweep:
"Report the perturbation at which fusion drops below the LiDAR-only baseline.
That number is the calibration accuracy the robot must sustain in production."
So it is printed, and written into the companion JSON as a first-class field,
rather than left for a reader to interpolate out of the CSV.

PERTURBATION IS INFERENCE TIME ONLY. `fit()` is always handed the unperturbed
extrinsic and the unperturbed cloud. Perturbing during fit measures whether a
model can LEARN to tolerate a miscalibrated rig, which is an augmentation
experiment and a different question from how a rig-calibrated model degrades
when the rig drifts in the field. The two are enforced by different mechanisms
because they arrive by different routes:

  - the extrinsic, by `runner.run` calling `extrinsic_fn(fit_ids[0],
    perturbed=False)` for fit and `extrinsic_fn(frame_id)` for inference. The
    perturbation lives behind that keyword.
  - the cloud and the image, by `PerturbedDataset` keying on which split a
    frame id belongs to, since fit frames and score frames are loaded through
    the same `load()`.

One column of the CSV a reader will trip over: `consistency` is 1.0 in every
row, at every magnitude, for every arm in this repo. It is structural and not
a bug. All three projection-dependent baselines derive one modality from the
other through the SAME uv the accumulator then re-derives, so the two views
cannot disagree: bl_cam2d resamples its 3D labels out of its own 2D map, and
bl_geom3d and bl_paint scatter their 3D labels into 2D. Their 3D mIoU still
falls under decalibration, because the labels move to the wrong POINTS even
while the two views of them agree. That gap between a pinned consistency curve
and a falling mIoU curve is why the crossover below is computed on 3D mIoU,
and why coverage is reported in the column beside it.

Outputs, one set per sweep, named after the sweep because run_all.sh drives
several into the same --out-dir:

    <out-dir>/<sweep>.csv     the tidy rows, one per operating point per arm
    <out-dir>/<sweep>.json    the same rows plus the crossover verdicts
    <out-dir>/<sweep>.png     mIoU against magnitude, crossover marked

eval/report.py finds the CSV by its `sweep` column, the JSON by its stem and
the plot by its suffix, so nothing here has to be registered anywhere.
"""

import argparse
import contextlib
import csv
import functools
import hashlib
import io
import json
import os
import sys
import tempfile
from dataclasses import dataclass, replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from baselines.common import base, runner  # noqa: E402
from eval.metrics import consistency, coverage, miou  # noqa: E402
from eval.plot_results import COLOR_3D, COLOR_REFERENCE, DPI, FIG_WIDE  # noqa: E402
from semseg.datasets import Dataset  # noqa: E402
from semseg.deskew import deskew, se3_exp  # noqa: E402
from semseg.projection import AXIS_RANDOM, perturb_extrinsic  # noqa: E402

# Problem statement section 6.3, quoted as the grid rather than paraphrased:
# rotation in degrees and translation in centimetres.
DECALIB_ROT_DEG = (0.1, 0.25, 0.5, 1.0, 2.0)
DECALIB_TRANS_CM = (1, 2, 5, 10)

# Swept PER AXIS, and that decomposition is the point of the sweep rather than
# a nicety. The camera optical frame is x right, y down, z forward, so:
#
#   roll  is about z, the optical axis. It rotates the image about the
#         principal point, so a centred point does not move at all and an
#         off-axis one moves in proportion to its distance from the centre,
#         not to its range.
#   pitch is about x. A lateral world displacement of a projected point that
#         grows LINEARLY WITH RANGE, which is the failure that actually
#         reaches a costmap at 30 m.
#   yaw   is about y. The horizontal counterpart of pitch, same range scaling.
#
# Measured in this repo, at the fixture's intrinsics: 0.5 deg of roll moves a
# centred point 0 px, while 0.5 deg of pitch moves it 9 px. A sweep over one
# unnamed delta-theta would average those together and report a degradation
# roughly a third of the real pitch sensitivity, which is to say it would
# average away the very effect the sweep exists to find, and it would do so
# quietly.
AXES = ("roll", "pitch", "yaw")

# The random-axis arm is run at each of these and reported as a spread. A real
# rig does not drift along a named axis, so this is the honest aggregate; the
# per-axis arms above are what make it interpretable.
SWEEP_SEEDS = (0, 1, 2)

# Section 6.3: "inject dt in {0, 10, 25, 50, 100, 200} ms between camera and
# LiDAR timestamps at three ego-speeds".
TIME_OFFSETS_MS = (0, 10, 25, 50, 100, 200)
EGO_SPEEDS_MPS = (0.3, 1.0, 2.0)

# Section 6.3: "LiDAR points randomly dropped 25/50/75 percent".
POINT_DROPOUT = (0.25, 0.5, 0.75)

SWEEP_DECALIB = "decalib"
SWEEP_TIME_OFFSET = "time_offset"
SWEEP_DROPOUT = "dropout"
SWEEP_DESKEW = "deskew"

# The two arms the crossover is defined between. bl_paint's RGB features
# arrive THROUGH the projection operator being perturbed; bl_geom3d never
# opens the image, so its 3D answer is the reference the fused curve falls
# towards. Named as constants because the crossover is meaningless without
# knowing which curve is which.
FUSED_BASELINE = "bl_paint"
LIDAR_ONLY_BASELINE = "bl_geom3d"

# Axis labels for the sweeps whose independent variable is not a rig axis.
AXIS_TIME_OFFSET = "time_offset"
AXIS_LIDAR_DROPOUT = "lidar_dropout"
AXIS_CAMERA_BLACK = "camera_black"
AXIS_CAMERA_SATURATED = "camera_saturated"
AXIS_DESKEW = "deskew"

# What to do to the image in the dropout sweep.
IMAGE_NOMINAL = "nominal"
IMAGE_BLACK = "black"
IMAGE_SATURATED = "saturated"

BLACK_LEVEL = 0
SATURATION_LEVEL = 255

UNIT_DEG = "deg"
UNIT_CM = "cm"
UNIT_MS = "ms"
UNIT_FRACTION = "fraction"
UNIT_ENABLED = "enabled"

CM_PER_M = 100.0
MS_PER_S = 1000.0

# Width of the md5 hex prefix mixed into the dropout seed. Pinned as a
# constant because narrowing it would quietly change every dropout draw.
FRAME_SEED_HEX_CHARS = 8

# Reference time the deskew arm compensates to. Zero, from the fixture's own
# pinned convention (semseg/datasets/fixture.py): `point_times` is seconds
# since scan start with t = 0 AT scan start, and "the sensor pose at time t is
# a translation of v * t and a yaw of wz * t relative to scan start". deskew()
# applies exp(xi * (t_i - t_ref)), so t_ref = 0 undoes exactly that motion. It
# is also the only value consistent with the image, which render_camera casts
# from a single static origin in the scan-start lidar frame. Any other value
# would leave a residual that looks like a finding.
DESKEW_T_REF_S = 0.0

# The lidar-only arm's 3D output cannot depend on the extrinsic: it never
# reads the image. So under decalib its 3D mIoU curve must be flat, and this
# tolerance is a self-check rather than a fudge. A curve that moves means
# something cached or re-derived the extrinsic where it should not have, which
# is the failure mode baselines/common/base.py warns about at length.
FLAT_REFERENCE_SWEEPS = (SWEEP_DECALIB,)
FLATNESS_TOL_MIOU = 1e-6

# Sweeps that need a populated `Frame.ego_twist`, and therefore a dataset that
# reports poses_available. Real GOOSE val ships no poses (spec 13.3), so these
# two refuse there rather than assuming a constant velocity.
SWEEPS_NEEDING_POSES = (SWEEP_TIME_OFFSET, SWEEP_DESKEW)

# Frozen column order (spec ch. 8). `baseline` is in it because a crossover is
# defined between two arms, so a row that did not say which arm produced it
# could not be grouped into a curve.
CSV_COLUMNS = ("sweep", "baseline", "axis", "magnitude", "unit", "seed", "speed_mps",
               "miou_2d", "miou_3d", "consistency", "coverage")

# Metric columns are rounded so the CSV diffs cleanly between runs. Six digits
# is well past the run-to-run reproducibility of any of them.
CSV_DIGITS = 6

DEFAULT_SPLIT = "score"
DEFAULT_OUT_DIR = os.path.join("results", "sweeps")

# The submission file every row writes is a throwaway: this module reads the
# payload runner.run RETURNS and never re-opens it, and one submission per row
# would be hundreds of files nobody reads. So they all go to one scratch name
# in a temporary directory, and --out-dir holds only the CSV, the JSON and the
# plot.
SCRATCH_SUBMISSION = "sweep_row_submission.json"

# Distinct from 1 so a driver can tell "this sweep is not runnable on this
# dataset" apart from "this sweep crashed". Matches bl_paint/run.py's
# NO_CALIB_EXIT, which means the same thing.
REFUSED_EXIT = 2

NO_CALIB_MESSAGE = (
    "every sweep needs T_cam_lidar and this dataset ships none.\n"
    "The metrics themselves are projection dependent: consistency and coverage are "
    "counted through pi(p) = K . T_cam_lidar . p, so there is nothing to measure "
    "without it.\n"
    "GOOSE distributes its rig calibration through the GOOSE-DB ROS bags' /tf_static, "
    "not through the annotated val zips (spec 13.3). Pass one with --calib, or run "
    "--dataset fixture, whose extrinsic is exact by construction.")

NO_POSES_MESSAGE = (
    "the {sweep} sweep needs a populated Frame.ego_twist and this dataset reports "
    "poses_available=False.\n"
    "Refusing rather than assuming a constant velocity: the ego twist IS the "
    "independent variable here, so a made-up one would make the whole curve a "
    "measurement of the assumption instead of of the data.\n"
    "Run --dataset fixture, which renders a moving sensor with a known twist and "
    "per-point timestamps, and is where every motion dependent number in this repo "
    "comes from.")

# Three distinct outcomes, and the two that report no magnitude are OPPOSITE
# findings. "fusion survived everything we swept" and "fusion was already
# losing before we started" must never both print as "no crossover": the first
# says the rig tolerance is looser than the grid, the second says it is tighter
# and the grid cannot see it.
NEVER_BELOW_REASON = ("fusion never falls below lidar-only within {largest:g} {unit}, "
                      "the largest perturbation swept, so the tolerance is wider "
                      "than this grid can measure")
ALREADY_BELOW_REASON = ("fusion is already below lidar-only at the smallest swept "
                        "perturbation ({smallest:g} {unit}), so the crossover is below "
                        "this grid and the sweep must be re-run finer")
NO_FUSED_ARM_REASON = ("the fused arm {fused} was not swept in this run, so there is no "
                       "curve to cross the lidar-only reference; pass --baseline {fused}")
NO_REFERENCE_REASON = ("the lidar-only arm {lidar_only} was not swept in this run, so "
                       "there is no reference to cross; pass --baseline {lidar_only}")
UNMEASURED_REASON = "the curve could not be read: {error}"

PLOT_SUFFIX = ".png"
CROSSOVER_MARKER_SIZE = 9


@dataclass(frozen=True)
class RowSpec:
    """One swept operating point: the CSV labels for it, plus every parameter
    needed to reproduce the perturbation it names.

    Frozen because a sweep result is only reproducible if the row that
    produced a number cannot be edited after the fact.

    `magnitude` and `unit` are the plottable independent variable, and every
    sweep has exactly one of them, which is what lets four different sweeps
    share one CSV schema and one crossover function. The mechanical fields
    below it are what actually gets applied.
    """
    sweep: str
    axis: str
    magnitude: float
    unit: str
    seed: int = 0

    # None rather than nan when the sweep has no ego speed, so the CSV cell is
    # empty and the JSON field is null instead of carrying a NaN that JSON
    # cannot represent.
    speed_mps: float = None

    # decalib
    rot_deg: float = 0.0
    trans_m: float = 0.0

    # time_offset
    offset_s: float = 0.0

    # dropout
    drop_fraction: float = 0.0
    image_mode: str = IMAGE_NOMINAL

    # deskew
    deskew_enabled: bool = False


@dataclass(frozen=True)
class Crossover:
    """The verdict of `crossover`, in the shape eval/report.py reads.

    `crossed` is always present, so the verdict is recognisable even when
    there is no magnitude to report. That matters: a companion JSON with no
    readable verdict renders as "no crossover verdict found", which report.py
    is careful to say is "not the same finding as no crossover", and the whole
    reason this dataclass carries a `reason` on every path is to keep that
    distinction alive all the way to the page.
    """
    crossed: bool
    magnitude: float = None
    unit: str = None
    axis: str = None
    baseline: str = None
    reason: str = None

    def to_dict(self) -> dict:
        return {"crossed": self.crossed, "magnitude": self.magnitude,
                "unit": self.unit, "axis": self.axis,
                "baseline": self.baseline, "reason": self.reason}


def crossover(magnitudes, fused_miou_3d, lidar_only_miou_3d,
              unit=None, axis=None, baseline=None) -> Crossover:
    """-> the perturbation at which the fused curve drops below the lidar-only
    reference, by linear interpolation between the two bracketing magnitudes.

    This is the headline deliverable of the whole repo. Problem statement
    section 6.3: "Report the perturbation at which fusion drops below the
    LiDAR-only baseline. That number is the calibration accuracy the robot
    must sustain in production."

    `lidar_only_miou_3d` is a scalar because the lidar-only arm is a reference
    level and not a curve in the swept variable. Under decalib that is exact:
    bl_geom3d's 3D output never reads the image, so it cannot move when the
    extrinsic does. Under the other sweeps the reference genuinely degrades
    too, and the caller is responsible for having reduced it to one number and
    for reporting its spread beside this verdict.

    Three outcomes, and the two without a magnitude are opposite findings that
    must never be collapsed into one message:

      - the curve crosses, on a swept point or between two: a magnitude.
        No special case is needed for landing exactly on a swept point. When
        fused[k] equals the reference the interpolation weight comes out
        exactly 1, so the formula returns magnitudes[k] itself.
      - the curve never falls below within the grid: no magnitude, and the
        reason says the tolerance is wider than the grid.
      - the curve is already below at the smallest swept magnitude: no
        magnitude, and the reason says the crossover is below the grid.

    A nan anywhere RAISES rather than being skipped. metrics.miou returns nan
    for an empty split, and `nan <= reference` is False, so a swallowed nan
    reads as "fusion never falls below" and publishes the most reassuring
    possible conclusion off the back of a failed measurement.
    """
    magnitudes = np.asarray(magnitudes, dtype=np.float64)
    fused = np.asarray(fused_miou_3d, dtype=np.float64)
    if magnitudes.shape != fused.shape:
        raise ValueError(f"{magnitudes.shape[0]} magnitudes but {fused.shape[0]} "
                         f"fused values; they must pair up")

    if magnitudes.size == 0:
        raise ValueError("crossover needs at least one swept magnitude")

    reference = float(lidar_only_miou_3d)
    if reference != reference:
        raise ValueError("the lidar-only reference mIoU is nan, so there is no level "
                         "to cross; the reference arm scored nothing")

    bad = np.flatnonzero(fused != fused)
    if bad.size > 0:
        raise ValueError(f"fused mIoU is nan at magnitude {magnitudes[bad[0]]:g}, so the "
                         f"curve cannot be read; a skipped nan would publish as "
                         f"'never falls below'")

    # Sorted rather than assumed ascending: "the FIRST crossing" only means
    # "the smallest perturbation that loses fusion" if the grid ascends, and
    # `--rot-deg 2.0 0.5` is a legal invocation.
    order = np.argsort(magnitudes, kind="stable")
    magnitudes, fused = magnitudes[order], fused[order]

    verdict = functools.partial(Crossover, unit=unit, axis=axis, baseline=baseline)

    if fused[0] <= reference:
        return verdict(crossed=False,
                       reason=ALREADY_BELOW_REASON.format(smallest=magnitudes[0],
                                                          unit=unit or "units"))

    for index in range(1, magnitudes.size):
        if fused[index] > reference:
            continue

        # fused[index - 1] is strictly above the reference and fused[index] is
        # at or below it, so the denominator is strictly positive and needs no
        # epsilon guard.
        span = fused[index - 1] - fused[index]
        weight = (fused[index - 1] - reference) / span
        crossing = magnitudes[index - 1] + weight * (magnitudes[index] - magnitudes[index - 1])
        return verdict(crossed=True, magnitude=float(crossing))

    return verdict(crossed=False,
                   reason=NEVER_BELOW_REASON.format(largest=magnitudes[-1],
                                                    unit=unit or "units"))


class PerturbedDataset(Dataset):
    """`dataset` with one RowSpec's SENSOR perturbation applied to the frames
    of `score_ids` and to no other frame.

    Keyed on the id set rather than on a flag, because that is the only place
    the inference-time-only guarantee can live: runner.run loads fit frames
    and score frames through this same `load`, so a wrapper that perturbed
    unconditionally would perturb the fit split too and quietly turn the sweep
    into an augmentation experiment. Nothing would crash. The curve would
    simply be flatter than the truth, and the published crossover would be
    optimistic.

    The extrinsic is NOT touched here. It travels by its own route, the
    `perturbed` keyword of the extrinsic_fn, because runner.run needs to ask
    for the unperturbed one explicitly when it fits.
    """

    def __init__(self, dataset, fit_ids, score_ids, spec: RowSpec):
        overlap = sorted(set(fit_ids) & set(score_ids))
        if overlap:
            raise ValueError(f"{len(overlap)} frames are in both splits, e.g. "
                             f"{overlap[:3]}; a frame in both would be perturbed "
                             f"during fit")

        self._dataset = dataset
        self._perturbed_ids = set(score_ids)
        self._spec = spec

        # Forwarded rather than inherited: a caller that gated on these before
        # wrapping must get the same answers afterwards.
        self.calib_available = getattr(dataset, "calib_available", True)
        self.poses_available = getattr(dataset, "poses_available", True)

    def frame_ids(self) -> list:
        return self._dataset.frame_ids()

    def load(self, frame_id: str):
        frame = self._dataset.load(frame_id)
        if frame_id not in self._perturbed_ids:
            return frame

        return perturb_frame(frame, self._spec)

    def extrinsic(self, frame_id: str) -> np.ndarray:
        return self._dataset.extrinsic(frame_id)


def perturb_frame(frame, spec: RowSpec):
    """-> a copy of `frame` with `spec`'s sensor perturbation applied.

    Three independent effects, applied in the order a real rig produces them:
    the scan is motion compensated or not, the cloud is displaced by whatever
    camera-to-lidar time offset the row names, and then points or pixels are
    dropped. The extrinsic is untouched throughout, which for the time-offset
    sweep is the whole methodological point (see `_shift_by_twist`).
    """
    points = frame.points
    point_times = frame.point_times
    intensity = frame.intensity
    labels_3d = frame.labels_3d_gt
    image = frame.image

    if spec.deskew_enabled:
        points = _deskewed(frame)

    if spec.offset_s != 0.0:
        points = _shift_by_twist(points, frame, spec)

    if spec.drop_fraction > 0.0:
        keep = _dropout_mask(frame, spec)
        points = points[keep]
        intensity = intensity[keep]
        point_times = None if point_times is None else point_times[keep]
        labels_3d = None if labels_3d is None else labels_3d[keep]

    if spec.image_mode != IMAGE_NOMINAL:
        image = _degraded_image(image, spec.image_mode)

    return replace(frame, points=points, intensity=intensity, image=image,
                   point_times=point_times, labels_3d_gt=labels_3d)


def _deskewed(frame):
    """The motion-compensated cloud, refusing rather than silently passing the
    raw one through.

    deskew() returns the input unchanged when either the per-point timestamps
    or the ego twist is absent, which is the right answer for a dataset that
    ships neither, and exactly the wrong answer here: the deskew arm would
    then be bit-identical to the no-deskew arm and the ablation would report
    "motion compensation makes no difference" while measuring nothing at all.
    """
    if frame.point_times is None or frame.ego_twist is None:
        raise ValueError(
            f"frame {frame.frame_id} carries no "
            f"{'point_times' if frame.point_times is None else 'ego_twist'}, so the "
            f"deskew arm has nothing to compensate against and would silently "
            f"duplicate the no-deskew arm")

    return deskew(frame.points, frame.point_times, frame.ego_twist, DESKEW_T_REF_S)


def _shift_by_twist(points, frame, spec: RowSpec):
    """The cloud carried through `spec.offset_s` of ego motion.

    The time offset is injected by DISPLACING THE CLOUD and never by touching
    the extrinsic, and the difference is not cosmetic. A camera-to-lidar
    timestamp skew means the two sensors observed the world at two different
    instants; the rig geometry was correct at both. Folding it into
    T_cam_lidar would model it as a calibration error, which produces a
    range-dependent error of the wrong shape and, worse, would make the
    time-offset sweep a second and less honest decalibration sweep.

    Only the LINEAR part of the twist is rescaled to the row's ego speed. The
    yaw rate is left as the rig reported it, so `speed_mps` is the one thing
    that changes between rows at a fixed offset and the column can attribute
    the effect. Rescaling the rotation too would move two things per row.
    """
    if frame.ego_twist is None:
        raise ValueError(f"frame {frame.frame_id} carries no ego_twist, so no ego "
                         f"displacement can be computed; the time_offset sweep should "
                         f"have refused this dataset on poses_available")

    twist = np.asarray(frame.ego_twist, dtype=np.float64).copy()
    if spec.speed_mps is not None:
        speed = float(np.linalg.norm(twist[:3]))
        if speed == 0.0:
            raise ValueError(f"frame {frame.frame_id} reports a stationary ego twist, so "
                             f"there is no direction of travel to rescale to "
                             f"{spec.speed_mps} m/s")

        twist[:3] *= spec.speed_mps / speed

    T = se3_exp(twist * spec.offset_s)
    shifted = np.asarray(points, dtype=np.float64) @ T[:3, :3].T + T[:3, 3]
    return shifted.astype(points.dtype, copy=False)


def _dropout_mask(frame, spec: RowSpec):
    """-> (N,) bool, the points kept at `spec.drop_fraction`.

    The draw is seeded from the row seed AND the frame id, so a row is
    reproducible across runs and across machines while still dropping a
    different subset in each frame. md5 of the frame id is used as a fixed bit
    mixer for the same reason semseg/datasets/__init__.py uses it for the
    split: Python's built-in hash() is salted per process, so it would move
    the dropout between two runs of the same command.
    """
    digest = hashlib.md5(frame.frame_id.encode("utf-8")).hexdigest()
    rng = np.random.default_rng([spec.seed, int(digest[:FRAME_SEED_HEX_CHARS], 16)])
    return rng.random(frame.points.shape[0]) >= spec.drop_fraction


def _degraded_image(image, mode: str):
    """A blacked-out or saturated camera.

    Both are constant images rather than scaled ones, because the failure
    being modelled is a sensor that has stopped carrying information, not one
    that is dim or bright. A tunnel entrance saturates the sensor well and a
    dropped exposure returns zeros; in both cases every pixel of the affected
    region reads the same and a colour feature has nothing left to say.
    """
    if mode == IMAGE_BLACK:
        return np.full_like(image, BLACK_LEVEL)

    if mode == IMAGE_SATURATED:
        return np.full_like(image, SATURATION_LEVEL)

    raise ValueError(f"unknown image mode {mode!r}; expected one of "
                     f"{(IMAGE_NOMINAL, IMAGE_BLACK, IMAGE_SATURATED)}")


def sweep_extrinsic(T, spec: RowSpec):
    """-> the extrinsic to PREDICT with at `spec`.

    Only the decalib sweep touches it. Stated as an early return rather than
    left implicit, because a sweep that perturbed the extrinsic as a side
    effect of measuring something else would confound two error sources in one
    column, and problem statement section 3 lists them as three INDEPENDENT
    layers precisely so they can be measured apart.
    """
    if spec.sweep != SWEEP_DECALIB:
        return T

    return perturb_extrinsic(T, spec.axis, spec.rot_deg, spec.trans_m, rng=spec.seed)


def _extrinsic_fn(frame_id, perturbed=True, *, dataset, spec):
    """The extrinsic_fn runner.run drives, with the sweep's perturbation
    behind the `perturbed` keyword.

    runner.run calls this with perturbed=False exactly once, for fit, and
    that is where the inference-time-only guarantee for the extrinsic is
    made structural instead of remembered.
    """
    T = dataset.extrinsic(frame_id)
    if not perturbed:
        return T

    return sweep_extrinsic(T, spec)


def decalib_grid(rot_deg=DECALIB_ROT_DEG, trans_cm=DECALIB_TRANS_CM,
                 axes=AXES, seeds=SWEEP_SEEDS) -> list:
    """Rotation and translation swept SEPARATELY, one perturbation at a time.

    Problem statement section 6.3 lists them as two grids, and keeping them
    apart is what makes one `magnitude` column meaningful: a row that moved
    both would have no single magnitude to plot against, and a rotation
    tolerance and a translation tolerance are two different production
    requirements.

    The named axes come first, then the random-axis arm at each seed.
    """
    specs = []
    for axis in axes:
        specs += [RowSpec(sweep=SWEEP_DECALIB, axis=axis, magnitude=float(value),
                          unit=UNIT_DEG, rot_deg=float(value))
                  for value in rot_deg]
        specs += [RowSpec(sweep=SWEEP_DECALIB, axis=axis, magnitude=float(value),
                          unit=UNIT_CM, trans_m=float(value) / CM_PER_M)
                  for value in trans_cm]

    for seed in seeds:
        specs += [RowSpec(sweep=SWEEP_DECALIB, axis=AXIS_RANDOM, magnitude=float(value),
                          unit=UNIT_DEG, seed=seed, rot_deg=float(value))
                  for value in rot_deg]
        specs += [RowSpec(sweep=SWEEP_DECALIB, axis=AXIS_RANDOM, magnitude=float(value),
                          unit=UNIT_CM, seed=seed, trans_m=float(value) / CM_PER_M)
                  for value in trans_cm]

    return specs


def time_offset_grid(offsets_ms=TIME_OFFSETS_MS, speeds=EGO_SPEEDS_MPS) -> list:
    """One curve per ego speed. 0 ms is in the grid on purpose: it is the
    unperturbed anchor every curve is read against, and without it a reader
    cannot tell a shallow curve from a low one."""
    return [RowSpec(sweep=SWEEP_TIME_OFFSET, axis=AXIS_TIME_OFFSET,
                    magnitude=float(offset), unit=UNIT_MS, speed_mps=float(speed),
                    offset_s=float(offset) / MS_PER_S)
            for speed in speeds for offset in offsets_ms]


def dropout_grid(fractions=POINT_DROPOUT, seeds=SWEEP_SEEDS) -> list:
    """Three curves: point dropout, a blacked camera and a saturated one.

    Each carries its own unperturbed anchor at magnitude 0, which is what
    makes the categorical camera arms interpolable at all and gives every
    curve a crossover verdict rather than a bare pair of numbers.

    The point-dropout anchor is repeated once per seed even though nothing is
    dropped there and the three rows must come out identical. That is not
    waste: it is a free determinism check, and it keeps each seed's curve
    self-anchored so the spread between seeds is read at a fixed baseline.
    """
    specs = []
    for seed in seeds:
        specs += [RowSpec(sweep=SWEEP_DROPOUT, axis=AXIS_LIDAR_DROPOUT,
                          magnitude=float(value), unit=UNIT_FRACTION, seed=seed,
                          drop_fraction=float(value))
                  for value in (0.0,) + tuple(fractions)]

    for axis, mode in ((AXIS_CAMERA_BLACK, IMAGE_BLACK),
                       (AXIS_CAMERA_SATURATED, IMAGE_SATURATED)):
        specs.append(RowSpec(sweep=SWEEP_DROPOUT, axis=axis, magnitude=0.0,
                             unit=UNIT_FRACTION))
        specs.append(RowSpec(sweep=SWEEP_DROPOUT, axis=axis, magnitude=1.0,
                             unit=UNIT_FRACTION, image_mode=mode))

    return specs


def deskew_grid() -> list:
    """With and without per-point motion compensation, as one two-point curve
    so the ablation plots as a difference rather than as two unrelated bars."""
    return [RowSpec(sweep=SWEEP_DESKEW, axis=AXIS_DESKEW, magnitude=0.0,
                    unit=UNIT_ENABLED),
            RowSpec(sweep=SWEEP_DESKEW, axis=AXIS_DESKEW, magnitude=1.0,
                    unit=UNIT_ENABLED, deskew_enabled=True)]


SWEEP_BUILDERS = {
    SWEEP_DECALIB: decalib_grid,
    SWEEP_TIME_OFFSET: time_offset_grid,
    SWEEP_DROPOUT: dropout_grid,
    SWEEP_DESKEW: deskew_grid,
}


def run_row(baseline_class, dataset, fit_ids, score_ids, spec: RowSpec,
            scratch_dir: str, limit=None, split=DEFAULT_SPLIT) -> dict:
    """Fit unperturbed, predict at `spec`, and reduce the submission to one
    CSV row.

    Takes a baseline CLASS rather than an instance so every row gets a fresh
    fit from the same starting point. A single instance reused across rows
    would carry the previous row's fitted state, and on a baseline that
    refits lazily the curve would depend on the order the grid happened to be
    enumerated in.

    The baseline is constructed with its own default seed and never with the
    sweep's. The sweep's seeds are PERTURBATION seeds, and holding the fit
    seed fixed is what makes a difference between two rows attributable to the
    perturbation rather than to a re-drawn subsample.
    """
    wrapped = PerturbedDataset(dataset, fit_ids, score_ids, spec)
    extrinsic_fn = functools.partial(_extrinsic_fn, dataset=dataset, spec=spec)

    # runner.run prints two paths per call, which over a full grid is hundreds
    # of lines naming a scratch file nobody will open. Swallowed here and
    # replaced by one line per row carrying the numbers instead, which is what
    # a sweep log is read for.
    with contextlib.redirect_stdout(io.StringIO()):
        payload = runner.run(baseline_class(), wrapped, fit_ids, score_ids, extrinsic_fn,
                             os.path.join(scratch_dir, SCRATCH_SUBMISSION),
                             limit=limit, jobs=1, split=split)

    counts = payload["consistency"]
    return {
        "sweep": spec.sweep,
        "baseline": payload["method"],
        "axis": spec.axis,
        "magnitude": spec.magnitude,
        "unit": spec.unit,
        "seed": spec.seed,
        "speed_mps": spec.speed_mps,
        "miou_2d": miou(payload["conf_2d"]),
        "miou_3d": miou(payload["conf_3d"]),
        "consistency": consistency(counts["matched"], counts["scorable"]),
        "coverage": coverage(counts["in_frustum"], counts["total_points"]),
    }


def curve_key(row) -> tuple:
    """The columns that identify one CURVE, which is every label except the
    baseline and the magnitude.

    One key function for all four sweeps: a curve is a set of rows that differ
    only in the swept magnitude, whether that magnitude is degrees about
    pitch, milliseconds at 2 m/s, or a dropped fraction at seed 1.
    """
    return (row["axis"], row["unit"], row["seed"], row["speed_mps"])


def curve_labels(keys) -> dict:
    """-> {key: label}, for the crossover map and the plot legend.

    A key component that is the SAME for every curve in the sweep is left out
    of its label. It cannot disambiguate anything, and "pitch (deg) seed 0"
    reads as though the seed mattered when the sweep only ever used one.
    Dropping only constant components cannot make two labels collide.
    """
    keys = list(keys)
    varies = [len({key[index] for key in keys}) > 1 for index in range(4)]

    labels = {}
    for key in keys:
        axis, unit, seed, speed = key
        parts = [axis]
        if varies[1]:
            parts.append(f"({unit})")
        if varies[3] and speed is not None:
            parts.append(f"at {speed:g} m/s")
        if varies[2]:
            parts.append(f"seed {seed}")

        labels[key] = " ".join(parts)

    return labels


def _reference_level(rows):
    """-> (mean, min, max) 3D mIoU of the lidar-only rows handed in.

    Reduced to a mean rather than to the value at the nominal magnitude
    because not every grid contains a nominal point: decalib starts at 0.1
    deg. The spread is returned with it and reported, so a reader can see for
    themselves whether reducing the reference to one number was honest.
    """
    values = np.array([row["miou_3d"] for row in rows], dtype=np.float64)
    if values.size == 0 or np.isnan(values).any():
        return float("nan"), float("nan"), float("nan")

    return float(values.mean()), float(values.min()), float(values.max())


def crossovers(rows, sweep: str, stream=None) -> dict:
    """-> {label: Crossover} over every curve of the fused arm.

    The reference is the lidar-only arm's rows for the SAME curve key, so a
    time-offset curve at 2 m/s is compared against the lidar-only arm at
    2 m/s and not against its average over every speed.

    Under decalib the reference must be flat: bl_geom3d's 3D output never
    reads the image, so it cannot move when the extrinsic does. A curve that
    moves is reported loudly rather than averaged over, because the likely
    cause is that something cached the extrinsic, which is the one failure
    baselines/common/base.py says is dangerous exactly because nothing
    crashes.
    """
    out = sys.stderr if stream is None else stream

    fused = _by_curve(rows, FUSED_BASELINE)
    reference = _by_curve(rows, LIDAR_ONLY_BASELINE)

    if not fused:
        return {FUSED_BASELINE: Crossover(
            crossed=False, baseline=FUSED_BASELINE,
            reason=NO_FUSED_ARM_REASON.format(fused=FUSED_BASELINE))}

    if not reference:
        return {FUSED_BASELINE: Crossover(
            crossed=False, baseline=FUSED_BASELINE,
            reason=NO_REFERENCE_REASON.format(lidar_only=LIDAR_ONLY_BASELINE))}

    labels = curve_labels(fused)
    verdicts = {}

    for key, curve in sorted(fused.items()):
        axis, unit = key[0], key[1]
        label = f"{FUSED_BASELINE} {labels[key]}"

        level, low, high = _reference_level(reference.get(key, []))
        if sweep in FLAT_REFERENCE_SWEEPS and high - low > FLATNESS_TOL_MIOU:
            print(f"WARNING: the {LIDAR_ONLY_BASELINE} reference is not flat over "
                  f"{labels[key]} ({low:.6f} to {high:.6f} 3D mIoU). Its 3D output does "
                  f"not read the image and cannot depend on the extrinsic, so something "
                  f"derived a prediction from a cached T_cam_lidar.", file=out)

        try:
            verdicts[label] = crossover([row["magnitude"] for row in curve],
                                        [row["miou_3d"] for row in curve],
                                        level, unit=unit, axis=axis,
                                        baseline=FUSED_BASELINE)
        except ValueError as error:
            # Turned into a verdict rather than allowed to abort the run: the
            # CSV is already written by the time this is called, and a reason
            # naming the measurement failure is a third distinct finding, not
            # a silent "no crossover".
            verdicts[label] = Crossover(crossed=False, unit=unit, axis=axis,
                                        baseline=FUSED_BASELINE,
                                        reason=UNMEASURED_REASON.format(error=error))

    return verdicts


def _by_curve(rows, baseline: str) -> dict:
    """{curve key: rows}, for one baseline, each list sorted by magnitude."""
    curves = {}
    for row in rows:
        if row["baseline"] != baseline:
            continue
        curves.setdefault(curve_key(row), []).append(row)

    return {key: sorted(curve, key=lambda row: row["magnitude"])
            for key, curve in curves.items()}


def _csv_cell(column: str, value) -> str:
    """One CSV cell. An absent or undefined number becomes an empty cell, so
    the CSV and the companion JSON agree: JSON has no NaN, the repo's existing
    convention (see eval/plot_results.py) is that an undefined metric arrives
    as null, and the string "nan" in one file next to null in the other would
    read as two different outcomes."""
    if value is None:
        return ""

    if isinstance(value, str) or isinstance(value, (int, np.integer)):
        return str(value)

    number = float(value)
    if number != number:
        return ""

    if column == "magnitude":
        return f"{number:g}"

    return f"{number:.{CSV_DIGITS}f}"


def write_csv(path: str, rows) -> None:
    """The tidy rows, in the frozen column order.

    Tidy and not wide: one row per operating point per arm, with the sweep
    named in a column. eval/report.py recognises a sweep CSV by that column
    alone, so this shape is what makes the file discoverable without a
    manifest.
    """
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        for row in rows:
            writer.writerow([_csv_cell(column, row[column]) for column in CSV_COLUMNS])


def _json_number(value):
    """nan and numpy scalars out, plain JSON in."""
    if value is None or isinstance(value, str):
        return value

    if isinstance(value, (int, np.integer)):
        return int(value)

    number = float(value)
    return None if number != number else number


def write_json(path: str, sweep: str, rows, verdicts, args, n_frames) -> None:
    """The companion JSON: the same rows, plus the crossover verdicts.

    The verdicts are the reason this file exists rather than the CSV alone.
    They are keyed by curve label under `crossover`, which is the shape
    eval/report.py's `crossover_entries` reads, and every entry carries a
    `reason` when it carries no magnitude so that no verdict can reach the
    page as "unreadable".
    """
    payload = {
        "sweep": sweep,
        "dataset": args.dataset,
        "split": args.split,
        "baselines": list(args.baseline),
        "fused_baseline": FUSED_BASELINE,
        "lidar_only_baseline": LIDAR_ONLY_BASELINE,
        "n_frames_scored": n_frames,
        "limit": args.limit,
        "grid": {"rot_deg": list(args.rot_deg), "trans_cm": list(args.trans_cm),
                 "time_offsets_ms": list(TIME_OFFSETS_MS),
                 "ego_speeds_mps": list(EGO_SPEEDS_MPS),
                 "point_dropout": list(POINT_DROPOUT),
                 "seeds": list(SWEEP_SEEDS)},
        "crossover": {label: verdict.to_dict() for label, verdict in verdicts.items()},
        "rows": [{column: _json_number(row[column]) for column in CSV_COLUMNS}
                 for row in rows],
    }

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def plot_sweep(path: str, sweep: str, rows, verdicts) -> None:
    """mIoU against magnitude, one line per curve, the lidar-only arm as the
    reference and the crossover marked.

    One panel per unit. A decalib run sweeps degrees and centimetres, and
    drawing both against one x axis would put 2 deg next to 2 cm and invite
    exactly the comparison the two grids exist to keep apart.
    """
    units = sorted({row["unit"] for row in rows})
    if not units:
        return

    fig, axes = plt.subplots(1, len(units), figsize=FIG_WIDE, squeeze=False)

    for panel, unit in zip(axes[0], units):
        panel_rows = [row for row in rows if row["unit"] == unit]
        _plot_panel(panel, sweep, panel_rows, verdicts, unit)

    fig.tight_layout()
    fig.savefig(path, dpi=DPI)

    # Closed here and not left to the interpreter: a full grid renders one of
    # these per sweep and eval/plot_results.py already carries the warning
    # about leaking figure handles in a sweep.
    plt.close(fig)


def _plot_panel(panel, sweep: str, rows, verdicts, unit: str) -> None:
    fused = _by_curve(rows, FUSED_BASELINE)
    reference = _by_curve(rows, LIDAR_ONLY_BASELINE)
    labels = curve_labels(list(fused) + list(reference))

    for key, curve in sorted(fused.items()):
        panel.plot([row["magnitude"] for row in curve],
                   [row["miou_3d"] for row in curve],
                   marker="o", markersize=3, color=COLOR_3D, alpha=0.8,
                   label=f"{FUSED_BASELINE} {labels[key]}")

    # Drawn as its measured curve, not as an axhline at its mean. Under
    # decalib it comes out horizontal by itself, which is the honest way to
    # show a flat reference; under dropout it genuinely degrades, and a
    # horizontal line there would be a drawn assertion that it does not.
    for key, curve in sorted(reference.items()):
        panel.plot([row["magnitude"] for row in curve],
                   [row["miou_3d"] for row in curve],
                   linestyle="--", marker="s", markersize=3, color=COLOR_REFERENCE,
                   label=f"{LIDAR_ONLY_BASELINE} {labels[key]}")

    for verdict in verdicts.values():
        if not verdict.crossed or verdict.unit != unit:
            continue

        panel.axvline(verdict.magnitude, color=COLOR_3D, linestyle=":", linewidth=1)
        panel.plot([verdict.magnitude], [0.0], marker="^", color=COLOR_3D,
                   markersize=CROSSOVER_MARKER_SIZE, clip_on=False,
                   label=f"crossover {verdict.magnitude:g} {unit}")

    panel.set_xlabel(f"{sweep} magnitude ({unit})")
    panel.set_ylabel("3D mIoU")
    panel.set_ylim(0.0, 1.0)
    panel.set_title(f"{sweep}, {unit}")
    panel.grid(True, alpha=0.3)
    panel.legend(fontsize=6, loc="lower left")


def print_verdicts(verdicts, stream=None) -> None:
    """The headline, printed rather than left in the JSON.

    Problem statement section 6.3 calls the crossover magnitude the
    calibration accuracy the robot must sustain in production, and run_all.sh
    tees this to a log expecting to find it there.
    """
    out = sys.stdout if stream is None else stream
    print("== crossover: the perturbation at which fusion drops below lidar-only ==",
          file=out)

    for label, verdict in sorted(verdicts.items()):
        if verdict.crossed:
            print(f"  {label}: fusion drops below lidar-only at "
                  f"{verdict.magnitude:g} {verdict.unit}", file=out)
            continue

        print(f"  {label}: no crossover, {verdict.reason}", file=out)


def build_parser() -> argparse.ArgumentParser:
    """The sweep's own flags rather than runner.add_common_args.

    That helper declares --out, a submission path, which this tool has no use
    for: a sweep writes a CSV, a JSON and a plot, and inheriting a flag it
    silently ignores would be an invitation to pass it and wonder where the
    file went.
    """
    parser = argparse.ArgumentParser(
        description="Robustness sweeps of problem statement section 6.3, with the "
                    "fusion-versus-lidar-only crossover as the headline output")

    parser.add_argument("--baseline", action="append", required=True,
                        help="repeatable, and at least two arms are needed for a "
                             f"crossover: {FUSED_BASELINE} is the fused curve and "
                             f"{LIDAR_ONLY_BASELINE} is the reference it falls towards")
    parser.add_argument("--sweep", required=True, choices=sorted(SWEEP_BUILDERS))
    parser.add_argument("--dataset", required=True, choices=sorted(runner.DATASET_MODULES))
    parser.add_argument("--root", help="dataset root; the fixture generates into it")
    parser.add_argument("--split", default=DEFAULT_SPLIT, choices=("fit", "score"),
                        help="which split to sweep over; the other one is fitted on")
    parser.add_argument("--rot-deg", nargs="+", type=float, default=list(DECALIB_ROT_DEG),
                        help="decalib rotation magnitudes in degrees")
    parser.add_argument("--trans-cm", nargs="+", type=float, default=list(DECALIB_TRANS_CM),
                        help="decalib translation magnitudes in centimetres")
    parser.add_argument("--limit", type=int, help="sweep only the first N frames of the split")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=0, help="dataset seed, for the fixture")
    parser.add_argument("--calib", help="4x4 T_cam_lidar as .npy or .json. REQUIRED on "
                                        "goose: the val zips ship no calibration at all")
    return parser


def _build_grid(args) -> list:
    """The row specs for the chosen sweep. Only decalib takes its grid from
    the CLI, because only its magnitudes are expensive enough that run_all.sh
    needs to trim them to two per axis."""
    if args.sweep == SWEEP_DECALIB:
        return decalib_grid(rot_deg=args.rot_deg, trans_cm=args.trans_cm)

    return SWEEP_BUILDERS[args.sweep]()


def _refusal(dataset, sweep: str):
    """-> the reason this sweep cannot run on this dataset, or None.

    Both refusals are the same argument in two places: a number computed from
    a fabricated rig or a fabricated velocity is a measurement of the
    fabrication, and section 6.3 makes those very quantities the independent
    variables of the headline sweep, so a made-up value would not be a small
    error, it would be the entire result.
    """
    if not runner.has_calibration(dataset):
        return NO_CALIB_MESSAGE

    if sweep in SWEEPS_NEEDING_POSES and not getattr(dataset, "poses_available", True):
        return NO_POSES_MESSAGE.format(sweep=sweep)

    return None


def main(argv=None):
    args = build_parser().parse_args(argv)

    dataset = runner.build_dataset(args)

    refusal = _refusal(dataset, args.sweep)
    if refusal:
        print(f"eval/sweep.py --sweep {args.sweep}: {refusal}", file=sys.stderr)
        return REFUSED_EXIT

    # --split names the split to SWEEP over, so the other one is fitted on.
    # Resolved here rather than inside run_row, which is called once per row.
    fit_ids, score_ids = runner.resolve_splits(dataset, args.split)

    specs = _build_grid(args)
    n_frames = len(score_ids[:args.limit] if args.limit else score_ids)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"{args.sweep}: {len(specs)} operating points x {len(args.baseline)} arms "
          f"over {n_frames} frames")

    rows = []
    with tempfile.TemporaryDirectory() as scratch_dir:
        for name in args.baseline:
            baseline_class = base.load(name)
            for spec in specs:
                row = run_row(baseline_class, dataset, fit_ids, score_ids, spec,
                              scratch_dir, limit=args.limit, split=args.split)
                rows.append(row)
                print(_row_line(row))

    stem = os.path.join(args.out_dir, args.sweep)

    # The CSV is written BEFORE the verdicts are computed. crossover() raises
    # on an unreadable curve by design, and hours of measurement must not be
    # lost to a verdict that could not be formed.
    write_csv(stem + ".csv", rows)

    verdicts = crossovers(rows, args.sweep)
    write_json(stem + ".json", args.sweep, rows, verdicts, args, n_frames)
    plot_sweep(stem + PLOT_SUFFIX, args.sweep, rows, verdicts)

    print_verdicts(verdicts)
    for suffix in (".csv", ".json", PLOT_SUFFIX):
        print(f"wrote {stem + suffix}")

    return 0


def _row_line(row) -> str:
    """One progress line per row, carrying the numbers. This is what replaces
    runner.run's two swallowed path lines, and it is what a sweep log is
    actually read for."""
    speed = "" if row["speed_mps"] is None else f" at {row['speed_mps']:g} m/s"
    return (f"{row['baseline']} {row['axis']} {row['magnitude']:g} {row['unit']}"
            f"{speed} seed {row['seed']}: "
            f"3D mIoU {row['miou_3d']:.4f}, 2D mIoU {row['miou_2d']:.4f}, "
            f"consistency {row['consistency']:.4f}")


if __name__ == "__main__":
    sys.exit(main())
