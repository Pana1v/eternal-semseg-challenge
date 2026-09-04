"""Tests for the fused arm.

Three of these are the exercise itself rather than plumbing checks, and they
are written to be able to fail:

  1. On the fixture, at the nominal extrinsic, bl_paint's 3D mIoU beats BOTH
     single-modality arms. That is the premise of problem statement section 7,
     and if painting does not clear a lidar-only floor then there is nothing
     for a real fusion architecture to be compared against.
  2. The gain is localised to the points the camera could see. Section 6.2
     asks how much of a reported gain comes from points the camera never saw,
     and at the NOMINAL rig the honest answer for this method is "none", which
     is checkable. Under a perturbed rig it is not none, because the frustum
     moves with the extrinsic and repaints points that a nominal-frustum
     partition still calls out-of-frustum. The invariant that survives
     decalibration is stated over the points no rig painted, in
     test_decalibration_degrades_3d.
  3. A 2 degree extrinsic perturbation degrades that 3D mIoU measurably. A
     fused arm whose score does not move when the rig moves is either caching
     the extrinsic or not using colour, and both failures are silent.

The two comparison arms are built in this file out of baselines/common's
feature extractors and classifier, which is exactly what spec section 10
defines bl_geom3d and bl_cam2d to be, rather than imported from their
packages. A test whose verdict depends on another module's current state
reports on that module too, and this one is about bl_paint.

The geometry-only reference is fitted through the same _subsample helper with
the same cap and seed as bl_paint's geometry model, so the two are the same
estimator and the out-of-frustum delta is exactly zero rather than zero plus
sampling noise.
"""

import json
import os
from dataclasses import dataclass, replace

import numpy as np
import pytest

from baselines.bl_paint import run
from baselines.bl_paint.baseline import (FIT_POINTS_PER_FRAME, FIT_SUBSAMPLE_SEED,
                                          FUSED_FEATURE_NAMES, PaintBaseline, _subsample)
from baselines.common import base, features
from baselines.common.nb import GaussianNB
from eval.io_formats import Accumulator, load_submission
from eval.metrics import miou
from semseg.datasets import split_frames
from semseg.datasets.fixture import FixtureDataset
from semseg.projection import perturb_extrinsic, project, scatter_to_image, zbuffer
from semseg.types import NUM_CLASSES, UNLABELED, Prediction

FIXTURE_FRAMES = 6
FIXTURE_SEED = 0

# Section 7 asks whether fusion "clearly" beats the floor, so the assertion
# carries a margin rather than testing a strict inequality that a rounding
# difference could satisfy. The measured margins on this fixture are +0.055
# over the lidar-only arm and +0.110 over the camera-only arm.
MIN_FUSION_MARGIN = 0.02

# bl_paint's out-of-frustum half is the same fitted model as the reference
# lidar-only arm, so the delta there is not merely small, it is zero. Kept as a
# tolerance rather than an equality only to allow for float summation order.
OUT_FRUSTUM_TOLERANCE = 1e-9

DECALIB_AXES = ("roll", "pitch", "yaw")
DECALIB_DEG = 2.0

# Measured whole-cloud drops at 2 degrees, on this fixture at seed 0, are 0.044
# (roll), 0.101 (pitch) and 0.152 (yaw). The threshold is what "measurable"
# means here: below a couple of points of mIoU the fixture could not bracket a
# crossover and the sweep would be reporting noise.
MIN_DECALIB_DROP = 0.02

# How much more of the decalibration damage lands in-frustum than out of it.
# The assertion this guards is descriptive: it says where the mIoU moved, and
# the load-bearing claim in the same test is the exact label-level invariance
# over points no rig painted. Measured in-frustum drop over out-of-frustum drop
# on this fixture at seed 0 is 1570x (roll), 222x (pitch) and 53x (yaw), so 10x
# is a floor and not a threshold fitted to the numbers.
DAMAGE_LOCALISATION_RATIO = 10.0

# Brightness step for the colour-sensitivity perturbation. Large enough to move
# a painted point across a Gaussian class boundary, small enough that it is a
# lighting change and not a new image.
BRIGHTEN_STEP = 60

CLI_LIMIT = 2


@dataclass
class World:
    """The fixture, loaded once. Frames are held rather than reloaded because
    every arm and every perturbation reads the same four score frames, and
    regenerating them per test dominates the runtime of the file."""
    dataset: FixtureDataset
    fit_frames: list
    score_frames: list
    T_nominal: np.ndarray


class GeomOnlyArm:
    """bl_geom3d as spec section 10 defines it: geom_features into the shared
    Naive Bayes, with the 2D output projected and nearest-filled.

    Only the 3D half is implemented, because that is all any assertion in this
    file reads from this arm.
    """

    def __init__(self):
        self.model = None

    def fit(self, frames, T_cam_lidar):
        rng = np.random.default_rng(FIT_SUBSAMPLE_SEED)
        X, y = [], []

        for frame in frames:
            geom = features.geom_features(frame.points, frame.intensity)
            rows = np.flatnonzero(frame.labels_3d_gt != UNLABELED)
            take = _subsample(rng, rows, FIT_POINTS_PER_FRAME)
            X.append(geom[take])
            y.append(frame.labels_3d_gt[take])

        self.model = GaussianNB(NUM_CLASSES).fit(np.vstack(X), np.concatenate(y))
        return self

    def predict(self, frame, T_cam_lidar):
        height, width = frame.image.shape[:2]
        labels_3d, conf_3d = self.model.predict(
            features.geom_features(frame.points, frame.intensity))

        uv, depth, in_frustum = project(frame.points, frame.K, T_cam_lidar, width, height)
        _owner, visible = zbuffer(uv, depth, in_frustum, width, height)
        seeded = in_frustum & visible

        return Prediction(
            labels_2d=scatter_to_image(uv, seeded, labels_3d, width, height, UNLABELED),
            labels_3d=labels_3d,
            conf_2d=scatter_to_image(uv, seeded, conf_3d, width, height, np.float32(0.0)),
            conf_3d=conf_3d)


class CamOnlyArm:
    """bl_cam2d as spec section 10 defines it: pixel_features into the same
    Naive Bayes, with the 3D output sampled from the 2D map at each point's
    pixel.

    Points outside the frustum have no pixel to sample, so this arm declines
    them with UNLABELED. That is the honest answer and it is also why its 3D
    mIoU has to be read next to its scored-point count: confusion() drops
    UNLABELED, so this arm is scored on the frustum subset only. See
    test_camera_arm_scores_fewer_points.
    """

    def __init__(self):
        self.model = None

    def fit(self, frames, T_cam_lidar):
        X, y = [], []

        for frame in frames:
            pixels = features.pixel_features(frame.image)
            labels = frame.labels_2d_gt.reshape(-1)
            keep = labels != UNLABELED
            X.append(pixels[keep])
            y.append(labels[keep])

        self.model = GaussianNB(NUM_CLASSES).fit(np.vstack(X), np.concatenate(y))
        return self

    def predict(self, frame, T_cam_lidar):
        height, width = frame.image.shape[:2]
        flat_labels, flat_conf = self.model.predict(features.pixel_features(frame.image))
        labels_2d = flat_labels.reshape(height, width)
        conf_2d = flat_conf.reshape(height, width)

        uv, _depth, in_frustum = project(frame.points, frame.K, T_cam_lidar, width, height)
        labels_3d = np.full(frame.points.shape[0], UNLABELED, dtype=np.uint8)
        conf_3d = np.zeros(frame.points.shape[0], dtype=np.float32)
        labels_3d[in_frustum] = labels_2d[uv[in_frustum, 1], uv[in_frustum, 0]]
        conf_3d[in_frustum] = conf_2d[uv[in_frustum, 1], uv[in_frustum, 0]]

        return Prediction(labels_2d=labels_2d, labels_3d=labels_3d,
                          conf_2d=conf_2d, conf_3d=conf_3d)


@pytest.fixture(scope="module")
def world():
    dataset = FixtureDataset(n_frames=FIXTURE_FRAMES, seed=FIXTURE_SEED)
    fit_ids, score_ids = split_frames(dataset.frame_ids())

    return World(dataset=dataset,
                 fit_frames=[dataset.load(frame_id) for frame_id in fit_ids],
                 score_frames=[dataset.load(frame_id) for frame_id in score_ids],
                 T_nominal=dataset.extrinsic(score_ids[0]))


@pytest.fixture(scope="module")
def arms(world):
    """All three arms, fitted once on the fit split at the nominal extrinsic.

    fit() is never handed a perturbed extrinsic anywhere in this file. Spec
    section 8 makes perturbation inference-time only, because fitting on a
    perturbed rig is an augmentation experiment and would flatten every
    decalibration curve in the repo.
    """
    paint = PaintBaseline()
    paint.fit(iter(world.fit_frames), world.T_nominal)

    return {"bl_paint": paint,
            "bl_geom3d": GeomOnlyArm().fit(world.fit_frames, world.T_nominal),
            "bl_cam2d": CamOnlyArm().fit(world.fit_frames, world.T_nominal)}


def score(world, arm, T_predict=None) -> Accumulator:
    """Fold one arm over the score split.

    The accumulator always gets the NOMINAL extrinsic even when the arm
    predicts under a perturbed one. The frustum masks and the range bins are
    properties of the true rig, so holding them fixed is what makes a
    perturbed mIoU comparable to the nominal one: otherwise the in-frustum
    matrix would be built over a different set of points at every magnitude
    and the curve would mix a moving prediction with a moving denominator.
    """
    if T_predict is None:
        T_predict = world.T_nominal

    acc = Accumulator(method="test", split="score")
    for frame in world.score_frames:
        acc.add(frame, arm.predict(frame, T_predict), world.T_nominal)

    return acc


def uncoloured(frame, T_cam_lidar) -> np.ndarray:
    """The points predict() will NOT paint under `T_cam_lidar`: outside the
    frustum, or inside it but behind whatever won the z-buffer.

    Recovered through the same two public calls bl_paint itself makes, so this
    is the arm's painted set and not a second derivation of it that could drift
    away from the first.
    """
    height, width = frame.image.shape[:2]
    uv, depth, in_frustum = project(frame.points, frame.K, T_cam_lidar, width, height)
    _owner, visible = zbuffer(uv, depth, in_frustum, width, height)

    return ~(in_frustum & visible)


def test_beats_both_single_modality_arms(world, arms):
    """Problem statement section 7: the naive fused floor must clearly beat
    appending nothing at all."""
    fused = miou(score(world, arms["bl_paint"]).conf_3d)
    lidar_only = miou(score(world, arms["bl_geom3d"]).conf_3d)
    camera_only = miou(score(world, arms["bl_cam2d"]).conf_3d)

    assert fused > lidar_only + MIN_FUSION_MARGIN
    assert fused > camera_only + MIN_FUSION_MARGIN


def test_camera_arm_scores_fewer_points(world, arms):
    """The qualifier that the previous test's second assertion has to be read
    with. bl_cam2d declines every point with no pixel, and confusion() drops
    UNLABELED, so its 3D mIoU is computed over the frustum subset while
    bl_paint's covers the whole cloud. The fused arm also wins on the equal
    denominator, which is the comparison that carries weight.
    """
    fused = score(world, arms["bl_paint"])
    camera_only = score(world, arms["bl_cam2d"])

    assert camera_only.conf_3d.sum() < fused.conf_3d.sum()
    assert camera_only.conf_3d_out_frustum.sum() == 0

    assert miou(fused.conf_3d_in_frustum) > miou(camera_only.conf_3d_in_frustum)


def test_frustum_split_localises_the_gain(world, arms):
    """Spec section 6.2's ablation: how much of the gain comes from points the
    camera never saw. Here, exactly none of it, by construction."""
    fused = score(world, arms["bl_paint"])
    lidar_only = score(world, arms["bl_geom3d"])

    gain_in = miou(fused.conf_3d_in_frustum) - miou(lidar_only.conf_3d_in_frustum)
    gain_out = miou(fused.conf_3d_out_frustum) - miou(lidar_only.conf_3d_out_frustum)

    assert gain_in > gain_out
    assert abs(gain_out) < OUT_FRUSTUM_TOLERANCE

    # both matrices are non-empty, so the comparison above is between two real
    # populations and not between two nans
    assert fused.conf_3d_in_frustum.sum() > 0
    assert fused.conf_3d_out_frustum.sum() > 0


def test_beats_lidar_only_in_2d(world, arms):
    """The 2D half of the same comparison. Both arms build their 2D map the
    same way, by scattering 3D labels and nearest-filling, so this measures the
    colour in the 3D labels and nothing else.
    """
    fused = miou(score(world, arms["bl_paint"]).conf_2d)
    lidar_only = miou(score(world, arms["bl_geom3d"]).conf_2d)

    assert fused > lidar_only


@pytest.mark.parametrize("axis", DECALIB_AXES)
def test_decalibration_degrades_3d(world, arms, axis):
    """Spec section 6.3, and the check that predict() reads the extrinsic it is
    handed on every call. A cached extrinsic gives a flat curve, no crash and a
    crossover that never happens.

    Roll is swept alongside the other two. The claim this file used to make,
    that roll would give a null result because an on-axis point does not move,
    does not survive measurement: off-axis error grows with distance from the
    principal point (see projection.AXIS_VECTORS), and roll still costs 0.044
    whole-cloud mIoU at 2 degrees on this fixture.
    """
    nominal = score(world, arms["bl_paint"])

    perturbed_T = perturb_extrinsic(world.T_nominal, axis, DECALIB_DEG, 0.0)
    perturbed = score(world, arms["bl_paint"], perturbed_T)

    assert miou(nominal.conf_3d) - miou(perturbed.conf_3d) > MIN_DECALIB_DROP

    # The damage is where the colour is, and this is the exact form of that
    # claim. It is NOT true of the out-of-frustum confusion matrix: score()
    # partitions on the nominal rig while predict() paints on the perturbed
    # one, so a decalibrated frustum sweeps over points the partition calls
    # out-of-frustum and repaints them. At 2 degrees of yaw 526 such points
    # enter the perturbed frustum and 30 flip class, which is the entire 0.004
    # of out-of-frustum mIoU movement. Over the points NO rig painted, the two
    # label arrays are identical, not merely close.
    for frame in world.score_frames:
        never_painted = uncoloured(frame, world.T_nominal) & uncoloured(frame, perturbed_T)
        assert never_painted.any()

        assert np.array_equal(
            arms["bl_paint"].predict(frame, world.T_nominal).labels_3d[never_painted],
            arms["bl_paint"].predict(frame, perturbed_T).labels_3d[never_painted])

    # And the mIoU that did move, moved in-frustum: the residual out-of-frustum
    # drop is the boundary crossings above and nothing else.
    drop_in = miou(nominal.conf_3d_in_frustum) - miou(perturbed.conf_3d_in_frustum)
    drop_out = miou(nominal.conf_3d_out_frustum) - miou(perturbed.conf_3d_out_frustum)

    assert drop_in > DAMAGE_LOCALISATION_RATIO * drop_out


def test_colour_changes_only_coloured_points(world, arms):
    """The other half of the same perturbation argument, from the image side.
    Brightening the image must move the labels of painted points and must not
    touch a single point the camera could not see.
    """
    frame = world.score_frames[0]
    brighter = replace(frame, image=np.clip(frame.image.astype(np.int16) + BRIGHTEN_STEP,
                                            0, 255).astype(np.uint8))

    before = arms["bl_paint"].predict(frame, world.T_nominal).labels_3d
    after = arms["bl_paint"].predict(brighter, world.T_nominal).labels_3d

    height, width = frame.image.shape[:2]
    uv, depth, in_frustum = project(frame.points, frame.K, world.T_nominal, width, height)
    _owner, visible = zbuffer(uv, depth, in_frustum, width, height)
    has_colour = in_frustum & visible

    assert (before[has_colour] != after[has_colour]).any()
    assert np.array_equal(before[~has_colour], after[~has_colour])


def test_prediction_contract(world, arms):
    """Shapes and dtypes, because eval/io_formats.py folds these arrays into
    confusion matrices without re-checking them."""
    frame = world.score_frames[0]
    pred = arms["bl_paint"].predict(frame, world.T_nominal)

    assert pred.labels_2d.shape == frame.image.shape[:2]
    assert pred.labels_2d.dtype == np.uint8
    assert pred.labels_3d.shape == (frame.points.shape[0],)
    assert pred.labels_3d.dtype == np.uint8

    assert pred.conf_3d.dtype == np.float32
    assert pred.conf_2d.shape == frame.image.shape[:2]
    assert np.all((pred.conf_3d >= 0.0) & (pred.conf_3d <= 1.0))
    assert np.all((pred.conf_2d >= 0.0) & (pred.conf_2d <= 1.0))

    # every point gets a label: this arm declines nothing in 3D, it only
    # changes which of its two models answers
    assert not (pred.labels_3d == UNLABELED).any()


def test_no_extrinsic_declines_2d(world, arms):
    """The baseline contract's None case (spec 1b). Geometry still answers in
    3D; the 2D map, which only exists via the projection, is declined whole
    rather than guessed at."""
    frame = world.score_frames[0]
    pred = arms["bl_paint"].predict(frame, None)

    assert np.all(pred.labels_2d == UNLABELED)
    assert pred.conf_2d is None
    assert not (pred.labels_3d == UNLABELED).any()


def test_fit_without_extrinsic_is_geometry_only(world):
    """A fit with no calibration builds no fused model, so every point is
    scored by geometry even when a later predict() is handed a rig. run.py
    refuses this combination; the class itself stays honest about it."""
    unfused = PaintBaseline()
    unfused.fit(iter(world.fit_frames), None)

    frame = world.score_frames[0]
    pred = unfused.predict(frame, world.T_nominal)

    assert np.all(pred.labels_2d == UNLABELED)

    geom_reference = GeomOnlyArm().fit(world.fit_frames, world.T_nominal)
    assert np.array_equal(pred.labels_3d,
                          geom_reference.predict(frame, world.T_nominal).labels_3d)


def test_predict_before_fit_raises(world):
    with pytest.raises(RuntimeError, match="before fit"):
        PaintBaseline().predict(world.score_frames[0], world.T_nominal)


def test_fit_requires_3d_labels(world):
    unlabelled = replace(world.fit_frames[0], labels_3d_gt=None)

    with pytest.raises(ValueError, match="labels_3d_gt"):
        PaintBaseline().fit([unlabelled], world.T_nominal)


def test_feature_layout_is_geometry_then_colour():
    """The columns are the whole method, so their order is a contract: geometry
    first and unchanged, colour appended. Anything that reorders them breaks
    the claim that this arm and bl_geom3d share a feature set."""
    assert FUSED_FEATURE_NAMES[:len(features.GEOM_FEATURE_NAMES)] == features.GEOM_FEATURE_NAMES
    assert FUSED_FEATURE_NAMES[len(features.GEOM_FEATURE_NAMES):] == features.RGB_FEATURE_NAMES


def test_registered_under_its_directory_name():
    assert base.load("bl_paint") is PaintBaseline
    assert PaintBaseline.name == "bl_paint"


def test_cli_writes_a_valid_submission(tmp_path):
    out_path = os.path.join(str(tmp_path), "submission.json")

    assert run.main(["--dataset", "fixture", "--split", "score",
                     "--out", out_path, "--limit", str(CLI_LIMIT)]) == 0

    payload = load_submission(out_path)
    assert payload["method"] == "bl_paint"
    assert payload["n_frames"] == CLI_LIMIT

    with open(out_path + ".meta.json") as f:
        meta = json.load(f)
    assert meta["method_name"] == "bl_paint"


def test_cli_refuses_without_calibration(monkeypatch):
    """A GOOSE run with no --calib must not produce a file at all. Stubbed
    here rather than driven through GooseDataset because the refusal is a
    property of run.py: it asks the dataset one question and writes nothing if
    the answer is no.
    """
    class NoCalibDataset:
        calib_available = False

        def frame_ids(self):
            raise AssertionError("run.py must refuse before it touches the frames")

    monkeypatch.setattr(run.runner, "build_dataset", lambda args: NoCalibDataset())

    assert run.main(["--dataset", "goose", "--root", "/nonexistent",
                     "--out", "/nonexistent/submission.json"]) == run.NO_CALIB_EXIT
