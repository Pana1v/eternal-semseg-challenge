"""Unit tests for the LiDAR-only arm.

Everything is asserted on the procedural fixture, because it is the only world
in this repo with exact ground truth in both modalities AND an exact known
extrinsic (interface spec 13.3), so it is the only place the 2D half of this
arm can be checked at all.

Two of the tests below exist to prove the section 6.2 ablation is intact rather
than to check a number: test_never_reads_image_content, which fails loudly if
this arm so much as indexes a pixel, and test_fit_ignores_extrinsic. Both guard
against a refactor that would leave every reported score looking reasonable.
"""

import dataclasses

import numpy as np
import pytest

from baselines.common import base
from baselines.bl_geom3d.baseline import Geom3dBaseline, MIN_POINTS_FOR_FEATURES
from eval.io_formats import Accumulator
from eval.metrics import confusion, miou
from semseg.datasets import split_frames
from semseg.datasets.fixture import (
    ARTIFICIAL_GROUND, ARTIFICIAL_STRUCTURES, NATURAL_GROUND,
    FixtureDataset, nominal_extrinsic,
)
from semseg.projection import perturb_extrinsic, project, zbuffer
from semseg.types import UNLABELED

N_FIXTURE_FRAMES = 6
FIXTURE_SEED = 0

# The two goose9 classes that are both ground. Asserted as a union because the
# claim under test is "a ground point is called ground": artificial_ground and
# natural_ground differ by surface type, which a lidar-only arm can only read
# off intensity, and telling asphalt from low grass is not what this baseline
# is a floor for.
GROUND_CLASSES = (ARTIFICIAL_GROUND, NATURAL_GROUND)

# "Predominantly" spelled as a number. Well clear of half, and well clear of
# the measured values (0.99 for ground, 0.95 for the wall) so the tests fail on
# a real regression rather than on fixture noise.
PREDOMINANT_FRAC = 0.7

# Margin over the majority-mode floor. Measured margin on this fixture is about
# 0.80 (3D mIoU 0.886 against a floor of 0.084), so this asserts the arm learned
# something substantial rather than merely tying the constant predictor by a
# hair, which a bare > would allow.
MIN_MIOU_OVER_MAJORITY = 0.20

# The decalibration magnitude the perturbation case uses, and the axis it uses
# it about. pitch is the axis semseg/projection.py names as the one that hurts:
# it shifts a projected point laterally in proportion to its range. 2.0 degrees
# is the top of DECALIB_ROT_DEG in the interface spec section 8 sweep.
PERTURB_AXIS = "pitch"
PERTURB_DEG = 2.0

# The 2D map has to move by at least this fraction of its pixels. A bare
# "not array_equal" would pass on a single changed pixel, and it would still
# pass if scatter_to_image's nearest fill smeared the perturbation away almost
# everywhere, which is the outcome worth catching. Measured value is 0.071.
MIN_2D_CHANGED_FRAC = 0.01

# One point short of what a plane fit needs, derived from the shipped constant
# rather than written as a literal so the test keeps testing the declining
# branch if that threshold ever rises.
TINY_CLOUD_POINTS = MIN_POINTS_FOR_FEATURES - 1


@pytest.fixture(scope="module")
def dataset():
    return FixtureDataset(root=None, n_frames=N_FIXTURE_FRAMES, seed=FIXTURE_SEED)


@pytest.fixture(scope="module")
def splits(dataset):
    return split_frames(dataset.frame_ids())


@pytest.fixture(scope="module")
def fitted(dataset, splits):
    """One fit for the whole module. Nothing below mutates the baseline, and
    interface spec section 9 forbids fitting on the split being scored, so the
    fit split is the only thing this ever sees."""
    fit_ids, _ = splits

    baseline = Geom3dBaseline()
    baseline.fit((dataset.load(frame_id) for frame_id in fit_ids), nominal_extrinsic())
    return baseline


@pytest.fixture(scope="module")
def scored_frame(dataset, splits):
    _, score_ids = splits
    return dataset.load(score_ids[0])


@pytest.fixture(scope="module")
def prediction(fitted, scored_frame):
    return fitted.predict(scored_frame, nominal_extrinsic())


def declared_majority(dataset, fit_ids) -> int:
    """The single class bl_prior --mode majority predicts everywhere.

    bl_prior is written in the same wave as this file and is not importable
    yet, so its RULE is reproduced here rather than its code imported:
    interface spec section 10 pins the majority class as a declared constant
    taken from the fit split and never fitted on the scored split. When
    bl_prior lands, this becomes an import of its class and this function goes
    away.
    """
    counts = np.zeros(9, dtype=np.int64)
    for frame_id in fit_ids:
        labels = dataset.load(frame_id).labels_3d_gt
        counts += np.bincount(labels[labels != UNLABELED], minlength=9)

    return int(counts.argmax())


class ShapeOnlyImage:
    """Stands in for frame.image and fails loudly on any read of a pixel.

    Only `.shape` is allowed through, which is what baseline._canvas_hw needs
    and argues for. Every other access raises, and it raises AssertionError
    with the consequence named rather than leaning on a TypeError from a
    missing dunder, so a failure says what went wrong instead of only that
    something did.

    Dunder lookups are let through as a normal AttributeError: numpy and pytest
    probe for __array_interface__ and friends, and an AssertionError raised
    inside a repr would hide the real failure.
    """

    def __init__(self, shape):
        self.shape = shape

    def _read(self, *args, **kwargs):
        raise AssertionError(
            "bl_geom3d read frame.image content; the section 6.2 ablation is void")

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)

        self._read()

    __getitem__ = _read
    __array__ = _read
    __iter__ = _read
    __len__ = _read


def test_registry_resolves_name():
    """The integration surface eval/sweep.py actually uses. Catches a broken
    import chain and a duplicate registration under two module paths, neither
    of which any other test here would notice."""
    assert base.load("bl_geom3d") is Geom3dBaseline
    assert Geom3dBaseline.name == "bl_geom3d"


def test_beats_majority_mode(dataset, splits, scored_frame, prediction):
    fit_ids, _ = splits
    majority = declared_majority(dataset, fit_ids)

    gt = scored_frame.labels_3d_gt
    ours = miou(confusion(gt, prediction.labels_3d))
    floor = miou(confusion(gt, np.full_like(gt, majority)))

    assert ours > floor + MIN_MIOU_OVER_MAJORITY, (
        f"3D mIoU {ours:.4f} does not clear the majority-mode floor {floor:.4f} "
        f"(class {majority}) by {MIN_MIOU_OVER_MAJORITY}")


def test_ground_points_get_ground_class(scored_frame, prediction):
    is_ground = np.isin(scored_frame.labels_3d_gt, GROUND_CLASSES)
    assert is_ground.any(), "the fixture frame has no ground points, so this proves nothing"

    called_ground = np.isin(prediction.labels_3d[is_ground], GROUND_CLASSES).mean()
    assert called_ground > PREDOMINANT_FRAC, (
        f"only {called_ground:.3f} of ground points were called ground")


def test_wall_points_get_artificial_structures(scored_frame, prediction):
    """The wall is the one structure whose verticality column is the reason it
    is separable at all, so this is the test that the PCA features are wired
    the right way round."""
    is_wall = scored_frame.labels_3d_gt == ARTIFICIAL_STRUCTURES
    assert is_wall.any(), "the fixture frame has no wall points, so this proves nothing"

    called_wall = (prediction.labels_3d[is_wall] == ARTIFICIAL_STRUCTURES).mean()
    assert called_wall > PREDOMINANT_FRAC, (
        f"only {called_wall:.3f} of wall points were called artificial_structures")


def test_unperturbed_extrinsic_is_stable(fitted, scored_frame, prediction):
    """The negative control for the perturbation case below. A zero-magnitude
    perturbation must change nothing at all, in either modality. Without this
    the changed-fraction assertion could be passing on nondeterminism rather
    than on the perturbation."""
    unmoved = fitted.predict(
        scored_frame, perturb_extrinsic(nominal_extrinsic(), PERTURB_AXIS, 0.0, 0.0))

    assert np.array_equal(unmoved.labels_3d, prediction.labels_3d)
    assert np.array_equal(unmoved.labels_2d, prediction.labels_2d)


def test_2deg_decalib_moves_2d_only(fitted, scored_frame, prediction):
    """The asymmetry this arm exists to provide (interface spec section 8).

    The 3D half is a tautology by construction, and that is the point: it is
    the flat reference line the fused arm's crossover is measured against, and
    a change here would mean the 3D path had started reading the extrinsic. The
    2D half is the one that can silently fail to move, which is what
    MIN_2D_CHANGED_FRAC is for.
    """
    perturbed = fitted.predict(
        scored_frame, perturb_extrinsic(nominal_extrinsic(), PERTURB_AXIS, PERTURB_DEG, 0.0))

    assert np.array_equal(perturbed.labels_3d, prediction.labels_3d), (
        f"a {PERTURB_DEG} degree extrinsic error changed the 3D labels, so the 3D path is "
        "reading T_cam_lidar and is no longer a calibration-free reference line")

    changed = float((perturbed.labels_2d != prediction.labels_2d).mean())
    assert changed > MIN_2D_CHANGED_FRAC, (
        f"a {PERTURB_DEG} degree extrinsic error moved only {changed:.4f} of the 2D pixels; "
        "the decalibration sweep would report a robustness this arm does not have")


def test_never_reads_image_content(fitted, scored_frame):
    """Proof, not a convention, that this is a lidar-only arm."""
    poisoned = dataclasses.replace(
        scored_frame, image=ShapeOnlyImage(scored_frame.image.shape))

    pred = fitted.predict(poisoned, nominal_extrinsic())
    assert pred.labels_2d.shape == scored_frame.image.shape[:2]


def test_no_extrinsic_declines_2d(fitted, scored_frame, prediction):
    """The real GOOSE case (interface spec 13.3): no calibration ships, so the
    2D answer is declined and the 3D answer is untouched. A fabricated default
    extrinsic here would turn every 2D number on real GOOSE into a fiction."""
    pred = fitted.predict(scored_frame, None)

    assert np.all(pred.labels_2d == UNLABELED)
    assert pred.conf_2d is None
    assert np.array_equal(pred.labels_3d, prediction.labels_3d)


def test_fit_ignores_extrinsic(dataset, splits, scored_frame):
    """Fitting under a decalibrated rig must produce the identical classifier,
    because no feature of this arm is a projection. Guards the promise the 3D
    reference line rests on."""
    fit_ids, _ = splits
    decalibrated = perturb_extrinsic(nominal_extrinsic(), PERTURB_AXIS, PERTURB_DEG, 0.1)

    other = Geom3dBaseline()
    other.fit((dataset.load(frame_id) for frame_id in fit_ids), decalibrated)
    reference = Geom3dBaseline()
    reference.fit((dataset.load(frame_id) for frame_id in fit_ids), nominal_extrinsic())

    assert np.array_equal(
        other.predict(scored_frame, nominal_extrinsic()).labels_3d,
        reference.predict(scored_frame, nominal_extrinsic()).labels_3d)


def test_2d_pairs_label_with_confidence(scored_frame, prediction):
    """Every seeded pixel must carry BOTH the label and the confidence of the
    z-buffer winner that seeded it.

    The two are transported by separate scatter_to_image calls and nothing in
    the shipped code forces them to use the same mask, so a mismatch is a live
    failure mode: the map would then report one point's label next to another
    point's certainty, a pair no point ever produced, and the 2D ECE would be
    measuring a chimera.
    """
    height, width = scored_frame.image.shape[:2]
    uv, depth, in_frustum = project(
        scored_frame.points, scored_frame.K, nominal_extrinsic(), width, height)
    owner, _visible = zbuffer(uv, depth, in_frustum, width, height)

    rows, cols = np.nonzero(owner >= 0)
    assert rows.size > 0, "no pixel was seeded, so this proves nothing"
    winners = owner[rows, cols]

    assert np.array_equal(prediction.labels_2d[rows, cols], prediction.labels_3d[winners])
    assert np.array_equal(prediction.conf_2d[rows, cols], prediction.conf_3d[winners])


def test_consistency_is_exactly_one(scored_frame, prediction):
    """Cross-modal consistency is 1.0 for this arm BY CONSTRUCTION, and that is
    pinned here rather than left as a remark in the module docstring.

    The 2D label of a pixel is the 3D label of the point that seeded it, and
    only seeding points are scorable, so the two agree by identity. Anything
    below 1.0 would mean the z-buffer winner used by eval/io_formats.py and the
    scatter seed used by predict() had drifted apart, which is a real bug that
    nothing else here would catch. Reading this number as a result would be
    reporting a tautology as a finding.
    """
    acc = Accumulator(method=Geom3dBaseline.name, split="score")
    acc.add(scored_frame, prediction, nominal_extrinsic())

    counts = acc.consistency
    assert counts["scorable"] > 0, "nothing was scorable, so the identity is untested"
    assert counts["matched"] == counts["scorable"]


def test_predict_before_fit_raises():
    frame = FixtureDataset(root=None, n_frames=1, seed=FIXTURE_SEED).load("fixture_0000")
    with pytest.raises(RuntimeError, match="before fit"):
        Geom3dBaseline().predict(frame, nominal_extrinsic())


def test_tiny_cloud_declines(fitted, scored_frame):
    """The modality-dropout arm of the sweep (interface spec section 8) hands
    out a gutted cloud on purpose, and semseg/types.py's validate() allows a
    point count of zero for exactly that reason. A cloud with no plane in it
    has no geometric features, so declining is the answer and crashing is not.
    """
    gutted = dataclasses.replace(
        scored_frame,
        points=scored_frame.points[:TINY_CLOUD_POINTS],
        intensity=scored_frame.intensity[:TINY_CLOUD_POINTS],
        point_times=scored_frame.point_times[:TINY_CLOUD_POINTS],
        labels_3d_gt=scored_frame.labels_3d_gt[:TINY_CLOUD_POINTS])

    pred = fitted.predict(gutted, nominal_extrinsic())

    assert np.all(pred.labels_3d == UNLABELED)
    assert np.all(pred.labels_2d == UNLABELED)
