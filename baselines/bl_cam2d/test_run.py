"""Unit tests for bl_cam2d.

Three of these are the point of the file and the rest hold the edges.

  - the camera-only arm must beat the chance floor in 2D, or the ablation has
    no lower arm to compare against.
  - it must decline, not guess, on the points it cannot see.
  - a 2 degree extrinsic perturbation must move its 3D labels while leaving
    its 2D labels byte identical. That asymmetry is the mirror image of
    bl_geom3d, whose 2D output is a projection of its 3D labels and so moves
    the other way round, and together the two bracket what the decalibration
    sweep of problem statement section 6.3 shows: a perturbation hurts
    exactly the modality that had to be projected to produce it.

Every geometric assertion here has a perturbation twin, because an equality
test that cannot fail proves nothing. The extrinsic and the intrinsics are
each nudged and each nudge is asserted to move the 3D labels.
"""

import dataclasses

import numpy as np
import pytest

from baselines.bl_cam2d import baseline as cam2d
from baselines.bl_cam2d.baseline import Cam2dBaseline, FIT_PIXELS_PER_FRAME
from baselines.common.features import PIXEL_FEATURE_NAMES
from eval.metrics import confusion, miou
from semseg.datasets.fixture import FixtureDataset
from semseg.projection import perturb_extrinsic, project
from semseg.types import NUM_CLASSES, UNLABELED

N_FIXTURE_FRAMES = 4
N_FIT_FRAMES = 3

# Absolute floor for the 2D mIoU on the held-out fixture frame. Measured at
# 1.0 there, because the fixture's per-class colours are separable and the
# camera-only arm is the arm the fixture is easiest for, so this is a wide
# margin and not a tuned threshold. It exists because the two prior
# comparisons below sit near 0.05: beating them is true but nearly
# unfalsifiable, and an assertion with no teeth is worse than no assertion.
MIN_MIOU_2D = 0.5

# Seed for the uniform chance floor, so the floor is the same number on every
# machine and the comparison is reproducible.
PRIOR_SEED = 0

# The section 6.3 decalibration sweep's largest rotation magnitude. Using the
# extreme of the published sweep rather than a smaller one keeps the test
# about the asymmetry and not about sensitivity.
PERTURB_ROT_DEG = 2.0

# Principal point nudge, in pixels, for the intrinsics perturbation twin.
# Roughly the same image-space displacement as PERTURB_ROT_DEG at the
# fixture's 160 px focal length.
PERTURB_CX_PX = 6.0

# Uniform brightness offset for the 2D perturbation case. Four sigma of the
# fixture's IMAGE_NOISE_SIGMA, so it is small next to the gaps between the
# class base colours and still crosses a Naive Bayes decision boundary
# somewhere. The dropout sweep's over-exposed camera mode is this, larger.
IMAGE_NUDGE = 25

# A perturbation only counts as observed if the output actually moved, and one
# flipped pixel out of 76800 would be indistinguishable from a rounding
# artefact.
MIN_CHANGED_FRACTION = 1e-3


@pytest.fixture(scope="module")
def fitted():
    """-> (baseline, held-out frame, its exact extrinsic).

    Module scoped because fitting is the slow part and none of the tests
    mutate the fitted classifier. The held-out frame is the last one and is
    never in the fit set, which is spec section 9's rule applied in miniature:
    nothing is ever fitted on the frame it is scored on.
    """
    dataset = FixtureDataset(root=None, n_frames=N_FIXTURE_FRAMES)
    frame_ids = dataset.frame_ids()

    fit_ids, score_id = frame_ids[:N_FIT_FRAMES], frame_ids[N_FIT_FRAMES]

    model = Cam2dBaseline()
    model.fit((dataset.load(frame_id) for frame_id in fit_ids), dataset.extrinsic(fit_ids[0]))

    return model, dataset.load(score_id), dataset.extrinsic(score_id)


def prior_miou_2d(labels_2d_gt, mode: str) -> float:
    """The chance floor bl_prior defines, recomputed here.

    Recomputed rather than imported on purpose. The constant-class variant
    used here picks the argmax of THIS frame's own ground truth histogram,
    which is strictly stronger than bl_prior --mode majority can be, since
    bl_prior's majority class is a declared constant and is never fitted on
    the split it is scored on. Beating this therefore implies beating
    bl_prior, without this test file depending on a sibling module.
    """
    if mode == "majority":
        best = int(np.bincount(labels_2d_gt.reshape(-1), minlength=NUM_CLASSES).argmax())
        pred = np.full_like(labels_2d_gt, best)
    elif mode == "uniform":
        rng = np.random.default_rng(PRIOR_SEED)
        pred = rng.integers(0, NUM_CLASSES, size=labels_2d_gt.shape).astype(np.uint8)
    else:
        raise ValueError(f"unknown prior mode {mode!r}")

    return miou(confusion(labels_2d_gt, pred))


def test_2d_beats_the_chance_floor(fitted):
    model, frame, T_cam_lidar = fitted
    pred = model.predict(frame, T_cam_lidar)

    measured = miou(confusion(frame.labels_2d_gt, pred.labels_2d))

    assert measured > prior_miou_2d(frame.labels_2d_gt, "majority")
    assert measured > prior_miou_2d(frame.labels_2d_gt, "uniform")
    assert measured > MIN_MIOU_2D

    # the accumulator folds conf_2d into the calibration counts, and
    # metrics.ece_accumulate raises on a confidence outside [0, 1] or on a nan
    # rather than clamping it, so the range is part of the contract
    assert pred.conf_2d.shape == pred.labels_2d.shape
    assert np.isfinite(pred.conf_2d).all()
    assert pred.conf_2d.min() >= 0.0 and pred.conf_2d.max() <= 1.0


def test_2d_never_reads_the_cloud(fitted):
    """The invariant that makes this the camera-only arm.

    Handing predict() a cloud of noise and then an empty cloud must leave the
    2D map byte identical. A 2D path that peeked at frame.points for a height
    or horizon prior would confound the section 6.2 comparison, and the peek
    would not show up anywhere in the results.
    """
    model, frame, T_cam_lidar = fitted
    reference = model.predict(frame, T_cam_lidar).labels_2d

    rng = np.random.default_rng(PRIOR_SEED)
    garbage = dataclasses.replace(
        frame,
        points=rng.normal(0.0, 20.0, size=frame.points.shape).astype(np.float32),
        labels_3d_gt=None)

    empty = dataclasses.replace(
        frame,
        points=np.zeros((0, 3), dtype=np.float32),
        intensity=np.zeros(0, dtype=np.float32),
        point_times=np.zeros(0, dtype=np.float32),
        labels_3d_gt=None)

    assert np.array_equal(model.predict(garbage, T_cam_lidar).labels_2d, reference)
    assert np.array_equal(model.predict(empty, T_cam_lidar).labels_2d, reference)


def test_out_of_frustum_stays_unlabeled(fitted):
    """Assumption A2, asserted rather than assumed.

    Most of the cloud has no pixel, so most of the 3D output must be
    UNLABELED. The second assertion is what keeps the first honest: a
    baseline that declined everything would satisfy it trivially.
    """
    model, frame, T_cam_lidar = fitted
    pred = model.predict(frame, T_cam_lidar)

    height, width = frame.image.shape[:2]
    _uv, _depth, in_frustum = project(frame.points, frame.K, T_cam_lidar, width, height)

    assert in_frustum.any() and not in_frustum.all()
    assert (pred.labels_3d[~in_frustum] == UNLABELED).all()
    assert (pred.labels_3d[in_frustum] != UNLABELED).any()


def test_declines_3d_without_extrinsic(fitted):
    """No calibration means no 3D answer at all, which is the real GOOSE case
    (spec 13.3). The 2D half must still be produced, because it is the half
    that needs no extrinsic."""
    model, frame, _T_cam_lidar = fitted
    pred = model.predict(frame, None)

    assert (pred.labels_3d == UNLABELED).all()
    assert (pred.labels_2d != UNLABELED).any()


def test_empty_cloud_predicts_empty(fitted):
    """The dropout sweep hands baselines a near-empty cloud on purpose
    (semseg.types.validate), so zero points must be a shape and not a crash."""
    model, frame, T_cam_lidar = fitted

    empty = dataclasses.replace(
        frame,
        points=np.zeros((0, 3), dtype=np.float32),
        intensity=np.zeros(0, dtype=np.float32),
        point_times=np.zeros(0, dtype=np.float32),
        labels_3d_gt=None)
    pred = model.predict(empty, T_cam_lidar)

    assert pred.labels_3d.shape == (0,)
    assert pred.conf_3d.shape == (0,)


def _moved_labels_3d(before, after) -> int:
    """Count of points whose 3D label changed, over the points BOTH runs
    labelled.

    Masking to the intersection is what makes the perturbation test mean
    something. A perturbation also moves the frustum boundary, so points flip
    into and out of UNLABELED for a reason that has nothing to do with the
    labels having moved. Comparing the raw arrays would report a difference
    from that churn alone and the test would pass without ever observing the
    effect it claims.
    """
    both = (before != UNLABELED) & (after != UNLABELED)
    return int((before[both] != after[both]).sum())


def test_pitch_perturbation_moves_3d_only(fitted):
    """The headline asymmetry, and the perturbation case for the 2D-to-3D
    resample.

    A camera-only arm's 2D map is computed before any projection happens, so
    decalibration cannot touch it. Its 3D labels are that same map resampled
    THROUGH the projection, so decalibration moves them. bl_geom3d is the
    mirror image and bl_paint is hurt in both modalities, which is what the
    section 6.3 sweep separates.
    """
    model, frame, T_cam_lidar = fitted
    nominal = model.predict(frame, T_cam_lidar)

    perturbed_T = perturb_extrinsic(T_cam_lidar, "pitch", PERTURB_ROT_DEG, 0.0)
    perturbed = model.predict(frame, perturbed_T)

    assert np.array_equal(perturbed.labels_2d, nominal.labels_2d)
    assert _moved_labels_3d(nominal.labels_3d, perturbed.labels_3d) > 0


def test_intrinsics_perturbation_moves_3d(fitted):
    """The second perturbation twin for the same transform.

    Nudging the principal point moves where a point lands in the image without
    touching the extrinsic, so it proves the resample reads frame.K per call
    rather than having cached a uv table from an earlier one.
    """
    model, frame, T_cam_lidar = fitted
    nominal = model.predict(frame, T_cam_lidar)

    shifted_K = frame.K.copy()
    shifted_K[0, 2] += PERTURB_CX_PX
    shifted = model.predict(dataclasses.replace(frame, K=shifted_K), T_cam_lidar)

    assert _moved_labels_3d(nominal.labels_3d, shifted.labels_3d) > 0


def test_image_nudge_moves_2d(fitted):
    """The perturbation case for the camera path itself.

    The three tests above assert 2D equality under geometric perturbation, and
    that equality would also hold if the 2D map were a constant. Brightening
    the image has to move it.
    """
    model, frame, T_cam_lidar = fitted
    nominal = model.predict(frame, T_cam_lidar).labels_2d

    brighter = np.clip(frame.image.astype(np.int16) + IMAGE_NUDGE, 0, 255).astype(np.uint8)
    nudged = model.predict(dataclasses.replace(frame, image=brighter), T_cam_lidar).labels_2d

    assert float((nudged != nominal).mean()) > MIN_CHANGED_FRACTION


def test_predict_before_fit_raises(fitted):
    """runner.run skips fit() when the fit split resolved to zero frames, and
    the failure has to name the baseline rather than surfacing as a TypeError
    from inside the classifier."""
    _model, frame, T_cam_lidar = fitted

    with pytest.raises(RuntimeError, match="before fit"):
        Cam2dBaseline().predict(frame, T_cam_lidar)


class _RecordingNB:
    """Stands in for nb.GaussianNB just long enough to record the shape of the
    matrix fit() hands it. Confined to this test file; nothing in the shipped
    module is stubbed."""

    last_shape = None

    def fit(self, X, y):
        _RecordingNB.last_shape = X.shape
        assert y.shape[0] == X.shape[0]
        return self


def test_fit_subsample_is_capped(monkeypatch):
    """FIT_PIXELS_PER_FRAME has to actually bind, per frame.

    Without the cap a GOOSE fit split accumulates hundreds of millions of
    feature rows before the classifier ever runs. The comparison against the
    full pixel count is what proves the cap did something on this fixture
    rather than being larger than the frame.
    """
    monkeypatch.setattr(cam2d, "GaussianNB", _RecordingNB)

    dataset = FixtureDataset(root=None, n_frames=N_FIXTURE_FRAMES)
    fit_ids = dataset.frame_ids()[:N_FIT_FRAMES]
    frames = [dataset.load(frame_id) for frame_id in fit_ids]

    Cam2dBaseline().fit(iter(frames), dataset.extrinsic(fit_ids[0]))

    all_pixels = sum(frame.labels_2d_gt.size for frame in frames)
    rows, columns = _RecordingNB.last_shape

    assert rows == N_FIT_FRAMES * FIT_PIXELS_PER_FRAME
    assert rows < all_pixels
    assert columns == len(PIXEL_FEATURE_NAMES)


def test_fit_without_2d_labels_raises():
    """A camera-only arm fitted on frames with no 2D ground truth has nothing
    to fit, and saying so beats fitting on an empty matrix."""
    dataset = FixtureDataset(root=None, n_frames=1)
    frame_id = dataset.frame_ids()[0]
    unlabelled = dataclasses.replace(dataset.load(frame_id), labels_2d_gt=None)

    with pytest.raises(ValueError, match="no labelled 2D pixels"):
        Cam2dBaseline().fit([unlabelled], dataset.extrinsic(frame_id))
