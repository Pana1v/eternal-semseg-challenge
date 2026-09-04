"""Unit tests for the chance floor.

bl_prior computes no metric and applies no geometric transform, so the
mandated perturbation cases land on the only two inputs its output depends on,
the seed and the frame id. Both are tested in both directions: a one-step
change in either must move the uniform draw, and the extrinsic must NOT move
it, because ignoring the extrinsic is the documented behaviour rather than an
oversight. A test that only checked shapes would pass on a baseline that
returned the same array for every frame.
"""

import numpy as np
import pytest

from baselines.bl_prior.baseline import (
    MAJORITY_CLASS, MAJORITY_CONFIDENCE, MAJORITY_MODE, UNIFORM_CONFIDENCE,
    UNIFORM_MODE, PriorBaseline, frame_seed,
)
from baselines.bl_prior.run import main
from baselines.common import base
from eval.io_formats import load_submission
from semseg.types import NUM_CLASSES, UNLABELED, Frame, validate

# Big enough that a uniform draw's per-class counts are tight around N/9:
# 60000 samples give a per-class standard deviation near 75 against an
# expectation of 6667, so a 5 percent band is roughly four sigma wide and the
# balance assertion below is not a coin flip.
TEST_HEIGHT = 200
TEST_WIDTH = 300
TEST_N_POINTS = 60000

BALANCE_REL_TOL = 0.05
BALANCE_RATIO_MAX = 1.1

# A frame id and the same id with one character changed, which is the smallest
# perturbation the per-frame seed derivation can be asked to respond to.
FRAME_ID = "seq_0007"
FRAME_ID_NEIGHBOUR = "seq_0008"


def make_frame(frame_id=FRAME_ID, height=TEST_HEIGHT, width=TEST_WIDTH,
               n_points=TEST_N_POINTS) -> Frame:
    """A minimal valid Frame. No ground truth: nothing in this module reads
    it, and predict() must not either."""
    rng = np.random.default_rng(0)
    frame = Frame(
        frame_id=frame_id,
        image=rng.integers(256, size=(height, width, 3), dtype=np.uint8),
        points=rng.normal(0.0, 10.0, size=(n_points, 3)).astype(np.float32),
        intensity=rng.uniform(0.0, 254.0, size=n_points).astype(np.float32),
        K=np.array([[500.0, 0.0, width / 2], [0.0, 500.0, height / 2], [0.0, 0.0, 1.0]]),
    )
    validate(frame)
    return frame


def identity_extrinsic() -> np.ndarray:
    return np.eye(4)


@pytest.mark.parametrize("mode", [UNIFORM_MODE, MAJORITY_MODE])
def test_shapes_and_dtypes(mode):
    frame = make_frame()
    pred = PriorBaseline(mode=mode).predict(frame, identity_extrinsic())

    assert pred.labels_2d.shape == (TEST_HEIGHT, TEST_WIDTH)
    assert pred.labels_3d.shape == (TEST_N_POINTS,)
    assert pred.labels_2d.dtype == np.uint8
    assert pred.labels_3d.dtype == np.uint8

    assert pred.conf_2d.shape == (TEST_HEIGHT, TEST_WIDTH)
    assert pred.conf_3d.shape == (TEST_N_POINTS,)
    assert pred.conf_2d.dtype == np.float32
    assert pred.conf_3d.dtype == np.float32


@pytest.mark.parametrize("mode", [UNIFORM_MODE, MAJORITY_MODE])
def test_fills_both_modalities(mode):
    """Both modes answer everywhere. A floor that declined anywhere would be
    scored on a subset it selected for itself."""
    frame = make_frame()
    pred = PriorBaseline(mode=mode).predict(frame, identity_extrinsic())

    assert not (pred.labels_2d == UNLABELED).any()
    assert not (pred.labels_3d == UNLABELED).any()
    assert pred.labels_2d.max() < NUM_CLASSES
    assert pred.labels_3d.max() < NUM_CLASSES


def test_uniform_confidence_is_one_over_nine():
    """Compared with allclose, not equality: float32(1/9) is not float64(1/9)
    and an exact test would fail for the wrong reason."""
    pred = PriorBaseline(mode=UNIFORM_MODE).predict(make_frame(), identity_extrinsic())

    assert np.allclose(pred.conf_2d, UNIFORM_CONFIDENCE)
    assert np.allclose(pred.conf_3d, UNIFORM_CONFIDENCE)


def test_majority_confidence_is_declared_constant():
    pred = PriorBaseline(mode=MAJORITY_MODE).predict(make_frame(), identity_extrinsic())

    assert np.allclose(pred.conf_2d, MAJORITY_CONFIDENCE)
    assert np.allclose(pred.conf_3d, MAJORITY_CONFIDENCE)


def test_uniform_counts_roughly_balanced():
    frame = make_frame()
    pred = PriorBaseline(mode=UNIFORM_MODE, seed=0).predict(frame, identity_extrinsic())

    for labels, total in ((pred.labels_2d, TEST_HEIGHT * TEST_WIDTH),
                          (pred.labels_3d, TEST_N_POINTS)):
        counts = np.bincount(labels.ravel(), minlength=NUM_CLASSES)
        expected = total / NUM_CLASSES

        assert counts.size == NUM_CLASSES
        assert (counts > 0).all()
        assert np.abs(counts - expected).max() < BALANCE_REL_TOL * expected
        assert counts.max() / counts.min() < BALANCE_RATIO_MAX


def test_majority_emits_exactly_one_class():
    frame = make_frame()
    pred = PriorBaseline(mode=MAJORITY_MODE).predict(frame, identity_extrinsic())

    assert np.array_equal(np.unique(pred.labels_2d), [MAJORITY_CLASS])
    assert np.array_equal(np.unique(pred.labels_3d), [MAJORITY_CLASS])


def test_same_seed_is_deterministic():
    frame = make_frame()
    first = PriorBaseline(mode=UNIFORM_MODE, seed=7).predict(frame, identity_extrinsic())
    second = PriorBaseline(mode=UNIFORM_MODE, seed=7).predict(frame, identity_extrinsic())

    assert np.array_equal(first.labels_2d, second.labels_2d)
    assert np.array_equal(first.labels_3d, second.labels_3d)


def test_seed_perturbation_moves_uniform():
    """PERTURBATION: one step in the seed has to change the draw, or the
    determinism test above is passing on a constant."""
    frame = make_frame()
    base_pred = PriorBaseline(mode=UNIFORM_MODE, seed=0).predict(frame, identity_extrinsic())
    moved = PriorBaseline(mode=UNIFORM_MODE, seed=1).predict(frame, identity_extrinsic())

    assert not np.array_equal(base_pred.labels_2d, moved.labels_2d)
    assert not np.array_equal(base_pred.labels_3d, moved.labels_3d)


def test_seed_does_not_move_majority():
    """The other half of the same property: majority is a declaration, so the
    seed is not allowed to touch it. A seed spread reported over this mode
    would be a spread of exactly zero, and that is correct."""
    frame = make_frame()
    first = PriorBaseline(mode=MAJORITY_MODE, seed=0).predict(frame, identity_extrinsic())
    second = PriorBaseline(mode=MAJORITY_MODE, seed=1).predict(frame, identity_extrinsic())

    assert np.array_equal(first.labels_2d, second.labels_2d)


def test_frame_id_perturbation_moves_uniform():
    """PERTURBATION: two frames of identical shape must not receive the same
    labels, which is what a single process-wide RNG drawn once would do."""
    prior = PriorBaseline(mode=UNIFORM_MODE, seed=0)
    first = prior.predict(make_frame(frame_id=FRAME_ID), identity_extrinsic())
    second = prior.predict(make_frame(frame_id=FRAME_ID_NEIGHBOUR), identity_extrinsic())

    assert not np.array_equal(first.labels_2d, second.labels_2d)
    assert not np.array_equal(first.labels_3d, second.labels_3d)


def test_frame_seed_responds_to_both_inputs():
    """PERTURBATION on the mixer itself, so a later change to
    FRAME_SEED_HEX_CHARS that collapsed it cannot pass silently."""
    assert frame_seed(0, FRAME_ID) != frame_seed(0, FRAME_ID_NEIGHBOUR)
    assert frame_seed(0, FRAME_ID) != frame_seed(1, FRAME_ID)
    assert frame_seed(0, FRAME_ID) == frame_seed(0, FRAME_ID)


def test_call_order_does_not_change_a_frame():
    """A member RNG advanced per call would make a frame's labels depend on
    how many frames preceded it, so reordering the split would relabel all of
    them, and under --jobs > 1 the labels would follow the worker assignment.
    This is the property the per-frame seed derivation exists for."""
    frame_a = make_frame(frame_id=FRAME_ID)
    frame_b = make_frame(frame_id=FRAME_ID_NEIGHBOUR)

    forward = PriorBaseline(mode=UNIFORM_MODE, seed=0)
    a_first = forward.predict(frame_a, identity_extrinsic()).labels_3d

    backward = PriorBaseline(mode=UNIFORM_MODE, seed=0)
    backward.predict(frame_b, identity_extrinsic())
    a_second = backward.predict(frame_a, identity_extrinsic()).labels_3d

    assert np.array_equal(a_first, a_second)


def test_extrinsic_is_ignored():
    """The one place a flat response is the requirement. A 2 degree yaw error
    and a 10 cm shift are the far ends of the decalibration sweep, and
    bl_prior's curve is flat across them by construction: it is the reference
    line the other arms are read against."""
    frame = make_frame()
    prior = PriorBaseline(mode=UNIFORM_MODE, seed=0)

    angle = np.radians(2.0)
    perturbed = np.eye(4)
    perturbed[:3, :3] = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                                  [np.sin(angle), np.cos(angle), 0.0],
                                  [0.0, 0.0, 1.0]])
    perturbed[:3, 3] = [0.1, 0.0, 0.0]

    nominal = prior.predict(frame, identity_extrinsic())
    shifted = prior.predict(frame, perturbed)
    uncalibrated = prior.predict(frame, None)

    assert np.array_equal(nominal.labels_3d, shifted.labels_3d)
    assert np.array_equal(nominal.labels_3d, uncalibrated.labels_3d)


def test_empty_cloud_is_allowed():
    """types.validate permits a zero point cloud on purpose, because the
    dropout sweep hands one over."""
    frame = make_frame(n_points=0)
    pred = PriorBaseline(mode=UNIFORM_MODE).predict(frame, identity_extrinsic())

    assert pred.labels_3d.shape == (0,)
    assert pred.conf_3d.shape == (0,)
    assert pred.labels_2d.shape == (TEST_HEIGHT, TEST_WIDTH)


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="unknown --mode"):
        PriorBaseline(mode="argmax")


def test_registered_under_its_directory_name():
    # the registry is already warm here, because importing this module
    # imported baseline.py and ran @register. This checks the name and the
    # lookup path, not a cold start.
    assert base.load("bl_prior") is PriorBaseline
    assert PriorBaseline.name == "bl_prior"


@pytest.mark.parametrize("mode", [UNIFORM_MODE, MAJORITY_MODE])
def test_main_writes_a_valid_submission(tmp_path, mode):
    """End to end through the shared driver on one fixture frame. This is what
    proves the registry wiring, the empty fit split and the Accumulator fold,
    none of which the unit tests above touch."""
    out_path = tmp_path / "submission.json"
    main(["--dataset", "fixture", "--split", "score", "--mode", mode,
          "--out", str(out_path), "--limit", "1"])

    payload = load_submission(str(out_path))

    assert payload["method"] == "bl_prior"
    assert payload["n_frames"] == 1
    assert payload["num_classes"] == NUM_CLASSES
    assert (tmp_path / "submission.json.meta.json").exists()

    # a floor that folded an all-zero matrix would score nan and look like a
    # missing run rather than a chance run
    assert payload["conf_3d"].sum() > 0
