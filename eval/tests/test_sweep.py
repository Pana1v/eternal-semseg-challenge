"""Tests for eval/sweep.py.

The sweep harness has one output a reader will quote out of context, the
crossover magnitude, and one guarantee nothing downstream can re-check, that
the perturbation was applied at inference time only. Those two carry the
weight here.

`crossover` is tested against hand-built curves rather than measured ones. The
whole point of the function is that the answer is arithmetic on a curve, so a
curve whose crossing is known by construction is the only test that can
separate a correct interpolation from a plausible one. The cases are chosen so
that the two which report NO magnitude are opposite findings, and the test
asserts their messages differ: "fusion survived the whole grid" and "fusion was
already losing at the smallest step" collapsing into one string would be the
single most misleading thing this module could do.

The fit-time guarantee is tested with a recording stub baseline, which is why
`run_row` takes a baseline class. The stub records both the extrinsic AND the
cloud it was fitted on, because the two arrive by different routes, the
extrinsic_fn's `perturbed` keyword and PerturbedDataset keying on the split,
and asserting only one leaves the other's route untested.

The decalib smoke run is a module-scoped fixture. It re-predicts a whole split
once per magnitude per arm, so running it once and asserting several things
about the result costs a fraction of what one sweep per test would.
"""

import csv
import json
from dataclasses import replace

import numpy as np
import pytest

from eval import sweep
from eval.report import crossover_entries, verdict_sentence
from eval.sweep import (
    AXIS_LIDAR_DROPOUT, AXIS_DESKEW, AXIS_TIME_OFFSET, CSV_COLUMNS,
    DESKEW_T_REF_S, FUSED_BASELINE, IMAGE_BLACK, IMAGE_SATURATED,
    LIDAR_ONLY_BASELINE, REFUSED_EXIT, RowSpec, SWEEP_DECALIB, SWEEP_DESKEW,
    SWEEP_DROPOUT, SWEEP_SEEDS, SWEEP_TIME_OFFSET, UNIT_DEG, UNIT_ENABLED,
    UNIT_FRACTION, UNIT_MS, crossover, decalib_grid, perturb_frame,
)
from semseg.datasets import split_frames
from semseg.datasets.fixture import (
    FixtureConfig, FixtureDataset, nominal_extrinsic,
)
from semseg.projection import project, zbuffer
from semseg.types import Prediction, UNLABELED

# A grid whose bracketing pair is obvious by eye, so a reader can check the
# expected crossovers below without running anything.
MAGNITUDES = (0.1, 0.5, 1.0, 2.0)

# The level every hand-built curve is compared against.
REFERENCE = 0.70

# Two frames is what the smoke run scores, matching run_all.sh's SWEEP_FRAMES.
SMOKE_FRAMES = 2

# The trimmed magnitude list the smoke run sweeps. Two rotations so a curve can
# interpolate, one translation so the second unit is exercised without doubling
# the runtime.
SMOKE_ROT_DEG = (0.5, 2.0)
SMOKE_TRANS_CM = (10.0,)

# Enough frames that split_frames puts several on each side. The fixture's
# split is md5 of the frame id, so these counts are fixed rather than lucky.
FIXTURE_FRAMES = 6

# The ego speed the deskew control runs at. At the fixture's default 1.0 m/s a
# mid-scan point moves about 5 cm, which is under a pixel at the fixture's
# intrinsics and so under this metric's noise floor: measured there, the deskew
# arm moves 3D mIoU by 0.002 in either direction depending on the frame.
# Amplifying the speed is what turns the ablation into a signal worth
# asserting. See test_deskew_sign_is_correct.
DESKEW_CONTROL_SPEED_MPS = 10.0

# Tolerance on the dropout fraction actually retained. Each point is dropped by
# an independent draw, so the kept share is binomial around 1 - fraction and
# not exact.
DROPOUT_SHARE_TOL = 0.02


class RecordingBaseline:
    """Records what fit() and predict() were handed, and predicts UNLABELED.

    Not registered, so there is no registry key to collide with the four real
    baselines: the test injects the class into run_row directly.

    The records are class level because run_row constructs its own instance,
    so the test cannot hold a reference to the object that gets fitted.
    """

    name = "bl_recording"

    fit_extrinsics = []
    fit_clouds = []
    predict_extrinsics = []

    @classmethod
    def reset(cls):
        cls.fit_extrinsics = []
        cls.fit_clouds = []
        cls.predict_extrinsics = []

    def fit(self, frames, T_cam_lidar):
        RecordingBaseline.fit_extrinsics.append(np.array(T_cam_lidar, dtype=np.float64))
        for frame in frames:
            RecordingBaseline.fit_clouds.append(np.array(frame.points))

    def predict(self, frame, T_cam_lidar):
        RecordingBaseline.predict_extrinsics.append(np.array(T_cam_lidar, dtype=np.float64))
        return Prediction(
            labels_2d=np.full(frame.image.shape[:2], UNLABELED, dtype=np.uint8),
            labels_3d=np.full(frame.points.shape[0], UNLABELED, dtype=np.uint8),
        )


class StubDataset:
    """The minimum a refusal check reads. Same shape as the stub dataset in
    baselines/bl_paint/test_run.py, which sets calib_available False on a
    dataset that is never loaded from.

    `load` raises rather than returning something, so any test that reaches a
    frame has proved the refusal did not fire.
    """

    def __init__(self, calib_available=True, poses_available=False):
        self.calib_available = calib_available
        self.poses_available = poses_available

    def frame_ids(self):
        return ["seq__0_0", "seq__1_1", "seq__2_2", "seq__3_3"]

    def load(self, frame_id):
        raise AssertionError("a refused sweep must not load a single frame")

    def extrinsic(self, frame_id):
        return nominal_extrinsic()


@pytest.fixture
def dataset():
    return FixtureDataset(n_frames=FIXTURE_FRAMES, seed=0)


@pytest.fixture
def frame(dataset):
    return dataset.load(dataset.frame_ids()[0])


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    """One trimmed decalib sweep over two fixture frames and both arms.

    Module scoped on purpose: this is the only test here that runs the real
    pipeline end to end, and it is the expensive one.
    """
    out_dir = tmp_path_factory.mktemp("decalib")
    argv = ["--sweep", SWEEP_DECALIB, "--dataset", "fixture", "--split", "score",
            "--rot-deg", *[str(v) for v in SMOKE_ROT_DEG],
            "--trans-cm", *[str(v) for v in SMOKE_TRANS_CM],
            "--limit", str(SMOKE_FRAMES), "--out-dir", str(out_dir),
            "--baseline", FUSED_BASELINE, "--baseline", LIDAR_ONLY_BASELINE]

    code = sweep.main(argv)
    stem = out_dir / SWEEP_DECALIB

    with open(str(stem) + ".json") as f:
        payload = json.load(f)

    return {"code": code, "dir": out_dir, "stem": stem,
            "rows": _read_csv(str(stem) + ".csv"), "payload": payload}


# ---------------------------------------------------------------------------
# crossover, the headline deliverable
# ---------------------------------------------------------------------------

def test_crossover_between_two_swept_points():
    """The ordinary case. The curve passes the reference between 0.5 and 1.0,
    and the answer is the interpolation, not either bracket."""
    verdict = crossover(MAGNITUDES, [0.90, 0.80, 0.60, 0.40], REFERENCE, unit=UNIT_DEG)

    assert verdict.crossed
    # 0.80 down to 0.60 across 0.5 to 1.0, and the reference 0.70 sits halfway
    assert verdict.magnitude == pytest.approx(0.75)
    assert verdict.reason is None


def test_crossover_exactly_on_a_swept_point():
    """A curve that meets the reference exactly on a grid point must report
    that point and not an interpolation a hair either side of it.

    The general formula gets this right because the weight comes out exactly 1,
    which is why there is no special case in the function to test separately.
    """
    verdict = crossover(MAGNITUDES, [0.90, 0.80, 0.70, 0.40], REFERENCE, unit=UNIT_DEG)

    assert verdict.crossed
    assert verdict.magnitude == pytest.approx(1.0)


def test_crossover_never_crosses():
    verdict = crossover(MAGNITUDES, [0.90, 0.88, 0.86, 0.85], REFERENCE, unit=UNIT_DEG)

    assert not verdict.crossed
    assert verdict.magnitude is None
    assert "never falls below" in verdict.reason
    # The largest swept magnitude is the number a reader needs: this finding is
    # bounded by the grid, not by the method.
    assert "2 deg" in verdict.reason


def test_crossover_already_below_at_the_smallest_magnitude():
    verdict = crossover(MAGNITUDES, [0.60, 0.50, 0.40, 0.30], REFERENCE, unit=UNIT_DEG)

    assert not verdict.crossed
    assert verdict.magnitude is None
    assert "already below" in verdict.reason
    assert "0.1 deg" in verdict.reason


def test_the_two_no_magnitude_messages_differ():
    """The load-bearing assertion of this file.

    "fusion never fell below within the swept range" and "fusion was already
    below at the smallest perturbation" are OPPOSITE findings: the first says
    the rig tolerance is looser than the grid, the second says it is tighter
    and the grid cannot see it. Reporting both as "no crossover" would tell a
    reader the calibration budget is fine in exactly the case where it is not.
    """
    never = crossover(MAGNITUDES, [0.90, 0.88, 0.86, 0.85], REFERENCE, unit=UNIT_DEG)
    already = crossover(MAGNITUDES, [0.60, 0.50, 0.40, 0.30], REFERENCE, unit=UNIT_DEG)

    assert never.reason != already.reason
    assert not never.crossed and not already.crossed

    # And neither may be mistaken for the other by a substring match, which is
    # how a downstream reader would go wrong.
    assert "already below" not in never.reason
    assert "never falls below" not in already.reason


def test_crossover_takes_the_first_crossing_of_a_noisy_curve():
    """A curve that dips below, recovers and falls again must report the
    SMALLEST perturbation that lost fusion. That is the production tolerance;
    the later crossing is not a weaker requirement."""
    verdict = crossover(MAGNITUDES, [0.90, 0.70, 0.80, 0.40], REFERENCE, unit=UNIT_DEG)

    assert verdict.crossed
    assert verdict.magnitude == pytest.approx(0.5)


def test_crossover_sorts_an_unordered_grid():
    """`--rot-deg 2.0 0.5` is a legal invocation, so "first crossing" has to
    mean smallest magnitude and not first in argument order."""
    ordered = crossover(MAGNITUDES, [0.90, 0.80, 0.60, 0.40], REFERENCE, unit=UNIT_DEG)
    shuffled = crossover([2.0, 0.5, 0.1, 1.0], [0.40, 0.80, 0.90, 0.60],
                         REFERENCE, unit=UNIT_DEG)

    assert shuffled.magnitude == pytest.approx(ordered.magnitude)


def test_crossover_raises_on_a_nan_in_the_curve():
    """metrics.miou returns nan for an empty split, and `nan <= reference` is
    False, so a swallowed nan would read as "never falls below" and publish the
    most reassuring possible conclusion off a failed measurement."""
    with pytest.raises(ValueError, match="nan at magnitude 1"):
        crossover(MAGNITUDES, [0.90, 0.80, float("nan"), 0.40], REFERENCE, unit=UNIT_DEG)


def test_crossover_raises_on_a_nan_reference():
    with pytest.raises(ValueError, match="reference mIoU is nan"):
        crossover(MAGNITUDES, [0.90, 0.80, 0.60, 0.40], float("nan"), unit=UNIT_DEG)


def test_crossover_rejects_mismatched_inputs():
    with pytest.raises(ValueError, match="must pair up"):
        crossover(MAGNITUDES, [0.90, 0.80], REFERENCE, unit=UNIT_DEG)

    with pytest.raises(ValueError, match="at least one swept magnitude"):
        crossover([], [], REFERENCE, unit=UNIT_DEG)


def test_every_no_magnitude_verdict_carries_a_reason():
    """eval/report.py renders a verdict with neither field as "unreadable", and
    a companion JSON with no readable verdict renders as "no crossover verdict
    found", which report.py itself calls a different finding from no crossover.
    So every no-magnitude path must supply a reason."""
    for values in ([0.90, 0.88, 0.86, 0.85], [0.60, 0.50, 0.40, 0.30]):
        verdict = crossover(MAGNITUDES, values, REFERENCE, unit=UNIT_DEG)
        assert verdict.magnitude is None
        assert verdict.reason


# ---------------------------------------------------------------------------
# the inference-time-only guarantee
# ---------------------------------------------------------------------------

def test_fit_sees_the_unperturbed_extrinsic(dataset, tmp_path):
    """Perturbing during fit is an augmentation experiment and a different
    question, so fit() must be handed the nominal rig even on a row whose whole
    purpose is a 2 degree error.

    Asserted through run_row rather than by reading the code: the guarantee is
    a property of how runner.run calls extrinsic_fn, and no test of this
    module's internals would catch a regression there.
    """
    RecordingBaseline.reset()
    fit_ids, score_ids = split_frames(dataset.frame_ids())

    spec = RowSpec(sweep=SWEEP_DECALIB, axis="pitch", magnitude=2.0, unit=UNIT_DEG,
                   rot_deg=2.0)
    sweep.run_row(RecordingBaseline, dataset, fit_ids, score_ids, spec, str(tmp_path),
                  limit=SMOKE_FRAMES)

    assert len(RecordingBaseline.fit_extrinsics) == 1
    np.testing.assert_allclose(RecordingBaseline.fit_extrinsics[0], nominal_extrinsic())

    # The control: predict() really did see a DIFFERENT matrix, so the
    # assertion above is not passing because nothing was perturbed at all.
    assert RecordingBaseline.predict_extrinsics
    for seen in RecordingBaseline.predict_extrinsics:
        assert not np.allclose(seen, nominal_extrinsic())


def test_fit_sees_the_unperturbed_cloud(dataset, tmp_path):
    """The cloud arrives by a different route from the extrinsic: the sweeps
    that perturb it do so through PerturbedDataset, which fit and predict both
    load through. So this needs asserting separately, and it is the route where
    an accidental fit-time perturbation would actually happen.
    """
    RecordingBaseline.reset()
    fit_ids, score_ids = split_frames(dataset.frame_ids())

    # Three quarters of the cloud dropped, which is impossible to miss in a
    # point count.
    spec = RowSpec(sweep=SWEEP_DROPOUT, axis=AXIS_LIDAR_DROPOUT, magnitude=0.75,
                   unit=UNIT_FRACTION, drop_fraction=0.75)
    sweep.run_row(RecordingBaseline, dataset, fit_ids, score_ids, spec, str(tmp_path),
                  limit=SMOKE_FRAMES)

    assert len(RecordingBaseline.fit_clouds) == len(fit_ids)
    for cloud, frame_id in zip(RecordingBaseline.fit_clouds, fit_ids):
        np.testing.assert_array_equal(cloud, dataset.load(frame_id).points)


def test_perturbed_dataset_leaves_fit_frames_alone(dataset):
    """The wrapper's contract, tested directly: a score frame comes back
    perturbed and a fit frame comes back untouched, from the same load()."""
    fit_ids, score_ids = split_frames(dataset.frame_ids())
    spec = RowSpec(sweep=SWEEP_DROPOUT, axis=AXIS_LIDAR_DROPOUT, magnitude=0.75,
                   unit=UNIT_FRACTION, drop_fraction=0.75)

    wrapped = sweep.PerturbedDataset(dataset, fit_ids, score_ids, spec)

    np.testing.assert_array_equal(wrapped.load(fit_ids[0]).points,
                                  dataset.load(fit_ids[0]).points)

    perturbed = wrapped.load(score_ids[0])
    assert perturbed.points.shape[0] < dataset.load(score_ids[0]).points.shape[0]

    # The per-point arrays have to be dropped with the points, or the Frame
    # contract breaks and the accumulator scores the wrong labels.
    assert perturbed.intensity.shape[0] == perturbed.points.shape[0]
    assert perturbed.labels_3d_gt.shape[0] == perturbed.points.shape[0]
    assert perturbed.point_times.shape[0] == perturbed.points.shape[0]


def test_perturbed_dataset_refuses_overlapping_splits(dataset):
    fit_ids, score_ids = split_frames(dataset.frame_ids())
    spec = RowSpec(sweep=SWEEP_DECALIB, axis="pitch", magnitude=1.0, unit=UNIT_DEG)

    with pytest.raises(ValueError, match="both splits"):
        sweep.PerturbedDataset(dataset, fit_ids, fit_ids + score_ids, spec)


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------

def test_time_offset_refuses_a_dataset_without_poses(monkeypatch, capsys, tmp_path):
    """Real GOOSE val ships no poses (spec 13.3), so this path is the normal
    outcome there and not an edge case. Assuming a constant velocity would make
    the ego twist, which IS the independent variable of this sweep, a property
    of the assumption rather than of the data."""
    _patch_dataset(monkeypatch, StubDataset(poses_available=False))

    code = sweep.main(_argv(SWEEP_TIME_OFFSET, tmp_path))

    assert code == REFUSED_EXIT

    message = capsys.readouterr().err
    assert "poses_available=False" in message
    # It has to name the alternative, or an operator is told only that they
    # cannot proceed.
    assert "--dataset fixture" in message


def test_deskew_also_refuses_a_dataset_without_poses(monkeypatch, capsys, tmp_path):
    """deskew() returns the raw cloud when the twist or the timestamps are
    absent, which is right for a dataset shipping neither and exactly wrong for
    the ablation: both arms would be bit-identical and the sweep would report
    "motion compensation makes no difference" while measuring nothing."""
    _patch_dataset(monkeypatch, StubDataset(poses_available=False))

    assert sweep.main(_argv(SWEEP_DESKEW, tmp_path)) == REFUSED_EXIT
    assert "poses_available=False" in capsys.readouterr().err


def test_decalib_does_not_need_poses(monkeypatch, tmp_path):
    """The refusal has to be scoped to the sweeps that need a twist. A blanket
    gate would make decalib, the headline sweep, unrunnable on any dataset
    without poses.

    StubDataset.load raises, so reaching a frame at all proves the refusal did
    not fire. That is the assertion.
    """
    _patch_dataset(monkeypatch, StubDataset(poses_available=False))

    with pytest.raises(AssertionError, match="must not load"):
        sweep.main(_argv(SWEEP_DECALIB, tmp_path))


def test_dropout_does_not_need_poses(monkeypatch, tmp_path):
    _patch_dataset(monkeypatch, StubDataset(poses_available=False))

    with pytest.raises(AssertionError, match="must not load"):
        sweep.main(_argv(SWEEP_DROPOUT, tmp_path))


def test_every_sweep_refuses_a_dataset_with_no_calibration(monkeypatch, capsys, tmp_path):
    """Consistency and coverage are counted THROUGH the projection operator, so
    every sweep here is projection dependent and none can run without a rig.
    Refusing beats emitting a curve measured against a fabricated one."""
    _patch_dataset(monkeypatch, StubDataset(calib_available=False, poses_available=True))

    for name in (SWEEP_DECALIB, SWEEP_TIME_OFFSET, SWEEP_DROPOUT, SWEEP_DESKEW):
        assert sweep.main(_argv(name, tmp_path)) == REFUSED_EXIT
        assert "ships none" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# the decalib smoke run, end to end
# ---------------------------------------------------------------------------

def test_smoke_writes_one_row_per_operating_point_per_arm(smoke):
    """The expected row count comes from the grid ENUMERATION and not from a
    literal, so adding an axis or a seed to the sweep cannot leave this test
    passing against a stale number. The random-axis arm is inside that
    enumeration, which is exactly the part a hand-counted literal would drop.
    """
    assert smoke["code"] == 0

    expected = len(decalib_grid(rot_deg=SMOKE_ROT_DEG, trans_cm=SMOKE_TRANS_CM)) * 2
    assert len(smoke["rows"]) == expected

    assert tuple(smoke["rows"][0].keys()) == CSV_COLUMNS


def test_smoke_fused_curve_moves_and_reference_stays_flat(smoke):
    """A flat fused curve would mean the baseline cached the extrinsic and the
    sweep measured nothing, which baselines/common/base.py calls the dangerous
    failure precisely because nothing crashes.

    The lidar-only arm's 3D output never reads the image, so it cannot depend
    on the extrinsic and its curve must be exactly flat. That is the control
    which makes the fused arm's movement attributable to the perturbation.
    """
    fused = {row["miou_3d"] for row in smoke["rows"]
             if row["baseline"] == FUSED_BASELINE}
    reference = {row["miou_3d"] for row in smoke["rows"]
                 if row["baseline"] == LIDAR_ONLY_BASELINE}

    assert len(fused) > 1
    assert len(reference) == 1


def test_smoke_companion_json_is_report_readable(smoke):
    """eval/report.py reads the verdicts out of the companion JSON by a fixed
    shape, so the smoke run asserts that shape rather than trusting it."""
    payload = smoke["payload"]
    assert payload["crossover"]

    entries = crossover_entries(payload)
    assert len(entries) == len(payload["crossover"])

    for label, verdict in entries:
        # The one sentence report.py emits when it can read neither a magnitude
        # nor a reason. No verdict this module writes may produce it.
        assert "unreadable" not in verdict_sentence(label, verdict)


def test_smoke_emits_a_plot(smoke):
    assert (smoke["dir"] / (SWEEP_DECALIB + sweep.PLOT_SUFFIX)).exists()


def test_smoke_json_rows_carry_no_nan(smoke):
    """JSON has no NaN. The repo's convention (eval/plot_results.py) is that an
    undefined metric arrives as null, so a consumer can json.load the file
    without a non-standard parser."""
    with open(str(smoke["stem"]) + ".json") as f:
        text = f.read()

    assert "NaN" not in text
    assert smoke["payload"]["rows"]


def test_smoke_reports_a_crossover_for_pitch_and_not_for_roll(smoke):
    """The finding the per-axis decomposition exists to produce.

    Roll is about the optical axis and moves a centred point not at all; pitch
    is a lateral shift proportional to range. So the production calibration
    tolerance is set by pitch, and a sweep over one unnamed delta-theta would
    have averaged the two together and reported neither.
    """
    verdicts = smoke["payload"]["crossover"]

    pitch = [v for label, v in verdicts.items() if label.startswith(f"{FUSED_BASELINE} pitch")
             and v["unit"] == UNIT_DEG]
    roll = [v for label, v in verdicts.items() if label.startswith(f"{FUSED_BASELINE} roll")
            and v["unit"] == UNIT_DEG]

    assert len(pitch) == 1 and len(roll) == 1
    assert pitch[0]["crossed"]
    assert not roll[0]["crossed"]

    # Within the swept range, and reported as a real number rather than a
    # bracket, which is what problem statement section 6.3 asks for.
    assert 0.0 < pitch[0]["magnitude"] <= max(SMOKE_ROT_DEG)


def test_smoke_pitch_costs_more_than_roll_at_the_same_magnitude(smoke):
    """The same assertion measured on the rows rather than on the verdict, so a
    regression in the crossover arithmetic cannot make both pass together."""
    largest = max(SMOKE_ROT_DEG)
    at_largest = {row["axis"]: float(row["miou_3d"]) for row in smoke["rows"]
                  if row["baseline"] == FUSED_BASELINE and row["unit"] == UNIT_DEG
                  and float(row["magnitude"]) == largest}

    assert at_largest["pitch"] < at_largest["roll"]


# ---------------------------------------------------------------------------
# the deskew arm's direction
# ---------------------------------------------------------------------------

def test_deskew_sign_is_correct():
    """t_ref and the sign of the correction, verified against the fixture's own
    geometry with a control that can fail.

    The oracle is ground truth on both sides: the fraction of z-buffer-visible
    points whose 3D ground-truth label matches the 2D ground-truth label of the
    pixel they land on. No baseline is involved, so this measures the
    projection geometry and nothing else.

    The control is the NEGATED twist. Compensating the wrong way must make
    agreement worse; without that arm, an assertion that deskew beats the raw
    cloud could pass on a correction that was merely small.
    """
    config = FixtureConfig(ego_speed_mps=DESKEW_CONTROL_SPEED_MPS)
    dataset = FixtureDataset(n_frames=1, seed=0, config=config)
    source = dataset.load(dataset.frame_ids()[0])

    raw = _gt_agreement(source.points, source)

    spec = RowSpec(sweep=SWEEP_DESKEW, axis=AXIS_DESKEW, magnitude=1.0,
                   unit=UNIT_ENABLED, deskew_enabled=True)
    compensated = _gt_agreement(perturb_frame(source, spec).points, source)

    backwards = _gt_agreement(
        sweep.deskew(source.points, source.point_times, -source.ego_twist,
                     DESKEW_T_REF_S),
        source)

    assert compensated > raw
    assert backwards < raw


def test_deskew_reference_time_is_scan_start():
    """The fixture pins t = 0 at scan start and stores points as if the sensor
    had never moved, so t_ref = 0 is what undoes exactly that motion. It is
    also the only value consistent with the image, which is cast from a static
    origin in the scan-start frame."""
    assert DESKEW_T_REF_S == 0.0


def test_deskew_arm_refuses_a_frame_with_no_timestamps(frame):
    """A silent pass-through would make both deskew arms bit-identical and the
    ablation would report a null result it had not measured."""
    spec = RowSpec(sweep=SWEEP_DESKEW, axis=AXIS_DESKEW, magnitude=1.0,
                   unit=UNIT_ENABLED, deskew_enabled=True)

    with pytest.raises(ValueError, match="point_times"):
        perturb_frame(replace(frame, point_times=None), spec)

    with pytest.raises(ValueError, match="ego_twist"):
        perturb_frame(replace(frame, ego_twist=None), spec)


def test_deskew_off_arm_is_the_raw_cloud(frame):
    """The no-deskew arm must be the cloud exactly as the sensor delivered it,
    or the ablation is measuring a difference between two corrections."""
    spec = RowSpec(sweep=SWEEP_DESKEW, axis=AXIS_DESKEW, magnitude=0.0,
                   unit=UNIT_ENABLED)

    np.testing.assert_array_equal(perturb_frame(frame, spec).points, frame.points)


# ---------------------------------------------------------------------------
# the time-offset mechanism
# ---------------------------------------------------------------------------

# A rigid shift is one vector for every point, but Frame.points is float32, so
# the recovered vector varies across the cloud by up to one unit in the last
# place at the frame's largest coordinate. Measured at 2.9e-6 m over a 40 m
# fixture extent; the same algebra in float64 spreads by 1.8e-15.
FLOAT32_RIGID_ATOL_M = 1e-5


def test_time_offset_displaces_the_cloud_and_not_the_extrinsic(frame):
    """A camera-to-lidar timestamp skew means the two sensors observed the
    world at two different instants; the rig geometry was right at both. So it
    must move the points and leave T_cam_lidar exactly alone, or the sweep is
    just a second and less honest decalibration sweep."""
    spec = RowSpec(sweep=SWEEP_TIME_OFFSET, axis=AXIS_TIME_OFFSET, magnitude=200.0,
                   unit=UNIT_MS, speed_mps=2.0, offset_s=0.2)

    assert not np.allclose(perturb_frame(frame, spec).points, frame.points)
    np.testing.assert_array_equal(
        sweep.sweep_extrinsic(nominal_extrinsic(), spec), nominal_extrinsic())


def test_time_offset_displacement_scales_with_ego_speed(frame):
    """The speed column scales the ego TRANSLATION and nothing else.

    Asserted on the difference between two speeds rather than on the ratio of
    their displacements, because the total displacement is NOT proportional to
    speed and expecting it to be would be the wrong test. The yaw rate is
    deliberately left as the rig reported it, so a point is displaced by
    (R - I)p from the rotation plus the ego translation, and the norm of that
    vector sum grows sub-linearly: measured on this frame, 0.112 m at 1 m/s and
    0.206 m at 2 m/s, not 0.225.

    The difference between the two, however, is exactly the extra translation:
    the rotational term is identical in both, so it cancels. That difference
    must be one constant vector, the same for every point, of length
    (v2 - v1) * offset.
    """
    offset_s = 0.1

    def displacement(speed):
        spec = RowSpec(sweep=SWEEP_TIME_OFFSET, axis=AXIS_TIME_OFFSET, magnitude=100.0,
                       unit=UNIT_MS, speed_mps=speed, offset_s=offset_s)
        return perturb_frame(frame, spec).points - frame.points

    extra = displacement(2.0) - displacement(1.0)

    # One rigid translation, so every point moved by the same vector. Broadcast
    # the reference explicitly: assert_allclose compares shapes before values,
    # so an (N, 3) against a (3,) is a shape mismatch rather than the value
    # comparison intended. The tolerance is set by float32 point storage, not by
    # the algebra: in float64 the spread is 1.8e-15, and the float32 round trip
    # widens it to 2.9e-6, which is 0.75 of one ULP at this frame's 40 m extent.
    np.testing.assert_allclose(
        extra, np.broadcast_to(extra[0], extra.shape), atol=FLOAT32_RIGID_ATOL_M)

    expected_m = (2.0 - 1.0) * offset_s
    assert float(np.linalg.norm(extra[0])) == pytest.approx(expected_m, rel=1e-3)


def test_time_offset_displacement_grows_with_ego_speed(frame):
    """Monotone in the speed column, which is the weaker claim the sweep
    actually rests on."""
    def shift(speed):
        spec = RowSpec(sweep=SWEEP_TIME_OFFSET, axis=AXIS_TIME_OFFSET, magnitude=100.0,
                       unit=UNIT_MS, speed_mps=speed, offset_s=0.1)
        moved = perturb_frame(frame, spec)
        return float(np.linalg.norm(moved.points - frame.points, axis=1).mean())

    assert shift(0.3) < shift(1.0) < shift(2.0)


def test_zero_offset_is_an_exact_no_op(frame):
    """0 ms is in the grid as every curve's unperturbed anchor, so it must
    return the cloud bit for bit. An anchor that had drifted would shift the
    whole curve and the crossover with it."""
    spec = RowSpec(sweep=SWEEP_TIME_OFFSET, axis=AXIS_TIME_OFFSET, magnitude=0.0,
                   unit=UNIT_MS, speed_mps=2.0, offset_s=0.0)

    np.testing.assert_array_equal(perturb_frame(frame, spec).points, frame.points)


def test_time_offset_grid_anchors_every_speed_at_zero(dataset):
    """A curve with no unperturbed point cannot be read: a shallow curve and a
    low one look identical without it."""
    by_speed = {}
    for spec in sweep.time_offset_grid():
        by_speed.setdefault(spec.speed_mps, []).append(spec.magnitude)

    assert set(by_speed) == set(sweep.EGO_SPEEDS_MPS)
    for magnitudes in by_speed.values():
        assert 0.0 in magnitudes


# ---------------------------------------------------------------------------
# the dropout arms
# ---------------------------------------------------------------------------

def test_dropout_is_reproducible_across_calls(frame):
    """Seeded from the row seed and an md5 of the frame id, never from Python's
    hash(), which is salted per process and would move the dropout between two
    runs of the same command."""
    spec = RowSpec(sweep=SWEEP_DROPOUT, axis=AXIS_LIDAR_DROPOUT, magnitude=0.5,
                   unit=UNIT_FRACTION, drop_fraction=0.5)

    np.testing.assert_array_equal(perturb_frame(frame, spec).points,
                                  perturb_frame(frame, spec).points)


def test_dropout_seeds_draw_different_subsets(frame):
    def kept(seed):
        spec = RowSpec(sweep=SWEEP_DROPOUT, axis=AXIS_LIDAR_DROPOUT, magnitude=0.5,
                       unit=UNIT_FRACTION, seed=seed, drop_fraction=0.5)
        return perturb_frame(frame, spec).points

    assert kept(0).shape[0] > 0
    assert not np.array_equal(kept(0), kept(1))


def test_dropout_fraction_is_honoured(frame):
    total = frame.points.shape[0]

    for fraction in sweep.POINT_DROPOUT:
        spec = RowSpec(sweep=SWEEP_DROPOUT, axis=AXIS_LIDAR_DROPOUT,
                       magnitude=fraction, unit=UNIT_FRACTION, drop_fraction=fraction)
        kept = perturb_frame(frame, spec).points.shape[0]

        assert kept / total == pytest.approx(1.0 - fraction, abs=DROPOUT_SHARE_TOL)


def test_camera_dropout_modes_are_constant_images(frame):
    """A blacked or saturated camera has stopped carrying information, so every
    pixel reads the same. A scaled image would model a dim camera, which is a
    different failure."""
    for mode, level in ((IMAGE_BLACK, 0), (IMAGE_SATURATED, 255)):
        spec = RowSpec(sweep=SWEEP_DROPOUT, axis="camera_" + mode, magnitude=1.0,
                       unit=UNIT_FRACTION, image_mode=mode)
        image = perturb_frame(frame, spec).image

        assert image.shape == frame.image.shape
        assert image.dtype == frame.image.dtype
        assert np.all(image == level)


def test_dropout_grid_anchors_every_curve_at_zero():
    by_axis = {}
    for spec in sweep.dropout_grid():
        by_axis.setdefault(spec.axis, []).append(spec.magnitude)

    assert set(by_axis) == {AXIS_LIDAR_DROPOUT, "camera_black", "camera_saturated"}
    for magnitudes in by_axis.values():
        assert 0.0 in magnitudes


# ---------------------------------------------------------------------------
# grid and label bookkeeping
# ---------------------------------------------------------------------------

def test_decalib_sweeps_rotation_and_translation_separately():
    """One magnitude column only means something if a row moves one thing. A
    row with both would have no single magnitude to plot against, and a
    rotation tolerance and a translation tolerance are two different production
    requirements."""
    for spec in decalib_grid():
        assert (spec.rot_deg == 0.0) or (spec.trans_m == 0.0)
        assert (spec.rot_deg != 0.0) or (spec.trans_m != 0.0)


def test_decalib_grid_covers_every_axis_and_the_random_arm():
    specs = decalib_grid()

    assert {spec.axis for spec in specs} == {"roll", "pitch", "yaw", "random"}
    assert {spec.seed for spec in specs if spec.axis == "random"} == set(SWEEP_SEEDS)

    # The named axes carry no seed spread: their perturbation is deterministic,
    # so repeating them per seed would be three identical rows.
    assert {spec.seed for spec in specs if spec.axis == "pitch"} == {0}


def test_decalib_grid_uses_the_problem_statement_magnitudes():
    """Problem statement section 6.3 fixes the grid, so it is asserted rather
    than left as a default a later edit could quietly widen."""
    assert sweep.DECALIB_ROT_DEG == (0.1, 0.25, 0.5, 1.0, 2.0)
    assert sweep.DECALIB_TRANS_CM == (1, 2, 5, 10)
    assert sweep.TIME_OFFSETS_MS == (0, 10, 25, 50, 100, 200)
    assert sweep.EGO_SPEEDS_MPS == (0.3, 1.0, 2.0)
    assert sweep.POINT_DROPOUT == (0.25, 0.5, 0.75)


def test_curve_labels_omit_components_that_never_vary():
    """A key component that is the same for every curve cannot disambiguate
    anything, and "pitch (deg) seed 0" reads as though the seed mattered when
    the sweep only used one."""
    labels = sweep.curve_labels([("pitch", UNIT_DEG, 0, None), ("roll", UNIT_DEG, 0, None)])

    assert set(labels.values()) == {"pitch", "roll"}


def test_curve_labels_stay_unique_when_a_component_varies():
    labels = sweep.curve_labels([("random", UNIT_DEG, 0, None),
                                 ("random", UNIT_DEG, 1, None)])

    assert len(set(labels.values())) == 2
    assert all("seed" in label for label in labels.values())


def test_csv_writes_an_empty_cell_for_an_undefined_metric(tmp_path):
    """JSON has no NaN and this repo's convention is that an undefined metric
    arrives as null, so the CSV agrees rather than carrying the string "nan"
    next to a null in the companion file."""
    path = tmp_path / "decalib.csv"
    row = {"sweep": SWEEP_DECALIB, "baseline": FUSED_BASELINE, "axis": "pitch",
           "magnitude": 1.0, "unit": UNIT_DEG, "seed": 0, "speed_mps": None,
           "miou_2d": float("nan"), "miou_3d": 0.5,
           "consistency": float("nan"), "coverage": 0.25}

    sweep.write_csv(str(path), [row])
    written = _read_csv(str(path))[0]

    assert written["miou_2d"] == ""
    assert written["speed_mps"] == ""
    assert written["consistency"] == ""
    assert written["miou_3d"] == "0.500000"
    assert written["magnitude"] == "1"


def test_crossovers_reports_a_readable_verdict_with_no_fused_arm():
    """Sweeping only the lidar-only arm is a legal run, and it must not reach
    report.py as an absent verdict, which renders as "no crossover verdict
    found" and is a different finding from no crossover."""
    verdicts = sweep.crossovers([_row(LIDAR_ONLY_BASELINE)], SWEEP_DECALIB)

    assert verdicts
    for verdict in verdicts.values():
        assert not verdict.crossed
        assert FUSED_BASELINE in verdict.reason


def test_crossovers_reports_a_readable_verdict_with_no_reference_arm():
    verdicts = sweep.crossovers([_row(FUSED_BASELINE)], SWEEP_DECALIB)

    assert verdicts
    for verdict in verdicts.values():
        assert not verdict.crossed
        assert LIDAR_ONLY_BASELINE in verdict.reason


def test_crossovers_warns_when_the_reference_is_not_flat(capsys):
    """bl_geom3d's 3D output never reads the image, so under decalib it cannot
    depend on the extrinsic. A reference that moves means something derived a
    prediction from a cached T_cam_lidar, which is the failure that produces a
    flat curve and a fictitious crossover."""
    rows = [_row(FUSED_BASELINE, magnitude=0.5, miou_3d=0.9),
            _row(FUSED_BASELINE, magnitude=2.0, miou_3d=0.4),
            _row(LIDAR_ONLY_BASELINE, magnitude=0.5, miou_3d=0.6),
            _row(LIDAR_ONLY_BASELINE, magnitude=2.0, miou_3d=0.8)]

    sweep.crossovers(rows, SWEEP_DECALIB)

    assert "not flat" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _patch_dataset(monkeypatch, stub):
    monkeypatch.setattr(sweep.runner, "build_dataset", lambda args: stub)


def _argv(name, out_dir):
    """A CLI invocation for the refusal and gating tests. --dataset goose names
    the real no-poses, no-calib case (spec 13.3); build_dataset is patched, so
    nothing is read from disk."""
    return ["--sweep", name, "--dataset", "goose", "--baseline", FUSED_BASELINE,
            "--rot-deg", "2.0", "--trans-cm", "10",
            "--out-dir", str(out_dir)]


def _row(baseline, magnitude=1.0, miou_3d=0.6):
    return {"sweep": SWEEP_DECALIB, "baseline": baseline, "axis": "pitch",
            "magnitude": magnitude, "unit": UNIT_DEG, "seed": 0, "speed_mps": None,
            "miou_2d": 0.4, "miou_3d": miou_3d, "consistency": 1.0, "coverage": 0.3}


def _gt_agreement(points, frame):
    """Fraction of z-buffer-visible points whose 3D ground-truth label matches
    the 2D ground-truth label of the pixel they land on.

    Ground truth on both sides, so this measures the projection geometry and
    not any baseline's skill.
    """
    height, width = frame.image.shape[:2]

    uv, depth, in_frustum = project(points, frame.K, nominal_extrinsic(), width, height)
    _owner, visible = zbuffer(uv, depth, in_frustum, width, height)

    label_2d = frame.labels_2d_gt[uv[in_frustum, 1], uv[in_frustum, 0]]
    label_3d = frame.labels_3d_gt[in_frustum]
    scorable = visible[in_frustum]

    return float((label_2d[scorable] == label_3d[scorable]).mean())


def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))
