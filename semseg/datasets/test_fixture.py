"""Unit tests for the procedural fixture and the split policy.

Every geometric claim the fixture makes is tested with a perturbation case:
change one input by a small amount and the output has to move. A fixture whose
agreement metric cannot fall would let the decalibration sweep report a flat
line and call it robustness, which is the one failure mode this file exists to
prevent.

The projection used here is written locally rather than imported from
semseg/projection.py on purpose. The fixture must be verifiable without the
module whose correctness the fixture is later used to check, otherwise a shared
sign error would cancel out and both would look right.
"""

import json
import os
import subprocess
import sys
import types as pytypes
from dataclasses import dataclass

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import semseg  # noqa: E402


def _install_types_stub():
    """semseg/types.py is another module's deliverable. Stub it here, never in
    the shipped module, so this file can run before that one lands."""
    try:
        import semseg.types  # noqa: F401
        return
    except ImportError:
        pass

    stub = pytypes.ModuleType("semseg.types")
    stub.NUM_CLASSES = 9
    stub.UNLABELED = 255

    @dataclass
    class Frame:
        frame_id: str
        image: np.ndarray
        points: np.ndarray
        intensity: np.ndarray
        K: np.ndarray
        point_times: np.ndarray = None
        ego_twist: np.ndarray = None
        labels_2d_gt: np.ndarray = None
        labels_3d_gt: np.ndarray = None

    stub.Frame = Frame
    sys.modules["semseg.types"] = stub
    semseg.types = stub


_install_types_stub()

from semseg.datasets import FIT_FRACTION, split_bucket, split_frames  # noqa: E402
from semseg.datasets import fixture as fx  # noqa: E402
from semseg.types import NUM_CLASSES  # noqa: E402

# Spec section 9 bar: a correctly calibrated painted point must land on a pixel
# of its own class.
MIN_PAINT_AGREEMENT = 0.80

# The fixture measures about 0.986, so the spec bar alone would still pass with
# the two modalities silently offset by a few pixels, which is exactly the
# defect that would bias every sweep. The tight floor is what catches that.
NOMINAL_PAINT_FLOOR = 0.95

# Measured drop at 2 deg of pitch is about 0.14 on the default scene. The bar
# is set well under that so it survives jitter, and well over zero so a fixture
# that had stopped responding to decalibration would fail here.
PITCH_PERTURB_DEG = 2.0
MIN_PITCH_DROP = 0.05

MIN_DEPTH_M = 0.5
DEPTH_TOL_M = 0.5

# Split buckets are measured values, printed from split_bucket and pasted here.
# Written from reasoning they would prove nothing.
KNOWN_FIT_ID = "fixture_0000"      # bucket 0.0939
KNOWN_SCORE_ID = "fixture_0001"    # bucket 0.9385
NEAR_HALF_ID = "fixture_0002"      # bucket 0.5043, just above FIT_FRACTION


def _project(points, K, T_cam_lidar, width, height):
    """-> (uv int64, depth float64, in_frustum bool). Same convention as spec
    section 3: u = K @ (T @ p)[:3], pixel = u[:2] / u[2]."""
    homog = np.concatenate([points.astype(np.float64), np.ones((len(points), 1))], axis=1)
    cam = homog @ T_cam_lidar.T
    depth = cam[:, 2]

    with np.errstate(invalid="ignore", divide="ignore"):
        image_plane = cam[:, :3] @ K.T
        u = np.rint(image_plane[:, 0] / image_plane[:, 2])
        v = np.rint(image_plane[:, 1] / image_plane[:, 2])

    in_frustum = (depth > MIN_DEPTH_M) & (u >= 0) & (u < width) & (v >= 0) & (v < height)

    uv = np.zeros((len(points), 2), dtype=np.int64)
    uv[in_frustum, 0] = u[in_frustum]
    uv[in_frustum, 1] = v[in_frustum]
    return uv, depth, in_frustum


def _zbuffer(uv, depth, in_frustum, width, height):
    """-> visible bool mask. Nearest point per pixel wins, written far to near
    so the last write is the nearest."""
    owner = np.full(width * height, -1, dtype=np.int64)
    candidates = np.flatnonzero(in_frustum)
    far_to_near = candidates[np.argsort(depth[candidates], kind="stable")][::-1]

    owner[uv[far_to_near, 1] * width + uv[far_to_near, 0]] = far_to_near

    visible = np.zeros(len(depth), dtype=bool)
    visible[owner[owner >= 0]] = True
    return visible


def _rotate_camera(T_cam_lidar, axis, degrees):
    """Perturb the extrinsic by a rotation applied in the CAMERA frame, which
    is how a decalibration presents itself: the lens has moved, not the world."""
    angle = np.radians(degrees)
    c, s = np.cos(angle), np.sin(angle)

    rot = {"pitch": [[1, 0, 0], [0, c, -s], [0, s, c]],
           "yaw": [[c, 0, s], [0, 1, 0], [-s, 0, c]],
           "roll": [[c, -s, 0], [s, c, 0], [0, 0, 1]]}[axis]

    perturbation = np.eye(4)
    perturbation[:3, :3] = rot
    return perturbation @ T_cam_lidar


def _paint_agreement(frame, T_cam_lidar, subset):
    """Fraction of `subset` whose 2D ground truth pixel carries its own class."""
    height, width = frame.labels_2d_gt.shape
    uv, _, in_frustum = _project(frame.points, frame.K, T_cam_lidar, width, height)

    scored = subset & in_frustum
    labels_2d = frame.labels_2d_gt[uv[scored, 1], uv[scored, 0]]
    return float((labels_2d == frame.labels_3d_gt[scored]).mean()), scored


def _visible_mask(frame, T_cam_lidar):
    height, width = frame.labels_2d_gt.shape
    uv, depth, in_frustum = _project(frame.points, frame.K, T_cam_lidar, width, height)
    return in_frustum & _zbuffer(uv, depth, in_frustum, width, height)


@pytest.fixture(scope="module")
def all_frames():
    dataset = fx.FixtureDataset()
    return [dataset.load(frame_id) for frame_id in dataset.frame_ids()]


@pytest.fixture(scope="module")
def frame(all_frames):
    return all_frames[0]


def test_determinism_from_seed():
    first = fx.FixtureDataset(seed=7).load(KNOWN_FIT_ID)
    second = fx.FixtureDataset(seed=7).load(KNOWN_FIT_ID)

    assert np.array_equal(first.points, second.points)
    assert np.array_equal(first.image, second.image)
    assert np.array_equal(first.labels_2d_gt, second.labels_2d_gt)
    assert np.array_equal(first.labels_3d_gt, second.labels_3d_gt)
    assert np.array_equal(first.intensity, second.intensity)


def test_seed_change_moves_render():
    """Perturbation case for the generator itself: a neighbouring seed must
    produce a different world, or the seed is not wired through."""
    baseline = fx.FixtureDataset(seed=7).load(KNOWN_FIT_ID)
    perturbed = fx.FixtureDataset(seed=8).load(KNOWN_FIT_ID)

    assert baseline.points.shape != perturbed.points.shape or \
        not np.array_equal(baseline.points, perturbed.points)
    assert not np.array_equal(baseline.image, perturbed.image)
    assert not np.array_equal(baseline.labels_2d_gt, perturbed.labels_2d_gt)


def test_frames_differ_from_each_other():
    dataset = fx.FixtureDataset()
    first = dataset.load(fx.FRAME_ID_FMT.format(0))
    second = dataset.load(fx.FRAME_ID_FMT.format(1))

    assert not np.array_equal(first.labels_2d_gt, second.labels_2d_gt)


def test_class_presence_in_both_modalities(all_frames):
    """Classes 1 to 7 must appear in BOTH modalities in EVERY frame, because a
    fitted baseline handed a frame with no humans in it learns nothing about
    humans and the sweep then measures the gap instead of the fusion.

    Sky is deliberately asymmetric: present in 2D, exactly absent in 3D. That
    is assumption A3 and the reason the label space is 9 with a documented
    fold_sky (spec section 1), and emitting sky-labelled points here would put
    support in the 3D confusion matrices that cannot physically exist.
    Class 0 other is unused by construction, so nothing is quietly absorbed.
    """
    for loaded in all_frames:
        frame_id = loaded.frame_id
        in_3d = set(np.unique(loaded.labels_3d_gt).tolist())
        in_2d = set(np.unique(loaded.labels_2d_gt).tolist())

        for cls in range(fx.ARTIFICIAL_STRUCTURES, fx.SKY):
            assert cls in in_3d, f"{frame_id}: class {cls} missing from the cloud"
            assert cls in in_2d, f"{frame_id}: class {cls} missing from the image"

        assert fx.SKY in in_2d
        assert fx.SKY not in in_3d
        assert fx.OTHER not in in_2d
        assert fx.OTHER not in in_3d


def test_intensity_is_unnormalised(frame):
    """Spec section 1b: real GOOSE intensity is 0 to 254 and not normalised. A
    normalised fixture would let bl_geom3d fit here and break on the real data."""
    assert frame.intensity.dtype == np.float32
    assert frame.intensity.min() >= 0.0
    assert frame.intensity.max() <= fx.INTENSITY_MAX
    assert frame.intensity.max() > 1.5


def test_point_times_follow_azimuth(frame):
    """The timestamp of a point must be recoverable from its own azimuth, or
    the time-offset sweep is displacing points by a time they never had."""
    azimuth = np.arctan2(frame.points[:, 1].astype(np.float64),
                         frame.points[:, 0].astype(np.float64))
    phase = np.mod(azimuth - fx.AZIMUTH_START_RAD, 2.0 * np.pi) / (2.0 * np.pi)
    expected = phase * fx.SCAN_PERIOD_S

    error = np.abs(expected - frame.point_times.astype(np.float64))
    wrapped = np.minimum(error, fx.SCAN_PERIOD_S - error)
    assert wrapped.max() < 1e-6

    assert frame.point_times.min() >= 0.0
    assert frame.point_times.max() < fx.SCAN_PERIOD_S


def test_scan_period_scales_times():
    """Perturbation case for the timestamp transform."""
    scene = fx.build_scene(np.random.default_rng(0))
    slow = fx.FixtureConfig(scan_period_s=2.0 * fx.SCAN_PERIOD_S)

    _, _, base_times, _ = fx.render_lidar(
        scene, fx.FixtureConfig(), np.random.default_rng(1))
    _, _, slow_times, _ = fx.render_lidar(scene, slow, np.random.default_rng(1))

    assert slow_times.max() == pytest.approx(2.0 * base_times.max(), rel=1e-5)


def test_uncompensated_skew_equals_ego_motion():
    """The pinned skew convention, checked analytically on one beam and one
    wall. A ray fired forward at mid-scan leaves from a sensor that has already
    travelled speed * t, so the stored range is short by exactly that. Whoever
    writes the deskew arm gets the sign from this test."""
    wall_face_x = 10.0
    wall = fx.Box(cls=fx.ARTIFICIAL_STRUCTURES,
                  center=np.array([wall_face_x + 2.0, 0.0, 0.0]),
                  half_extent=np.array([2.0, 6.0, 3.0]))

    speed = 4.0
    still = fx.FixtureConfig(n_beams=1, n_azimuth=4, vfov_min_deg=0.0, vfov_max_deg=0.0,
                             ego_speed_mps=0.0, ego_yaw_rate_rps=0.0)
    moving = fx.FixtureConfig(n_beams=1, n_azimuth=4, vfov_min_deg=0.0, vfov_max_deg=0.0,
                              ego_speed_mps=speed, ego_yaw_rate_rps=0.0)

    still_pts, _, still_times, still_labels = fx.render_lidar(
        [wall], still, np.random.default_rng(3))
    moving_pts, _, moving_times, _ = fx.render_lidar([wall], moving, np.random.default_rng(3))

    assert still_labels.tolist() == [fx.ARTIFICIAL_STRUCTURES]
    assert still_times[0] == pytest.approx(0.5 * fx.SCAN_PERIOD_S, abs=1e-6)

    still_range = float(np.linalg.norm(still_pts[0]))
    moving_range = float(np.linalg.norm(moving_pts[0]))

    assert still_range == pytest.approx(wall_face_x, abs=5.0 * fx.LIDAR_RANGE_NOISE_M)
    assert still_range - moving_range == pytest.approx(speed * still_times[0], abs=1e-3)


def test_paint_agreement_at_nominal(all_frames):
    """Spec section 9: at the true extrinsic a painted point lands on a pixel
    of its own class. Asserted on every frame, because the sweeps average over
    frames and a bar met only by frame 0 would be a bar met by nothing."""
    for frame in all_frames:
        nominal = fx.nominal_extrinsic()
        visible = _visible_mask(frame, nominal)
        agreement, scored = _paint_agreement(frame, nominal, visible)

        assert scored.sum() > 1000, f"{frame.frame_id}: too few points to measure anything"
        assert agreement >= MIN_PAINT_AGREEMENT, frame.frame_id
        assert agreement >= NOMINAL_PAINT_FLOOR, frame.frame_id


def test_paint_agreement_drops_under_pitch(all_frames):
    """The test that proves this fixture can measure a fusion crossover at all.

    The point set is fixed to the nominally visible points and only the
    extrinsic used for projection changes, so the denominator cannot drift and
    manufacture or mask the drop. The image is the one rendered at the nominal
    extrinsic, exactly as eval/sweep.py sees it: the world does not move, the
    calibration is wrong.
    """
    for frame in all_frames:
        nominal = fx.nominal_extrinsic()
        visible = _visible_mask(frame, nominal)

        at_nominal, _ = _paint_agreement(frame, nominal, visible)
        at_pitch, _ = _paint_agreement(
            frame, _rotate_camera(nominal, "pitch", PITCH_PERTURB_DEG), visible)

        assert at_pitch < at_nominal - MIN_PITCH_DROP, (
            f"{frame.frame_id}: agreement barely moved: "
            f"{at_nominal:.4f} -> {at_pitch:.4f}")


def test_pitch_hurts_more_than_roll(all_frames):
    """Spec section 8 sweeps per axis because a rotation about the optical axis
    is nearly harmless while the same rotation about pitch is a lateral shift
    proportional to range. If the fixture did not reproduce that, a per-axis
    sweep on it would be measuring nothing."""
    for frame in all_frames:
        nominal = fx.nominal_extrinsic()
        visible = _visible_mask(frame, nominal)

        at_nominal, _ = _paint_agreement(frame, nominal, visible)
        at_pitch, _ = _paint_agreement(
            frame, _rotate_camera(nominal, "pitch", PITCH_PERTURB_DEG), visible)
        at_roll, _ = _paint_agreement(
            frame, _rotate_camera(nominal, "roll", PITCH_PERTURB_DEG), visible)

        assert at_nominal - at_pitch > at_nominal - at_roll, frame.frame_id


def test_render_depth_finds_occluded_points(frame):
    """The rendered per-pixel range is an exact visibility oracle, and a sparse
    point z-buffer is not: a point behind the wall often lands on a pixel no
    wall point occupies, survives the z-buffer, and disagrees. Gating on the
    rendered range must therefore raise agreement, which is the evidence that
    the residual disagreement is occlusion and silhouette quantisation rather
    than a calibration error baked into the fixture."""
    nominal = fx.nominal_extrinsic()
    config = fx.FixtureConfig()
    scene = fx.build_scene(np.random.default_rng(fx.DEFAULT_SEED))
    _, _, ray_range = fx.render_camera(scene, frame.K, nominal, config,
                                       np.random.default_rng(0))

    height, width = frame.labels_2d_gt.shape
    uv, _, _ = _project(frame.points, frame.K, nominal, width, height)
    camera_origin = -nominal[:3, :3].T @ nominal[:3, 3]
    point_range = np.linalg.norm(frame.points.astype(np.float64) - camera_origin, axis=1)

    visible = _visible_mask(frame, nominal)
    unoccluded = visible & (np.abs(point_range - ray_range[uv[:, 1], uv[:, 0]]) < DEPTH_TOL_M)

    ungated, _ = _paint_agreement(frame, nominal, visible)
    gated, scored = _paint_agreement(frame, nominal, unoccluded)

    assert scored.sum() > 1000
    assert unoccluded.sum() < visible.sum(), "no point was occluded, so the gate is untested"
    assert gated > ungated


def test_camera_depth_ordering():
    """Two boxes on the same ray: the near one owns the pixel. The perturbation
    is swapping their classes, which must flip the rendered label, so the test
    cannot pass by painting a constant."""
    near_face_x = 9.5
    near = fx.Box(cls=fx.VEHICLE, center=np.array([near_face_x + 0.5, 0.0, -0.05]),
                  half_extent=np.array([0.5, 2.0, 2.0]))
    far = fx.Box(cls=fx.ARTIFICIAL_STRUCTURES, center=np.array([20.0, 0.0, -0.05]),
                 half_extent=np.array([0.5, 2.0, 2.0]))

    config = fx.FixtureConfig()
    K = fx.intrinsics(config)
    row, col = config.image_height // 2, config.image_width // 2

    _, labels, ray_range = fx.render_camera([near, far], K, fx.nominal_extrinsic(),
                                            config, np.random.default_rng(0))
    assert labels[row, col] == fx.VEHICLE
    assert ray_range[row, col] == pytest.approx(
        near_face_x - fx.NOMINAL_CAM_ORIGIN_M[0], abs=0.05)

    near.cls, far.cls = far.cls, near.cls
    _, swapped, _ = fx.render_camera([near, far], K, fx.nominal_extrinsic(),
                                     config, np.random.default_rng(0))
    assert swapped[row, col] == fx.ARTIFICIAL_STRUCTURES


def test_moving_a_primitive_moves_the_labels():
    """Perturbation case for the geometry itself: nudge one primitive and the
    2D ground truth must change, but only locally. A global change would mean
    the renderer is keyed on something other than the primitive positions."""
    config = fx.FixtureConfig()
    K = fx.intrinsics(config)
    scene = fx.build_scene(np.random.default_rng(fx.DEFAULT_SEED))

    _, before, _ = fx.render_camera(scene, K, fx.nominal_extrinsic(), config,
                                    np.random.default_rng(0))

    human = next(p for p in scene if isinstance(p, fx.Cylinder))
    human.center_xy = human.center_xy + np.array([0.0, 0.2])

    _, after, _ = fx.render_camera(scene, K, fx.nominal_extrinsic(), config,
                                   np.random.default_rng(0))

    changed = int((before != after).sum())
    assert changed > 0
    assert changed < 0.05 * before.size


def test_extrinsic_is_rigid_and_a_fresh_copy():
    dataset = fx.FixtureDataset()
    T = dataset.extrinsic(KNOWN_FIT_ID)

    assert T.shape == (4, 4)
    assert T.dtype == np.float64
    assert np.allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3))
    assert np.linalg.det(T[:3, :3]) == pytest.approx(1.0)
    assert np.allclose(T[3, :], [0.0, 0.0, 0.0, 1.0])

    # a sweep perturbs the matrix it is handed, so handing out module state
    # would leak the perturbation into every later frame
    T[0, 3] += 5.0
    assert not np.allclose(T, dataset.extrinsic(KNOWN_FIT_ID))


def test_unknown_frame_id_raises():
    dataset = fx.FixtureDataset()
    with pytest.raises(KeyError):
        dataset.load("not_a_frame")
    with pytest.raises(KeyError):
        dataset.extrinsic("not_a_frame")


def _assert_frame_contract(frame):
    """Spec section 2, asserted rather than assumed. np.array_equal compares
    values and not dtypes, so the round-trip test alone would pass on a frame
    whose labels came back int64 or whose image decoded to (H, W); the breakage
    would then surface inside somebody else's np.bincount."""
    n_points = len(frame.points)
    height, width = frame.image.shape[:2]

    assert frame.points.shape == (n_points, 3)
    assert frame.points.dtype == np.float32
    assert frame.intensity.shape == (n_points,)
    assert frame.intensity.dtype == np.float32
    assert frame.point_times.shape == (n_points,)
    assert frame.point_times.dtype == np.float32
    assert frame.labels_3d_gt.shape == (n_points,)
    assert frame.labels_3d_gt.dtype == np.uint8

    assert frame.image.shape == (height, width, 3)
    assert frame.image.dtype == np.uint8
    assert frame.labels_2d_gt.shape == (height, width)
    assert frame.labels_2d_gt.dtype == np.uint8

    assert frame.K.shape == (3, 3)
    assert frame.K.dtype == np.float64
    assert frame.ego_twist.shape == (6,)
    assert frame.ego_twist.dtype == np.float64

    # an id outside the label space would silently widen a 9x9 confusion matrix
    assert frame.labels_2d_gt.max() < NUM_CLASSES
    assert frame.labels_3d_gt.max() < NUM_CLASSES


def test_frame_contract_procedural(frame):
    _assert_frame_contract(frame)


def test_frame_contract_from_disk(tmp_path):
    out_dir = str(tmp_path / "fixture")
    fx.write_fixture(out_dir, n_frames=1, seed=0)
    _assert_frame_contract(fx.FixtureDataset(root=out_dir).load(fx.FRAME_ID_FMT.format(0)))


def test_disk_round_trip_is_exact(tmp_path):
    """run_all.sh materialises the fixture once and every tool reads it back,
    so the read-back frame has to be the same bytes the generator produced."""
    out_dir = str(tmp_path / "fixture")
    written = fx.write_fixture(out_dir, n_frames=2, seed=3)

    assert written == [fx.FRAME_ID_FMT.format(0), fx.FRAME_ID_FMT.format(1)]

    on_disk = fx.FixtureDataset(root=out_dir)
    assert on_disk.frame_ids() == written
    assert on_disk.seed == 3

    for frame_id in written:
        expected = fx.FixtureDataset(seed=3, n_frames=2).load(frame_id)
        actual = on_disk.load(frame_id)

        assert np.array_equal(actual.image, expected.image)
        assert np.array_equal(actual.labels_2d_gt, expected.labels_2d_gt)
        assert np.array_equal(actual.labels_3d_gt, expected.labels_3d_gt)
        assert np.array_equal(actual.points, expected.points)
        assert np.array_equal(actual.intensity, expected.intensity)
        assert np.array_equal(actual.point_times, expected.point_times)
        assert np.allclose(actual.K, expected.K)
        assert np.allclose(actual.ego_twist, expected.ego_twist)

    with open(os.path.join(out_dir, fx.FIXTURE_META_NAME)) as f:
        meta = json.load(f)
    assert meta["label_space"] == "goose9"
    assert meta["fixture_version"] == fx.FIXTURE_VERSION

    # disk-mode extrinsic() returns the nominal matrix and never reads this
    # field, so without the assertion the written value is decoration
    assert np.allclose(np.array(meta["T_cam_lidar"]), fx.nominal_extrinsic())


def test_missing_or_stale_fixture_raises(tmp_path):
    """Boundary validation: a tool pointed at an unmaterialised fixture must
    fail loudly rather than generate a different world than the rest of the
    pipeline scored."""
    empty = str(tmp_path / "empty")
    os.makedirs(empty)
    with pytest.raises(FileNotFoundError):
        fx.FixtureDataset(root=empty)

    stale = str(tmp_path / "stale")
    fx.write_fixture(stale, n_frames=1, seed=0)
    meta_path = os.path.join(stale, fx.FIXTURE_META_NAME)
    with open(meta_path) as f:
        meta = json.load(f)
    meta["fixture_version"] = fx.FIXTURE_VERSION + 1
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    with pytest.raises(ValueError):
        fx.FixtureDataset(root=stale)


def test_split_frames_known_assignment():
    """Measured md5 buckets, pasted from split_bucket. Built-in hash is salted
    per process, so this assertion is what catches a switch to it."""
    frame_ids = [fx.FRAME_ID_FMT.format(i) for i in range(8)]
    fit_ids, score_ids = split_frames(frame_ids)

    assert split_bucket(KNOWN_FIT_ID) == pytest.approx(0.0939, abs=1e-4)
    assert split_bucket(KNOWN_SCORE_ID) == pytest.approx(0.9385, abs=1e-4)

    assert KNOWN_FIT_ID in fit_ids
    assert KNOWN_SCORE_ID in score_ids
    assert set(fit_ids).isdisjoint(score_ids)
    assert sorted(fit_ids + score_ids) == sorted(frame_ids)
    assert fit_ids == [fid for fid in frame_ids if fid in set(fit_ids)]


def test_split_frames_stable_across_processes():
    """The reason for md5 over hash(). PYTHONHASHSEED changes the built-in
    hash, so a split built on it would move between two runs of the same
    command and a fitted baseline would end up scoring on frames it fitted."""
    code = (f"import sys; sys.path.insert(0, {REPO_ROOT!r});"
            "from semseg.datasets import split_frames;"
            "print(split_frames(['fixture_%04d' % i for i in range(64)]))")

    outputs = []
    for hash_seed in ("0", "1", "random"):
        env = dict(os.environ, PYTHONHASHSEED=hash_seed)
        result = subprocess.run([sys.executable, "-c", code], env=env,
                                capture_output=True, text=True, check=True)
        outputs.append(result.stdout)

    assert len(set(outputs)) == 1
    in_process = str(split_frames([fx.FRAME_ID_FMT.format(i) for i in range(64)]))
    assert outputs[0].strip() == in_process


def test_split_fraction_perturbation():
    """Perturbation case for the split: nudge the fraction and the split moves,
    in the one direction it is allowed to move."""
    many = [f"frame_{i:05d}" for i in range(200)]

    half_fit, _ = split_frames(many, fit_fraction=0.5)
    more_fit, _ = split_frames(many, fit_fraction=0.6)

    assert len(more_fit) > len(half_fit)
    assert set(half_fit).issubset(more_fit)

    # a bucket just above FIT_FRACTION has to cross when the fraction rises
    assert split_bucket(NEAR_HALF_ID) > FIT_FRACTION
    assert NEAR_HALF_ID not in split_frames([NEAR_HALF_ID], fit_fraction=FIT_FRACTION)[0]
    assert NEAR_HALF_ID in split_frames([NEAR_HALF_ID], fit_fraction=0.6)[0]

    assert split_frames(many, fit_fraction=0.0)[0] == []
    assert split_frames(many, fit_fraction=1.0)[1] == []


def test_split_frames_rejects_bad_fraction():
    with pytest.raises(ValueError):
        split_frames(["a"], fit_fraction=1.5)
    with pytest.raises(ValueError):
        split_frames(["a"], fit_fraction=-0.1)
