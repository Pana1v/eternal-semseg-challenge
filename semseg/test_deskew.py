"""Unit tests for per-point motion compensation.

Two independent checks carry most of the weight here. First, `se3_exp` is
compared against scipy's generic matrix exponential of the same 4x4 generator,
which is a genuinely separate implementation of the same map and so catches any
error in the closed-form coefficients. Second, `deskew` is compared per point
against `se3_exp` at that point's own dt: the two share the coefficient helper
but not the assembly, so a transpose or a sign flip in the vectorised path is
invisible to any test that only exercises one of them.

The perturbation cases are the point of the file: a nonzero twist with nonzero
timestamps must move points, and a zero twist or a zero dt must be an exact
no-op. A deskew that quietly returns its input would otherwise pass a whole
suite of shape assertions.
"""

import numpy as np
import pytest
from scipy.linalg import expm

from semseg.deskew import deskew, se3_exp, OMEGA_EPS_RAD_PER_S

# a fast turn while driving: 2 m/s forward with a 30 deg/s yaw rate, in the
# lidar body frame [vx, vy, vz, wx, wy, wz]
TURNING_TWIST = np.array([2.0, 0.0, 0.0, 0.0, 0.0, np.radians(30.0)])

# one revolution of a 10 Hz spinning lidar
SCAN_PERIOD_S = 0.1


def _generator(xi):
    """The 4x4 se(3) generator, for scipy's expm to exponentiate."""
    G = np.zeros((4, 4))
    G[:3, :3] = np.array([[0.0, -xi[5], xi[4]],
                          [xi[5], 0.0, -xi[3]],
                          [-xi[4], xi[3], 0.0]])
    G[:3, 3] = xi[:3]
    return G


def _apply(T, points):
    return points @ T[:3, :3].T + T[:3, 3]


def _scan(n=8, radius=12.0):
    """A ring of points, as a spinning sensor would sample it, with the
    timestamps spread over one revolution."""
    azimuth = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    points = np.stack([radius * np.cos(azimuth),
                       radius * np.sin(azimuth),
                       np.zeros(n)], axis=1)
    times = np.linspace(0.0, SCAN_PERIOD_S, n, endpoint=False)
    return points, times


# --- se3_exp -----------------------------------------------------------------

def test_zero_twist_is_the_identity():
    np.testing.assert_array_equal(se3_exp(np.zeros(6)), np.eye(4))


def test_pure_translation_twist():
    T = se3_exp(np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0]))

    np.testing.assert_allclose(T[:3, :3], np.eye(3), atol=1e-15)
    np.testing.assert_allclose(T[:3, 3], [1.0, 2.0, 3.0], atol=1e-15)


def test_pure_rotation_twist_turns_x_into_y():
    T = se3_exp(np.array([0.0, 0.0, 0.0, 0.0, 0.0, np.pi / 2.0]))

    np.testing.assert_allclose(T[:3, :3] @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(T[:3, 3], np.zeros(3), atol=1e-15)


def test_matches_scipy_matrix_exponential():
    # an independent implementation of the same map, so this is the test that
    # would catch a wrong V(theta) coefficient
    for xi in (TURNING_TWIST,
               TURNING_TWIST * 0.01,
               np.array([0.3, -0.2, 0.05, 0.4, -0.7, 1.1]),
               np.array([0.0, 0.0, 0.0, 0.0, 0.0, np.pi])):
        np.testing.assert_allclose(se3_exp(xi), expm(_generator(xi)), atol=1e-12)


def test_negated_twist_is_the_inverse():
    T = se3_exp(TURNING_TWIST)

    np.testing.assert_allclose(T @ se3_exp(-TURNING_TWIST), np.eye(4), atol=1e-12)


def test_two_half_steps_equal_one_whole_step():
    half = se3_exp(TURNING_TWIST * 0.5)

    np.testing.assert_allclose(half @ half, se3_exp(TURNING_TWIST), atol=1e-12)


def test_every_twist_component_moves_the_result():
    base = se3_exp(TURNING_TWIST)
    for i in range(6):
        nudged = TURNING_TWIST.copy()
        nudged[i] += 1e-3
        moved = se3_exp(nudged)

        assert not np.array_equal(base, moved), f"component {i} does not move the pose"
        assert np.abs(moved - base).max() > 1e-6


def test_tiny_rotation_stays_below_the_pure_translation_branch():
    # just under the epsilon: the axis is undefined there, so the result has to
    # come back as a clean pure translation rather than a nan
    xi = np.array([1.0, 0.0, 0.0, 0.0, 0.0, OMEGA_EPS_RAD_PER_S / 2.0])
    T = se3_exp(xi)

    assert np.isfinite(T).all()
    np.testing.assert_allclose(T[:3, 3], [1.0, 0.0, 0.0], atol=1e-15)


def test_bad_twist_shape_raises():
    with pytest.raises(ValueError):
        se3_exp(np.zeros(3))


# --- deskew, the no-op cases -------------------------------------------------

def test_zero_twist_is_an_exact_no_op():
    points, times = _scan()

    corrected = deskew(points, times, np.zeros(6), t_ref=0.0)

    np.testing.assert_array_equal(corrected, points)


def test_zero_dt_is_an_exact_no_op():
    # every point stamped at the reference time, with the robot turning hard:
    # the twist is irrelevant because no time has passed for any point
    points, _ = _scan()
    times = np.full(points.shape[0], 0.05)

    corrected = deskew(points, times, TURNING_TWIST, t_ref=0.05)

    np.testing.assert_array_equal(corrected, points)


def test_zero_times_with_zero_reference_is_an_exact_no_op():
    points, _ = _scan()

    corrected = deskew(points, np.zeros(points.shape[0]), TURNING_TWIST, t_ref=0.0)

    np.testing.assert_array_equal(corrected, points)


def test_absent_times_or_twist_returns_the_raw_cloud():
    points, times = _scan()

    for a, b in ((None, TURNING_TWIST), (times, None), (None, None)):
        corrected = deskew(points, a, b, t_ref=0.0)
        np.testing.assert_array_equal(corrected, points)
        assert corrected is not points          # a copy, so callers cannot alias it


# --- deskew, the cases that must move ----------------------------------------

def test_nonzero_twist_and_times_move_the_points():
    points, times = _scan()

    corrected = deskew(points, times, TURNING_TWIST, t_ref=0.0)

    assert not np.allclose(corrected, points)
    # one scan period at 2 m/s is 20 cm, and the first point is stamped at
    # t_ref so it must not have moved at all
    np.testing.assert_array_equal(corrected[0], points[0])
    assert np.abs(corrected[1:] - points[1:]).max() > 1e-3


def test_forward_motion_pulls_a_point_nearer():
    # sensor moving at 1 m/s along its own x, point measured 1 s before the
    # reference time, so by t_ref the sensor has closed 1 m on it
    points = np.array([[10.0, 0.0, 0.0]])
    twist = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    corrected = deskew(points, np.array([0.0]), twist, t_ref=1.0)

    np.testing.assert_allclose(corrected, [[9.0, 0.0, 0.0]], atol=1e-12)


def test_yaw_rate_swings_a_point_the_right_way():
    # sensor yawing +90 deg/s. A point seen dead ahead 1 s before the
    # reference must sit 90 deg clockwise in the reference frame, because the
    # frame itself turned the other way.
    points = np.array([[1.0, 0.0, 0.0]])
    twist = np.array([0.0, 0.0, 0.0, 0.0, 0.0, np.pi / 2.0])

    corrected = deskew(points, np.array([0.0]), twist, t_ref=1.0)

    np.testing.assert_allclose(corrected, [[0.0, -1.0, 0.0]], atol=1e-12)


def test_uniform_times_agree_with_se3_exp():
    # the whole cloud shares one dt, so the correction collapses to a single
    # rigid transform and se3_exp must reproduce it. This is the cross-check
    # between the batch path and the single-pose path.
    points, _ = _scan()
    t = 0.037
    times = np.full(points.shape[0], t)

    corrected = deskew(points, times, TURNING_TWIST, t_ref=0.0)

    np.testing.assert_allclose(corrected, _apply(se3_exp(TURNING_TWIST * t), points),
                               atol=1e-12)


def test_per_point_times_agree_with_se3_exp_point_by_point():
    # the same cross-check with a different dt for every point, which is the
    # case the vectorised coefficients actually exist for. The loop is in the
    # test, never in the module.
    points, times = _scan()
    t_ref = SCAN_PERIOD_S / 2.0

    corrected = deskew(points, times, TURNING_TWIST, t_ref)

    for i, t in enumerate(times):
        expected = _apply(se3_exp(TURNING_TWIST * (t - t_ref)), points[i:i + 1])
        np.testing.assert_allclose(corrected[i], expected[0], atol=1e-12)


def test_perturbing_one_timestamp_moves_only_that_point():
    points, times = _scan(n=5)
    nudged = times.copy()
    nudged[2] += 1e-3

    before = deskew(points, times, TURNING_TWIST, t_ref=0.0)
    after = deskew(points, nudged, TURNING_TWIST, t_ref=0.0)

    assert np.linalg.norm(after[2] - before[2]) > 1e-4
    np.testing.assert_array_equal(np.delete(after, 2, axis=0), np.delete(before, 2, axis=0))


def test_correction_grows_with_the_time_offset():
    points = np.array([[12.0, 0.0, 0.0]])
    displacements = []
    for offset_ms in (10.0, 25.0, 50.0, 100.0, 200.0):
        corrected = deskew(points, np.array([offset_ms / 1000.0]), TURNING_TWIST, t_ref=0.0)
        displacements.append(float(np.linalg.norm(corrected - points)))

    assert displacements == sorted(displacements)
    assert displacements[0] > 0.0


def test_pure_rotation_preserves_every_range():
    # a rotation-only twist cannot change how far away a point is, whatever its
    # timestamp, so this fails on any coefficient error that leaks scale
    points, times = _scan()
    twist = np.array([0.0, 0.0, 0.0, 0.1, -0.2, 0.9])

    corrected = deskew(points, times, twist, t_ref=0.0)

    np.testing.assert_allclose(np.linalg.norm(corrected, axis=1),
                               np.linalg.norm(points, axis=1), atol=1e-12)


# --- deskew, boundaries ------------------------------------------------------

def test_empty_cloud_survives():
    corrected = deskew(np.zeros((0, 3)), np.zeros(0), TURNING_TWIST, t_ref=0.0)

    assert corrected.shape == (0, 3)


def test_float32_cloud_stays_float32():
    points, times = _scan()
    corrected = deskew(points.astype(np.float32), times.astype(np.float32),
                       TURNING_TWIST, t_ref=0.0)

    assert corrected.dtype == np.float32


def test_mismatched_shapes_raise():
    points, times = _scan()

    with pytest.raises(ValueError):
        deskew(points[:, :2], times, TURNING_TWIST, t_ref=0.0)
    with pytest.raises(ValueError):
        deskew(points, times[:-1], TURNING_TWIST, t_ref=0.0)
    with pytest.raises(ValueError):
        deskew(points, times, TURNING_TWIST[:3], t_ref=0.0)
