"""Unit tests for the projection operator.

Every geometric transform here gets a perturbation case: one input moves by a
small amount and the output has to move with it. A projection test that only
checks the nominal case passes just as happily against a function that ignores
its extrinsic argument, which is the one bug that would silently void every
decalibration sweep in the repo.
"""

import numpy as np
import pytest

from semseg.projection import (
    project, zbuffer, paint, scatter_to_image, perturb_extrinsic,
    MIN_DEPTH_M, AXIS_VECTORS, AXIS_RANDOM, DEFAULT_PERTURB_SEED,
)

WIDTH = 100
HEIGHT = 100
FILL_LABEL = 255


def _K(fx=100.0, fy=100.0, cx=50.0, cy=50.0):
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def _wide_K():
    """A longer focal length, so a sub-degree rotation is worth several whole
    pixels and the decalibration test is not fighting the floor()."""
    return _K(fx=500.0, fy=500.0, cx=250.0, cy=250.0)


def _grid_points(n=7, depth=10.0):
    """A lateral grid at fixed depth, all of it inside a 500x500 frustum."""
    span = np.linspace(-2.0, 2.0, n)
    x, y = np.meshgrid(span, span)
    return np.stack([x.ravel(), y.ravel(), np.full(x.size, depth)], axis=1)


def _unique_colour_image(height=HEIGHT, width=WIDTH):
    rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    return np.stack([rows % 256, cols % 256, (rows + cols) % 256], axis=2).astype(np.uint8)


# --- project -----------------------------------------------------------------

def test_on_axis_point_hits_principal_point():
    # camera optical frame, so +z is straight ahead through the lens
    points = np.array([[0.0, 0.0, 10.0]])
    uv, depth, in_frustum = project(points, _K(), np.eye(4), WIDTH, HEIGHT)

    assert in_frustum.tolist() == [True]
    assert uv[0].tolist() == [50, 50]
    np.testing.assert_allclose(depth[0], 10.0, atol=1e-6)


def test_optical_frame_axes_point_the_documented_way():
    # x right, y down: +x must land right of centre and +y below it
    points = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 10.0]])
    uv, _, in_frustum = project(points, _K(), np.eye(4), WIDTH, HEIGHT)

    assert in_frustum.all()
    assert uv[0, 0] > 50 and uv[0, 1] == 50
    assert uv[1, 1] > 50 and uv[1, 0] == 50


def test_min_depth_is_strict():
    points = np.array([[0.0, 0.0, MIN_DEPTH_M - 0.01],
                       [0.0, 0.0, MIN_DEPTH_M],
                       [0.0, 0.0, MIN_DEPTH_M + 0.01]])
    _, _, in_frustum = project(points, _K(), np.eye(4), WIDTH, HEIGHT)

    assert in_frustum.tolist() == [False, False, True]


def test_in_frustum_is_reproducible_from_returned_depth():
    # a caller re-deriving the near test from the float32 depth must get the
    # same mask back, so the two never disagree on a borderline point
    points = np.array([[0.0, 0.0, MIN_DEPTH_M],
                       [0.0, 0.0, np.nextafter(MIN_DEPTH_M, 1.0)],
                       [0.0, 0.0, 3.0]])
    _, depth, in_frustum = project(points, _K(), np.eye(4), WIDTH, HEIGHT)

    assert np.array_equal(in_frustum, (depth > MIN_DEPTH_M))


def test_points_behind_and_off_image_are_excluded_and_zeroed():
    points = np.array([[0.0, 0.0, -10.0],     # behind the camera
                       [0.0, 0.0, 0.0],       # at the pinhole
                       [50.0, 0.0, 10.0],     # way off to the right
                       [0.0, 0.0, 10.0]])     # the only good one
    uv, depth, in_frustum = project(points, _K(), np.eye(4), WIDTH, HEIGHT)

    assert in_frustum.tolist() == [False, False, False, True]
    assert uv[:3].sum() == 0                     # invalid rows are 0
    assert np.isfinite(depth).all()


def test_empty_cloud_survives():
    # a 75 percent dropout on a small fixture can empty the frustum for real
    uv, depth, in_frustum = project(np.zeros((0, 3)), _K(), np.eye(4), WIDTH, HEIGHT)

    assert uv.shape == (0, 2)
    assert depth.shape == (0,)
    assert in_frustum.shape == (0,)


def test_bad_shapes_raise():
    with pytest.raises(ValueError):
        project(np.zeros((4, 2)), _K(), np.eye(4), WIDTH, HEIGHT)
    with pytest.raises(ValueError):
        project(np.zeros((4, 3)), np.eye(4), np.eye(4), WIDTH, HEIGHT)
    with pytest.raises(ValueError):
        project(np.zeros((4, 3)), _K(), np.eye(3), WIDTH, HEIGHT)


def test_decalibrated_projection_moves_the_pixels():
    points = _grid_points()
    K = _wide_K()
    nominal = np.eye(4)
    decalibrated = perturb_extrinsic(nominal, "pitch", rot_deg=0.5, trans_m=0.0)

    uv_a, _, frustum_a = project(points, K, nominal, 500, 500)
    uv_b, _, frustum_b = project(points, K, decalibrated, 500, 500)

    assert frustum_a.all() and frustum_b.all()
    assert not np.array_equal(uv_a, uv_b)

    # a pitch error is a vertical shift, and at 10 m with fx = 500 it is
    # several whole pixels, so essentially every point has to move
    moved = (uv_a[:, 1] != uv_b[:, 1]).mean()
    assert moved > 0.9


def test_translated_extrinsic_moves_the_pixels():
    points = _grid_points()
    K = _wide_K()
    shifted = perturb_extrinsic(np.eye(4), "yaw", rot_deg=0.0, trans_m=0.10)

    uv_a, _, _ = project(points, K, np.eye(4), 500, 500)
    uv_b, _, _ = project(points, K, shifted, 500, 500)

    assert not np.array_equal(uv_a, uv_b)


def test_perturbed_intrinsics_move_the_pixels():
    # K is an input too, and a focal length that is off by one percent is a
    # real calibration failure mode
    points = _grid_points()
    uv_a, _, _ = project(points, _wide_K(), np.eye(4), 500, 500)
    uv_b, _, _ = project(points, _K(fx=505.0, fy=500.0, cx=250.0, cy=250.0), np.eye(4), 500, 500)

    assert not np.array_equal(uv_a, uv_b)


# --- perturb_extrinsic -------------------------------------------------------

def test_half_degree_pitch_moves_a_ten_metre_point_nine_cm():
    # problem statement section 3, layer 3, item 1: "A 0.5 degree rotational
    # error puts a 10 m point off by ~9 cm laterally"
    point = np.array([0.0, 0.0, 10.0])
    T = perturb_extrinsic(np.eye(4), "pitch", rot_deg=0.5, trans_m=0.0)

    moved = T[:3, :3] @ point + T[:3, 3]
    offset = moved - point
    distance = np.linalg.norm(offset)

    assert round(float(distance), 2) == 0.09
    assert 0.085 < distance < 0.090

    # pitch is about x, so the error is a vertical image shift
    assert abs(offset[1]) > 0.99 * distance


def test_half_degree_yaw_moves_the_same_point_horizontally():
    point = np.array([0.0, 0.0, 10.0])
    T = perturb_extrinsic(np.eye(4), "yaw", rot_deg=0.5, trans_m=0.0)

    offset = (T[:3, :3] @ point + T[:3, 3]) - point

    assert round(float(np.linalg.norm(offset)), 2) == 0.09
    assert abs(offset[0]) > 0.99 * np.linalg.norm(offset)


def test_roll_is_nearly_harmless_and_pitch_is_not():
    on_axis = np.array([0.0, 0.0, 10.0])
    off_axis = np.array([1.0, 0.0, 10.0])

    T_roll = perturb_extrinsic(np.eye(4), "roll", rot_deg=0.5, trans_m=0.0)
    T_pitch = perturb_extrinsic(np.eye(4), "pitch", rot_deg=0.5, trans_m=0.0)

    # roll is about the optical axis, so a point on that axis does not move
    assert np.linalg.norm(T_roll[:3, :3] @ on_axis - on_axis) < 1e-12

    # off axis it does move, but by an amount set by the 1 m offset from the
    # principal ray, not by the 10 m range: an order of magnitude less than
    # the same rotation applied as pitch
    roll_off = np.linalg.norm(T_roll[:3, :3] @ off_axis - off_axis)
    pitch_on = np.linalg.norm(T_pitch[:3, :3] @ on_axis - on_axis)
    assert roll_off > 0.0
    assert roll_off < 0.2 * pitch_on


def test_rotation_magnitude_is_monotonic():
    point = np.array([0.0, 0.0, 10.0])
    distances = []
    for rot_deg in (0.1, 0.25, 0.5, 1.0, 2.0):
        T = perturb_extrinsic(np.eye(4), "pitch", rot_deg=rot_deg, trans_m=0.0)
        distances.append(np.linalg.norm(T[:3, :3] @ point - point))

    assert distances == sorted(distances)
    assert distances[0] > 0.0


def test_translation_runs_along_the_named_axis():
    T = perturb_extrinsic(np.eye(4), "pitch", rot_deg=0.0, trans_m=0.05)

    np.testing.assert_allclose(T[:3, :3], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(T[:3, 3], AXIS_VECTORS["pitch"] * 0.05, atol=1e-12)


def test_zero_perturbation_is_the_identity():
    T = np.eye(4)
    T[:3, 3] = [0.1, -0.2, 0.3]
    for axis in list(AXIS_VECTORS) + [AXIS_RANDOM]:
        np.testing.assert_allclose(
            perturb_extrinsic(T, axis, rot_deg=0.0, trans_m=0.0), T, atol=1e-12)


def test_perturbation_is_applied_in_the_camera_frame():
    # a non-identity extrinsic must be left-multiplied: the residual belongs to
    # the camera, not to the lidar, or the axis names stop meaning anything
    T = np.eye(4)
    T[:3, 3] = [0.5, 0.0, 0.0]
    delta_rot = perturb_extrinsic(np.eye(4), "yaw", rot_deg=1.0, trans_m=0.0)

    np.testing.assert_allclose(perturb_extrinsic(T, "yaw", rot_deg=1.0, trans_m=0.0),
                               delta_rot @ T, atol=1e-12)


def test_random_axis_is_seeded_and_seeds_differ():
    T_a = perturb_extrinsic(np.eye(4), AXIS_RANDOM, 0.5, 0.02, rng=np.random.default_rng(1))
    T_b = perturb_extrinsic(np.eye(4), AXIS_RANDOM, 0.5, 0.02, rng=np.random.default_rng(1))
    T_c = perturb_extrinsic(np.eye(4), AXIS_RANDOM, 0.5, 0.02, rng=np.random.default_rng(2))

    np.testing.assert_array_equal(T_a, T_b)
    assert not np.array_equal(T_a, T_c)

    # an int seed is accepted, and an absent rng falls back to a named seed
    np.testing.assert_array_equal(
        T_a, perturb_extrinsic(np.eye(4), AXIS_RANDOM, 0.5, 0.02, rng=1))
    np.testing.assert_array_equal(
        perturb_extrinsic(np.eye(4), AXIS_RANDOM, 0.5, 0.02),
        perturb_extrinsic(np.eye(4), AXIS_RANDOM, 0.5, 0.02, rng=DEFAULT_PERTURB_SEED))


def test_random_axis_still_rotates_by_the_asked_angle():
    T = perturb_extrinsic(np.eye(4), AXIS_RANDOM, rot_deg=0.5, trans_m=0.0, rng=3)
    cos_angle = np.clip((np.trace(T[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)

    np.testing.assert_allclose(np.degrees(np.arccos(cos_angle)), 0.5, atol=1e-9)


def test_unknown_axis_raises():
    with pytest.raises(ValueError):
        perturb_extrinsic(np.eye(4), "tilt", 0.5, 0.0)


# --- zbuffer -----------------------------------------------------------------

def test_zbuffer_keeps_the_near_point_and_drops_the_occluded_one():
    # the far point is listed FIRST, so a "first write wins" implementation
    # fails this test
    points = np.array([[0.0, 0.0, 20.0], [0.0, 0.0, 5.0]])
    uv, depth, in_frustum = project(points, _K(), np.eye(4), WIDTH, HEIGHT)

    assert uv[0].tolist() == uv[1].tolist()      # same pixel, so one occludes

    owner, visible = zbuffer(uv, depth, in_frustum, WIDTH, HEIGHT)

    assert visible.tolist() == [False, True]
    assert owner[50, 50] == 1
    assert (owner == -1).sum() == WIDTH * HEIGHT - 1


def test_zbuffer_winner_follows_the_depth_order():
    # perturbation: swap which point is nearer and the winner has to flip
    near_second = np.array([[0.0, 0.0, 20.0], [0.0, 0.0, 5.0]])
    near_first = np.array([[0.0, 0.0, 5.0], [0.0, 0.0, 20.0]])

    _, visible_a = zbuffer(*project(near_second, _K(), np.eye(4), WIDTH, HEIGHT),
                           WIDTH, HEIGHT)
    _, visible_b = zbuffer(*project(near_first, _K(), np.eye(4), WIDTH, HEIGHT),
                           WIDTH, HEIGHT)

    assert visible_a.tolist() == [False, True]
    assert visible_b.tolist() == [True, False]


def test_zbuffer_marks_every_point_that_owns_its_own_pixel():
    points = np.array([[0.0, 0.0, 10.0], [1.0, 0.0, 10.0], [0.0, 1.0, 10.0]])
    uv, depth, in_frustum = project(points, _K(), np.eye(4), WIDTH, HEIGHT)
    owner, visible = zbuffer(uv, depth, in_frustum, WIDTH, HEIGHT)

    assert visible.all()
    assert (owner >= 0).sum() == 3
    for i in range(3):
        assert owner[uv[i, 1], uv[i, 0]] == i


def test_zbuffer_ignores_out_of_frustum_points():
    points = np.array([[0.0, 0.0, 10.0], [0.0, 0.0, -10.0]])
    uv, depth, in_frustum = project(points, _K(), np.eye(4), WIDTH, HEIGHT)
    _, visible = zbuffer(uv, depth, in_frustum, WIDTH, HEIGHT)

    assert visible.tolist() == [True, False]


def test_zbuffer_with_nothing_in_frustum():
    owner, visible = zbuffer(np.zeros((3, 2), dtype=np.int32), np.zeros(3),
                             np.zeros(3, dtype=bool), WIDTH, HEIGHT)

    assert owner.shape == (HEIGHT, WIDTH)
    assert (owner == -1).all()
    assert not visible.any()


# --- paint -------------------------------------------------------------------

def test_paint_reads_the_pixel_the_point_lands_on():
    image = _unique_colour_image()
    points = np.array([[0.0, 0.0, 10.0]])
    uv, _, in_frustum = project(points, _K(), np.eye(4), WIDTH, HEIGHT)

    rgb = paint(uv, in_frustum, image)

    np.testing.assert_array_equal(rgb[0], image[50, 50])


def test_paint_moves_when_the_point_moves_one_pixel():
    # fx = 100 at z = 10, so 0.1 m of x is exactly one pixel
    image = _unique_colour_image()
    here = np.array([[0.0, 0.0, 10.0]])
    one_pixel_right = np.array([[0.1, 0.0, 10.0]])

    uv_a, _, frustum_a = project(here, _K(), np.eye(4), WIDTH, HEIGHT)
    uv_b, _, frustum_b = project(one_pixel_right, _K(), np.eye(4), WIDTH, HEIGHT)

    assert uv_b[0, 0] == uv_a[0, 0] + 1
    assert not np.array_equal(paint(uv_a, frustum_a, image),
                              paint(uv_b, frustum_b, image))


def test_paint_returns_zero_outside_the_frustum():
    image = np.full((HEIGHT, WIDTH, 3), 200, dtype=np.uint8)
    points = np.array([[0.0, 0.0, 10.0], [0.0, 0.0, -10.0]])
    uv, _, in_frustum = project(points, _K(), np.eye(4), WIDTH, HEIGHT)

    rgb = paint(uv, in_frustum, image)

    assert rgb[0].tolist() == [200, 200, 200]
    assert rgb[1].tolist() == [0, 0, 0]


def test_paint_with_nothing_in_frustum():
    image = _unique_colour_image()
    rgb = paint(np.zeros((2, 2), dtype=np.int32), np.zeros(2, dtype=bool), image)

    assert rgb.shape == (2, 3)
    assert not rgb.any()


def test_paint_rejects_a_non_rgb_image():
    with pytest.raises(ValueError):
        paint(np.zeros((1, 2), dtype=np.int32), np.ones(1, dtype=bool),
              np.zeros((HEIGHT, WIDTH), dtype=np.uint8))


# --- scatter_to_image --------------------------------------------------------

def _two_seed_setup():
    """Seeds at pixels (10, 10) and (90, 90), with an out-of-frustum point
    carrying a value that must never reach the dense map."""
    uv = np.array([[10, 10], [0, 0], [90, 90]], dtype=np.int32)
    in_frustum = np.array([True, False, True])
    values = np.array([3, 9, 7], dtype=np.uint8)
    return uv, in_frustum, values


def test_scatter_keeps_seeds_and_fills_by_nearest():
    uv, in_frustum, values = _two_seed_setup()

    dense = scatter_to_image(uv, in_frustum, values, WIDTH, HEIGHT, FILL_LABEL)

    assert dense.shape == (HEIGHT, WIDTH)
    assert dense[10, 10] == 3
    assert dense[90, 90] == 7
    assert dense[0, 0] == 3            # nearest seed is (10, 10)
    assert dense[99, 99] == 7          # nearest seed is (90, 90)
    assert (dense == FILL_LABEL).sum() == 0


def test_scatter_ignores_values_of_out_of_frustum_points():
    # the middle value is indexed by row, so a function that scatters `values`
    # instead of `values[in_frustum]` writes 9 somewhere and fails here
    uv, in_frustum, values = _two_seed_setup()

    dense = scatter_to_image(uv, in_frustum, values, WIDTH, HEIGHT, FILL_LABEL)

    assert 9 not in np.unique(dense)


def test_scatter_moves_when_a_seed_value_changes():
    uv, in_frustum, values = _two_seed_setup()
    nudged = values.copy()
    nudged[0] = 4

    before = scatter_to_image(uv, in_frustum, values, WIDTH, HEIGHT, FILL_LABEL)
    after = scatter_to_image(uv, in_frustum, nudged, WIDTH, HEIGHT, FILL_LABEL)

    assert not np.array_equal(before, after)
    assert after[10, 10] == 4


def test_scatter_moves_when_a_seed_moves():
    uv, in_frustum, values = _two_seed_setup()
    moved = uv.copy()
    moved[0] = [40, 40]

    before = scatter_to_image(uv, in_frustum, values, WIDTH, HEIGHT, FILL_LABEL)
    after = scatter_to_image(moved, in_frustum, values, WIDTH, HEIGHT, FILL_LABEL)

    assert not np.array_equal(before, after)
    assert after[40, 40] == 3


def test_scatter_extrapolates_to_every_pixel_from_one_seed():
    # pinning the documented reading of "pixels with no point anywhere take
    # the fill value": the fill is for a map with NO seed at all, so a single
    # seed claims the whole image. That matters downstream. A lidar gets no
    # return from sky (spec chapter 1), so a 3D-only prediction turned into a
    # 2D one by this function labels the sky region with whatever ground or
    # vegetation class was nearest, and never with UNLABELED. Changing that to
    # a radius-limited fill has to be a deliberate edit to this test.
    dense = scatter_to_image(np.array([[10, 10]], dtype=np.int32), np.ones(1, dtype=bool),
                             np.array([3], dtype=np.uint8), WIDTH, HEIGHT, FILL_LABEL)

    assert (dense == 3).all()
    assert (dense == FILL_LABEL).sum() == 0


def test_scatter_with_no_seeds_is_all_fill():
    dense = scatter_to_image(np.zeros((2, 2), dtype=np.int32), np.zeros(2, dtype=bool),
                             np.zeros(2, dtype=np.uint8), WIDTH, HEIGHT, FILL_LABEL)

    assert (dense == FILL_LABEL).all()


def test_scatter_carries_float_values_too():
    # the same function turns a per-point confidence into a dense conf map
    uv = np.array([[10, 10]], dtype=np.int32)
    dense = scatter_to_image(uv, np.ones(1, dtype=bool), np.array([0.25], dtype=np.float32),
                             WIDTH, HEIGHT, np.float32(0.0))

    assert dense.dtype == np.float32
    np.testing.assert_allclose(dense[70, 70], 0.25)


def test_scatter_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        scatter_to_image(np.zeros((3, 2), dtype=np.int32), np.ones(3, dtype=bool),
                         np.zeros(2, dtype=np.uint8), WIDTH, HEIGHT, FILL_LABEL)
