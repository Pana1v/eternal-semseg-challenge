"""Unit tests for the shared feature extractors.

Everything is asserted on ONE scene, not on a separate scene per structure.
A scene holding only a wall would let the wall test pass while ground_plane
was quietly fitting the wall, and the height column would be measured from
the wrong surface with nobody noticing.
"""

import numpy as np
import pytest

from baselines.common.features import (
    GEOM_FEATURE_NAMES, PIXEL_FEATURE_NAMES, RGB_MAX,
    geom_features, ground_plane, local_pca, pixel_features, rgb_features,
)

HEIGHT, VERTICALITY, PLANARITY, RANGE, INTENSITY = range(len(GEOM_FEATURE_NAMES))

GROUND_SIDE = 40      # 1600 ground points
WALL_X_M = 3.0
WALL_SIDE = 11        # 121 wall points
CLUMP_N = 300
CLUMP_CENTRE_Z_M = 3.0


def scene(seed=0):
    """-> (points, intensity, kind) with kind in {"ground", "wall", "clump"}.

    The ground patch is deliberately by far the densest structure (1600
    points against 121 and 300), because RANSAC returns the plane with the
    most inliers and the tests below only mean anything if that plane is the
    ground. A wall is also perfectly planar, so density is the only thing
    separating the two.
    """
    rng = np.random.default_rng(seed)

    axis = np.linspace(-4.0, 4.0, GROUND_SIDE)
    grid_x, grid_y = np.meshgrid(axis, axis, indexing="ij")
    ground = np.column_stack([grid_x.ravel(), grid_y.ravel(), np.zeros(grid_x.size)])

    wall_y = np.linspace(-1.0, 1.0, WALL_SIDE)
    wall_z = np.linspace(0.0, 2.0, WALL_SIDE)
    span_y, span_z = np.meshgrid(wall_y, wall_z, indexing="ij")
    wall = np.column_stack([np.full(span_y.size, WALL_X_M), span_y.ravel(), span_z.ravel()])

    # a vegetation like clump, so "high planarity" is asserted against
    # something that is genuinely not planar rather than against nothing
    clump = rng.normal(0.0, 0.3, size=(CLUMP_N, 3)) + np.array([0.0, 0.0, CLUMP_CENTRE_Z_M])

    points = np.vstack([ground, wall, clump])
    kind = np.array(["ground"] * len(ground) + ["wall"] * len(wall) + ["clump"] * len(clump))

    # spread across the raw GOOSE range of 0 to 254 (interface spec 1b) so a
    # sneaked in normalisation would show up as a scaled column
    intensity = np.linspace(0.0, 254.0, points.shape[0])
    return points, intensity, kind


def interior_ground(points, kind):
    """Ground away from the patch edge and from the wall, where a k nearest
    neighbourhood is entirely ground."""
    return (kind == "ground") & (np.abs(points[:, 0]) < 2.0) & (np.abs(points[:, 1]) < 2.0)


def test_ground_plane_finds_the_ground():
    points, _, kind = scene()
    normal, offset, inliers = ground_plane(points)

    np.testing.assert_allclose(np.abs(normal), [0.0, 0.0, 1.0], atol=1e-9)
    assert normal[2] > 0.0            # flipped to point up, not down
    assert abs(offset) < 1e-9
    assert inliers[kind == "ground"].all()

    off_ground = (kind == "wall") & (points[:, 2] > 0.5)
    assert not inliers[off_ground].any()
    assert not inliers[kind == "clump"].any()


def test_ground_normal_moves_on_tilt():
    """Perturbation case for the geometric transform: tilting the scene by 2
    degrees must tilt the fitted normal by the same 2 degrees. A fit that
    always returned +z would pass the test above."""
    points, _, _ = scene()
    normal, offset, _ = ground_plane(points)

    angle = np.radians(2.0)
    rotate_x = np.array([[1.0, 0.0, 0.0],
                         [0.0, np.cos(angle), -np.sin(angle)],
                         [0.0, np.sin(angle), np.cos(angle)]])
    tilted_normal, tilted_offset, _ = ground_plane(points @ rotate_x.T)

    moved_deg = np.degrees(np.arccos(np.clip(normal @ tilted_normal, -1.0, 1.0)))
    assert moved_deg > 0.5
    assert abs(moved_deg - 2.0) < 0.1
    assert abs(tilted_offset - offset) < 1e-9   # the plane still passes through the origin


def test_geom_on_ground_and_wall():
    points, intensity, kind = scene()
    features = geom_features(points, intensity)

    assert features.shape == (points.shape[0], len(GEOM_FEATURE_NAMES))

    ground = interior_ground(points, kind)
    assert np.abs(features[ground, HEIGHT]).max() < 1e-9
    assert features[ground, VERTICALITY].max() < 0.05
    assert features[ground, PLANARITY].mean() > 0.7

    # Per point planarity on a perfect plane is high but not 1: with 16
    # neighbours on a regular lattice the two in plane eigenvalues are not
    # equal, so (l2 - l3) / l1 sits around 0.75. What makes the column useful
    # is the gap to a non planar structure, asserted below.
    assert features[ground, PLANARITY].min() > 0.4
    assert features[kind == "clump", PLANARITY].mean() < 0.4

    upper_wall = (kind == "wall") & (points[:, 2] > 0.5)
    assert features[upper_wall, VERTICALITY].min() > 0.9
    np.testing.assert_allclose(features[upper_wall, HEIGHT], points[upper_wall, 2], atol=1e-9)


def test_geom_range_and_raw_intensity():
    points, intensity, _ = scene()
    features = geom_features(points, intensity)

    np.testing.assert_allclose(features[:, RANGE], np.linalg.norm(points, axis=1))

    # raw, not rescaled: a Naive Bayes with per feature variances is invariant
    # to a per feature scale, so any rescale here would be an unexplained
    # transform on the way in
    np.testing.assert_allclose(features[:, INTENSITY], intensity)


def test_height_moves_on_nudge():
    """Perturbation case: lifting one ground point must lift its height
    column by the same amount and leave the other ground rows alone."""
    points, intensity, kind = scene()
    before = geom_features(points, intensity)

    ground = np.flatnonzero(interior_ground(points, kind))
    target = int(ground[100])

    lifted = points.copy()
    lifted[target, 2] += 0.1
    after = geom_features(lifted, intensity)

    assert after[target, HEIGHT] - before[target, HEIGHT] > 0.09
    others = ground[ground != target]
    assert np.abs(after[others, HEIGHT] - before[others, HEIGHT]).max() < 1e-9


def test_local_pca_handles_tiny_cloud():
    """Fewer points than KNN_K: cKDTree reports a missing neighbour as index
    n, which would index out of bounds if k were not clamped."""
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                       [1.0, 1.0, 0.0], [0.5, 0.5, 0.0]])
    verticality, planarity = local_pca(points)

    assert verticality.shape == (5,)
    assert planarity.shape == (5,)
    assert np.all(np.isfinite(verticality))
    assert np.all(np.isfinite(planarity))
    assert verticality.max() < 1e-9    # a flat cloud has a vertical normal


def test_geom_rejects_bad_shapes():
    points, intensity, _ = scene()

    with pytest.raises(ValueError):
        geom_features(points[:, :2], intensity)
    with pytest.raises(ValueError):
        geom_features(points, intensity[:-1])
    with pytest.raises(ValueError):
        ground_plane(points[:2])


def test_rgb_features_normalise():
    rgb = np.array([[0, 128, 255], [255, 255, 255]], dtype=np.uint8)
    features = rgb_features(rgb)

    np.testing.assert_allclose(features, np.array([[0.0, 128.0 / RGB_MAX, 1.0], [1.0, 1.0, 1.0]]))

    with pytest.raises(ValueError):
        rgb_features(rgb[:, :2])


def test_rgb_moves_on_one_level():
    """Perturbation case at the smallest step a uint8 image can take."""
    rgb = np.array([[10, 20, 30]], dtype=np.uint8)
    before = rgb_features(rgb)
    after = rgb_features(rgb + np.array([[1, 0, 0]], dtype=np.uint8))

    assert after[0, 0] - before[0, 0] == pytest.approx(1.0 / RGB_MAX)
    np.testing.assert_allclose(after[0, 1:], before[0, 1:])


def test_pixel_features_grid():
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    image[2, 3] = (255, 0, 0)
    features = pixel_features(image)

    assert features.shape == (24, len(PIXEL_FEATURE_NAMES))

    row_norm = features[:, 3].reshape(4, 6)
    col_norm = features[:, 4].reshape(4, 6)

    # row major order, so a reshape recovers the grid: row varies down and
    # column varies across, both normalised to end at 1.0
    np.testing.assert_allclose(row_norm[:, 0], [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
    np.testing.assert_allclose(col_norm[0], np.arange(6) / 5.0)
    assert np.allclose(row_norm, row_norm[:, :1])

    np.testing.assert_allclose(features[2 * 6 + 3, :3], [1.0, 0.0, 0.0])


def test_pixel_features_move_on_one_pixel():
    """Perturbation case: changing one pixel changes that pixel's row and
    nothing else, which is what makes a reshape back to (H, W) safe."""
    image = np.full((3, 3, 3), 100, dtype=np.uint8)
    before = pixel_features(image)

    image[1, 2, 1] += 1
    after = pixel_features(image)

    changed = np.flatnonzero(np.abs(after - before).sum(axis=1) > 0)
    assert changed.tolist() == [1 * 3 + 2]
    assert after[changed[0], 1] - before[changed[0], 1] == pytest.approx(1.0 / RGB_MAX)


def test_pixel_features_single_column():
    """A one pixel wide image would divide by zero when normalising the
    column index."""
    features = pixel_features(np.zeros((3, 1, 3), dtype=np.uint8))

    assert features.shape == (3, len(PIXEL_FEATURE_NAMES))
    np.testing.assert_allclose(features[:, 4], 0.0)


def test_pixel_features_rejects_bad_shape():
    with pytest.raises(ValueError):
        pixel_features(np.zeros((4, 6), dtype=np.uint8))
