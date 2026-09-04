"""The per point and per pixel feature extractors the baselines compose.

Separate from nb.py because that separation is what makes the section 6.2
ablation an ablation. bl_geom3d asks for geom_features, bl_cam2d asks for
pixel_features, bl_paint asks for geom_features concatenated with
rgb_features of the painted colour, and all three then feed the identical
nb.GaussianNB. The only thing that differs between the three arms is which
columns of this file they request, so a difference in score is a difference
in information, not in machinery.
"""

import numpy as np
from scipy.spatial import cKDTree

# RANSAC ground fit. 200 iterations is enough because the ground is by far the
# largest coplanar structure in an off-road scene, so the probability that no
# sampled triple lands on it is negligible.
GROUND_RANSAC_ITERS = 200

# GOOSE drives off road, where soil, gravel and low grass are genuinely rough
# at the 10 cm scale. A tighter band fits one facet of the terrain instead of
# the terrain, and then everything downhill of it reads as an obstacle.
GROUND_DIST_THRESH_M = 0.15

# The plane fit is deterministic by default: a robustness sweep must vary the
# perturbation and nothing else, and a reseeded RANSAC would add its own
# spread to every curve.
GROUND_RANSAC_SEED = 0

GROUND_MIN_NORMAL_NORM = 1e-6  # three collinear samples span no plane

KNN_K = 16      # neighbours per point for the local PCA
EIG_EPS = 1e-12  # a neighbourhood whose points all coincide has no spread

UP_AXIS = np.array([0.0, 0.0, 1.0])

RGB_MAX = 255.0  # the images are uint8, so this is a real maximum

# Column meanings, so a reader can map a column index of the returned matrix
# to a quantity without counting np.column_stack arguments.
GEOM_FEATURE_NAMES = ("height_above_ground_m", "verticality", "planarity", "range_m", "intensity")
RGB_FEATURE_NAMES = ("r", "g", "b")
PIXEL_FEATURE_NAMES = RGB_FEATURE_NAMES + ("row_norm", "col_norm")


def ground_plane(points: np.ndarray, iters: int = GROUND_RANSAC_ITERS,
                  thresh: float = GROUND_DIST_THRESH_M, seed: int = GROUND_RANSAC_SEED):
    """RANSAC the dominant plane.

    -> (normal (3,) unit, offset float, inlier_mask (N,) bool) for the plane
    `normal . p + offset = 0`, so the signed distance of a point is just
    `p @ normal + offset`.

    The normal is flipped to point up (+z in the lidar frame) because the
    sign of a cross product depends on the order the three samples happened
    to be drawn in. Without the flip, height above ground would carry a
    random sign per frame and the height column would be noise.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), got {points.shape}")
    if points.shape[0] < 3:
        raise ValueError(f"need at least 3 points to fit a plane, got {points.shape[0]}")

    rng = np.random.default_rng(seed)
    triples = rng.integers(0, points.shape[0], size=(iters, 3))

    # Degenerate fallback: the up plane through the origin, which makes height
    # plain z. Only reachable when every sampled triple was collinear, i.e. on
    # a cloud with no plane in it at all.
    best_normal, best_offset, best_count = UP_AXIS.copy(), 0.0, -1

    for triple in triples:
        a, b, c = points[triple]
        normal = np.cross(b - a, c - a)
        norm = float(np.linalg.norm(normal))
        if norm < GROUND_MIN_NORMAL_NORM:
            continue

        normal = normal / norm
        offset = -float(normal @ a)
        count = int(np.count_nonzero(np.abs(points @ normal + offset) < thresh))
        if count > best_count:
            best_normal, best_offset, best_count = normal, offset, count

    if best_normal[2] < 0.0:
        best_normal, best_offset = -best_normal, -best_offset

    inliers = np.abs(points @ best_normal + best_offset) < thresh
    return best_normal, best_offset, inliers


def local_pca(points: np.ndarray, k: int = KNN_K):
    """-> (verticality (N,), planarity (N,)) from the eigenvalues of the
    covariance of each point's k nearest neighbours.

    Batched into a single np.linalg.eigh over an (N, 3, 3) stack. A per point
    loop costs minutes on one 170k point scan, and the sweeps call this dozens
    of times per baseline.
    """
    points = np.asarray(points, dtype=np.float64)
    n = points.shape[0]

    # cKDTree returns index n for a neighbour that does not exist, which would
    # index out of bounds, so a cloud smaller than k asks only for what exists.
    k = min(k, n)

    tree = cKDTree(points)
    _, idx = tree.query(points, k=k)
    idx = idx.reshape(n, k)

    neighbours = points[idx]                                   # (N, k, 3)
    centred = neighbours - neighbours.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", centred, centred) / k

    eigvals, eigvecs = np.linalg.eigh(cov)                     # ascending
    local_normal = eigvecs[:, :, 0]                            # least spread direction

    # A wall's local normal is horizontal and flat ground's is vertical, so
    # 1 - |n . up| reads 1 on a wall and 0 on the ground. Taking the absolute
    # value first because eigh's sign convention is arbitrary.
    verticality = 1.0 - np.abs(local_normal @ UP_AXIS)

    # Weinmann's planarity (l2 - l3) / l1 with l1 >= l2 >= l3: on a plane the
    # smallest eigenvalue collapses while the other two stay comparable, so
    # this goes to 1, and on a wire or a vegetation clump it does not.
    largest = np.maximum(eigvals[:, 2], EIG_EPS)
    planarity = (eigvals[:, 1] - eigvals[:, 0]) / largest

    return verticality, planarity


def geom_features(points: np.ndarray, intensity: np.ndarray, k: int = KNN_K) -> np.ndarray:
    """-> (N, 5) float64, columns named by GEOM_FEATURE_NAMES.

    The complete lidar-only feature set of interface spec section 10: height
    above the fitted ground plane, local verticality and planarity, range and
    intensity.

    Intensity is passed through in raw sensor units. A Gaussian Naive Bayes
    with per feature variances is exactly invariant to a per feature rescale
    (the log 2 pi sigma^2 term is identical for every class and cancels in the
    posterior), so a normalisation would buy nothing while risking pushing a
    column's variance down into nb.VAR_FLOOR. There is also no honest
    constant to divide by: GOOSE intensity is 0 to 254 (interface spec 1b) and
    undefined elsewhere.
    """
    points = np.asarray(points, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64).reshape(-1)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), got {points.shape}")
    if intensity.shape[0] != points.shape[0]:
        raise ValueError(f"intensity has {intensity.shape[0]} values for {points.shape[0]} points")

    normal, offset, _ = ground_plane(points)
    height = points @ normal + offset

    verticality, planarity = local_pca(points, k)
    ranges = np.linalg.norm(points, axis=1)

    return np.column_stack([height, verticality, planarity, ranges, intensity])


def rgb_features(rgb: np.ndarray) -> np.ndarray:
    """-> (N, 3) float64 colour in [0, 1], columns RGB_FEATURE_NAMES.

    Unlike lidar intensity there is an honest constant here, because the
    images are uint8. Dividing by it puts bl_paint's painted colour on exactly
    the same scale as bl_cam2d's pixel colour, so the fused and camera-only
    arms read the same numbers and the section 6.2 comparison cannot be
    confounded by a scale difference between them.
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError(f"rgb must be (N, 3), got {rgb.shape}")

    return rgb / RGB_MAX


def pixel_features(image: np.ndarray) -> np.ndarray:
    """-> (H*W, 5) float64 in row major order, columns PIXEL_FEATURE_NAMES, so
    that reshape(H, W) of any per row result recovers the image grid.

    The camera-only arm gets normalised row and column on top of colour
    because a bare RGB Naive Bayes cannot tell grey asphalt from grey sky.
    Row is a horizon prior and it is the cheapest one there is: sky is up,
    ground is down. It is also the honest reason bl_cam2d is a floor and not
    a segmentation model.
    """
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image must be (H, W, 3), got {image.shape}")

    height, width = image.shape[:2]
    rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")

    # max(.., 1): a one pixel wide image would otherwise divide by zero
    row_norm = rows.reshape(-1) / max(height - 1, 1)
    col_norm = cols.reshape(-1) / max(width - 1, 1)

    colour = rgb_features(image.reshape(-1, 3))
    return np.column_stack([colour, row_norm, col_norm])
