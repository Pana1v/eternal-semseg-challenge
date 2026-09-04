"""The projection operator, kept in one module of its own because every other
module in the repo is downstream of it.

Problem statement section 2: "That consistency constraint is the whole problem.
Everything hard about this task lives in the projection operator
pi(p) = K . T_cam_lidar . p". Section 3 layer 2 traces the causality: feature
association between the two modalities is done through this operator, so if the
extrinsic or the timestamps are wrong, image features get attached to the wrong
points and the fusion gain evaporates.

That is why projection is not inlined into the baselines. eval/sweep.py
perturbs one argument here (`T_cam_lidar`, via `perturb_extrinsic`) and re-runs
everything downstream unchanged. A baseline that reimplemented the projection
inline, or cached the extrinsic, would be immune to the sweep and would report
a robustness it does not have.

Frame convention throughout: `T_cam_lidar` is 4x4 float64 and maps a point in
the lidar frame to the camera OPTICAL frame, which is x right, y down,
z forward. Projection is then `u = K @ (T_cam_lidar @ p_homog)[:3]` and
`pixel = u[:2] / u[2]`.

Everything here is vectorised over points. There is no Python loop over points
anywhere in this file, including the z-buffer.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt

# A return closer than this is the vehicle itself or the sensor housing, and
# near z = 0 the perspective division amplifies any calibration residual
# without bound. The test is strict: depth > MIN_DEPTH_M.
MIN_DEPTH_M = 0.5

# Guard for the perspective division. A pinhole K has last row [0, 0, 1] so the
# homogeneous w equals the camera-frame z, but K arrives from a caller and we
# must not emit inf pixels if it does not.
W_EPS = 1e-12

# The three camera optical axes, under the names the decalibration sweep uses.
# The camera optical frame is x right, y down, z forward, so:
#   roll  is about z, the optical axis. Nearly harmless: it rotates the image
#         about the principal point, and an on-axis point does not move at all.
#         Off-axis error grows with distance from the principal point, not with
#         range.
#   pitch is about x. A vertical image shift, and the one that hurts: the
#         lateral world error of a projected point grows linearly with range.
#   yaw   is about y. The horizontal counterpart of pitch, same range scaling.
# Problem statement section 3, layer 3, item 1 quantifies the consequence: a
# 0.5 degree rotational error puts a 10 m point off by about 9 cm laterally.
AXIS_VECTORS = {
    "roll": np.array([0.0, 0.0, 1.0]),
    "pitch": np.array([1.0, 0.0, 0.0]),
    "yaw": np.array([0.0, 1.0, 0.0]),
}
AXIS_RANDOM = "random"

# A random-axis perturbation with no rng would make the `seed` column of the
# sweep CSV a lie, so an absent rng gets a named seed rather than entropy.
DEFAULT_PERTURB_SEED = 0


def project(points, K, T_cam_lidar, width, height):
    """-> (uv, depth, in_frustum)

    uv: (N, 2) int32 pixel coords, column then row. Invalid rows are 0.
    depth: (N,) float32 camera-frame z, for every point including the ones
        that miss the image, because the range-stratified metrics need the
        range of out-of-frustum points too.
    in_frustum: (N,) bool, depth > MIN_DEPTH_M and the pixel inside the image.

    Assumption A2 of the problem statement is that the camera and the lidar see
    the same scene, and it is false: the lidar is 360 degrees and the camera is
    a frustum, so on real data most points come back with in_frustum False.
    That is the expected outcome here, not an error.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), got {points.shape}")

    K = np.asarray(K, dtype=np.float64)
    T_cam_lidar = np.asarray(T_cam_lidar, dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"K must be (3, 3), got {K.shape}")
    if T_cam_lidar.shape != (4, 4):
        raise ValueError(f"T_cam_lidar must be (4, 4), got {T_cam_lidar.shape}")

    # R @ p + t for every point at once, without materialising homogeneous rows
    p_cam = points @ T_cam_lidar[:3, :3].T + T_cam_lidar[:3, 3]

    # depth is returned as float32, and `ahead` is derived from the returned
    # value rather than the float64 intermediate, so a caller who re-tests
    # depth > MIN_DEPTH_M gets exactly this in_frustum back
    depth = p_cam[:, 2].astype(np.float32)
    ahead = depth > MIN_DEPTH_M

    u_homog = p_cam @ K.T
    w = u_homog[:, 2]
    w_safe = np.where(np.abs(w) > W_EPS, w, 1.0)
    uv_float = u_homog[:, :2] / w_safe[:, None]

    # the inside test is done in float and only in-frustum rows are cast: a
    # point just past MIN_DEPTH_M can project to 1e12, and casting that (or a
    # nan) to int32 is undefined behaviour in numpy
    inside = ((uv_float[:, 0] >= 0.0) & (uv_float[:, 0] < width) &
              (uv_float[:, 1] >= 0.0) & (uv_float[:, 1] < height))
    in_frustum = ahead & inside

    # floor, not round: a sample landing at u = 0.6 belongs to pixel 0, and
    # rounding would credit it to pixel 1
    uv = np.zeros((points.shape[0], 2), dtype=np.int32)
    uv[in_frustum] = np.floor(uv_float[in_frustum]).astype(np.int32)

    return uv, depth, in_frustum


def zbuffer(uv, depth, in_frustum, width, height):
    """-> (owner, visible)

    owner: (H, W) int64 index of the nearest point per pixel, -1 where empty.
    visible: (N,) bool, True only for points that own their pixel.

    Occlusion handling is not optional. A point behind a wall still projects
    onto the wall's pixel, so without a z-buffer the cross-modal consistency
    metric scores that occluded point against the occluder's label and reports
    a disagreement that is an artefact of the projection, not a model error.
    The same holds for painting: the point would be given the wall's colour.

    Nearest wins, resolved without a loop: sort by pixel index and then by
    depth, so the first row of each pixel's run is its nearest point, and take
    those first rows with np.unique on the sorted pixel indices.
    """
    uv = np.asarray(uv)
    depth = np.asarray(depth, dtype=np.float64)
    in_frustum = np.asarray(in_frustum, dtype=bool)

    owner = np.full(width * height, -1, dtype=np.int64)
    visible = np.zeros(in_frustum.shape[0], dtype=bool)

    candidates = np.flatnonzero(in_frustum)
    if candidates.size == 0:
        return owner.reshape(height, width), visible

    flat_pixel = uv[candidates, 1].astype(np.int64) * width + uv[candidates, 0]

    # lexsort's last key is primary, so this is "by pixel, then by depth"
    order = np.lexsort((depth[candidates], flat_pixel))
    pixel_sorted = flat_pixel[order]

    # the input is already sorted ascending, so return_index gives the first
    # row of each pixel's run, which is its minimum depth
    _, first = np.unique(pixel_sorted, return_index=True)
    winners = candidates[order[first]]

    owner[pixel_sorted[first]] = winners
    visible[winners] = True

    return owner.reshape(height, width), visible


def paint(points_uv, in_frustum, image):
    """-> (N, 3) uint8 RGB per point, 0 outside the frustum.

    Points that miss the image get 0 rather than a nearest-pixel guess. This is
    the honest answer for assumption A2: there is no camera evidence for those
    points, and inventing some would let the fused baseline claim a gain on
    points the camera never saw, which is exactly the ablation the problem
    statement section 6.2 asks us to separate out.

    Pass a `visible` mask from zbuffer as `in_frustum` when occlusion matters,
    which for painting it does: otherwise an occluded point is painted with the
    colour of whatever is in front of it.
    """
    points_uv = np.asarray(points_uv)
    in_frustum = np.asarray(in_frustum, dtype=bool)
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image must be (H, W, 3), got {image.shape}")

    rgb = np.zeros((points_uv.shape[0], 3), dtype=np.uint8)
    if not in_frustum.any():
        return rgb

    rgb[in_frustum] = image[points_uv[in_frustum, 1], points_uv[in_frustum, 0]]
    return rgb


def scatter_to_image(uv, in_frustum, values, width, height, fill):
    """Sparse point values -> dense (H, W) map by nearest-neighbour fill.

    Used to turn a 3D-only prediction into a 2D one (bl_geom3d owes a 2D output
    and only has per-point labels). The lidar covers the image sparsely, so
    every unseeded pixel takes the value of the nearest seeded pixel by exact
    Euclidean distance. When there is no seed anywhere the whole map is `fill`.

    Duplicate pixels resolve to the last point in array order, which is
    arbitrary. Pass a zbuffer-visible selection when that matters.
    """
    uv = np.asarray(uv)
    in_frustum = np.asarray(in_frustum, dtype=bool)
    values = np.asarray(values)
    if values.shape[0] != uv.shape[0]:
        raise ValueError(f"values has {values.shape[0]} rows, uv has {uv.shape[0]}")

    dense = np.full((height, width), fill, dtype=values.dtype)

    # the early return is correctness, not a speed shortcut: with no seed the
    # distance transform's input is all foreground and its returned indices are
    # undefined, so there is nothing to look up
    if not in_frustum.any():
        return dense

    rows = uv[in_frustum, 1]
    cols = uv[in_frustum, 0]
    dense[rows, cols] = values[in_frustum]

    seeded = np.zeros((height, width), dtype=bool)
    seeded[rows, cols] = True

    # distance_transform_edt measures the distance to the nearest zero, so
    # feeding it ~seeded makes every pixel point at the nearest seeded pixel
    _, indices = distance_transform_edt(~seeded, return_distances=True, return_indices=True)
    return dense[indices[0], indices[1]]


def perturb_extrinsic(T, axis, rot_deg, trans_m, rng=None):
    """-> a 4x4 copy of `T` with a calibration residual applied.

    This is what eval/sweep.py calls, and the decalibration sweep is the whole
    reason projection is a module. Problem statement section 6.3: report the
    perturbation at which fusion drops below the LiDAR-only baseline, because
    that number is the calibration accuracy the robot must sustain in
    production.

    The residual is applied on the LEFT, `delta @ T`, so it is expressed in the
    camera optical frame (x right, y down, z forward) and not in the lidar
    frame. That is what makes the axis names mean what they say, and it makes
    the world error of a projected point directly readable: a camera-frame
    rotation of theta moves a point at range r by 2 r sin(theta / 2), so
    0.5 degrees at 10 m is about 9 cm, exactly the figure in problem statement
    section 3, layer 3, item 1.

    `axis` is "roll" (about z, the optical axis), "pitch" (about x), "yaw"
    (about y), or "random". `rot_deg` rotates about the named axis and
    `trans_m` translates along that same axis, one convention for both so a
    sweep row is describable by one axis name. Note the consequence for roll:
    its translation runs along the optical axis, so it is a depth shift and the
    roll curve is expected to be nearly flat in both magnitudes. That is
    physics, not a plumbing bug.

    "random" draws an independent uniform direction for the rotation axis and
    for the translation, from `rng`, which may be a Generator or an int seed.
    An absent rng falls back to DEFAULT_PERTURB_SEED so the result stays
    reproducible and the sweep CSV's seed column stays true.
    """
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"T must be (4, 4), got {T.shape}")

    if axis == AXIS_RANDOM:
        if rng is None:
            rng = np.random.default_rng(DEFAULT_PERTURB_SEED)
        elif isinstance(rng, (int, np.integer)):
            rng = np.random.default_rng(rng)
        rot_axis = _unit_vector(rng)
        trans_axis = _unit_vector(rng)
    elif axis in AXIS_VECTORS:
        rot_axis = AXIS_VECTORS[axis]
        trans_axis = AXIS_VECTORS[axis]
    else:
        valid = ", ".join(list(AXIS_VECTORS) + [AXIS_RANDOM])
        raise ValueError(f"axis must be one of {valid}, got {axis!r}")

    delta = np.eye(4)
    delta[:3, :3] = _rodrigues(rot_axis, np.radians(rot_deg))
    delta[:3, 3] = trans_axis * trans_m

    return delta @ T


def _rodrigues(axis, angle_rad):
    """Rotation matrix about a unit axis by a signed angle."""
    u = np.asarray(axis, dtype=np.float64)
    U = np.array([[0.0, -u[2], u[1]],
                  [u[2], 0.0, -u[0]],
                  [-u[1], u[0], 0.0]])
    return np.eye(3) + np.sin(angle_rad) * U + (1.0 - np.cos(angle_rad)) * (U @ U)


def _unit_vector(rng):
    """A direction drawn uniformly on the sphere. Normalised gaussians, not
    uniform Euler angles, which would clump at the poles."""
    v = rng.standard_normal(3)
    return v / np.linalg.norm(v)
