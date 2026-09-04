"""Per-point motion compensation of a spinning lidar scan.

Problem statement section 3, layer 3, item 3, which is the third of the three
independent error sources that make projection inexact: "A spinning LiDAR
samples points over the whole revolution. Without deskewing against ego-motion,
the 'point cloud at time t' is a fiction, and per-point timestamps are
sometimes fabricated by the driver rather than measured."

That is the whole reason this module exists and is separate from projection.py.
A scan is not an instantaneous observation. At 10 Hz the first and last point
of a revolution are 100 ms apart, and at 1 m/s that is a decimetre of ego
translation sitting inside a single "frame". Project the raw scan and the error
is not even constant across the image: it is a smooth function of azimuth, so
it survives averaging and it biases the cross-modal consistency metric rather
than just adding noise to it. The deskew ablation of section 6.3 measures
exactly that, which is why this is a first-class module with tests and not a
utility hidden inside a baseline.

The correction is a rigid motion per point, not a single rigid motion for the
scan: each point is carried by the motion the sensor underwent between that
point's own timestamp and the reference time, integrated as a constant twist
through the SE(3) exponential map.

The twist is taken in the lidar body frame, matching Frame.ego_twist in
semseg/types.py. Nothing here loops over points.
"""

import numpy as np

# Below this the rotational part of the twist is treated as absent and the
# motion is a pure translation. Not a convenience: with |w| at zero the axis
# w / |w| is undefined, and a stationary or purely translating robot is the
# common case, not an edge case.
OMEGA_EPS_RAD_PER_S = 1e-9

# Separate guard, on the per-point angle rather than on the rate. theta = |w|
# times dt is zero for any point whose timestamp is the reference time, even
# when the robot is turning hard, so the coefficient limits below are hit on
# ordinary data and not only on a degenerate twist.
THETA_EPS_RAD = 1e-9


def se3_exp(xi):
    """Exponential map se(3) -> SE(3) over unit time.

    xi is (6,) [vx, vy, vz, wx, wy, wz]. The result is the pose reached by
    holding that twist constant for one unit of time, so for a twist in
    metres/second and radians/second, `se3_exp(xi * t)` is the pose after t
    seconds. Rotation and translation are coupled: the sensor rotates while it
    translates, so the translation is the rotated integral V(theta) @ v and not
    simply v. Getting that wrong shows up as a small azimuth-dependent bias,
    which is the hardest kind of error to see in a picture.
    """
    xi = np.asarray(xi, dtype=np.float64)
    if xi.shape != (6,):
        raise ValueError(f"xi must be (6,), got {xi.shape}")

    v, w = xi[:3], xi[3:]
    theta = float(np.linalg.norm(w))

    T = np.eye(4)
    if theta < OMEGA_EPS_RAD_PER_S:
        T[:3, 3] = v
        return T

    U = _skew(w / theta)
    UU = U @ U
    sin_t, one_minus_cos, c1, c2 = _exp_coeffs(theta)

    T[:3, :3] = np.eye(3) + sin_t * U + one_minus_cos * UU
    T[:3, 3] = v + c1 * (U @ v) + c2 * (UU @ v)
    return T


def deskew(points, point_times, ego_twist, t_ref):
    """-> (N, 3) points expressed in the sensor frame as it stood at `t_ref`.

    `point_times` is per-point, in seconds, on the same clock as `t_ref`. Each
    point is moved by the rigid motion the sensor underwent between its own
    timestamp and the reference time:

        p_ref = exp(xi * (t_i - t_ref)) @ p_i

    which is the composition exp(xi * t_ref)^-1 @ exp(xi * t_i) collapsed into
    one exponential, legal here because a constant twist commutes with itself.
    A point sampled before `t_ref` gets a negative dt and is carried forward,
    one sampled after gets a positive dt and is carried back.

    A frame with no per-point timestamps or no ego motion cannot be deskewed,
    and the honest answer is then the raw cloud rather than a crash: the
    datasets that ship neither (see docs/SENSORS.md) are exactly the ones where
    the deskew ablation cannot be run, and that is a property of the data. It
    is reported there, not papered over with an assumed velocity. Malformed
    inputs, as opposed to absent ones, still raise.
    """
    source = np.asarray(points)
    if source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), got {source.shape}")

    # a float32 cloud stays float32, so this can be dropped into a Frame
    out_dtype = source.dtype if source.dtype.kind == "f" else np.float64

    if point_times is None or ego_twist is None:
        return source.astype(out_dtype, copy=True)

    p = source.astype(np.float64)
    point_times = np.asarray(point_times, dtype=np.float64)
    ego_twist = np.asarray(ego_twist, dtype=np.float64)
    if point_times.shape != (p.shape[0],):
        raise ValueError(f"point_times must be ({p.shape[0]},), got {point_times.shape}")
    if ego_twist.shape != (6,):
        raise ValueError(f"ego_twist must be (6,), got {ego_twist.shape}")

    dt = point_times - t_ref
    v, w = ego_twist[:3], ego_twist[3:]
    omega = float(np.linalg.norm(w))

    if omega < OMEGA_EPS_RAD_PER_S:
        return (p + dt[:, None] * v).astype(out_dtype, copy=False)

    # the rotation axis is shared by every point and only the angle scales
    # with dt, so one 3x3 skew serves the whole cloud and the per-point work
    # is four scalar coefficient arrays
    U = _skew(w / omega)
    UU = U @ U
    theta = omega * dt
    sin_t, one_minus_cos, c1, c2 = _exp_coeffs(theta)

    # row i of p @ U.T is U @ p_i, which is how Rodrigues gets applied to a
    # whole cloud without building N separate rotation matrices
    rotated = p + sin_t[:, None] * (p @ U.T) + one_minus_cos[:, None] * (p @ UU.T)

    scaled_v = dt[:, None] * v
    translated = scaled_v + (c1 * dt)[:, None] * (U @ v) + (c2 * dt)[:, None] * (UU @ v)

    return (rotated + translated).astype(out_dtype, copy=False)


def _exp_coeffs(theta):
    """sin(theta), 1 - cos(theta), (1 - cos(theta))/theta and
    (theta - sin(theta))/theta, elementwise, with the theta -> 0 limits taken
    from the Taylor series.

    The limits are what make a zero twist or a zero dt an exact no-op rather
    than a nan: at theta = 0 all four coefficients come out exactly zero, so
    the point is returned bit for bit. `theta` is signed, and both quotients
    have the same parity as the exact matrix terms they multiply, so a point
    with a negative dt is carried the other way and not mirrored.
    """
    theta = np.asarray(theta, dtype=np.float64)
    small = np.abs(theta) < THETA_EPS_RAD
    denom = np.where(small, 1.0, theta)

    sin_t = np.sin(theta)
    one_minus_cos = 1.0 - np.cos(theta)
    c1 = np.where(small, 0.5 * theta, one_minus_cos / denom)
    c2 = np.where(small, theta * theta / 6.0, (theta - sin_t) / denom)

    return sin_t, one_minus_cos, c1, c2


def _skew(u):
    """The matrix that takes a cross product with u."""
    return np.array([[0.0, -u[2], u[1]],
                     [u[2], 0.0, -u[0]],
                     [-u[1], u[0], 0.0]])
