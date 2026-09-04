"""Procedural bi-modal fixture: one outdoor scene, rendered into BOTH a lidar
cloud and an RGB image from the SAME geometric primitives, with exact ground
truth in both modalities and an exact known extrinsic.

Why this file carries the repo. Real GOOSE val ships no calibration and no
poses (spec section 1b), so the fused arm, the cross-modal consistency metric
and every robustness sweep of problem statement section 6.3 cannot run on it
without a calibration supplied by hand. This fixture is therefore where every
projection-dependent result in this repo comes from, and `run_all.sh` drives
the whole harness against it from a clean checkout with zero downloads.

The scientific property, stated plainly. Both modalities are rendered from the
same primitives through the same pinhole camera and the same known
T_cam_lidar, so a correctly calibrated painted point lands on a pixel of its
own class, and a decalibrated one does not. Agreement is near total at the
nominal extrinsic and degrades as the extrinsic is perturbed. That is what
makes the decalibration and time-offset sweeps measure a real crossover
instead of measuring plumbing, which is all an isotropic-blob fixture can do.

Sky is reproduced, not papered over. A camera ray that hits nothing is class 8;
a lidar ray that hits nothing is no return at all. So class 8 is one of the
largest classes in the 2D ground truth and is exactly empty in 3D, which is
assumption A3 of the problem statement and the reason the label space is nine
classes with a documented `fold_sky` (spec section 1). Class 0 `other` is
unused by construction, so nothing in this fixture is quietly absorbed into it.

Skew convention, pinned here because eval/sweep.py's deskew and time-offset
arms are written against it and a sign error there would look like a finding:

  - `point_times` is seconds since scan start, t = 0 is scan start, derived
    from the beam azimuth. The scan starts pointing backwards, so the
    camera-visible azimuths carry mid-scan timestamps, as on a real rig.
  - `points` are UNCOMPENSATED. Each point is stored as `range * direction` in
    the instantaneous sensor frame, that is, as if the sensor had never moved
    during the sweep. The sensor really does move, so the raw cloud is skewed
    and per-point motion compensation genuinely improves it.
  - `ego_twist` is [vx, vy, vz, wx, wy, wz] in the scan-start lidar frame.
    The sensor pose at time t is a translation of v * t and a yaw of wz * t
    relative to scan start, so a deskew applies T(t) to the stored point.

Rendering method. The image is produced by casting one ray per pixel through
the same intersection code the lidar uses, rather than by rasterising
triangles. Same primitives, same K, same extrinsic, and depth ordering comes
out exact instead of painter's-algorithm approximate.

Frames: lidar frame is x forward, y left, z up, origin at the sensor, ground
plane at z = -LIDAR_HEIGHT_M. Camera optical frame is x right, y down,
z forward.
"""

import json
import math
import os
from dataclasses import dataclass

import numpy as np
from PIL import Image

from semseg.datasets import Dataset
from semseg.types import Frame, NUM_CLASSES

# goose9 ids used by the scene. Spelled out locally so a reader can see which
# classes the fixture actually generates without cross-referencing.
OTHER = 0
ARTIFICIAL_STRUCTURES = 1
ARTIFICIAL_GROUND = 2
NATURAL_GROUND = 3
OBSTACLE = 4
VEHICLE = 5
VEGETATION = 6
HUMAN = 7
SKY = 8

# Sensor rig
LIDAR_HEIGHT_M = 1.8
IMAGE_WIDTH = 320
IMAGE_HEIGHT = 240
HFOV_DEG = 90.0
N_BEAMS = 40
N_AZIMUTH = 720
VFOV_MIN_DEG = -25.0
VFOV_MAX_DEG = 10.0
SCAN_PERIOD_S = 0.1

# The sweep starts pointing backwards so that the camera-visible azimuths land
# mid-scan, which is where a real rig's time offset actually bites.
AZIMUTH_START_RAD = -math.pi

# Camera centre in the lidar frame, and the axis permutation from lidar
# (x fwd, y left, z up) to camera optical (x right, y down, z fwd).
NOMINAL_CAM_ORIGIN_M = (0.10, 0.0, -0.05)
R_CAM_FROM_LIDAR = ((0.0, -1.0, 0.0),
                    (0.0, 0.0, -1.0),
                    (1.0, 0.0, 0.0))

# Ego motion during the sweep. Nonzero by default: a fixture with a still
# sensor makes the deskew ablation a no-op and hides the bug it exists to find.
EGO_SPEED_MPS = 1.0
EGO_YAW_RATE_RPS = 0.05

# Scene extent. The ground is a finite disc, so the horizon sits at a real
# radius and rays past it see sky. The radius is deliberately close to where
# returns actually land, so the ground/sky boundary is crossed by points under
# a small pitch error rather than sitting out beyond the useful range.
GROUND_RADIUS_M = 55.0
ROAD_HALF_WIDTH_M = 3.0
MAX_RANGE_M = 80.0

# Ray bookkeeping. MIN_HIT_RANGE_M rejects hits inside the sensor housing and
# the self-intersection a surface would otherwise report at t ~ 0.
RAY_EPS = 1e-9
MIN_HIT_RANGE_M = 0.3

# Structure: a long facade off to the right, tall enough that its top edge is
# a near-horizontal image boundary many lidar points sit against.
WALL_CENTER = (19.0, -9.0)
WALL_HALF_EXTENT_M = (15.0, 0.3, 2.0)
WALL_JITTER_M = 0.5

# Anchors are fixed and only jittered, so every mandatory class is present in
# both modalities in EVERY frame by construction. Free-range placement makes
# class coverage a property of the seed, which would flake the presence test
# and, worse, hand a downstream baseline a frame with no humans in it.
POSITION_JITTER_M = 0.5
VEHICLE_ANCHORS = ((11.0, 1.6), (21.0, -1.6))
VEHICLE_HALF_EXTENT_M = (2.1, 0.9, 0.75)
VEHICLE_YAW_JITTER_DEG = 4.0

HUMAN_ANCHORS = ((7.0, -4.2), (12.5, 4.4))
HUMAN_RADIUS_M = 0.32
HUMAN_HEIGHT_M = 1.75

OBJECT_ANCHORS = ((5.0, -2.6), (6.5, 2.7), (9.0, -5.2), (11.5, 5.6))
OBJECT_HALF_EXTENT_M = (0.22, 0.22, 0.35)
OBJECT_JITTER_M = 0.3

VEG_ANCHORS = ((8.5, 7.0), (13.0, -6.2), (17.0, 8.5), (21.0, -6.5),
               (26.0, 9.5), (31.0, -7.0), (15.0, 12.0))
VEG_RADII_MIN_M = (0.8, 0.8, 1.2)
VEG_RADII_MAX_M = (1.8, 1.8, 3.2)
VEG_JITTER_M = 1.2

# Vegetation returns are scattered rather than surface-exact, so the range is
# noisy while the label stays exact. The label is what the sweeps score, so
# noising the range costs nothing and keeps the cloud from looking analytic.
VEG_RANGE_NOISE_M = 0.05
LIDAR_RANGE_NOISE_M = 0.01

# Per-class appearance. Distinct base colours plus per-pixel noise: distinct
# enough that a camera-only Naive Bayes can separate them, noisy enough that it
# cannot reach a meaningless 1.0.
CLASS_BASE_COLOR = ((60, 60, 60),        # other, unused by this scene
                    (182, 170, 150),     # artificial_structures
                    (92, 92, 98),        # artificial_ground
                    (108, 132, 72),      # natural_ground
                    (222, 122, 40),      # obstacle
                    (58, 92, 200),       # vehicle
                    (40, 118, 52),       # vegetation
                    (228, 62, 118),      # human
                    (150, 190, 232))     # sky
IMAGE_NOISE_SIGMA = 6.0

# Intensity stays unnormalised in 0 to 254, matching the real GOOSE .bin files
# (spec section 1b). A normalised fixture would let a baseline fitted here
# break silently on the real data.
INTENSITY_BY_CLASS = (40.0, 90.0, 120.0, 70.0, 165.0, 205.0, 55.0, 100.0, 0.0)
INTENSITY_NOISE_SIGMA = 6.0
INTENSITY_MAX = 254.0

# On-disk layout
FIXTURE_META_NAME = "fixture_meta.json"
FIXTURE_VERSION = 1
FRAME_ID_FMT = "fixture_{:04d}"
CLOUD_FMT = "{}_cloud.npz"
IMAGE_FMT = "{}_image.png"
LABEL_2D_FMT = "{}_labelids.png"
FRAME_META_FMT = "{}_meta.json"

DEFAULT_N_FRAMES = 6
DEFAULT_SEED = 0

# Each frame gets its own RNG stream. The stride keeps neighbouring seeds from
# producing overlapping streams, which would make frames correlated.
SEED_STRIDE = 1000


@dataclass(frozen=True)
class FixtureConfig:
    """Rig and sweep geometry. Frozen because a render is only reproducible if
    the config that produced it cannot be edited afterwards."""
    image_width: int = IMAGE_WIDTH
    image_height: int = IMAGE_HEIGHT
    hfov_deg: float = HFOV_DEG
    n_beams: int = N_BEAMS
    n_azimuth: int = N_AZIMUTH
    vfov_min_deg: float = VFOV_MIN_DEG
    vfov_max_deg: float = VFOV_MAX_DEG
    scan_period_s: float = SCAN_PERIOD_S
    ego_speed_mps: float = EGO_SPEED_MPS
    ego_yaw_rate_rps: float = EGO_YAW_RATE_RPS

    def to_dict(self) -> dict:
        return {
            "image_width": self.image_width, "image_height": self.image_height,
            "hfov_deg": self.hfov_deg, "n_beams": self.n_beams,
            "n_azimuth": self.n_azimuth, "vfov_min_deg": self.vfov_min_deg,
            "vfov_max_deg": self.vfov_max_deg, "scan_period_s": self.scan_period_s,
            "ego_speed_mps": self.ego_speed_mps,
            "ego_yaw_rate_rps": self.ego_yaw_rate_rps,
        }


@dataclass
class Primitive:
    """A surface that answers two questions: where a ray hits it, and what
    class the hit point carries. Class is per hit rather than per primitive so
    the ground can be one surface with a road strip painted through it."""
    cls: int

    def intersect(self, origins: np.ndarray, dirs: np.ndarray) -> np.ndarray:
        """-> (M,) float64 ray parameter of the nearest hit, inf on a miss.
        `origins` and `dirs` are both (M, 3) and `dirs` is unit length, so the
        ray parameter is a range in metres."""
        raise NotImplementedError

    def classes(self, points: np.ndarray) -> np.ndarray:
        return np.full(len(points), self.cls, dtype=np.uint8)


@dataclass
class Ground(Primitive):
    """Horizontal disc. `cls` is the terrain fallback and the road strip
    overrides it, so a hit near the centreline reads artificial_ground."""
    z: float
    radius: float
    road_half_width: float

    def intersect(self, origins, dirs):
        t = np.full(len(dirs), np.inf)

        # only a downward ray can reach the disc, and the sensor is above it
        down = dirs[:, 2] < -RAY_EPS
        t_plane = (self.z - origins[down, 2]) / dirs[down, 2]
        hit = origins[down] + t_plane[:, None] * dirs[down]

        inside = np.hypot(hit[:, 0], hit[:, 1]) <= self.radius
        t[down] = np.where(inside & (t_plane > MIN_HIT_RANGE_M), t_plane, np.inf)
        return t

    def classes(self, points):
        road = np.abs(points[:, 1]) <= self.road_half_width
        return np.where(road, ARTIFICIAL_GROUND, NATURAL_GROUND).astype(np.uint8)


@dataclass
class Box(Primitive):
    """Yawed axis-aligned box, by the slab method in its own frame."""
    center: np.ndarray
    half_extent: np.ndarray
    yaw: float = 0.0

    def intersect(self, origins, dirs):
        rot = _rot_z(self.yaw)

        # rot.T applied on the right, so both arrays stay (M, 3)
        o_local = (origins - self.center) @ rot
        d_local = dirs @ rot

        inv = 1.0 / _no_zeros(d_local)
        t1 = (-self.half_extent - o_local) * inv
        t2 = (self.half_extent - o_local) * inv

        t_near = np.minimum(t1, t2).max(axis=1)
        t_far = np.maximum(t1, t2).min(axis=1)

        # t_near < 0 means the origin is inside the box, so the exit face is
        # the visible one
        t_hit = np.where(t_near > MIN_HIT_RANGE_M, t_near, t_far)
        valid = (t_far >= t_near) & (t_hit > MIN_HIT_RANGE_M)
        return np.where(valid, t_hit, np.inf)


@dataclass
class Ellipsoid(Primitive):
    """Vegetation clump. Solved in the unit-sphere space obtained by dividing
    by the radii, where the ray parameter is unchanged."""
    center: np.ndarray
    radii: np.ndarray

    def intersect(self, origins, dirs):
        o_local = (origins - self.center) / self.radii
        d_local = dirs / self.radii

        a = np.einsum("ij,ij->i", d_local, d_local)
        b = 2.0 * np.einsum("ij,ij->i", o_local, d_local)
        c = np.einsum("ij,ij->i", o_local, o_local) - 1.0

        disc = b * b - 4.0 * a * c
        root = np.sqrt(np.maximum(disc, 0.0))
        t_near = (-b - root) / (2.0 * a)
        t_far = (-b + root) / (2.0 * a)

        t_hit = np.where(t_near > MIN_HIT_RANGE_M, t_near, t_far)
        return np.where((disc >= 0.0) & (t_hit > MIN_HIT_RANGE_M), t_hit, np.inf)


@dataclass
class Cylinder(Primitive):
    """Upright finite cylinder with caps. The caps matter: without a top cap a
    ray from a sensor above the cylinder passes through the head and paints the
    ground behind it, which shows up as a hole in the image."""
    center_xy: np.ndarray
    z_min: float
    z_max: float
    radius: float

    def intersect(self, origins, dirs):
        ox = origins[:, 0] - self.center_xy[0]
        oy = origins[:, 1] - self.center_xy[1]
        dx, dy, dz = dirs[:, 0], dirs[:, 1], dirs[:, 2]

        a = dx * dx + dy * dy
        b = 2.0 * (ox * dx + oy * dy)
        c = ox * ox + oy * oy - self.radius * self.radius

        disc = b * b - 4.0 * a * c
        root = np.sqrt(np.maximum(disc, 0.0))
        a_safe = _no_zeros(a)

        candidates = []
        for t_side in ((-b - root) / (2.0 * a_safe), (-b + root) / (2.0 * a_safe)):
            z = origins[:, 2] + t_side * dz
            ok = (disc >= 0.0) & (a > RAY_EPS) & (t_side > MIN_HIT_RANGE_M)
            ok &= (z >= self.z_min) & (z <= self.z_max)
            candidates.append(np.where(ok, t_side, np.inf))

        for z_cap in (self.z_min, self.z_max):
            t_cap = (z_cap - origins[:, 2]) / _no_zeros(dz)
            hx = ox + t_cap * dx
            hy = oy + t_cap * dy
            ok = (np.abs(dz) > RAY_EPS) & (t_cap > MIN_HIT_RANGE_M)
            ok &= (hx * hx + hy * hy) <= self.radius * self.radius
            candidates.append(np.where(ok, t_cap, np.inf))

        return np.minimum.reduce(candidates)


def _rot_z(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _no_zeros(values: np.ndarray) -> np.ndarray:
    """Replace near-zero divisors with a signed epsilon, so a ray parallel to a
    slab yields a huge ray parameter instead of a nan that would poison the
    min-reduce downstream."""
    tiny = np.abs(values) < RAY_EPS
    return np.where(tiny, np.where(values < 0.0, -RAY_EPS, RAY_EPS), values)


def cast(primitives, origins: np.ndarray, dirs: np.ndarray):
    """-> (t, cls, hit). Nearest hit over every primitive.

    t is inf and cls is 0 where the ray hit nothing; `hit` is the mask that
    says which. Both modalities go through this one function, which is what
    makes them consistent by construction rather than by agreement of two
    renderers.
    """
    n_rays = len(dirs)
    best_t = np.full(n_rays, np.inf)
    owner = np.full(n_rays, -1, dtype=np.int32)

    for index, prim in enumerate(primitives):
        t = prim.intersect(origins, dirs)
        closer = t < best_t
        best_t[closer] = t[closer]
        owner[closer] = index

    cls = np.zeros(n_rays, dtype=np.uint8)
    for index, prim in enumerate(primitives):
        selected = owner == index
        if not selected.any():
            continue
        points = origins[selected] + best_t[selected][:, None] * dirs[selected]
        cls[selected] = prim.classes(points)

    return best_t, cls, owner >= 0


def build_scene(rng) -> list:
    """The primitives, in no particular order since `cast` sorts by range.

    Every mandatory goose9 class gets at least one instance at a near range
    inside the camera frustum. Randomness is jitter around fixed anchors, so
    class coverage is a property of the construction and not of the seed.
    """
    ground_z = -LIDAR_HEIGHT_M
    prims = [Ground(cls=NATURAL_GROUND, z=ground_z, radius=GROUND_RADIUS_M,
                    road_half_width=ROAD_HALF_WIDTH_M)]

    wall_x = WALL_CENTER[0] + rng.uniform(-WALL_JITTER_M, WALL_JITTER_M)
    wall_y = WALL_CENTER[1] + rng.uniform(-WALL_JITTER_M, WALL_JITTER_M)
    prims.append(Box(cls=ARTIFICIAL_STRUCTURES,
                     center=np.array([wall_x, wall_y, ground_z + WALL_HALF_EXTENT_M[2]]),
                     half_extent=np.array(WALL_HALF_EXTENT_M)))

    for anchor_x, anchor_y in VEHICLE_ANCHORS:
        x = anchor_x + rng.uniform(-POSITION_JITTER_M, POSITION_JITTER_M)
        y = anchor_y + rng.uniform(-POSITION_JITTER_M, POSITION_JITTER_M)
        yaw = math.radians(rng.uniform(-VEHICLE_YAW_JITTER_DEG, VEHICLE_YAW_JITTER_DEG))
        prims.append(Box(cls=VEHICLE,
                         center=np.array([x, y, ground_z + VEHICLE_HALF_EXTENT_M[2]]),
                         half_extent=np.array(VEHICLE_HALF_EXTENT_M),
                         yaw=yaw))

    for anchor_x, anchor_y in HUMAN_ANCHORS:
        x = anchor_x + rng.uniform(-POSITION_JITTER_M, POSITION_JITTER_M)
        y = anchor_y + rng.uniform(-POSITION_JITTER_M, POSITION_JITTER_M)
        prims.append(Cylinder(cls=HUMAN, center_xy=np.array([x, y]),
                              z_min=ground_z, z_max=ground_z + HUMAN_HEIGHT_M,
                              radius=HUMAN_RADIUS_M))

    for anchor_x, anchor_y in OBJECT_ANCHORS:
        x = anchor_x + rng.uniform(-OBJECT_JITTER_M, OBJECT_JITTER_M)
        y = anchor_y + rng.uniform(-OBJECT_JITTER_M, OBJECT_JITTER_M)
        prims.append(Box(cls=OBSTACLE,
                         center=np.array([x, y, ground_z + OBJECT_HALF_EXTENT_M[2]]),
                         half_extent=np.array(OBJECT_HALF_EXTENT_M),
                         yaw=rng.uniform(0.0, math.pi)))

    for anchor_x, anchor_y in VEG_ANCHORS:
        x = anchor_x + rng.uniform(-VEG_JITTER_M, VEG_JITTER_M)
        y = anchor_y + rng.uniform(-VEG_JITTER_M, VEG_JITTER_M)
        radii = np.array([rng.uniform(lo, hi)
                          for lo, hi in zip(VEG_RADII_MIN_M, VEG_RADII_MAX_M)])

        # sunk into the terrain so the clump reads as growing out of it rather
        # than floating above it
        center_z = ground_z + 0.75 * radii[2]
        prims.append(Ellipsoid(cls=VEGETATION,
                               center=np.array([x, y, center_z]), radii=radii))

    return prims


def nominal_extrinsic() -> np.ndarray:
    """-> T_cam_lidar, 4x4 float64. A fresh array every call, so a sweep that
    perturbs the returned matrix in place cannot corrupt module state."""
    rot = np.array(R_CAM_FROM_LIDAR, dtype=np.float64)
    origin = np.array(NOMINAL_CAM_ORIGIN_M, dtype=np.float64)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rot
    T[:3, 3] = -rot @ origin
    return T


def intrinsics(config: FixtureConfig) -> np.ndarray:
    """-> K, 3x3 float64. Square pixels, principal point at the exact image
    centre, so the horizon of a level camera lands on the centre row."""
    focal = 0.5 * config.image_width / math.tan(0.5 * math.radians(config.hfov_deg))
    return np.array([[focal, 0.0, 0.5 * (config.image_width - 1)],
                     [0.0, focal, 0.5 * (config.image_height - 1)],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def _camera_rays(K: np.ndarray, T_cam_lidar: np.ndarray, width: int, height: int):
    """One ray per pixel, in the lidar frame. Rays go through integer pixel
    coordinates, which is where projection.py's rounded uv lands, so a point
    and the pixel it projects into refer to the same ray."""
    rot = T_cam_lidar[:3, :3]
    origin = -rot.T @ T_cam_lidar[:3, 3]

    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    pixels = np.stack([cols.ravel(), rows.ravel(), np.ones(width * height)], axis=1)

    dirs_cam = pixels @ np.linalg.inv(K).T
    dirs = dirs_cam @ rot
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    return np.broadcast_to(origin, dirs.shape), dirs


def render_camera(primitives, K, T_cam_lidar, config: FixtureConfig, rng):
    """-> (image (H, W, 3) uint8 RGB, labels_2d (H, W) uint8 goose9,
    ray_range (H, W) float32 metres from the camera centre, inf where sky).

    ray_range is returned because it is the exact visibility oracle for this
    scene: a painted point whose range disagrees with the range of the surface
    the pixel actually shows was occluded, and a sparse point z-buffer cannot
    always tell.
    """
    height, width = config.image_height, config.image_width
    origins, dirs = _camera_rays(K, T_cam_lidar, width, height)

    t, cls, hit = cast(primitives, origins, dirs)

    labels = np.where(hit, cls, SKY).astype(np.uint8)
    colors = np.array(CLASS_BASE_COLOR, dtype=np.float64)[labels]
    noise = rng.normal(0.0, IMAGE_NOISE_SIGMA, size=colors.shape)
    image = np.clip(colors + noise, 0.0, 255.0).astype(np.uint8)

    return (image.reshape(height, width, 3),
            labels.reshape(height, width),
            np.where(hit, t, np.inf).astype(np.float32).reshape(height, width))


def _lidar_beams(config: FixtureConfig):
    """-> (dirs_local, times). Beam directions in the instantaneous sensor
    frame and the timestamp each one fires at."""
    azimuth = AZIMUTH_START_RAD + 2.0 * math.pi * np.arange(config.n_azimuth) / config.n_azimuth
    elevation = np.radians(np.linspace(config.vfov_min_deg, config.vfov_max_deg, config.n_beams))

    az, el = np.meshgrid(azimuth, elevation)
    az, el = az.ravel(), el.ravel()

    dirs = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=1)
    times = (az - AZIMUTH_START_RAD) / (2.0 * math.pi) * config.scan_period_s
    return dirs, times


def render_lidar(primitives, config: FixtureConfig, rng):
    """-> (points (N, 3) float32, intensity (N,) float32, times (N,) float32,
    labels (N,) uint8).

    Points are uncompensated, in the convention pinned in the module docstring:
    the sensor really moves during the sweep, the ray is cast from where the
    sensor actually was, and the point is then stored as if it had not moved.
    """
    dirs_local, times = _lidar_beams(config)

    speed, yaw_rate = config.ego_speed_mps, config.ego_yaw_rate_rps
    origins = np.zeros((len(dirs_local), 3))
    origins[:, 0] = speed * times

    yaw = yaw_rate * times
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    dirs_world = np.stack([cos_yaw * dirs_local[:, 0] - sin_yaw * dirs_local[:, 1],
                           sin_yaw * dirs_local[:, 0] + cos_yaw * dirs_local[:, 1],
                           dirs_local[:, 2]], axis=1)

    t, cls, hit = cast(primitives, origins, dirs_world)
    hit &= t <= MAX_RANGE_M

    t = t[hit]
    cls = cls[hit]
    dirs_local = dirs_local[hit]
    times = times[hit]

    t = t + rng.normal(0.0, LIDAR_RANGE_NOISE_M, size=len(t))
    foliage = cls == VEGETATION
    t[foliage] += rng.normal(0.0, VEG_RANGE_NOISE_M, size=int(foliage.sum()))

    intensity = np.array(INTENSITY_BY_CLASS)[cls] + rng.normal(0.0, INTENSITY_NOISE_SIGMA, size=len(cls))

    return (np.ascontiguousarray(t[:, None] * dirs_local, dtype=np.float32),
            np.clip(intensity, 0.0, INTENSITY_MAX).astype(np.float32),
            times.astype(np.float32),
            cls)


def make_frame(frame_id: str, index: int, seed: int, config: FixtureConfig) -> Frame:
    """One fully rendered frame. The RNG is drawn in a fixed order (scene, then
    lidar, then camera), which is the whole of what makes a seed reproducible.
    """
    rng = np.random.default_rng(seed + SEED_STRIDE * index)
    primitives = build_scene(rng)

    K = intrinsics(config)
    points, intensity, times, labels_3d = render_lidar(primitives, config, rng)
    image, labels_2d, _ = render_camera(primitives, K, nominal_extrinsic(), config, rng)

    twist = np.array([config.ego_speed_mps, 0.0, 0.0, 0.0, 0.0, config.ego_yaw_rate_rps],
                     dtype=np.float64)

    return Frame(frame_id=frame_id, image=image, points=points, intensity=intensity,
                 K=K, point_times=times, ego_twist=twist,
                 labels_2d_gt=labels_2d, labels_3d_gt=labels_3d)


class FixtureDataset(Dataset):
    """The procedural fixture, either generated in memory or read back from a
    directory written by `write_fixture`.

    `root=None` generates. A `root` that exists but holds no fixture metadata
    raises instead of silently generating, because a tool pointed at an
    unmaterialised fixture would otherwise score a different world than the one
    the rest of the pipeline used.
    """

    def __init__(self, root: str = None, n_frames: int = DEFAULT_N_FRAMES,
                 seed: int = DEFAULT_SEED, config: FixtureConfig = None):
        self.root = root
        self.config = config or FixtureConfig()
        self.seed = seed
        self.n_frames = n_frames

        if root is None:
            self._frame_ids = [FRAME_ID_FMT.format(i) for i in range(n_frames)]
            return

        meta_path = os.path.join(root, FIXTURE_META_NAME)
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"no fixture at {root}: {meta_path} is missing. "
                "Run semseg.datasets.fixture.write_fixture first.")

        with open(meta_path) as f:
            meta = json.load(f)

        if meta.get("fixture_version") != FIXTURE_VERSION:
            raise ValueError(f"{meta_path}: fixture_version is "
                             f"{meta.get('fixture_version')}, expected {FIXTURE_VERSION}")

        self.seed = meta["seed"]
        self.config = FixtureConfig(**meta["config"])
        self._frame_ids = list(meta["frame_ids"])
        self.n_frames = len(self._frame_ids)

    def frame_ids(self) -> list[str]:
        return list(self._frame_ids)

    def load(self, frame_id: str) -> Frame:
        if frame_id not in self._frame_ids:
            raise KeyError(f"{frame_id} is not a frame of this fixture")

        if self.root is None:
            return make_frame(frame_id, self._frame_ids.index(frame_id), self.seed, self.config)

        return self._read_frame(frame_id)

    def extrinsic(self, frame_id: str) -> np.ndarray:
        if frame_id not in self._frame_ids:
            raise KeyError(f"{frame_id} is not a frame of this fixture")

        # exact and constant by construction, which is assumption A1 holding.
        # The decalibration sweep is what tests A1 failing.
        return nominal_extrinsic()

    def _read_frame(self, frame_id: str) -> Frame:
        cloud = np.load(os.path.join(self.root, CLOUD_FMT.format(frame_id)))
        image = np.asarray(Image.open(os.path.join(self.root, IMAGE_FMT.format(frame_id))))
        labels_2d = np.asarray(Image.open(os.path.join(self.root, LABEL_2D_FMT.format(frame_id))))

        with open(os.path.join(self.root, FRAME_META_FMT.format(frame_id))) as f:
            meta = json.load(f)

        return Frame(frame_id=frame_id, image=image,
                     points=cloud["points"], intensity=cloud["intensity"],
                     K=np.array(meta["K"], dtype=np.float64),
                     point_times=cloud["point_times"],
                     ego_twist=np.array(meta["ego_twist"], dtype=np.float64),
                     labels_2d_gt=labels_2d, labels_3d_gt=cloud["labels_3d"])


def write_fixture(out_dir: str, n_frames: int = DEFAULT_N_FRAMES,
                  seed: int = DEFAULT_SEED, config: FixtureConfig = None) -> list[str]:
    """Materialise the fixture to disk and return the frame ids written.

    npz for the cloud and its 3D labels, png for the image and the 2D label
    map, json for K, the extrinsic and the meta. run_all.sh calls this once and
    every tool afterwards reads the same bytes, so a baseline and the scorer
    cannot disagree about what the world was.
    """
    config = config or FixtureConfig()
    os.makedirs(out_dir, exist_ok=True)

    frame_ids = []
    for index in range(n_frames):
        frame_id = FRAME_ID_FMT.format(index)
        frame = make_frame(frame_id, index, seed, config)

        np.savez_compressed(os.path.join(out_dir, CLOUD_FMT.format(frame_id)),
                            points=frame.points, intensity=frame.intensity,
                            point_times=frame.point_times, labels_3d=frame.labels_3d_gt)
        Image.fromarray(frame.image, mode="RGB").save(
            os.path.join(out_dir, IMAGE_FMT.format(frame_id)))
        Image.fromarray(frame.labels_2d_gt, mode="L").save(
            os.path.join(out_dir, LABEL_2D_FMT.format(frame_id)))

        with open(os.path.join(out_dir, FRAME_META_FMT.format(frame_id)), "w") as f:
            json.dump({"frame_id": frame_id,
                       "K": frame.K.tolist(),
                       "T_cam_lidar": nominal_extrinsic().tolist(),
                       "ego_twist": frame.ego_twist.tolist(),
                       "n_points": int(len(frame.points))}, f, indent=2)

        frame_ids.append(frame_id)

    with open(os.path.join(out_dir, FIXTURE_META_NAME), "w") as f:
        json.dump({"fixture_version": FIXTURE_VERSION,
                   "label_space": "goose9",
                   "num_classes": NUM_CLASSES,
                   "seed": seed,
                   "frame_ids": frame_ids,
                   "config": config.to_dict(),
                   "T_cam_lidar": nominal_extrinsic().tolist(),
                   "notes": "class 8 sky is 2D only, class 0 other is unused"}, f, indent=2)

    return frame_ids
