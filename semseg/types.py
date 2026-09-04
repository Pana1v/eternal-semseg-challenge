"""The data contract every other module is written against: one bi-modal
observation (`Frame`), one prediction over it (`Prediction`), and the boundary
validator that refuses a malformed one.

Kept separate from labels.py because this file says what a Frame *holds* while
labels.py says what a label value *means*. Nothing here imports a dataset, a
baseline or a metric, so both of those can depend on it without a cycle.
"""

from dataclasses import dataclass

import numpy as np

# The goose9 label space (see labels.py for the names and the mapping).
NUM_CLASSES = 9

# Sentinel for pixels and points with no ground truth, and for a prediction a
# baseline declines to make. Excluded from every confusion matrix, which is why
# it must stay outside 0..NUM_CLASSES-1 rather than being folded into `other`.
UNLABELED = 255

# Fields validate() insists are present, and the optional ones it checks only
# when they are not None.
REQUIRED_ARRAYS = ("image", "points", "intensity", "K")
OPTIONAL_ARRAYS = ("point_times", "ego_twist", "labels_2d_gt", "labels_3d_gt")

# Length of the ego twist vector: [vx, vy, vz, wx, wy, wz].
TWIST_LENGTH = 6


@dataclass
class Frame:
    """One time-synchronised bi-modal observation. Everything the model may
    see. The extrinsic is deliberately NOT a member: it is a parameter of the
    prediction call so eval/sweep.py can perturb it per call."""
    frame_id: str
    image: np.ndarray            # (H, W, 3) uint8, RGB
    points: np.ndarray           # (N, 3) float32, lidar frame
    intensity: np.ndarray        # (N,) float32
    K: np.ndarray                # (3, 3) float64 pinhole intrinsics
    point_times: np.ndarray = None   # (N,) float32 seconds relative to scan start, or None
    ego_twist: np.ndarray = None     # (6,) [vx,vy,vz,wx,wy,wz] in lidar frame, or None
    labels_2d_gt: np.ndarray = None  # (H, W) uint8 in goose9, UNLABELED where none
    labels_3d_gt: np.ndarray = None  # (N,) uint8 in goose9, UNLABELED where none


@dataclass
class Prediction:
    labels_2d: np.ndarray        # (H, W) uint8 goose9 or UNLABELED
    labels_3d: np.ndarray        # (N,) uint8 goose9 or UNLABELED
    conf_2d: np.ndarray = None   # (H, W) float32 in [0,1], max-class prob; None -> ECE skipped
    conf_3d: np.ndarray = None   # (N,) float32 in [0,1]


def _check_measurements(frame_id, name, array, length, matches):
    """A per-point or fixed-length float vector. float32 and float64 are both
    accepted because adapters read whatever the file holds and the metrics do
    not care, unlike the label arrays below."""
    if array.ndim != 1 or array.shape[0] != length:
        raise ValueError(
            f"frame {frame_id}: {name} must be ({length},) to match {matches}, got {array.shape}"
        )

    if array.dtype.kind != "f":
        raise ValueError(f"frame {frame_id}: {name} must be floating point, got {array.dtype}")


def _check_labels(frame_id, name, array, shape, matches):
    """A ground truth label array. uint8 is not cosmetic: UNLABELED is 255, so a
    narrower or signed dtype aliases the sentinel onto a real class."""
    if array.shape != shape:
        raise ValueError(
            f"frame {frame_id}: {name} must be {shape} to match {matches}, got {array.shape}"
        )

    if array.dtype != np.uint8:
        raise ValueError(f"frame {frame_id}: {name} must be uint8, got {array.dtype}")


def validate(frame: Frame) -> None:
    """Raise ValueError naming the first field whose shape or dtype disagrees
    with the contract above.

    Frames are built by dataset adapters out of files on disk, so this is the
    system boundary and validation belongs exactly here. A projection or a
    confusion matrix built from a mismatched Frame does not crash, it quietly
    scores the wrong points against the wrong labels, which is the one failure
    mode this repo cannot afford. Nothing deeper re-checks.

    Point count is deliberately allowed to be zero: the dropout sweep hands
    baselines a near-empty cloud on purpose.
    """
    for name in REQUIRED_ARRAYS:
        value = getattr(frame, name)
        if not isinstance(value, np.ndarray):
            raise ValueError(
                f"frame {frame.frame_id}: {name} must be a numpy array, got {type(value).__name__}"
            )

    for name in OPTIONAL_ARRAYS:
        value = getattr(frame, name)
        if value is not None and not isinstance(value, np.ndarray):
            raise ValueError(
                f"frame {frame.frame_id}: {name} must be a numpy array or None, got {type(value).__name__}"
            )

    image = frame.image
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"frame {frame.frame_id}: image must be (H, W, 3), got {image.shape}")

    if image.dtype != np.uint8:
        raise ValueError(f"frame {frame.frame_id}: image must be uint8 RGB, got {image.dtype}")

    height, width = image.shape[:2]

    points = frame.points
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"frame {frame.frame_id}: points must be (N, 3), got {points.shape}")

    if points.dtype.kind != "f":
        raise ValueError(f"frame {frame.frame_id}: points must be floating point, got {points.dtype}")

    n_points = points.shape[0]

    if frame.K.shape != (3, 3):
        raise ValueError(f"frame {frame.frame_id}: K must be (3, 3), got {frame.K.shape}")

    if frame.K.dtype.kind != "f":
        raise ValueError(f"frame {frame.frame_id}: K must be floating point, got {frame.K.dtype}")

    _check_measurements(frame.frame_id, "intensity", frame.intensity, n_points, "the point count")

    if frame.point_times is not None:
        _check_measurements(frame.frame_id, "point_times", frame.point_times,
                            n_points, "the point count")

    if frame.ego_twist is not None:
        _check_measurements(frame.frame_id, "ego_twist", frame.ego_twist,
                            TWIST_LENGTH, "[vx,vy,vz,wx,wy,wz]")

    if frame.labels_3d_gt is not None:
        _check_labels(frame.frame_id, "labels_3d_gt", frame.labels_3d_gt,
                      (n_points,), "the point count")

    if frame.labels_2d_gt is not None:
        _check_labels(frame.frame_id, "labels_2d_gt", frame.labels_2d_gt,
                      (height, width), "the image")
