"""Accumulation and file formats for the semseg challenge (spec ch. 6). Kept
separate from metrics.py so the math there stays pure: this file owns both
boundaries, raw label arrays on the way in and validated JSON on the way out.

The Accumulator is the only object in the repo that touches raw label arrays.
Everything downstream reads confusion matrices instead, so a committed
submission stays a few kilobytes and the scorer never needs the label files.

submission.json:

    {
      "submission_version": 1,
      "method": "bl_paint",
      "label_space": "goose9",
      "num_classes": 9,
      "split": "score",
      "n_frames": 480,
      "conf_2d": [[...]],               9x9 int, rows = gt, cols = pred
      "conf_2d_boundary": [[...]],      the same, near gt class boundaries only
      "conf_3d": [[...]],
      "conf_3d_by_range": {"0-5m": [[...]], "5-15m": [[...]],
                           "15-30m": [[...]], "30m+": [[...]]},
      "conf_3d_in_frustum": [[...]],
      "conf_3d_out_frustum": [[...]],
      "ece_2d": {"counts": [...], "conf_sum": [...], "correct": [...]},
      "ece_3d": {"counts": [...], "conf_sum": [...], "correct": [...]},
      "projection_available": bool(self.projection_available),
      "consistency": {"matched": 812431, "scorable": 1044902,
                      "in_frustum": 1102300, "total_points": 3117655},
      "frame_ids": ["...", "..."]
    }

The round trip is deliberately asymmetric. `Accumulator.to_dict` emits plain
nested lists of ints so the file stays diffable JSON, while `load_submission`
hands back (9, 9) int64 arrays, because every consumer of a matrix does
arithmetic on it.

Pixel convention, since the projection returns one array for both axes:
`uv[:, 0]` is the column and `uv[:, 1]` is the row, from `pixel = u[:2] / u[2]`
in spec ch. 3.
"""

import json
import os

import numpy as np

from eval.metrics import (confusion, boundary_confusion, ece_accumulate, EceAccum,
                          RANGE_BINS, RANGE_BIN_NAMES, ECE_BINS)
from semseg.projection import project, zbuffer
from semseg.types import Frame, Prediction, NUM_CLASSES, UNLABELED

SUBMISSION_VERSION = 1

# The one label space this repo scores in (spec ch. 1). A submission in any
# other space is not comparable, so it is rejected rather than quietly remapped.
LABEL_SPACE = "goose9"

# Every top-level 9x9 matrix the schema requires. conf_3d_by_range is validated
# separately because it is nested one level deeper.
MATRIX_FIELDS = ("conf_2d", "conf_2d_boundary", "conf_3d",
                 "conf_3d_in_frustum", "conf_3d_out_frustum")

ECE_FIELDS = ("ece_2d", "ece_3d")

# These are the EceAccum field names as well as the sidecar keys, which is what
# lets to_dict copy one straight into the other.
ECE_ARRAYS = ("counts", "conf_sum", "correct")

# Validated as a chain, so this tuple order IS the invariant: a matched point
# is scorable, a scorable point is in the frustum, and every point is counted.
CONSISTENCY_FIELDS = ("matched", "scorable", "in_frustum", "total_points")

META_SUFFIX = ".meta.json"
JSON_INDENT = 2


class FormatError(ValueError):
    pass


def _zero_matrix() -> np.ndarray:
    return np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)


def _fold_ece(conf, gt, pred) -> EceAccum:
    """One frame's calibration counts. The binning itself belongs to
    metrics.ece_accumulate: two copies of the bin edges would be two things to
    keep in step. UNLABELED is dropped on both sides under the same rule
    confusion() applies, so a declined prediction is not charged here either.

    An out-of-range or nan confidence raises from ece_accumulate rather than
    being clamped here. A baseline emitting one has a bug, and a silently
    clamped confidence would show up as a calibration result.
    """
    valid = (gt != UNLABELED) & (pred != UNLABELED)
    return ece_accumulate(np.asarray(conf)[valid], gt[valid] == pred[valid])


class Accumulator:
    """Folds per-frame predictions into the committable sidecar."""

    def __init__(self, method: str, split: str):
        self.method = method
        self.split = split
        self.frame_ids = []

        # True until a frame arrives with no extrinsic. Recorded rather than
        # inferred from an all-zero frustum matrix, because "no camera" and "a
        # camera that saw nothing" are different runs and a reader must be able
        # to tell them apart.
        self.projection_available = True

        self.conf_2d = _zero_matrix()
        self.conf_2d_boundary = _zero_matrix()

        self.conf_3d = _zero_matrix()
        self.conf_3d_by_range = {name: _zero_matrix() for name in RANGE_BIN_NAMES}
        self.conf_3d_in_frustum = _zero_matrix()
        self.conf_3d_out_frustum = _zero_matrix()

        self.ece_2d = EceAccum.zeros()
        self.ece_3d = EceAccum.zeros()

        self.consistency = {name: 0 for name in CONSISTENCY_FIELDS}

    def add(self, frame: Frame, pred: Prediction, T_cam_lidar) -> None:
        # The image carries its own resolution. GOOSE shipped 1000x2048 in the
        # sampled frame, but that is not a promise (spec ch. 1b).
        height, width = frame.image.shape[:2]

        # T_cam_lidar is None on a dataset that ships no extrinsic, which real
        # GOOSE val does not (spec 13.3). Everything that needs a projection is
        # then genuinely unanswerable, so it is DECLINED rather than computed
        # against a fabricated rig: no frustum split, no consistency counts.
        # The 3D confusion, its range bins and both ECE histograms need no
        # projection at all and are still exactly right, which is what keeps a
        # lidar-only arm and the chance floor runnable on the public release.
        if T_cam_lidar is None:
            self.projection_available = False
            self._fold_2d(frame, pred)
            self._fold_3d(frame, pred, in_frustum=None)
            self.frame_ids.append(frame.frame_id)
            return

        uv, depth, in_frustum = project(frame.points, frame.K, T_cam_lidar, width, height)

        # Only the per-point flag is needed here. The dense owner map is what
        # produced it and is of no further use to a confusion matrix.
        _owner, visible = zbuffer(uv, depth, in_frustum, width, height)

        self._fold_2d(frame, pred)
        self._fold_3d(frame, pred, in_frustum)
        self._fold_consistency(pred, uv, in_frustum, visible)

        self.frame_ids.append(frame.frame_id)

    def _fold_2d(self, frame: Frame, pred: Prediction) -> None:
        """Skipped when the frame carries no 2D ground truth. Consistency needs
        none (spec 6.1), so an unlabelled robot log is still worth folding in.
        """
        if frame.labels_2d_gt is None:
            return

        self.conf_2d += confusion(frame.labels_2d_gt, pred.labels_2d)
        self.conf_2d_boundary += boundary_confusion(frame.labels_2d_gt, pred.labels_2d)

        if pred.conf_2d is not None:
            self.ece_2d += _fold_ece(pred.conf_2d, frame.labels_2d_gt, pred.labels_2d)

    def _fold_3d(self, frame: Frame, pred: Prediction, in_frustum) -> None:
        if frame.labels_3d_gt is None:
            return

        gt, labels = frame.labels_3d_gt, pred.labels_3d
        self.conf_3d += confusion(gt, labels)

        # Range is measured from the lidar origin, never camera depth. The bins
        # exist because point density falls off as 1/r^2 (spec 6.1), which is a
        # property of the sensor and not of the camera frustum.
        point_range = np.linalg.norm(frame.points, axis=1)

        for name, (near, far) in zip(RANGE_BIN_NAMES, RANGE_BINS):
            # Half-open [near, far), so the bins partition every point exactly
            # once and the bin matrices sum back to conf_3d.
            in_bin = (point_range >= near) & (point_range < far)
            self.conf_3d_by_range[name] += confusion(gt[in_bin], labels[in_bin])

        # Spec 6.2 asks how much of a reported fusion gain comes from points the
        # camera never saw. That is only answerable if the split is kept from
        # the start, because the frustum mask cannot be recovered from a matrix.
        # With no extrinsic there is no frustum to split on, and both matrices
        # stay at zero, which reads downstream as an undefined mIoU rather than
        # as a score of nothing.
        if in_frustum is not None:
            self.conf_3d_in_frustum += confusion(gt[in_frustum], labels[in_frustum])
            self.conf_3d_out_frustum += confusion(gt[~in_frustum], labels[~in_frustum])

        if pred.conf_3d is not None:
            self.ece_3d += _fold_ece(pred.conf_3d, gt, labels)

    def _fold_consistency(self, pred: Prediction, uv, in_frustum, visible) -> None:
        """Cross-modal consistency counts (spec 6.1). No ground truth is
        involved, which is what makes this the one metric computable on an
        unlabelled log.
        """
        counts = self.consistency
        counts["total_points"] += int(in_frustum.size)
        counts["in_frustum"] += int(in_frustum.sum())

        # project() zeroes the uv rows it rejected, so the 2D label must be
        # gathered on the frustum subset. An unmasked gather would read pixel
        # (0, 0) for every point the camera never saw (assumption A2: most
        # points have no pixel at all).
        label_2d = pred.labels_2d[uv[in_frustum, 1], uv[in_frustum, 0]]
        label_3d = pred.labels_3d[in_frustum]

        # A point is scorable only when all three of these hold, and dropping
        # any one of them changes what the ratio means:
        #   visible, so the point owns its pixel in the z-buffer. Without it an
        #     occluded point is scored against the occluder's label.
        #   labelled in 3D, because UNLABELED never matches anything, so a
        #     baseline that declines to predict would otherwise shrink its own
        #     denominator and look consistent by abstaining.
        #   labelled in 2D, the same argument from the other modality. Sky is a
        #     real class in goose9 (spec ch. 1) and is scored; UNLABELED is not.
        scorable = visible[in_frustum] & (label_3d != UNLABELED) & (label_2d != UNLABELED)

        counts["scorable"] += int(scorable.sum())
        counts["matched"] += int((scorable & (label_3d == label_2d)).sum())

    def to_dict(self) -> dict:
        payload = {
            "submission_version": SUBMISSION_VERSION,
            "method": self.method,
            "label_space": LABEL_SPACE,
            "num_classes": NUM_CLASSES,
            "split": self.split,
            "n_frames": len(self.frame_ids),
        }

        for field in MATRIX_FIELDS:
            payload[field] = getattr(self, field).tolist()

        payload["conf_3d_by_range"] = {name: matrix.tolist()
                                       for name, matrix in self.conf_3d_by_range.items()}

        for field in ECE_FIELDS:
            accum = getattr(self, field)
            payload[field] = {key: getattr(accum, key).tolist() for key in ECE_ARRAYS}

        payload["consistency"] = {name: int(self.consistency[name])
                                  for name in CONSISTENCY_FIELDS}

        # Carried so a reader can distinguish a run with no camera geometry from
        # one whose camera saw nothing. Both leave the frustum matrices at zero.
        payload["projection_available"] = bool(self.projection_available)
        payload["frame_ids"] = list(self.frame_ids)

        return payload


def _require(payload: dict, field: str, path: str):
    if field not in payload:
        raise FormatError(f"{path}: missing required field '{field}'")
    return payload[field]


def _as_matrix(raw, field: str, path: str) -> np.ndarray:
    try:
        matrix = np.asarray(raw)
    except ValueError as exc:
        # A ragged nested list is the shape a hand-edited file corrupts into.
        raise FormatError(f"{path}: field '{field}' is not a rectangular matrix: {exc}")

    if matrix.shape != (NUM_CLASSES, NUM_CLASSES):
        raise FormatError(f"{path}: field '{field}' must be "
                          f"{NUM_CLASSES}x{NUM_CLASSES}, got shape {matrix.shape}")

    if not np.issubdtype(matrix.dtype, np.integer):
        raise FormatError(f"{path}: field '{field}' must hold integer counts, "
                          f"got dtype {matrix.dtype}")

    if (matrix < 0).any():
        raise FormatError(f"{path}: field '{field}' holds a negative count ({matrix.min()})")

    return matrix.astype(np.int64)


def _as_ece(raw, field: str, path: str) -> dict:
    if not isinstance(raw, dict):
        raise FormatError(f"{path}: field '{field}' must be an object holding {ECE_ARRAYS}")

    parsed = {}
    for key in ECE_ARRAYS:
        if key not in raw:
            raise FormatError(f"{path}: missing required field '{field}.{key}'")

        # All three are read as floats: metrics.ece() only ever divides them,
        # and re-typing the counts would silently truncate a malformed file.
        values = np.asarray(raw[key], dtype=np.float64)
        if values.shape != (ECE_BINS,):
            raise FormatError(f"{path}: field '{field}.{key}' must have {ECE_BINS} "
                              f"entries, got shape {values.shape}")

        parsed[key] = values

    return parsed


def _as_consistency(raw, path: str) -> dict:
    if not isinstance(raw, dict):
        raise FormatError(f"{path}: field 'consistency' must be an object "
                          f"holding {CONSISTENCY_FIELDS}")

    counts = {}
    for name in CONSISTENCY_FIELDS:
        if name not in raw:
            raise FormatError(f"{path}: missing required field 'consistency.{name}'")

        value = raw[name]

        # np.int64 is deliberately rejected too. Accepting it here would only
        # move the failure into json.dump, which reports "Object of type int64
        # is not JSON serializable" and names no field at all.
        if not isinstance(value, int):
            raise FormatError(f"{path}: field 'consistency.{name}' must be a plain "
                              f"JSON-serialisable int, got {type(value).__name__}")

        if value < 0:
            raise FormatError(f"{path}: field 'consistency.{name}' is negative ({value})")

        counts[name] = value

    # The counts must nest, since each is a subset of the next. A violation
    # means the writer mislabelled one of them, and a consistency ratio read
    # against the wrong denominator is worse than no number at all.
    for inner, outer in zip(CONSISTENCY_FIELDS, CONSISTENCY_FIELDS[1:]):
        if counts[inner] > counts[outer]:
            raise FormatError(f"{path}: field 'consistency.{inner}' ({counts[inner]}) "
                              f"exceeds 'consistency.{outer}' ({counts[outer]})")

    return counts


def _parse(payload: dict, path: str) -> dict:
    """Validates every field the schema requires and returns the payload with
    the matrices as int64 arrays. One function, so the writer and the reader
    cannot drift apart: save_submission validates with it too.
    """
    version = _require(payload, "submission_version", path)
    if version != SUBMISSION_VERSION:
        raise FormatError(f"{path}: field 'submission_version' is {version}, "
                          f"this scorer only reads {SUBMISSION_VERSION}")

    label_space = _require(payload, "label_space", path)
    if label_space != LABEL_SPACE:
        raise FormatError(f"{path}: field 'label_space' is '{label_space}', "
                          f"expected '{LABEL_SPACE}'")

    num_classes = _require(payload, "num_classes", path)
    if num_classes != NUM_CLASSES:
        raise FormatError(f"{path}: field 'num_classes' is {num_classes}, "
                          f"expected {NUM_CLASSES}")

    parsed = dict(payload)

    for field in MATRIX_FIELDS:
        parsed[field] = _as_matrix(_require(payload, field, path), field, path)

    by_range = _require(payload, "conf_3d_by_range", path)
    if not isinstance(by_range, dict):
        raise FormatError(f"{path}: field 'conf_3d_by_range' must be an object "
                          f"keyed by range bin name")

    parsed["conf_3d_by_range"] = {}
    for name in RANGE_BIN_NAMES:
        if name not in by_range:
            raise FormatError(f"{path}: field 'conf_3d_by_range' is missing range bin '{name}'")

        parsed["conf_3d_by_range"][name] = _as_matrix(
            by_range[name], f"conf_3d_by_range.{name}", path)

    for field in ECE_FIELDS:
        parsed[field] = _as_ece(_require(payload, field, path), field, path)

    parsed["consistency"] = _as_consistency(_require(payload, "consistency", path), path)

    # A mismatch here means the writer lost or double-counted frames, which
    # would quietly rescale every per-frame number the scorer reports.
    n_frames = _require(payload, "n_frames", path)
    frame_ids = _require(payload, "frame_ids", path)
    if len(frame_ids) != n_frames:
        raise FormatError(f"{path}: field 'n_frames' is {n_frames} but 'frame_ids' "
                          f"holds {len(frame_ids)} entries")

    return parsed


def save_submission(path: str, payload: dict) -> None:
    """Writes an Accumulator.to_dict() payload as diffable JSON. Validated
    first, so a broken writer fails here rather than hours later in the scorer,
    and nothing is created on disk when validation fails.

    `payload` must hold JSON types only, which is what to_dict() produces. A
    hand-built dict carrying numpy scalars or arrays is not accepted: the whole
    point of the sidecar is that it is a plain diffable file.
    """
    _parse(payload, path)

    with open(path, "w") as f:
        json.dump(payload, f, indent=JSON_INDENT)
        f.write("\n")


def load_submission(path: str) -> dict:
    """Returns the payload with every matrix as a (9, 9) int64 array and every
    ECE array as float64. Raises FormatError naming the offending field.

    There is no `--gt` counterpart to this in the scorer: ground truth is
    already folded into the matrices by the Accumulator (spec ch. 7).
    """
    with open(path) as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise FormatError(f"{path}: top level must be a JSON object, "
                          f"got {type(payload).__name__}")

    return _parse(payload, path)


def load_compute_meta(submission_path: str):
    """Reads the `<submission>.meta.json` sidecar a baseline writes next to its
    submission. Returns None when there is none: compute is an optional,
    self-declared KPI (spec 6.4) and its absence must never fail a scoring run.
    It is reported beside the score and never folded into it, because ranking on
    a self-declared number would be trivially gameable.
    """
    path = submission_path + META_SUFFIX
    if not os.path.exists(path):
        return None

    with open(path) as f:
        meta = json.load(f)

    return {
        "runtime_sec_total": meta.get("runtime_sec_total"),
        "runtime_sec_per_frame": meta.get("runtime_sec_per_frame"),
        "peak_rss_mb": meta.get("peak_rss_mb"),
        "method_name": meta.get("method_name"),
        "machine": meta.get("machine"),
    }
