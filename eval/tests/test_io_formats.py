"""Unit tests for eval/io_formats.py: the Accumulator's per-frame folding, the
partition invariants that catch a flipped mask, and every validation failure
load_submission is required to raise.

A happy-path suite proves nothing about a validator, so each rejection is its
own case and each asserts the message names the offending field. Every metric
and the projection itself get a perturbation case: change one input a little
and the output has to move.

These tests run against the real semseg.projection and eval.metrics, so the
partition and z-buffer cases below check the shipped projection too, not a
stand-in for it.
"""

import json
import os
import re

import numpy as np
import pytest


from eval.io_formats import (Accumulator, FormatError, SUBMISSION_VERSION,
                             load_compute_meta, load_submission, save_submission)
from eval.metrics import ECE_BINS, RANGE_BIN_NAMES
from semseg.types import NUM_CLASSES, UNLABELED, Frame, Prediction

# Tiny synthetic world. Small enough to reason about by hand, big enough that
# every range bin and both sides of the frustum are populated.
IMAGE_H = 24
IMAGE_W = 32
FOCAL_PX = 20.0

# One range per bin edge region: 2/4 in 0-5m, 8/12 in 5-15m, 20/25 in 15-30m,
# 35/50 in 30m+. None sits near a bin edge, so the partition test is not
# measuring float rounding.
RANGES_M = (2.0, 4.0, 8.0, 12.0, 20.0, 25.0, 35.0, 50.0)

# Half the horizontal field of view is atan(16 / 20) = 38.6 degrees, so 0, 15
# and -20 land inside the image and 60, 130, 200 do not.
AZIMUTHS_DEG = (0.0, 15.0, -20.0, 60.0, 130.0, 200.0)

# Points landing exactly on a RANGE_BINS edge, on the optical axis so their
# norm is exact in float32. Without them a bin edge that lost its `>=` drops
# nothing and the partition invariant cannot see the bug.
EDGE_RANGES_M = (5.0, 15.0, 30.0)

# Big enough that the visible azimuth window barely overlaps the unperturbed
# one, so the frustum counts cannot coincide by luck.
YAW_PERTURB_DEG = 90.0

CLASS_CYCLE = 7          # fewer than NUM_CLASSES, so no label is out of range
SKY_CLASS = 8
GROUND_CLASS = 3
VEGETATION_CLASS = 6

# Rotates the lidar frame (x forward, y left, z up) into the camera optical
# frame (x right, y down, z forward).
LIDAR_TO_CAM_R = np.array([[0.0, -1.0, 0.0],
                           [0.0, 0.0, -1.0],
                           [1.0, 0.0, 0.0]])

INTRINSICS = np.array([[FOCAL_PX, 0.0, IMAGE_W / 2.0],
                       [0.0, FOCAL_PX, IMAGE_H / 2.0],
                       [0.0, 0.0, 1.0]])


def _extrinsic(yaw_deg: float = 0.0) -> np.ndarray:
    """T_cam_lidar for a camera bolted to the lidar, looking along +x of the
    lidar frame and yawed by `yaw_deg` about the lidar's up axis.
    """
    angle = np.radians(yaw_deg)
    yaw = np.array([[np.cos(angle), np.sin(angle), 0.0],
                    [-np.sin(angle), np.cos(angle), 0.0],
                    [0.0, 0.0, 1.0]])

    T = np.eye(4)
    T[:3, :3] = LIDAR_TO_CAM_R @ yaw
    return T


def _make_points() -> np.ndarray:
    points = []
    for azimuth in AZIMUTHS_DEG:
        angle = np.radians(azimuth)
        for i, distance in enumerate(RANGES_M):
            height = 0.4 * ((i % 3) - 1)     # stays well inside the vertical fov
            points.append((distance * np.cos(angle), distance * np.sin(angle), height))

    # Appended last, so the indices of the sweep above stay stable.
    for distance in EDGE_RANGES_M:
        points.append((distance, 0.0, 0.0))

    return np.array(points, dtype=np.float32)


def _make_labels_2d(offset: int):
    gt = np.full((IMAGE_H, IMAGE_W), SKY_CLASS, dtype=np.uint8)
    horizon = IMAGE_H // 2 + offset
    gt[horizon:, : IMAGE_W // 2] = GROUND_CLASS
    gt[horizon:, IMAGE_W // 2:] = VEGETATION_CLASS
    gt[0, 0] = UNLABELED                     # GOOSE leaves pixels unlabelled

    pred = np.full((IMAGE_H, IMAGE_W), SKY_CLASS, dtype=np.uint8)
    pred[horizon + 2:, : IMAGE_W // 2 + 3] = GROUND_CLASS
    pred[horizon + 2:, IMAGE_W // 2 + 3:] = VEGETATION_CLASS
    pred[-1, -1] = UNLABELED                 # a baseline declining to predict
    return gt, pred


def _make_labels_3d(count: int, offset: int):
    gt = (np.arange(count) + offset) % CLASS_CYCLE
    pred = gt.copy()
    pred[::5] = (pred[::5] + 1) % CLASS_CYCLE

    gt[0] = UNLABELED
    pred[3] = UNLABELED
    return gt.astype(np.uint8), pred.astype(np.uint8)


def _make_frame(frame_id: str, offset: int, points: np.ndarray = None):
    points = _make_points() if points is None else points
    count = len(points)

    gt_2d, pred_2d = _make_labels_2d(offset)
    gt_3d, pred_3d = _make_labels_3d(count, offset)

    frame = Frame(
        frame_id=frame_id,
        image=np.zeros((IMAGE_H, IMAGE_W, 3), dtype=np.uint8),
        points=points,
        intensity=np.linspace(0.0, 254.0, count, dtype=np.float32),
        K=INTRINSICS,
        labels_2d_gt=gt_2d,
        labels_3d_gt=gt_3d,
    )

    # linspace touches both 0.0 and exactly 1.0, so the top ECE bin edge is
    # exercised by the ordinary fixture and not only by its own test.
    pred = Prediction(
        labels_2d=pred_2d,
        labels_3d=pred_3d,
        conf_2d=np.linspace(0.0, 1.0, IMAGE_H * IMAGE_W,
                            dtype=np.float32).reshape(IMAGE_H, IMAGE_W),
        conf_3d=np.linspace(0.0, 1.0, count, dtype=np.float32),
    )
    return frame, pred


def _accumulate(frames=("f000", "f001"), T_cam_lidar=None) -> Accumulator:
    acc = Accumulator("bl_paint", "score")
    T = _extrinsic() if T_cam_lidar is None else T_cam_lidar

    for offset, frame_id in enumerate(frames):
        frame, pred = _make_frame(frame_id, offset)
        acc.add(frame, pred, T)

    return acc


def _valid_payload() -> dict:
    """One schema-valid payload, rebuilt per case so a mutation cannot leak
    into the next one. Built by the Accumulator rather than hand-written, which
    also asserts that what the writer emits is what the reader accepts.
    """
    return _accumulate().to_dict()


def _matrix(payload: dict, field: str) -> np.ndarray:
    return np.asarray(payload[field], dtype=np.int64)


def _write(tmp_path, payload: dict, name: str = "submission.json") -> str:
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return str(path)


def test_to_dict_has_the_whole_schema():
    payload = _accumulate().to_dict()

    assert payload["submission_version"] == SUBMISSION_VERSION
    assert payload["label_space"] == "goose9"
    assert payload["num_classes"] == NUM_CLASSES
    assert payload["method"] == "bl_paint"
    assert payload["split"] == "score"
    assert payload["n_frames"] == 2
    assert payload["frame_ids"] == ["f000", "f001"]

    for field in ("conf_2d", "conf_2d_boundary", "conf_3d",
                  "conf_3d_in_frustum", "conf_3d_out_frustum"):
        assert _matrix(payload, field).shape == (NUM_CLASSES, NUM_CLASSES)
        assert all(isinstance(v, int) for row in payload[field] for v in row)

    assert set(payload["conf_3d_by_range"]) == set(RANGE_BIN_NAMES)
    for name in RANGE_BIN_NAMES:
        assert np.asarray(payload["conf_3d_by_range"][name]).shape == (NUM_CLASSES, NUM_CLASSES)

    for field in ("ece_2d", "ece_3d"):
        for key in ("counts", "conf_sum", "correct"):
            assert len(payload[field][key]) == ECE_BINS


def test_range_bins_partition_conf_3d():
    payload = _accumulate().to_dict()

    total = sum(np.asarray(payload["conf_3d_by_range"][name], dtype=np.int64)
                for name in RANGE_BIN_NAMES)

    # RANGE_BINS starts at 0.0 and ends at inf, so the four half-open bins are
    # a complete partition. A flipped comparison or an off-by-one at a bin edge
    # breaks this and breaks nothing else in the suite.
    assert np.array_equal(total, _matrix(payload, "conf_3d"))

    for name in RANGE_BIN_NAMES:
        assert np.asarray(payload["conf_3d_by_range"][name]).sum() > 0


def _single_point_frame(point):
    """One labelled point and no 2D ground truth, so only the 3D folds move."""
    frame = Frame(
        frame_id="p000",
        image=np.zeros((IMAGE_H, IMAGE_W, 3), dtype=np.uint8),
        points=np.array([point], dtype=np.float32),
        intensity=np.zeros(1, dtype=np.float32),
        K=INTRINSICS,
        labels_3d_gt=np.array([GROUND_CLASS], dtype=np.uint8),
    )
    pred = Prediction(
        labels_2d=np.full((IMAGE_H, IMAGE_W), GROUND_CLASS, dtype=np.uint8),
        labels_3d=np.array([GROUND_CLASS], dtype=np.uint8),
    )
    return frame, pred


@pytest.mark.parametrize("point,expected", [
    ((3.0, 0.0, 0.0), "0-5m"),
    ((0.4, 12.0, 0.0), "5-15m"),      # small x, large norm: a per-axis bin gets this wrong
    ((0.0, 0.0, 20.0), "15-30m"),     # straight up, so the camera never sees it
    ((40.0, 0.0, 0.0), "30m+"),
])
def test_range_bin_uses_the_lidar_norm(point, expected):
    acc = Accumulator("bl_geom3d", "score")
    frame, pred = _single_point_frame(point)
    acc.add(frame, pred, _extrinsic())

    for name in RANGE_BIN_NAMES:
        assert acc.conf_3d_by_range[name].sum() == (1 if name == expected else 0)


def test_frustum_split_partitions_conf_3d():
    payload = _accumulate().to_dict()

    inside = _matrix(payload, "conf_3d_in_frustum")
    outside = _matrix(payload, "conf_3d_out_frustum")

    assert np.array_equal(inside + outside, _matrix(payload, "conf_3d"))

    # Both sides must be populated or the partition holds trivially. This is
    # also assumption A2 made visible: most points have no pixel.
    assert inside.sum() > 0
    assert outside.sum() > 0


def test_boundary_confusion_is_a_pixel_subset():
    payload = _accumulate().to_dict()

    boundary = _matrix(payload, "conf_2d_boundary")
    overall = _matrix(payload, "conf_2d")

    assert boundary.sum() > 0
    assert (boundary <= overall).all()

    # Strictly fewer, or the boundary matrix is just conf_2d under another
    # name. The fixture has large uniform blocks far from any gt boundary.
    assert boundary.sum() < overall.sum()


def test_consistency_counts_nest():
    counts = _accumulate().to_dict()["consistency"]

    assert counts["matched"] <= counts["scorable"]
    assert counts["scorable"] <= counts["in_frustum"]
    assert counts["in_frustum"] <= counts["total_points"]
    assert counts["total_points"] == 2 * len(_make_points())
    assert counts["in_frustum"] > 0


def _consistency_only_frame(points, label_2d, label_3d):
    """A frame with no ground truth at all, so only the consistency counts move.
    Consistency needs no labels (spec 6.1), which is what makes it computable on
    an unlabelled robot log.
    """
    count = len(points)
    frame = Frame(
        frame_id="c000",
        image=np.zeros((IMAGE_H, IMAGE_W, 3), dtype=np.uint8),
        points=np.asarray(points, dtype=np.float32),
        intensity=np.zeros(count, dtype=np.float32),
        K=INTRINSICS,
    )
    pred = Prediction(
        labels_2d=np.full((IMAGE_H, IMAGE_W), label_2d, dtype=np.uint8),
        labels_3d=np.full(count, label_3d, dtype=np.uint8),
    )
    return frame, pred


def test_agreeing_labels_are_all_matched():
    acc = Accumulator("bl_paint", "score")
    frame, pred = _consistency_only_frame([(5.0, 0.0, 0.0)], GROUND_CLASS, GROUND_CLASS)
    acc.add(frame, pred, _extrinsic())

    counts = acc.to_dict()["consistency"]
    assert counts == {"matched": 1, "scorable": 1, "in_frustum": 1, "total_points": 1}


def test_disagreeing_labels_are_scorable_but_unmatched():
    acc = Accumulator("bl_paint", "score")
    frame, pred = _consistency_only_frame([(5.0, 0.0, 0.0)], VEGETATION_CLASS, GROUND_CLASS)
    acc.add(frame, pred, _extrinsic())

    counts = acc.to_dict()["consistency"]
    assert counts["scorable"] == 1
    assert counts["matched"] == 0


@pytest.mark.parametrize("label_2d,label_3d", [(UNLABELED, GROUND_CLASS),
                                               (GROUND_CLASS, UNLABELED)])
def test_unlabeled_either_side_is_not_scorable(label_2d, label_3d):
    acc = Accumulator("bl_paint", "score")
    frame, pred = _consistency_only_frame([(5.0, 0.0, 0.0)], label_2d, label_3d)
    acc.add(frame, pred, _extrinsic())

    counts = acc.to_dict()["consistency"]
    assert counts["in_frustum"] == 1
    assert counts["scorable"] == 0
    assert counts["matched"] == 0


def test_occluded_point_is_not_scorable():
    # Two points on the optical axis land on the same pixel. Without the
    # z-buffer condition the far one is scored against the near one's label.
    acc = Accumulator("bl_paint", "score")
    frame, pred = _consistency_only_frame([(5.0, 0.0, 0.0), (10.0, 0.0, 0.0)],
                                          GROUND_CLASS, GROUND_CLASS)
    acc.add(frame, pred, _extrinsic())

    counts = acc.to_dict()["consistency"]
    assert counts["in_frustum"] == 2
    assert counts["scorable"] == 1
    assert counts["matched"] == 1


def _ece_frame(gt_label, pred_label, confidence):
    """One labelled point at a known confidence, so the ECE arrays can be
    checked against a hand-computed bin rather than only for movement.
    """
    frame = Frame(
        frame_id="e000",
        image=np.zeros((IMAGE_H, IMAGE_W, 3), dtype=np.uint8),
        points=np.array([(5.0, 0.0, 0.0)], dtype=np.float32),
        intensity=np.zeros(1, dtype=np.float32),
        K=INTRINSICS,
        labels_3d_gt=np.array([gt_label], dtype=np.uint8),
    )
    pred = Prediction(
        labels_2d=np.full((IMAGE_H, IMAGE_W), GROUND_CLASS, dtype=np.uint8),
        labels_3d=np.array([pred_label], dtype=np.uint8),
        conf_3d=np.array([confidence], dtype=np.float32),
    )
    return frame, pred


def _ece_3d(gt_label, pred_label, confidence) -> dict:
    acc = Accumulator("bl_geom3d", "score")
    frame, pred = _ece_frame(gt_label, pred_label, confidence)
    acc.add(frame, pred, _extrinsic())
    return acc.to_dict()["ece_3d"]


HALF_CONF = 0.5
HALF_CONF_BIN = 7          # 0.5 * 15 bins = 7.5, so it lands in bin 7


def test_ece_bins_a_correct_point():
    ece = _ece_3d(GROUND_CLASS, GROUND_CLASS, HALF_CONF)

    assert sum(ece["counts"]) == 1
    assert ece["counts"][HALF_CONF_BIN] == 1
    assert ece["conf_sum"][HALF_CONF_BIN] == pytest.approx(HALF_CONF)
    assert ece["correct"][HALF_CONF_BIN] == 1


def test_ece_marks_a_wrong_point_incorrect():
    ece = _ece_3d(GROUND_CLASS, VEGETATION_CLASS, HALF_CONF)

    # Same bin and same confidence as the correct point above. Only `correct`
    # separates a calibrated head from an overconfident one, so a suite that
    # checks the counts alone cannot see that column at all.
    assert ece["counts"][HALF_CONF_BIN] == 1
    assert ece["correct"][HALF_CONF_BIN] == 0


@pytest.mark.parametrize("gt_label,pred_label", [(UNLABELED, GROUND_CLASS),
                                                 (GROUND_CLASS, UNLABELED)])
def test_ece_drops_unlabeled_elements(gt_label, pred_label):
    # An element with no ground truth, or a prediction the baseline declined to
    # make, is not a calibration data point. Binning it would charge the head
    # for a confidence it never claimed.
    assert sum(_ece_3d(gt_label, pred_label, HALF_CONF)["counts"]) == 0


def test_ece_skipped_without_confidence():
    acc = Accumulator("bl_paint", "score")
    frame, pred = _make_frame("f000", 0)
    pred.conf_2d = None
    pred.conf_3d = None
    acc.add(frame, pred, _extrinsic())

    payload = acc.to_dict()
    for field in ("ece_2d", "ece_3d"):
        assert sum(payload[field]["counts"]) == 0
        assert sum(payload[field]["conf_sum"]) == 0.0

    # The confusion matrices are still folded in: a missing confidence costs
    # the ECE, not the score.
    assert _matrix(payload, "conf_2d").sum() > 0


def test_confidence_of_one_lands_in_the_top_bin():
    acc = Accumulator("bl_paint", "score")
    frame, pred = _make_frame("f000", 0)
    pred.conf_3d = np.ones(len(frame.points), dtype=np.float32)
    pred.conf_2d = None
    acc.add(frame, pred, _extrinsic())

    counts = acc.to_dict()["ece_3d"]["counts"]
    assert counts[-1] == sum(counts)
    assert counts[-1] > 0


def test_ece_moves_when_a_confidence_moves():
    frame, pred = _make_frame("f000", 0)

    before = Accumulator("bl_paint", "score")
    before.add(frame, pred, _extrinsic())

    nudged = Prediction(labels_2d=pred.labels_2d, labels_3d=pred.labels_3d,
                        conf_2d=pred.conf_2d, conf_3d=pred.conf_3d.copy())
    nudged.conf_3d[10] = 1.0

    after = Accumulator("bl_paint", "score")
    after.add(frame, nudged, _extrinsic())

    assert (before.to_dict()["ece_3d"]["conf_sum"]
            != after.to_dict()["ece_3d"]["conf_sum"])


def test_moving_a_point_across_a_bin_edge_moves_only_the_bins():
    points = _make_points()
    moved = points.copy()

    # Index 10 sits at azimuth 15 degrees and 8 m, so it is in the frustum and
    # in the 5-15m bin. Halving its range keeps the label and the pixel but
    # changes the bin it lands in.
    moved[10] = moved[10] * 0.25

    before = Accumulator("bl_geom3d", "score")
    frame, pred = _make_frame("f000", 0, points)
    before.add(frame, pred, _extrinsic())

    after = Accumulator("bl_geom3d", "score")
    frame_moved, pred_moved = _make_frame("f000", 0, moved)
    after.add(frame_moved, pred_moved, _extrinsic())

    assert np.array_equal(before.conf_3d, after.conf_3d)
    assert not np.array_equal(before.conf_3d_by_range["5-15m"],
                              after.conf_3d_by_range["5-15m"])
    assert not np.array_equal(before.conf_3d_by_range["0-5m"],
                              after.conf_3d_by_range["0-5m"])


def test_perturbing_the_extrinsic_moves_the_frustum():
    # The only test that proves T_cam_lidar is threaded through add() at all: a
    # label perturbation passes even when the extrinsic is ignored entirely.
    before = _accumulate(frames=("f000",)).to_dict()
    after = _accumulate(frames=("f000",),
                        T_cam_lidar=_extrinsic(YAW_PERTURB_DEG)).to_dict()

    assert before["consistency"]["in_frustum"] != after["consistency"]["in_frustum"]
    assert not np.array_equal(_matrix(before, "conf_3d_in_frustum"),
                              _matrix(after, "conf_3d_in_frustum"))

    # The whole-cloud matrix does not depend on the extrinsic, only the split.
    assert np.array_equal(_matrix(before, "conf_3d"), _matrix(after, "conf_3d"))


def test_round_trip_is_identical(tmp_path):
    payload = _accumulate().to_dict()
    path = str(tmp_path / "submission.json")

    save_submission(path, payload)
    loaded = load_submission(path)

    for field in ("conf_2d", "conf_2d_boundary", "conf_3d",
                  "conf_3d_in_frustum", "conf_3d_out_frustum"):
        assert np.array_equal(loaded[field], _matrix(payload, field))
        assert loaded[field].dtype == np.int64

    for name in RANGE_BIN_NAMES:
        assert np.array_equal(loaded["conf_3d_by_range"][name],
                              np.asarray(payload["conf_3d_by_range"][name]))

    for field in ("ece_2d", "ece_3d"):
        for key in ("counts", "conf_sum", "correct"):
            assert np.allclose(loaded[field][key], payload[field][key])

    assert loaded["consistency"] == payload["consistency"]
    assert loaded["frame_ids"] == payload["frame_ids"]


def test_one_flipped_label_changes_the_saved_matrices(tmp_path):
    frame, pred = _make_frame("f000", 0)

    before = Accumulator("bl_paint", "score")
    before.add(frame, pred, _extrinsic())

    flipped = Prediction(labels_2d=pred.labels_2d, labels_3d=pred.labels_3d.copy(),
                         conf_2d=pred.conf_2d, conf_3d=pred.conf_3d)
    flipped.labels_3d[10] = (flipped.labels_3d[10] + 1) % CLASS_CYCLE

    after = Accumulator("bl_paint", "score")
    after.add(frame, flipped, _extrinsic())

    path_before = str(tmp_path / "before.json")
    path_after = str(tmp_path / "after.json")
    save_submission(path_before, before.to_dict())
    save_submission(path_after, after.to_dict())

    assert not np.array_equal(load_submission(path_before)["conf_3d"],
                              load_submission(path_after)["conf_3d"])


def test_one_flipped_pixel_changes_the_2d_matrices(tmp_path):
    frame, pred = _make_frame("f000", 0)

    before = Accumulator("bl_cam2d", "score")
    before.add(frame, pred, _extrinsic())

    flipped = Prediction(labels_2d=pred.labels_2d.copy(), labels_3d=pred.labels_3d,
                         conf_2d=pred.conf_2d, conf_3d=pred.conf_3d)
    horizon = IMAGE_H // 2      # a gt class boundary under any definition of one
    flipped.labels_2d[horizon, :] = VEGETATION_CLASS

    after = Accumulator("bl_cam2d", "score")
    after.add(frame, flipped, _extrinsic())

    assert not np.array_equal(before.conf_2d, after.conf_2d)
    assert not np.array_equal(before.conf_2d_boundary, after.conf_2d_boundary)


def _set_cell(payload: dict, field: str, row: int, column: int, value) -> None:
    payload[field][row][column] = value


# (case id, one mutation of a valid payload, the field the message must name)
VALIDATION_CASES = (
    ("version_missing", lambda p: p.pop("submission_version"), "submission_version"),
    ("version_wrong", lambda p: p.update(submission_version=99), "submission_version"),
    ("label_space_wrong", lambda p: p.update(label_space="goose8"), "label_space"),
    ("num_classes_wrong", lambda p: p.update(num_classes=8), "num_classes"),
    ("matrix_missing", lambda p: p.pop("conf_3d"), "conf_3d"),
    ("matrix_too_small", lambda p: p.update(conf_2d=p["conf_2d"][:8]), "conf_2d"),
    ("matrix_ragged", lambda p: p.update(conf_3d=[row[:8] for row in p["conf_3d"][:8]]
                                         + [p["conf_3d"][8]]), "conf_3d"),
    ("matrix_not_integral", lambda p: _set_cell(p, "conf_3d", 0, 0, 0.5), "conf_3d"),
    ("matrix_negative", lambda p: _set_cell(p, "conf_2d_boundary", 1, 2, -3),
     "conf_2d_boundary"),
    ("range_bins_missing", lambda p: p.pop("conf_3d_by_range"), "conf_3d_by_range"),
    ("range_bin_missing", lambda p: p["conf_3d_by_range"].pop("30m+"), "30m+"),
    ("range_bin_too_small",
     lambda p: p["conf_3d_by_range"].update({"5-15m": p["conf_3d_by_range"]["5-15m"][:8]}),
     "conf_3d_by_range.5-15m"),
    ("ece_block_missing", lambda p: p.pop("ece_2d"), "ece_2d"),
    ("ece_array_missing", lambda p: p["ece_3d"].pop("correct"), "ece_3d.correct"),
    ("ece_array_too_short",
     lambda p: p["ece_2d"].update(counts=p["ece_2d"]["counts"][:-1]), "ece_2d.counts"),
    ("consistency_missing", lambda p: p.pop("consistency"), "consistency"),
    ("consistency_field_missing", lambda p: p["consistency"].pop("in_frustum"),
     "consistency.in_frustum"),
    ("consistency_negative", lambda p: p["consistency"].update(total_points=-1),
     "consistency.total_points"),
    ("consistency_not_an_int", lambda p: p["consistency"].update(scorable=1.5),
     "consistency.scorable"),
    ("consistency_not_nested",
     lambda p: p["consistency"].update(matched=p["consistency"]["scorable"] + 1),
     "consistency.matched"),
    ("frame_count_mismatch", lambda p: p.update(n_frames=99), "n_frames"),
)


@pytest.mark.parametrize("mutate,field",
                         [case[1:] for case in VALIDATION_CASES],
                         ids=[case[0] for case in VALIDATION_CASES])
def test_validation_failure_names_the_field(tmp_path, mutate, field):
    payload = _valid_payload()
    mutate(payload)
    path = _write(tmp_path, payload)

    with pytest.raises(FormatError, match=re.escape(field)):
        load_submission(path)


def test_valid_payload_loads(tmp_path):
    # The counterweight to the cases above: the same payload, unmutated, must
    # pass, or every rejection above could be firing for the wrong reason.
    path = _write(tmp_path, _valid_payload())
    assert load_submission(path)["n_frames"] == 2


def test_non_object_top_level_is_rejected(tmp_path):
    path = tmp_path / "submission.json"
    path.write_text("[1, 2, 3]")

    with pytest.raises(FormatError, match="top level"):
        load_submission(str(path))


def test_save_rejects_a_broken_payload_before_writing(tmp_path):
    payload = _valid_payload()
    payload["label_space"] = "cityscapes19"
    path = str(tmp_path / "submission.json")

    with pytest.raises(FormatError, match="label_space"):
        save_submission(path, payload)

    assert not os.path.exists(path)


def test_numpy_counts_are_rejected(tmp_path):
    # Deliberate strictness. A numpy scalar would sail past a looser check and
    # then fail inside json.dump, which names no field at all. Writers go
    # through Accumulator.to_dict(), which casts.
    payload = _valid_payload()
    payload["consistency"]["matched"] = np.int64(payload["consistency"]["matched"])
    path = str(tmp_path / "submission.json")

    with pytest.raises(FormatError, match=re.escape("consistency.matched")):
        save_submission(path, payload)

    assert not os.path.exists(path)


def test_compute_meta_is_optional(tmp_path):
    path = str(tmp_path / "submission.json")
    save_submission(path, _valid_payload())

    # Absent sidecar: None, never an exception. Compute is a self-declared KPI
    # and its absence must not fail a scoring run.
    assert load_compute_meta(path) is None


def test_compute_meta_is_read_when_present(tmp_path):
    path = str(tmp_path / "submission.json")
    sidecar = tmp_path / "submission.json.meta.json"
    sidecar.write_text(json.dumps({
        "runtime_sec_total": 42.0,
        "runtime_sec_per_frame": 0.0875,
        "peak_rss_mb": 1024.0,
        "method_name": "bl_paint",
        "machine": "workstation",
    }))

    meta = load_compute_meta(path)
    assert meta["runtime_sec_per_frame"] == pytest.approx(0.0875)
    assert meta["method_name"] == "bl_paint"


def test_compute_meta_fields_default_to_none(tmp_path):
    path = str(tmp_path / "submission.json")
    sidecar = tmp_path / "submission.json.meta.json"
    sidecar.write_text(json.dumps({"runtime_sec_total": 1.0}))

    meta = load_compute_meta(path)
    assert meta["runtime_sec_total"] == pytest.approx(1.0)
    assert meta["peak_rss_mb"] is None
