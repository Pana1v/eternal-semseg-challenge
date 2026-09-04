"""Segmentation scoring math (problem statement section 6.1): confusion
matrices, the IoU family, boundary IoU, expected calibration error and
cross-modal consistency.

Pure math and zero I/O, kept separate from score.py for the same reason the
gloc scorer splits them: every formula here is checkable against a small
hand-computed matrix, and nothing in this file needs a label file, a frame or
a path to be exercised.

Everything downstream derives from a confusion matrix. That is why the
committed submission carries matrices instead of label arrays: the artefact
stays kilobytes and the scorer never re-reads the raw ground truth.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from semseg.types import NUM_CLASSES, UNLABELED

# 3D IoU is reported per range bin because lidar point density falls off as
# 1/r^2: one pooled number hides that the model is failing at 30 m while the
# near field carries the average (section 6.1).
RANGE_BINS = ((0.0, 5.0), (5.0, 15.0), (15.0, 30.0), (30.0, float("inf")))
RANGE_BIN_NAMES = ("0-5m", "5-15m", "15-30m", "30m+")

ECE_BINS = 15

# Half-width of the boundary band, in pixels. The 3x3 morphological gradient
# already marks the 1 px ridge either side of a class change, so the band ends
# up BOUNDARY_WIDTH_PX + 1 pixels deep on each side of the edge.
BOUNDARY_WIDTH_PX = 3
GRADIENT_KERNEL_PX = 3


def confusion(gt, pred, num_classes=NUM_CLASSES) -> np.ndarray:
    """(C, C) int64 counts, rows = ground truth, cols = prediction.

    UNLABELED in EITHER array is dropped. A pixel with no ground truth cannot
    be scored, and a baseline that declines to predict must not be charged as
    though it had guessed: charging it would reward guessing.

    Counted with one np.bincount over gt * C + pred. A Python loop over the
    two million pixels of a single frame would dominate the runtime of the
    whole scorer.
    """
    gt = np.asarray(gt)
    pred = np.asarray(pred)

    # compared before flattening: (2, 3) and (3, 2) both ravel to (6,) and
    # would silently score a transposed prediction
    if gt.shape != pred.shape:
        raise ValueError(f"gt shape {gt.shape} != pred shape {pred.shape}")

    gt = gt.reshape(-1)
    pred = pred.reshape(-1)
    keep = (gt != UNLABELED) & (pred != UNLABELED)
    gt_kept = gt[keep].astype(np.int64)
    pred_kept = pred[keep].astype(np.int64)

    # gt * C + pred folds two labels into one index, so a label outside
    # [0, C) does not fail loudly, it wraps into some other cell and is
    # counted as a real prediction for the wrong class. Refuse: a quietly
    # corrupted confusion matrix is worse than a crash.
    if gt_kept.size > 0:
        lo = min(int(gt_kept.min()), int(pred_kept.min()))
        hi = max(int(gt_kept.max()), int(pred_kept.max()))
        if lo < 0 or hi >= num_classes:
            raise ValueError(
                f"label range {lo}..{hi} outside [0, {num_classes}) after "
                f"dropping UNLABELED={UNLABELED}"
            )

    flat = np.bincount(gt_kept * num_classes + pred_kept,
                       minlength=num_classes * num_classes)
    return flat.reshape(num_classes, num_classes).astype(np.int64)


def iou_per_class(conf) -> np.ndarray:
    """(C,) float64 IoU per class, np.nan for a class absent from both the
    ground truth and the prediction.

    That nan is the reason this module exists on its own. A class with zero
    support AND zero predictions has an undefined IoU, literally 0/0. Scoring
    it 0.0 drags the mean down by 1/C for every class a split happens not to
    contain, so the same method reports a different mIoU on two splits of the
    same data and no cross-split or cross-dataset comparison means anything.
    Assumption A5 already warns that mIoU is only loosely tied to task
    utility; scoring absent classes as failures makes it worse.

    A class with zero support but nonzero predictions is DEFINED, and its IoU
    is 0.0: the model invented a class that is not in the scene, which is a
    real error and not missing data.
    """
    conf = np.asarray(conf, dtype=np.float64)
    true_pos = np.diag(conf)
    support = conf.sum(axis=1)
    predicted = conf.sum(axis=0)
    union = support + predicted - true_pos

    iou = np.full(conf.shape[0], np.nan)
    defined = union > 0
    iou[defined] = true_pos[defined] / union[defined]
    return iou


def miou(conf) -> float:
    """Mean IoU over the classes that are defined.

    Spelled out rather than np.nanmean so an all-nan matrix (an empty split)
    returns nan quietly instead of emitting a RuntimeWarning in the middle of
    a scoring run.
    """
    iou = iou_per_class(conf)
    defined = iou[~np.isnan(iou)]
    if defined.size == 0:
        return float("nan")

    return float(defined.mean())


def fwiou(conf) -> float:
    """IoU weighted by ground-truth class frequency.

    Reported next to mIoU to expose whether a gain came from the dominant
    class (section 6.1): a method that improves only on `natural_ground`
    moves fwIoU a lot and mIoU barely at all.
    """
    conf = np.asarray(conf, dtype=np.float64)
    total = conf.sum()
    if total == 0:
        return float("nan")

    support = conf.sum(axis=1)
    iou = iou_per_class(conf)

    # a class with support > 0 always has union > 0, so its IoU is never nan
    # here. Restricting the sum to those classes is what keeps the undefined
    # ones out: their weight is 0, but 0 * nan is nan, not 0.
    present = support > 0
    return float((support[present] * iou[present]).sum() / total)


def macc(conf) -> float:
    """Mean per-class recall. Same nan rule as IoU: a class with no ground
    truth has no recall to measure, so it is left out of the mean rather than
    counted as a miss."""
    conf = np.asarray(conf, dtype=np.float64)
    support = conf.sum(axis=1)
    present = support > 0
    if not present.any():
        return float("nan")

    return float((np.diag(conf)[present] / support[present]).mean())


def all_acc(conf) -> float:
    """Global accuracy over every scored pixel or point."""
    conf = np.asarray(conf, dtype=np.float64)
    total = conf.sum()
    if total == 0:
        return float("nan")

    return float(np.diag(conf).sum() / total)


def boundary_confusion(gt_2d, pred_2d, width=BOUNDARY_WIDTH_PX) -> np.ndarray:
    """Confusion restricted to pixels within `width` of a ground-truth class
    boundary. Boundary IoU is then just miou() of the result, so there is no
    separate function for it.

    Boundaries are where a segmentation head is actually weak and where a
    costmap cares most: the leaf-against-pillar edge is the difference between
    drive through and stop. A neighbourhood holding more than one class is a
    boundary, and that is exactly what a morphological gradient (dilate minus
    erode, so max minus min over the neighbourhood) is nonzero on for a label
    image.

    UNLABELED regions raise the gradient at their own border too, and that is
    left alone deliberately: the annotation border genuinely is a place where
    the label changes, and confusion() drops the unlabelled pixels themselves
    in any case.
    """
    gt_2d = np.asarray(gt_2d)
    pred_2d = np.asarray(pred_2d)
    if gt_2d.shape != pred_2d.shape:
        raise ValueError(f"gt shape {gt_2d.shape} != pred shape {pred_2d.shape}")

    if gt_2d.ndim != 2:
        raise ValueError(f"boundary confusion needs a 2D label image, got {gt_2d.ndim}D")

    kernel = np.ones((GRADIENT_KERNEL_PX, GRADIENT_KERNEL_PX), np.uint8)
    gradient = cv2.morphologyEx(gt_2d.astype(np.uint8), cv2.MORPH_GRADIENT, kernel)
    band = gradient > 0

    if width > 0:
        span = 2 * width + 1
        band = cv2.dilate(band.astype(np.uint8), np.ones((span, span), np.uint8)) > 0

    # index the original arrays, not the uint8 copy handed to cv2: a label of
    # 300 casts to 44 and would then sail past confusion()'s range guard
    return confusion(gt_2d[band], pred_2d[band])


@dataclass
class EceAccum:
    """Per-bin calibration counts, folded across frames.

    Field names match the submission.json keys so the mapping from this object
    to the sidecar is obvious to the writer of that file.
    """
    counts: np.ndarray      # (num_bins,) int64, elements landing in each bin
    conf_sum: np.ndarray    # (num_bins,) float64, sum of confidences per bin
    correct: np.ndarray     # (num_bins,) float64, correct elements per bin

    @staticmethod
    def zeros(num_bins: int = ECE_BINS) -> "EceAccum":
        return EceAccum(
            counts=np.zeros(num_bins, dtype=np.int64),
            conf_sum=np.zeros(num_bins, dtype=np.float64),
            correct=np.zeros(num_bins, dtype=np.float64),
        )

    def __add__(self, other: "EceAccum") -> "EceAccum":
        return EceAccum(
            counts=self.counts + other.counts,
            conf_sum=self.conf_sum + other.conf_sum,
            correct=self.correct + other.correct,
        )


def ece_accumulate(conf, correct, num_bins: int = ECE_BINS) -> EceAccum:
    """Fold one frame's confidences into per-bin sums, ready to be added into
    a running EceAccum.

    This is the companion to ece(): the scorer never holds every frame's
    pixels at once, so calibration has to accumulate as counts rather than as
    arrays.

    Pass only scorable elements. UNLABELED is excluded upstream under the same
    rule confusion() applies, so a declined prediction is not charged here
    either.

    Bin i covers [i / num_bins, (i + 1) / num_bins), with a confidence of
    exactly 1.0 folded into the last bin.
    """
    conf = np.asarray(conf, dtype=np.float64)
    correct = np.asarray(correct)
    if conf.shape != correct.shape:
        raise ValueError(f"conf shape {conf.shape} != correct shape {correct.shape}")

    conf = conf.reshape(-1)
    correct = correct.reshape(-1).astype(np.float64)

    # written as "not all in range" rather than "any out of range" so that a
    # nan confidence is rejected as well: nan fails every comparison, and
    # (nan * num_bins).astype(int64) is a garbage bin index, not an error
    if not np.all((conf >= 0.0) & (conf <= 1.0)):
        raise ValueError("confidence must be in [0, 1] and not nan to be binned")

    index = np.minimum((conf * num_bins).astype(np.int64), num_bins - 1)
    return EceAccum(
        counts=np.bincount(index, minlength=num_bins).astype(np.int64),
        conf_sum=np.bincount(index, weights=conf, minlength=num_bins),
        correct=np.bincount(index, weights=correct, minlength=num_bins),
    )


def ece(bin_counts, bin_conf_sum, bin_correct) -> float:
    """Expected Calibration Error: the count-weighted mean gap between mean
    confidence and mean accuracy, per bin.

    A head feeding a costmap has to be trustworthy, not just accurate
    (section 6.1). A 0.99-confidence `natural_ground` on what is actually a
    person is a different failure from a 0.4-confidence one.

    Takes accumulated sums rather than raw arrays so frames compose: the ECE
    of a split is not the mean of its per-frame ECEs.
    """
    counts = np.asarray(bin_counts, dtype=np.float64)
    conf_sum = np.asarray(bin_conf_sum, dtype=np.float64)
    correct = np.asarray(bin_correct, dtype=np.float64)
    if not counts.shape == conf_sum.shape == correct.shape:
        raise ValueError(
            f"ece bin arrays disagree: counts {counts.shape}, "
            f"conf_sum {conf_sum.shape}, correct {correct.shape}"
        )

    total = counts.sum()
    if total == 0:
        return float("nan")

    filled = counts > 0
    accuracy = correct[filled] / counts[filled]
    confidence = conf_sum[filled] / counts[filled]
    return float((counts[filled] / total * np.abs(accuracy - confidence)).sum())


def consistency(matched, scorable) -> float:
    """Fraction of scorable lidar points whose 3D label matches the 2D label
    of the pixel they project into. Needs no ground truth, so it also runs on
    unlabelled robot logs (section 6.1).

    nan when nothing was scorable. That is a real outcome and not a zero: a
    frame where no point landed on a visible pixel has no consistency to
    report, and calling it 0.0 would punish a method for the geometry of the
    rig.
    """
    if scorable == 0:
        return float("nan")

    return float(matched) / float(scorable)


def coverage(scorable, total) -> float:
    """Fraction of points that had a visible pixel at all.

    Always reported next to consistency. Assumption A2 of the problem
    statement is that the camera and the lidar do NOT see the same scene: the
    lidar is 360 degrees and the camera is a frustum, so most points have no
    pixel. Consistency 0.95 over 3 percent of the cloud and consistency 0.95
    over 80 percent are entirely different results, and the bare ratio cannot
    tell them apart.
    """
    if total == 0:
        return float("nan")

    return float(scorable) / float(total)
