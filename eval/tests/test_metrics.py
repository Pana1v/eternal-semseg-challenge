"""Hand-computed cases for eval/metrics.py.

Every expected number below is worked out in the comment above it, from the
matrix itself, never copied back from the code's own output: a test that
agrees with the implementation by construction proves nothing. Every metric
also carries a perturbation case that moves one input by the smallest amount
that can matter and asserts the metric follows.
"""

import sys
import types as pytypes

import numpy as np
import pytest

# semseg/types.py owns the two label-space constants metrics.py imports. It is
# a separate module and may not be on disk yet, so it is stood in for here
# rather than duplicated into the shipped module.
try:
    from semseg.types import NUM_CLASSES, UNLABELED
except ImportError:
    _stub_types = pytypes.ModuleType("semseg.types")
    _stub_types.NUM_CLASSES = 9
    _stub_types.UNLABELED = 255
    sys.modules.setdefault("semseg", pytypes.ModuleType("semseg"))
    sys.modules["semseg.types"] = _stub_types
    from semseg.types import NUM_CLASSES, UNLABELED

# spec sections 1 and 2 freeze both values; a real semseg/types.py that
# disagrees has to fail loudly here instead of quietly rebasing these cases
assert NUM_CLASSES == 9
assert UNLABELED == 255

from eval.metrics import (
    BOUNDARY_WIDTH_PX, ECE_BINS, RANGE_BINS, RANGE_BIN_NAMES,
    EceAccum, all_acc, boundary_confusion, confusion, consistency, coverage,
    ece, ece_accumulate, fwiou, iou_per_class, macc, miou,
)

# The 2x2 case:
#   gt   = [0, 0, 0, 1]
#   pred = [0, 0, 1, 1]
#   conf = [[2, 1],
#           [0, 1]]
GT_2X2 = np.array([0, 0, 0, 1], dtype=np.uint8)
PRED_2X2 = np.array([0, 0, 1, 1], dtype=np.uint8)

# The 3x3 case:
#   gt   = [0, 0, 0, 0, 0, 1, 1, 1, 2, 2]
#   pred = [0, 0, 0, 1, 2, 1, 1, 0, 2, 0]
#   conf = [[3, 1, 1],
#           [1, 2, 0],
#           [1, 0, 1]]
GT_3X3 = np.array([0, 0, 0, 0, 0, 1, 1, 1, 2, 2], dtype=np.uint8)
PRED_3X3 = np.array([0, 0, 0, 1, 2, 1, 1, 0, 2, 0], dtype=np.uint8)
CONF_3X3 = np.array([[3, 1, 1], [1, 2, 0], [1, 0, 1]], dtype=np.int64)


def test_confusion_2x2_hand_computed():
    conf = confusion(GT_2X2, PRED_2X2, num_classes=2)
    assert conf.tolist() == [[2, 1], [0, 1]]
    assert conf.dtype == np.int64


def test_iou_family_2x2_hand_computed():
    conf = confusion(GT_2X2, PRED_2X2, num_classes=2)
    iou = iou_per_class(conf)

    # class 0: tp=2, gt=3, pred=2, union=3+2-2=3 -> 2/3
    # class 1: tp=1, gt=1, pred=2, union=1+2-1=2 -> 1/2
    assert iou[0] == pytest.approx(2 / 3)
    assert iou[1] == pytest.approx(1 / 2)

    # miou    = (2/3 + 1/2) / 2                = 7/12
    # fwiou   = (3 * 2/3 + 1 * 1/2) / 4        = 5/8
    # macc    = (2/3 + 1/1) / 2                = 5/6
    # all_acc = (2 + 1) / 4                    = 3/4
    assert miou(conf) == pytest.approx(7 / 12)
    assert fwiou(conf) == pytest.approx(5 / 8)
    assert macc(conf) == pytest.approx(5 / 6)
    assert all_acc(conf) == pytest.approx(3 / 4)


def test_confusion_3x3_hand_computed():
    conf = confusion(GT_3X3, PRED_3X3, num_classes=3)
    assert conf.tolist() == CONF_3X3.tolist()
    assert conf.sum() == 10


def test_iou_family_3x3_hand_computed():
    conf = CONF_3X3

    # column sums (predictions): 3+1+1=5, 1+2+0=3, 1+0+1=2
    # class 0: tp=3, gt=5, pred=5, union=5+5-3=7 -> 3/7
    # class 1: tp=2, gt=3, pred=3, union=3+3-2=4 -> 1/2
    # class 2: tp=1, gt=2, pred=2, union=2+2-1=3 -> 1/3
    iou = iou_per_class(conf)
    assert iou[0] == pytest.approx(3 / 7)
    assert iou[1] == pytest.approx(1 / 2)
    assert iou[2] == pytest.approx(1 / 3)

    # miou    = (3/7 + 1/2 + 1/3) / 3          = (18+21+14)/42 / 3 = 53/126
    # fwiou   = (5*3/7 + 3*1/2 + 2*1/3) / 10   = (90+63+28)/42 / 10 = 181/420
    # macc    = (3/5 + 2/3 + 1/2) / 3          = (18+20+15)/30 / 3 = 53/90
    # all_acc = (3 + 2 + 1) / 10               = 3/5
    assert miou(conf) == pytest.approx(53 / 126)
    assert fwiou(conf) == pytest.approx(181 / 420)
    assert macc(conf) == pytest.approx(53 / 90)
    assert all_acc(conf) == pytest.approx(3 / 5)


def test_unlabeled_dropped_from_either_side():
    # only the three pairs with a label on BOTH sides survive: (0,0), (1,1), (1,1)
    gt = np.array([0, 1, UNLABELED, 0, 1], dtype=np.uint8)
    pred = np.array([0, 1, 0, UNLABELED, 1], dtype=np.uint8)
    conf = confusion(gt, pred, num_classes=2)
    assert conf.tolist() == [[1, 0], [0, 2]]
    assert conf.sum() == 3


def test_out_of_range_label_raises_instead_of_wrapping():
    # gt=0, pred=4 with C=3 folds to index 0*3+4 = 4, which is cell (1, 1) of
    # the 3x3 matrix: a fabricated true positive for class 1, silently.
    with pytest.raises(ValueError):
        confusion(np.array([0], dtype=np.uint8), np.array([4], dtype=np.uint8), num_classes=3)

    # same wrap the other way round: gt=4 lands well past the last row
    with pytest.raises(ValueError):
        confusion(np.array([4], dtype=np.uint8), np.array([0], dtype=np.uint8), num_classes=3)


def test_confusion_rejects_transposed_shapes():
    # (2, 3) and (3, 2) both ravel to (6,), so the check has to happen first
    gt = np.zeros((2, 3), dtype=np.uint8)
    pred = np.zeros((3, 2), dtype=np.uint8)
    with pytest.raises(ValueError):
        confusion(gt, pred, num_classes=2)


def test_empty_confusion_is_nan_everywhere():
    gt = np.full(3, UNLABELED, dtype=np.uint8)
    pred = np.array([0, 1, 2], dtype=np.uint8)
    conf = confusion(gt, pred, num_classes=3)

    assert conf.sum() == 0
    assert np.isnan(miou(conf))
    assert np.isnan(fwiou(conf))
    assert np.isnan(macc(conf))
    assert np.isnan(all_acc(conf))


def test_absent_class_iou_is_nan_not_zero():
    # class 2 has neither ground truth nor predictions: 0/0, undefined
    #   conf = [[3, 1, 0],
    #           [1, 2, 0],
    #           [0, 0, 0]]
    # class 0: tp=3, gt=4, pred=4, union=5 -> 3/5
    # class 1: tp=2, gt=3, pred=3, union=4 -> 1/2
    conf = np.array([[3, 1, 0], [1, 2, 0], [0, 0, 0]], dtype=np.int64)
    iou = iou_per_class(conf)

    assert np.isnan(iou[2])
    assert miou(conf) == pytest.approx((3 / 5 + 1 / 2) / 2)  # 0.55

    # scoring the absent class 0.0 would report (3/5 + 1/2 + 0)/3 = 0.3667,
    # which is the bug this nan exists to prevent
    assert miou(conf) > (3 / 5 + 1 / 2) / 3


def test_zero_support_with_predictions_scores_zero():
    # class 2 has no ground truth but one prediction: defined, and wrong
    #   conf = [[3, 1, 1],
    #           [1, 2, 0],
    #           [0, 0, 0]]
    # class 0: tp=3, gt=5, pred=4, union=6 -> 1/2
    # class 1: tp=2, gt=3, pred=3, union=4 -> 1/2
    # class 2: tp=0, gt=0, pred=1, union=1 -> 0
    conf = np.array([[3, 1, 1], [1, 2, 0], [0, 0, 0]], dtype=np.int64)
    iou = iou_per_class(conf)

    assert not np.isnan(iou[2])
    assert iou[2] == 0.0

    # miou = (1/2 + 1/2 + 0) / 3 = 1/3, so the invented class does count
    assert miou(conf) == pytest.approx(1 / 3)

    # macc still skips it: there is no recall to measure without ground truth
    assert macc(conf) == pytest.approx((3 / 5 + 2 / 3) / 2)


def test_fwiou_and_miou_diverge_with_dominance():
    """Same divergence, both directions, driven only by which class dominates."""
    # dominant class predicted well:
    #   conf = [[90, 0], [5, 5]]
    #   class 0: tp=90, gt=90, pred=95, union=95 -> 18/19
    #   class 1: tp=5,  gt=10, pred=5,  union=10 -> 1/2
    #   miou  = (18/19 + 1/2) / 2 = 55/76           = 0.7237
    #   fwiou = (90 * 18/19 + 10 * 1/2) / 100       = 0.9026
    good_is_dominant = np.array([[90, 0], [5, 5]], dtype=np.int64)
    assert miou(good_is_dominant) == pytest.approx(55 / 76)
    assert fwiou(good_is_dominant) == pytest.approx((90 * 18 / 19 + 5.0) / 100)
    assert fwiou(good_is_dominant) - miou(good_is_dominant) > 0.15

    # dominant classes predicted badly, the perfect class is rare:
    #   conf = [[500, 0, 500], [0, 100, 0], [0, 0, 1000]]
    #   class 0: tp=500,  gt=1000, pred=500,  union=1000 -> 1/2
    #   class 1: tp=100,  gt=100,  pred=100,  union=100  -> 1
    #   class 2: tp=1000, gt=1000, pred=1500, union=1500 -> 2/3
    #   miou  = (1/2 + 1 + 2/3) / 3                          = 13/18 = 0.7222
    #   fwiou = (1000/2 + 100 + 1000 * 2/3) / 2100           = 0.6032
    good_is_rare = np.array([[500, 0, 500], [0, 100, 0], [0, 0, 1000]], dtype=np.int64)
    assert miou(good_is_rare) == pytest.approx(13 / 18)
    assert fwiou(good_is_rare) == pytest.approx((500 + 100 + 1000 * 2 / 3) / 2100)
    assert fwiou(good_is_rare) < miou(good_is_rare)


def test_confusion_perturbation_moves_one_count():
    pred = PRED_3X3.copy()
    pred[0] = 1  # a gt-0 pixel that was right is now called class 1

    delta = confusion(GT_3X3, pred, num_classes=3) - CONF_3X3
    assert delta[0, 0] == -1
    assert delta[0, 1] == 1
    assert np.abs(delta).sum() == 2


def _perturbed_3x3():
    """One extra gt-0 pixel predicted as class 1: the smallest change a
    confusion matrix can carry."""
    conf = CONF_3X3.copy()
    conf[0, 1] += 1
    return conf


def test_iou_per_class_perturbation():
    # the perturbed matrix is [[3, 2, 1], [1, 2, 0], [1, 0, 1]]:
    #   class 0: tp=3, gt=6, pred=5, union=6+5-3=8 -> 3/8, was 3/7
    #   class 1: tp=2, gt=3, pred=4, union=3+4-2=5 -> 2/5, was 1/2
    # one extra count moves two classes, because it is one class's recall and
    # another class's precision at the same time
    iou = iou_per_class(_perturbed_3x3())
    assert iou[0] == pytest.approx(3 / 8)
    assert iou[1] == pytest.approx(2 / 5)
    assert iou[0] < iou_per_class(CONF_3X3)[0]
    assert iou[1] < iou_per_class(CONF_3X3)[1]


def test_miou_perturbation():
    # (3/8 + 2/5 + 1/3) / 3 = (45+48+40)/120 / 3 = 133/360 = 0.36944,
    # was 53/126 = 0.42063
    moved = miou(_perturbed_3x3())
    assert moved == pytest.approx(133 / 360)
    assert moved < miou(CONF_3X3)


def test_fwiou_perturbation():
    # (6 * 3/8 + 3 * 2/5 + 2 * 1/3) / 11 = (135+72+40)/60 / 11 = 247/660,
    # was 181/420
    moved = fwiou(_perturbed_3x3())
    assert moved == pytest.approx(247 / 660)
    assert moved < fwiou(CONF_3X3)


def test_macc_perturbation():
    # recall of class 0 drops from 3/5 to 3/6
    moved = macc(_perturbed_3x3())
    assert moved == pytest.approx((3 / 6 + 2 / 3 + 1 / 2) / 3)
    assert moved < macc(CONF_3X3)


def test_all_acc_perturbation():
    # 6 correct out of 11 now, was 6 out of 10
    moved = all_acc(_perturbed_3x3())
    assert moved == pytest.approx(6 / 11)
    assert moved < all_acc(CONF_3X3)


# Boundary cases run on a 24x24 image split down the middle: class 0 for
# x < 12, class 1 for x >= 12.
BOUNDARY_SIZE = 24
BOUNDARY_SPLIT = 12


def _split_image():
    gt = np.zeros((BOUNDARY_SIZE, BOUNDARY_SIZE), dtype=np.uint8)
    gt[:, BOUNDARY_SPLIT:] = 1
    return gt


def test_boundary_band_is_width_plus_one_px_deep():
    """The 3x3 gradient already marks columns 11 and 12, so dilating by
    BOUNDARY_WIDTH_PX=3 gives columns 8..15: four pixels either side."""
    gt = _split_image()
    conf = boundary_confusion(gt, gt.copy())

    band_cols = 2 * (BOUNDARY_WIDTH_PX + 1)
    assert conf.shape == (NUM_CLASSES, NUM_CLASSES)
    assert conf.sum() == BOUNDARY_SIZE * band_cols
    assert conf[0, 0] == BOUNDARY_SIZE * (BOUNDARY_WIDTH_PX + 1)
    assert conf[1, 1] == BOUNDARY_SIZE * (BOUNDARY_WIDTH_PX + 1)


def test_boundary_width_zero_is_the_gradient_ridge_only():
    gt = _split_image()
    conf = boundary_confusion(gt, gt.copy(), width=0)
    assert conf.sum() == BOUNDARY_SIZE * 2  # columns 11 and 12


def test_interior_error_leaves_boundary_untouched():
    """The paired positive assertion matters: without it, a boundary_confusion
    that returned a constant would pass this test."""
    gt = _split_image()
    pred = gt.copy()
    pred[5, 2] = 1  # x=2 is 6 px clear of the band, which reaches x=8

    assert np.array_equal(boundary_confusion(gt, pred), boundary_confusion(gt, gt.copy()))
    assert not np.array_equal(confusion(gt, pred), confusion(gt, gt.copy()))


def test_boundary_perturbation_moves_boundary_confusion():
    gt = _split_image()
    clean = boundary_confusion(gt, gt.copy())

    pred = gt.copy()
    pred[5, BOUNDARY_SPLIT] = 0  # one pixel wrong, right on the edge

    moved = boundary_confusion(gt, pred)
    assert moved[1, 0] == clean[1, 0] + 1
    assert moved[1, 1] == clean[1, 1] - 1
    assert miou(moved) < miou(clean)


def test_boundary_keeps_unlabeled_out_of_the_band():
    """An UNLABELED region still raises the gradient, but its own pixels are
    dropped by confusion(), so only the labelled half of the band scores."""
    gt = _split_image()
    gt[:, BOUNDARY_SPLIT:] = UNLABELED
    conf = boundary_confusion(gt, np.zeros_like(gt))
    assert conf.sum() == BOUNDARY_SIZE * (BOUNDARY_WIDTH_PX + 1)


def test_boundary_rejects_non_2d_input():
    flat = np.zeros(16, dtype=np.uint8)
    with pytest.raises(ValueError):
        boundary_confusion(flat, flat.copy())


def test_ece_overconfident_scores_worse_than_calibrated():
    # 100 samples, half of them correct, in one bin either way.
    # calibrated:    mean confidence 0.50, accuracy 0.50 -> gap 0.00
    # overconfident: mean confidence 0.99, accuracy 0.50 -> gap 0.49
    correct = np.array([1] * 50 + [0] * 50, dtype=bool)
    calibrated = ece_accumulate(np.full(100, 0.5), correct)
    overconfident = ece_accumulate(np.full(100, 0.99), correct)

    cal = ece(calibrated.counts, calibrated.conf_sum, calibrated.correct)
    over = ece(overconfident.counts, overconfident.conf_sum, overconfident.correct)

    assert cal == pytest.approx(0.0)
    assert over == pytest.approx(0.49)
    assert over > cal


def test_ece_weights_bins_by_count():
    """Two bins with unequal counts, so the count weighting is load bearing:
    an unweighted mean of the per-bin gaps would report 0.2."""
    # 90 samples at confidence 0.9, all correct -> accuracy 1.0, gap 0.10
    # 10 samples at confidence 0.3, none correct -> accuracy 0.0, gap 0.30
    # ece = 0.9 * 0.10 + 0.1 * 0.30 = 0.09 + 0.03 = 0.12
    conf = np.concatenate([np.full(90, 0.9), np.full(10, 0.3)])
    correct = np.array([True] * 90 + [False] * 10)
    acc = ece_accumulate(conf, correct)

    assert acc.counts.sum() == 100
    assert ece(acc.counts, acc.conf_sum, acc.correct) == pytest.approx(0.12)
    assert ece(acc.counts, acc.conf_sum, acc.correct) != pytest.approx((0.1 + 0.3) / 2)


def test_ece_accumulation_composes_across_frames():
    conf_a = np.array([0.10, 0.90, 0.50])
    correct_a = np.array([False, True, True])
    conf_b = np.array([0.20, 0.95])
    correct_b = np.array([True, False])

    folded = ece_accumulate(conf_a, correct_a) + ece_accumulate(conf_b, correct_b)
    together = ece_accumulate(np.concatenate([conf_a, conf_b]),
                              np.concatenate([correct_a, correct_b]))

    assert np.array_equal(folded.counts, together.counts)
    assert folded.conf_sum == pytest.approx(together.conf_sum)
    assert folded.correct == pytest.approx(together.correct)
    assert ece(folded.counts, folded.conf_sum, folded.correct) == pytest.approx(
        ece(together.counts, together.conf_sum, together.correct))


def test_ece_accum_zeros_is_an_identity():
    single = ece_accumulate(np.array([0.3, 0.7]), np.array([True, False]))
    summed = EceAccum.zeros() + single
    assert np.array_equal(summed.counts, single.counts)
    assert summed.conf_sum == pytest.approx(single.conf_sum)


def test_ece_bins_and_edges():
    # bin i covers [i/15, (i+1)/15); 1.0 folds into the last bin
    acc = ece_accumulate(np.array([0.0, 1.0]), np.array([True, True]))
    assert acc.counts.shape == (ECE_BINS,)
    assert acc.counts[0] == 1
    assert acc.counts[ECE_BINS - 1] == 1


def test_ece_empty_is_nan():
    empty = EceAccum.zeros()
    assert np.isnan(ece(empty.counts, empty.conf_sum, empty.correct))


def test_ece_rejects_unbinnable_confidence():
    correct = np.array([True])
    for bad in (1.5, -0.1, float("nan")):
        with pytest.raises(ValueError):
            ece_accumulate(np.array([bad]), correct)


def test_ece_perturbation_one_more_correct():
    correct = np.array([1] * 50 + [0] * 50, dtype=bool)
    acc = ece_accumulate(np.full(100, 0.99), correct)
    base = ece(acc.counts, acc.conf_sum, acc.correct)

    bumped = acc.correct.copy()
    bumped[ECE_BINS - 1] += 1  # 51 of 100 correct -> gap 0.48, was 0.49

    moved = ece(acc.counts, acc.conf_sum, bumped)
    assert moved == pytest.approx(0.48)
    assert moved < base


def test_ece_accumulate_perturbation_one_nudged_confidence():
    # both samples stay in bin 7 ([0.4667, 0.5333)), so only conf_sum moves
    correct = np.array([True, False])
    base = ece_accumulate(np.array([0.50, 0.50]), correct)
    nudged = ece_accumulate(np.array([0.52, 0.50]), correct)

    assert np.array_equal(base.counts, nudged.counts)
    assert nudged.conf_sum.sum() == pytest.approx(base.conf_sum.sum() + 0.02)
    assert ece(nudged.counts, nudged.conf_sum, nudged.correct) == pytest.approx(0.01)
    assert ece(base.counts, base.conf_sum, base.correct) == pytest.approx(0.0)


def test_ece_rejects_mismatched_bin_arrays():
    with pytest.raises(ValueError):
        ece(np.zeros(ECE_BINS), np.zeros(ECE_BINS), np.zeros(ECE_BINS - 1))


def test_consistency_and_coverage_hand_computed():
    assert consistency(8, 10) == pytest.approx(0.8)
    assert consistency(0, 10) == 0.0
    assert np.isnan(consistency(0, 0))

    assert coverage(10, 100) == pytest.approx(0.1)
    assert np.isnan(coverage(0, 0))


def test_consistency_is_unreadable_without_coverage():
    """Assumption A2: most points have no pixel, so the same ratio over two
    very different fractions of the cloud is two different results."""
    assert consistency(30, 100) == pytest.approx(consistency(300, 1000))
    assert coverage(100, 3000) != pytest.approx(coverage(1000, 3000))


def test_consistency_perturbation():
    assert consistency(9, 10) > consistency(8, 10)
    assert consistency(8, 11) < consistency(8, 10)


def test_coverage_perturbation():
    assert coverage(11, 100) > coverage(10, 100)
    assert coverage(10, 101) < coverage(10, 100)


def test_range_bins_tile_without_gaps():
    """A gap between bins would silently drop points from the stratified
    report rather than fail."""
    assert len(RANGE_BINS) == len(RANGE_BIN_NAMES)
    assert RANGE_BINS[0][0] == 0.0
    assert RANGE_BINS[-1][1] == float("inf")
    for (_, upper), (lower_next, _) in zip(RANGE_BINS, RANGE_BINS[1:]):
        assert upper == lower_next
