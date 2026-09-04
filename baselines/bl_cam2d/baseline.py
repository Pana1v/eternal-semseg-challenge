"""bl_cam2d: the CAMERA-ONLY arm of the section 6.2 ablation.

Separate from run.py because run.py is a CLI and this is the method. The
sweep harness imports the class through baselines.common.base and never wants
argparse; a reader looking for what bl_cam2d actually computes should not have
to read flag parsing to find it.

What it is. The shared nb.GaussianNB over features.pixel_features, which is
colour plus normalised row and column and nothing else. Same classifier as
bl_geom3d and bl_paint, different columns, which is what makes the three arms
an ablation of FUSION rather than of architecture (see nb.py).

The one invariant that defines this file: the 2D prediction NEVER reads
frame.points. Not for a height prior, not for a horizon estimate, not at all.
A camera-only arm that peeked at the cloud would make the section 6.2
lidar-only vs camera-only vs fused comparison meaningless, and the peek would
be invisible in the results. test_run.py pins this by handing predict() a
cloud of garbage and asserting the 2D map is byte identical.

The 3D output, and why most of it is UNLABELED. Each point is projected with
semseg.projection.project, the z-buffer picks the point that actually owns its
pixel, and that point takes the 2D label of the pixel. Everything else is left
UNLABELED. Leaving them UNLABELED is the point, not a shortcoming: assumption
A2 of the problem statement says most lidar points have no pixel at all (an
azimuth-wedge bound on the real GOOSE cloud puts a 90 degree camera at 23.2
percent of the points, spec 13.5), and a camera-only method genuinely has
nothing to say about the other 77 percent. Filling them with a guess, a
nearest-neighbour fill, or the majority class would inflate the 3D score with
information the camera never had. The accumulator's in_frustum and
out_frustum split (spec section 6, driven by the section 6.2 question "how
much of the reported gain comes from points the camera never saw") then shows
the abstention directly: bl_cam2d's conf_3d_out_frustum comes out all zeros,
because UNLABELED is dropped from every confusion matrix.

One number a reader of the sweep output will trip over: bl_cam2d's cross-modal
consistency is 1.0 by construction, at every perturbation magnitude. Its 3D
labels ARE a resample of its own 2D map at the same uv, and the accumulator
recomputes that uv from the same extrinsic, so the two modalities cannot
disagree. Its 3D mIoU still degrades under decalibration, because the labels
move to the wrong points even though the two views of them agree. That gap
between a flat consistency curve and a falling mIoU curve is the reason
consistency is always reported next to its coverage and never alone.
"""

import numpy as np

from baselines.common.base import Baseline, register
from baselines.common.features import pixel_features
from baselines.common.nb import GaussianNB
from semseg.projection import project, zbuffer
from semseg.types import Frame, Prediction, UNLABELED

# Labelled pixels drawn per fit frame. A full GOOSE frame is 2048x1000, so
# fitting on every pixel of a few hundred frames would accumulate half a
# billion feature rows before GaussianNB.fit ever runs. A diagonal Gaussian
# needs a per-class mean and variance and nothing else, and both converge long
# before 20k samples of a class, so the extra rows buy precision in the fourth
# decimal place at a cost the sweeps pay dozens of times over.
FIT_PIXELS_PER_FRAME = 20000

# The subsample is drawn from a named seed rather than from entropy. A
# robustness sweep must vary the perturbation and nothing else; a reseeded
# subsample would add its own spread to every curve and make a flat result look
# noisy (same argument as features.GROUND_RANSAC_SEED).
FIT_SUBSAMPLE_SEED = 0

# Confidence recorded for a point this baseline declines to label. The label is
# UNLABELED, which eval/io_formats._fold_ece drops from the calibration counts
# on both sides, so this value is never scored. It still has to be inside
# [0, 1] or metrics.ece_accumulate would raise on it.
DECLINED_CONFIDENCE = 0.0


@register
class Cam2dBaseline(Baseline):
    """Camera-only segmentation. Fit on labelled pixels, predict every pixel,
    then resample the 2D map onto the points the camera can actually see."""

    name = "bl_cam2d"

    def __init__(self):
        self._nb = None

    def fit(self, frames, T_cam_lidar: np.ndarray) -> None:
        """Fit the shared Naive Bayes on a subsample of labelled pixels.

        T_cam_lidar is accepted to satisfy the contract and deliberately
        unused: fitting a camera-only arm involves no projection, so there is
        nothing here for the decalibration sweep to perturb even in principle.
        That is itself a result the sweep reports, since bl_paint's fit does
        depend on the extrinsic.

        `frames` is consumed as an iterable exactly once, because the driver
        hands over a generator (see runner._load_frames).
        """
        rng = np.random.default_rng(FIT_SUBSAMPLE_SEED)
        feature_blocks, label_blocks = [], []

        for frame in frames:
            if frame.labels_2d_gt is None:
                continue

            # pixel_features is row major (H*W, 5), so the ground truth has to
            # be flattened the same way or every feature row would be paired
            # with a label from somewhere else in the image
            features = pixel_features(frame.image)
            labels = frame.labels_2d_gt.reshape(-1)

            # filtered BEFORE the subsample, not after. Drawing first and then
            # dropping UNLABELED would silently keep far fewer than
            # FIT_PIXELS_PER_FRAME rows on a real GOOSE frame, where large
            # regions carry no ground truth, and the fit sample size would then
            # depend on how much of the frame happened to be annotated.
            labelled = np.flatnonzero(labels != UNLABELED)
            if labelled.size == 0:
                continue

            drawn = rng.choice(labelled, size=min(FIT_PIXELS_PER_FRAME, labelled.size),
                               replace=False)
            feature_blocks.append(features[drawn])
            label_blocks.append(labels[drawn])

        if not feature_blocks:
            raise ValueError(
                f"{self.name}: the fit split carried no labelled 2D pixels at all, so there "
                f"is nothing to fit. A camera-only arm needs labels_2d_gt; check that the "
                f"dataset adapter is returning it."
            )

        self._nb = GaussianNB().fit(np.vstack(feature_blocks), np.concatenate(label_blocks))

    def predict(self, frame: Frame, T_cam_lidar: np.ndarray) -> Prediction:
        if self._nb is None:
            raise RuntimeError(
                f"{self.name}.predict called before fit. This baseline is fitted, so a run "
                f"whose fit split resolved to zero frames has nothing to predict with."
            )

        height, width = frame.image.shape[:2]

        # The camera-only half. frame.points is not read on this path and must
        # never be: see the module docstring. Everything the 2D map knows comes
        # from the image and from where in the image a pixel sits.
        flat_labels, flat_conf = self._nb.predict(pixel_features(frame.image))
        labels_2d = flat_labels.reshape(height, width)
        conf_2d = flat_conf.reshape(height, width)

        labels_3d, conf_3d = self._resample_3d(frame, T_cam_lidar, labels_2d, conf_2d)

        return Prediction(labels_2d=labels_2d, labels_3d=labels_3d,
                          conf_2d=conf_2d, conf_3d=conf_3d)

    def _resample_3d(self, frame: Frame, T_cam_lidar, labels_2d, conf_2d):
        """-> (labels_3d (N,) uint8, conf_3d (N,) float32).

        UNLABELED everywhere the camera has no evidence, which is the majority
        of a real cloud. Three separate reasons a point ends up declined, all of
        them honest:

          - no extrinsic at all. GOOSE's val zips ship none (spec 13.3), and a
            fabricated default would turn every 3D number here into a fiction.
          - outside the frustum. The lidar is 360 degrees and the camera is
            not (assumption A2).
          - inside the frustum but not the z-buffer owner of its pixel. That
            pixel shows whatever is in front of this point, so labelling the
            point from it would score it against the occluder.
        """
        n_points = frame.points.shape[0]
        labels_3d = np.full(n_points, UNLABELED, dtype=np.uint8)
        conf_3d = np.full(n_points, DECLINED_CONFIDENCE, dtype=np.float32)

        if T_cam_lidar is None:
            return labels_3d, conf_3d

        height, width = labels_2d.shape
        uv, depth, in_frustum = project(frame.points, frame.K, T_cam_lidar, width, height)

        # only the per-point flag is wanted; the dense owner map is what
        # produced it and has no further use here
        _owner, visible = zbuffer(uv, depth, in_frustum, width, height)

        rows, cols = uv[visible, 1], uv[visible, 0]
        labels_3d[visible] = labels_2d[rows, cols]
        conf_3d[visible] = conf_2d[rows, cols]

        return labels_3d, conf_3d
