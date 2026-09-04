"""The LiDAR-only arm of the problem statement section 6.2 ablation: geometric
features off the raw cloud, through the shared Naive Bayes core, and nothing
else.

Separate from run.py so the class is importable without argparse running.
eval/sweep.py holds one instance across dozens of perturbation magnitudes and
never touches the CLI, and the fused arm's own test suite compares against this
class directly.

THIS ARM MUST NEVER READ frame.image CONTENT. If it does, the section 6.2
ablation is void: the whole point of the three arms sharing one classifier
(baselines/common/nb.py) is that a score difference between them is a
difference in INFORMATION, not in machinery. A lidar-only arm that has quietly
seen a pixel is no longer a lidar-only arm, the "does fusion pay for itself"
row of the section 6.2 table becomes unanswerable, and nothing crashes to tell
you. The only thing read from the image here is its `.shape`, in _canvas_hw,
and that is argued for at the function.

How a lidar-only method answers a 2D question. It has no camera model of its
own, so it projects its OWN 3D labels through T_cam_lidar and nearest-fills the
gaps with semseg.projection.scatter_to_image. That is the honest construction:
the answer is still made of nothing but lidar evidence, and the projection is
declared rather than hidden.

The consequence is the asymmetry that makes this module a control, and it is
deliberate. bl_geom3d's 2D score DEPENDS on the extrinsic, because the 2D map
is a projection of the 3D one. Its 3D score does NOT, because no step of the
3D path touches T_cam_lidar. So under the decalibration sweep of problem
statement section 6.3 this arm's 2D curve bends and its 3D curve stays exactly
flat. Both halves are useful:

  - the flat 3D curve is the reference line the crossover is measured against.
    Section 6.3 asks for "the perturbation at which fusion drops below the
    LiDAR-only baseline", and that magnitude is only well defined because the
    line it crosses does not itself move.
  - the bending 2D curve is the positive control on the same run. A sweep in
    which NOTHING moves is indistinguishable from a sweep that is not wired up,
    which is the failure mode baselines/common/base.py warns about: the curve
    is simply flat and the crossover never happens.

One number this arm produces must never be quoted, and the reason is the same
construction. Cross-modal consistency (problem statement section 6.1) counts
lidar points whose 3D label matches the 2D label of the pixel they project
into. Here the 2D label of that pixel WAS this point's 3D label, so the two
agree by identity and the measured consistency is exactly 1.0 at every
perturbation magnitude. That is a tautology, not a result: it says the
projection is self-consistent, which it is by construction, and says nothing
whatever about the scene. Consistency is only informative for the fused arm,
where the two modalities are genuinely separate sources.
"""

import numpy as np

from baselines.common.base import Baseline, register
from baselines.common.features import geom_features
from baselines.common.nb import GaussianNB
from semseg.projection import project, scatter_to_image, zbuffer
from semseg.types import Prediction, UNLABELED

# Per-frame cap on the rows handed to the classifier, and the seed that draws
# them. This is a memory bound, not a statistical choice: a GOOSE fit split is
# roughly 480 frames of about 170k points (interface spec 13.1, 13.4), which is
# 80M rows of 5 float64 columns, over 3 GB of feature matrix before the
# classifier has seen anything. A diagonal Gaussian's mean and variance
# converge long before 20k samples per frame, so the cap costs nothing
# measurable. The fixture's frames are smaller than the cap and are therefore
# used whole.
FIT_POINTS_PER_FRAME = 20000
FIT_SUBSAMPLE_SEED = 0

# Confidence written into a 2D pixel that no lidar point reached at all, which
# happens only when the frustum is empty. Zero and not nan: eval/metrics.py's
# ECE bins on [0, 1] and a nan would poison the accumulated sums.
CONF_UNSEEDED = 0.0

# ground_plane needs three points to span a plane and local_pca needs a
# neighbourhood, so below this a cloud has no geometric features at all and
# this arm declines rather than raising. The modality-dropout sweep (section
# 6.3) hands out a deliberately gutted cloud, and semseg/types.py's validate()
# allows a point count of zero for exactly that reason.
MIN_POINTS_FOR_FEATURES = 3


def _canvas_hw(frame):
    """-> (height, width) of the 2D answer this frame expects.

    This reads the image's SHAPE and never one of its pixels. The distinction
    is the whole licence for the line: a 2D prediction has to be the size of
    the map it will be scored against, and that size is rig metadata, fixed
    before the vehicle left the yard, not evidence about the scene. Taking it
    from frame.labels_2d_gt would be worse in the way that matters, because
    ground truth is not available at inference time.

    The convention is the harness's own, not this module's invention:
    eval/io_formats.py's Accumulator.add resolves the canvas exactly this way,
    so a baseline that sized its output any other way would fold a mismatched
    matrix.
    """
    return frame.image.shape[:2]


@register
class Geom3dBaseline(Baseline):
    """Tier 0 LiDAR-only baseline (interface spec section 10).

    Features are the five columns of baselines.common.features.geom_features:
    height above the RANSAC ground plane, local PCA verticality and planarity,
    range and intensity. The classifier is the shared
    baselines.common.nb.GaussianNB, unmodified.
    """

    name = "bl_geom3d"

    def __init__(self):
        self._nb = None

    def fit(self, frames, T_cam_lidar) -> None:
        """Fit the shared Naive Bayes on the geometric features of every
        ground-truth-labelled point in the fit split.

        Called ONLY on the fit split (interface spec section 9), and the driver
        in baselines/common/runner.py guarantees the extrinsic it passes is the
        unperturbed one.

        T_cam_lidar is accepted and then ignored, which is the contract of
        interface spec section 3 rather than an oversight. A lidar-only arm has
        no use for an extrinsic, and reading one here would be the first step
        toward the ablation leaking: fitting against a projection would make
        this arm's 3D score move under the decalibration sweep, and the
        crossover of section 6.3 would then be measured against a line that
        wanders.
        """
        rng = np.random.default_rng(FIT_SUBSAMPLE_SEED)
        blocks, targets = [], []

        for frame in frames:
            if frame.labels_3d_gt is None:
                continue

            if frame.points.shape[0] < MIN_POINTS_FOR_FEATURES:
                continue

            # Features first, subsample second, and the order is not
            # interchangeable. Verticality and planarity are neighbourhood
            # statistics over the k nearest points, so thinning the cloud
            # before the PCA would compute them over a sparser world than
            # predict() ever sees, and the fitted means would describe a
            # different sensor.
            features = geom_features(frame.points, frame.intensity)

            rows = np.flatnonzero(frame.labels_3d_gt != UNLABELED)
            if rows.size > FIT_POINTS_PER_FRAME:
                rows = rng.choice(rows, size=FIT_POINTS_PER_FRAME, replace=False)

            blocks.append(features[rows])
            targets.append(frame.labels_3d_gt[rows])

        if not blocks:
            raise ValueError(
                f"{self.name}.fit: the fit split held no frame with 3D ground truth and at "
                f"least {MIN_POINTS_FOR_FEATURES} points, so there is nothing to fit")

        self._nb = GaussianNB().fit(np.vstack(blocks), np.concatenate(targets))

    def predict(self, frame, T_cam_lidar) -> Prediction:
        """-> Prediction with per-point labels and confidences, and a 2D map
        projected from them.

        T_cam_lidar is read here and only here, once per call, and nothing
        derived from it is kept on self. eval/sweep.py changes it between
        successive calls (interface spec section 3) and a cached uv table or
        z-buffer would silently flatten the 2D curve.

        T_cam_lidar of None means the dataset ships no calibration, which is
        the real GOOSE case (interface spec 13.3). The 3D answer is unaffected,
        which is exactly why this arm is the one that still runs there, and the
        2D answer is declined as UNLABELED rather than guessed at from a
        fabricated rig.
        """
        if self._nb is None:
            raise RuntimeError(f"{self.name}.predict called before fit")

        height, width = _canvas_hw(frame)
        declined_2d = np.full((height, width), UNLABELED, dtype=np.uint8)

        if frame.points.shape[0] < MIN_POINTS_FOR_FEATURES:
            return Prediction(labels_2d=declined_2d,
                              labels_3d=np.full(frame.points.shape[0], UNLABELED, dtype=np.uint8),
                              conf_2d=None, conf_3d=None)

        labels_3d, conf_3d = self._nb.predict(geom_features(frame.points, frame.intensity))

        if T_cam_lidar is None:
            return Prediction(labels_2d=declined_2d, labels_3d=labels_3d,
                              conf_2d=None, conf_3d=conf_3d)

        uv, depth, in_frustum = project(frame.points, frame.K, T_cam_lidar, width, height)

        # The z-buffer, and not the bare frustum mask, is what seeds the 2D
        # map. Several points land on one pixel and the nearest one is the one
        # the camera would actually have seen, so without this a wall pixel can
        # be seeded by the vegetation standing behind the wall. scatter_to_image
        # resolves duplicates by array order, which is arbitrary, so the choice
        # has to be made here.
        _owner, visible = zbuffer(uv, depth, in_frustum, width, height)

        labels_2d = scatter_to_image(uv, visible, labels_3d, width, height, UNLABELED)

        # The confidence is transported by the SAME nearest-neighbour operator
        # as the label, because the two have to describe one prediction. A pixel
        # that reports a neighbour's label and its own untransported confidence
        # would be reporting a pair that no point ever produced. The honest
        # caveat, which belongs in docs/BASELINES.md and not only here: away
        # from the seeded pixels this arm's 2D confidence is the posterior of a
        # point some distance away, so its 2D ECE measures the calibration of a
        # transported posterior. That is a real property of a lidar-only method
        # answering a 2D question, not an artefact of the harness.
        conf_2d = scatter_to_image(uv, visible, conf_3d, width, height, CONF_UNSEEDED)

        return Prediction(labels_2d=labels_2d, labels_3d=labels_3d,
                          conf_2d=conf_2d, conf_3d=conf_3d)
