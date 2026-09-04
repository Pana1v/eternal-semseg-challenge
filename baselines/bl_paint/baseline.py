"""bl_paint: bl_geom3d's geometric features with the painted RGB of each point
appended, fed to the same Naive Bayes core. The fused arm of the ablation, and
the naive early-fusion floor.

Problem statement section 7 names this baseline explicitly and says why it is
worth writing: "The naive point-painting baseline matters. If a full fusion
architecture does not clearly beat 'append RGB to the point feature vector',
the added complexity is not justified." So this file is a floor to clear, not a
proposal. Anything with cross-modal attention in it has to beat these numbers
before its complexity is paid for.

Why it is exactly bl_geom3d plus colour, and nothing else. Section 6.2 asks
whether fusion pays for itself, and that question is only answerable if the
lidar-only, camera-only and fused arms run the identical classifier
(baselines/common/nb.py) over the identical geometry features
(baselines/common/features.py) and differ solely in which feature columns they
are handed. Change the classifier here as well and the resulting delta measures
fusion plus architecture, with no way left to separate the two. So this module
adds three columns and touches nothing else, including the 2D output path,
which is bl_geom3d's scatter-and-nearest-fill so that the two 3D-driven 2D
arms stay comparable.

The frustum, handled honestly, and the choice this file makes. The lidar is
360 degrees and the camera is a frustum, so most points have no pixel at all
(assumption A2; the fixture puts 29 percent of points in frustum and GOOSE's
own azimuth wedge bound puts a 90 degree camera at 23 percent, spec 13.5).
Those points have no colour, and there are two ways to cope:

  (a) one model over geometry plus colour plus a has_colour indicator, with a
      neutral fill for the missing colour, or
  (b) two models, one over geometry plus colour for the points that have a
      pixel and one over geometry alone for the rest.

This file implements (b). A neutral fill teaches the classifier that "no
colour" is itself a colour: every out-of-frustum point would land on the same
fill value, so the fill's per-class variance collapses towards nb.VAR_FLOOR
and the colour columns turn into a near-deterministic vote for whichever class
happens to dominate the out-of-frustum half. The indicator column cannot undo
that, because a diagonal Gaussian has no interaction terms and so cannot learn
"ignore columns 5 to 7 when column 8 is zero". Option (a) would have cost one
model instead of two and a slightly shorter predict(), which is not worth
paying a corrupted colour likelihood for on three quarters of the cloud.

One detail of (b) that is a deliberate deviation from the literal wording
"geometry alone for the rest": the geometry-only model is fitted on ALL
labelled fit points, not only on the out-of-frustum ones. That makes this
baseline's out-of-frustum half the same estimator as bl_geom3d, so the
section 6.2 in-frustum vs out-of-frustum split attributes every point of gain
to colour. Fitting it on out-of-frustum points only would have changed the
class prior on that half as well, and the ablation would then be moving two
variables at once.
"""

import numpy as np

from baselines.common import features
from baselines.common.base import Baseline, register
from baselines.common.nb import GaussianNB
from semseg.projection import paint, project, scatter_to_image, zbuffer
from semseg.types import NUM_CLASSES, UNLABELED, Prediction

# Rows of geometry features taken from one fit frame. GOOSE's fit split is
# roughly 480 frames of about 111k points (spec 13.4), so an uncapped geometry
# matrix would be about 2 GB of float64 to estimate 9 by 5 means and variances.
# A diagonal Gaussian's estimates are already converged long before that.
FIT_POINTS_PER_FRAME = 20000

# The subsample is drawn from one seeded generator in frame order, so a fit is
# reproducible. Anything that varies per run would add its own spread to every
# sweep curve and be indistinguishable from the perturbation under study.
FIT_SUBSAMPLE_SEED = 0

# Confidence of a pixel no lidar point reached. Zero rather than nan: the
# accumulator's ECE folding rejects a nan or out-of-range confidence, and
# rightly so, since a clamped one would surface as a calibration result.
CONF_2D_FILL = np.float32(0.0)

# Column layout of the fused feature matrix, so a reader can name a column
# without counting np.hstack arguments. Colour last keeps columns 0 to 4
# identical to bl_geom3d's matrix.
FUSED_FEATURE_NAMES = features.GEOM_FEATURE_NAMES + features.RGB_FEATURE_NAMES


def _subsample(rng, rows, cap: int = FIT_POINTS_PER_FRAME) -> np.ndarray:
    """Cap one frame's contribution to a fit matrix. Returns row indices."""
    if rows.size <= cap:
        return rows

    return rng.choice(rows, size=cap, replace=False)


@register
class PaintBaseline(Baseline):
    """Two fitted diagonal Gaussians over one shared geometry feature set.

    Which of the two a point is scored by is decided per call from the
    extrinsic handed to predict(), never cached, because eval/sweep.py's whole
    decalibration sweep is that argument moving.
    """

    name = "bl_paint"

    def __init__(self, seed: int = FIT_SUBSAMPLE_SEED):
        self.seed = seed

        # geometry only, fitted on every labelled fit point, so this half is
        # bl_geom3d's estimator
        self._nb_geom = None

        # geometry plus painted colour, fitted only on fit points that had a
        # pixel of their own. Its class prior is therefore the in-frustum
        # prior and not the cloud's, which is inherent to option (b): a model
        # that only ever sees coloured points can only be trained on them.
        self._nb_fused = None

    def fit(self, frames, T_cam_lidar) -> None:
        rng = np.random.default_rng(self.seed)

        geom_x, geom_y, fused_x, fused_y = [], [], [], []

        for frame in frames:
            if frame.labels_3d_gt is None:
                raise ValueError(
                    f"frame {frame.frame_id} carries no labels_3d_gt; bl_paint fits on 3D "
                    "ground truth and cannot be fitted on an unlabelled frame")

            labels = frame.labels_3d_gt
            geom = features.geom_features(frame.points, frame.intensity)
            labelled = labels != UNLABELED

            take = _subsample(rng, np.flatnonzero(labelled))
            geom_x.append(geom[take])
            geom_y.append(labels[take])

            if T_cam_lidar is None:
                continue

            has_colour, _uv, rgb = self._paint_points(frame, T_cam_lidar)
            coloured = labelled & has_colour
            if not coloured.any():
                continue

            take = _subsample(rng, np.flatnonzero(coloured))
            fused_x.append(np.hstack([geom[take], rgb[take]]))
            fused_y.append(labels[take])

        if not geom_x:
            raise ValueError("bl_paint was fitted on zero frames")

        self._nb_geom = GaussianNB(NUM_CLASSES).fit(np.vstack(geom_x), np.concatenate(geom_y))

        # No fused model means no calibration was available at fit time, so
        # every point will be scored by geometry alone. run.py refuses that
        # case rather than writing a submission labelled bl_paint that is
        # secretly bl_geom3d (spec 13.3: the fused arm is skipped with a
        # stated reason, never approximated).
        if fused_x:
            self._nb_fused = GaussianNB(NUM_CLASSES).fit(np.vstack(fused_x),
                                                         np.concatenate(fused_y))

    def predict(self, frame, T_cam_lidar) -> Prediction:
        if self._nb_geom is None:
            raise RuntimeError("bl_paint.predict was called before fit; there is no fitted "
                               "classifier to score with")

        height, width = frame.image.shape[:2]
        geom = features.geom_features(frame.points, frame.intensity)

        if T_cam_lidar is None or self._nb_fused is None:
            return self._geometry_only(geom, height, width)

        has_colour, uv, rgb = self._paint_points(frame, T_cam_lidar)

        labels_3d = np.full(geom.shape[0], UNLABELED, dtype=np.uint8)
        conf_3d = np.zeros(geom.shape[0], dtype=np.float32)

        # Both branches are guarded because either can be empty for real
        # reasons: a GOOSE cloud can put nothing in a narrow frustum, and the
        # dropout sweep can leave almost no points at all.
        uncoloured = ~has_colour
        if uncoloured.any():
            labels_3d[uncoloured], conf_3d[uncoloured] = self._nb_geom.predict(geom[uncoloured])

        if has_colour.any():
            fused = np.hstack([geom[has_colour], rgb[has_colour]])
            labels_3d[has_colour], conf_3d[has_colour] = self._nb_fused.predict(fused)

        # The 2D output is bl_geom3d's, unchanged: scatter the 3D labels into
        # the pixels their points own and nearest-fill the rest. Copied in
        # behaviour on purpose, so the two 3D-driven 2D arms differ only by the
        # colour in their 3D labels and the 2D column of the ablation table
        # stays a comparison rather than two different constructions.
        labels_2d = scatter_to_image(uv, has_colour, labels_3d, width, height, UNLABELED)
        conf_2d = scatter_to_image(uv, has_colour, conf_3d, width, height, CONF_2D_FILL)

        return Prediction(labels_2d=labels_2d, labels_3d=labels_3d,
                          conf_2d=conf_2d, conf_3d=conf_3d)

    def _paint_points(self, frame, T_cam_lidar):
        """-> (has_colour (N,) bool, uv (N, 2) int32, rgb (N, 3) float64 in
        [0, 1], zero where there is no colour).

        A point inside the frustum that loses the z-buffer does NOT have a
        colour. It projects onto the occluder's pixel, so painting it would
        hand it the colour of whatever stands in front of it and the fused
        model would be asked to classify a vegetation point that is the colour
        of a wall. Those points go to the geometry-only model, which is the
        honest half of frustum handling: the camera did not see them either.

        The painted set is a function of the extrinsic handed to predict() and
        not of the true rig, so decalibration moves it: a perturbed extrinsic
        repaints points that a nominal-frustum partition still calls
        out-of-frustum, and only points this returns uncoloured under BOTH rigs
        keep their labels under a sweep.
        """
        height, width = frame.image.shape[:2]

        uv, depth, in_frustum = project(frame.points, frame.K, T_cam_lidar, width, height)
        _owner, visible = zbuffer(uv, depth, in_frustum, width, height)

        has_colour = in_frustum & visible
        return has_colour, uv, features.rgb_features(paint(uv, has_colour, frame.image))

    def _geometry_only(self, geom, height, width) -> Prediction:
        """The no-calibration path. Points still get a label from geometry, but
        the 2D map is entirely UNLABELED and conf_2d is None, because every
        route from a 3D label to a pixel runs through the projection. Spec 1b:
        a plausible default extrinsic here would turn the 2D half of the result
        into a fiction, so this declines instead.
        """
        labels_3d, conf_3d = self._nb_geom.predict(geom)

        return Prediction(labels_2d=np.full((height, width), UNLABELED, dtype=np.uint8),
                          labels_3d=labels_3d, conf_2d=None, conf_3d=conf_3d)
