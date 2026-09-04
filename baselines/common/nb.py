"""Diagonal covariance Gaussian Naive Bayes: the one classifier all four
baselines share.

Why sharing it is the whole point, and not just an economy. Problem statement
section 6.2 demands a lidar-only vs camera-only vs fused ablation. That
comparison only measures FUSION if the three arms run the identical
classifier and differ solely in which feature columns they are given. Three
different architectures would confound "fusion helps" with "architecture B is
better", and no sweep run afterwards could separate the two effects again.
bl_geom3d, bl_cam2d and bl_paint therefore all instantiate this class and
change nothing but their call to baselines/common/features.py.

Why a few dozen lines of numpy and not sklearn: the runtime image
(docker/runtime.Dockerfile) does not carry sklearn, a diagonal Gaussian has a
closed form fit with no iteration in it, and a dependency the harness cannot
import inside its own container is worse than no dependency at all.
"""

import numpy as np

from semseg.types import NUM_CLASSES

# A feature column can be exactly constant within a class: the sweep's
# blacked-out camera mode makes every RGB column zero, and a synthetic ground
# plane makes the height column zero. Zero variance is a divide by zero and
# then a NaN posterior, so every fitted variance is floored. Far below the
# variance of any real feature column, so it never biases a class that did
# have spread.
VAR_FLOOR = 1e-9

LOG_2PI = float(np.log(2.0 * np.pi))


class GaussianNB:
    """fit(X, y) then predict_proba(X) or predict(X).

    All likelihood arithmetic is in log space: a product of one likelihood
    per feature per class underflows float64 long before a full feature set
    is consumed.
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        self.num_classes = num_classes
        self.mean = None       # (C, F)
        self.var = None        # (C, F), floored
        self.log_prior = None  # (C,), -inf for a class absent from the fit data

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).astype(np.int64).reshape(-1)
        if X.ndim != 2 or y.shape[0] != X.shape[0]:
            raise ValueError(f"need X (M, F) and one label per row, got {X.shape} and {y.shape}")
        if y.shape[0] == 0:
            raise ValueError("cannot fit on an empty feature matrix")

        counts = np.bincount(y, minlength=self.num_classes).astype(np.float64)
        self.mean = np.zeros((self.num_classes, X.shape[1]))
        self.var = np.full((self.num_classes, X.shape[1]), VAR_FLOOR)
        for c in np.flatnonzero(counts):
            in_class = X[y == c]
            self.mean[c] = in_class.mean(axis=0)
            self.var[c] = np.maximum(in_class.var(axis=0), VAR_FLOOR)

        # A class absent from the fit split gets prior zero, so log prior
        # -inf, which keeps it out of every prediction without ever producing
        # a NaN: -inf plus a finite log likelihood stays -inf, and exp(-inf)
        # is exactly 0.0. Computed by masked assignment rather than
        # np.log(counts / total) so the absent classes never raise a divide
        # warning on the way to the same answer.
        self.log_prior = np.full(self.num_classes, -np.inf)
        present = counts > 0
        self.log_prior[present] = np.log(counts[present] / counts.sum())
        return self

    def _log_joint(self, X: np.ndarray) -> np.ndarray:
        inv_var = 1.0 / self.var
        # (x - mu)^2 / var expands to x^2/var - 2 x mu/var + mu^2/var, so the
        # whole log likelihood is two matmuls plus a per class constant. The
        # readable (M, C, F) broadcast would allocate over a gigabyte per
        # temporary on one full resolution GOOSE image (2048 x 1000 pixels,
        # 9 classes, 5 features) and this runs once per sweep point.
        quad = ((X * X) @ inv_var.T
                - 2.0 * (X @ (self.mean * inv_var).T)
                + np.sum(self.mean * self.mean * inv_var, axis=1))
        log_det = np.sum(np.log(self.var), axis=1) + X.shape[1] * LOG_2PI
        return self.log_prior - 0.5 * (log_det + quad)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """-> (M, C) posteriors, each row summing to 1."""
        log_joint = self._log_joint(np.asarray(X, dtype=np.float64))

        # Subtract the row maximum before exponentiating. The pivot is finite
        # because at least one class was present at fit time, so the row sum
        # is at least 1.0 and the normalisation cannot divide by zero.
        log_joint = log_joint - log_joint.max(axis=1, keepdims=True)
        proba = np.exp(log_joint)
        return proba / proba.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray):
        """-> (labels uint8 in goose9, confidence float32).

        Confidence is the max posterior, which is what eval/metrics.py's ECE
        expects: a costmap consumer needs to know how much to trust the
        label it was handed, not the full distribution.
        """
        proba = self.predict_proba(X)
        return proba.argmax(axis=1).astype(np.uint8), proba.max(axis=1).astype(np.float32)
