"""The two declared priors and the one class that emits them.

Separate from run.py because run.py is a CLI and this is the method: the sweep
harness and the unit tests construct PriorBaseline directly and must not have
to go through argparse to do it.

Why a chance floor is not optional. mIoU on nine classes with one dominant
class is exactly the metric that flatters a model which learned nothing, and
problem statement section 6.1 asks for frequency-weighted IoU precisely
because a mean hides that. Section 8's second success criterion is a margin
over the better single modality, and a margin is only a number if the bottom
of the scale is known. So this baseline exists to answer, in the same units as
every other row of the results table, "what does knowing nothing score here".

The two modes are informative in different directions, and reporting both is
the point:

- uniform draws labels uniformly over the nine classes. Its mIoU is close to
  worthless while its confidence, a flat 1/9, is very nearly honest for a
  uniform guesser over nine classes, so its ECE lands near zero. A near-zero
  ECE next to a near-zero mIoU is the reminder that calibration and accuracy
  are independent axes.
- majority answers one declared class everywhere. On a dataset with a dominant
  class it posts a large allAcc and a large frequency-weighted IoU while its
  mIoU stays at roughly one ninth of one class's IoU, which is the failure
  the interface spec's section 1 warns about, made concrete. Its declared
  confidence is deliberately over-confident so the ECE column shows what an
  uncalibrated head looks like.
"""

import hashlib
from dataclasses import dataclass

import numpy as np

from baselines.common.base import Baseline, register
from semseg.types import NUM_CLASSES, Prediction

UNIFORM_MODE = "uniform"
MAJORITY_MODE = "majority"

# The class `majority` answers everywhere. DECLARED A PRIORI and never fitted,
# measured or argmaxed on the split being scored: a "majority" class read off
# the scored split's own histogram is a fitted model with one parameter, and it
# would make the floor move whenever the split moved.
#
# Vegetation is declared from GOOSE's documented character, an off-road dataset
# whose six of eight val sequences are forest and field tracks, and from
# nothing else. The one frame measured in interface spec section 13.5 would
# argue for artificial_ground on pixel share, but that frame is from
# neubiberg_sunny, the atypical suburban sequence, so following it would be
# fitting on the scored split and fitting on its least representative frame at
# the same time.
MAJORITY_CLASS = 6

# uniform over nine classes, so the max-class probability of an honest uniform
# guesser is exactly 1/9 and its ECE should come out near zero.
UNIFORM_CONFIDENCE = 1.0 / NUM_CLASSES

# What `majority` claims. Also declared rather than fitted, and deliberately
# over-confident: no single number can be right for both modalities anyway,
# since vegetation is around 70 percent of points and 14 percent of pixels in
# the worked frame, and calibrating it per modality would mean reading the
# scored split. 0.9 makes the ECE gap large and obvious, which is what the
# floor is here to demonstrate.
MAJORITY_CONFIDENCE = 0.9

DEFAULT_MODE = UNIFORM_MODE
DEFAULT_SEED = 0

# Width of the md5 hex prefix used to seed a frame's draw. Pinned as a constant
# because narrowing it would silently change every label this baseline has ever
# emitted.
FRAME_SEED_HEX_CHARS = 16


@dataclass(frozen=True)
class PriorMode:
    """One declared prior: what it answers and how sure it says it is.

    Frozen because a mode is a declaration, not state. Keeping the confidence
    beside the name as data means there is no branch anywhere that could pair
    the wrong number with the wrong mode.
    """
    name: str
    confidence: float


MODES = {
    UNIFORM_MODE: PriorMode(name=UNIFORM_MODE, confidence=UNIFORM_CONFIDENCE),
    MAJORITY_MODE: PriorMode(name=MAJORITY_MODE, confidence=MAJORITY_CONFIDENCE),
}

MODE_NAMES = tuple(sorted(MODES))


def frame_seed(seed: int, frame_id: str) -> int:
    """(seed, frame id) -> a deterministic seed for that one frame's draw.

    Per frame and not per call, which matters twice. A member RNG advanced on
    each predict() would make a frame's labels depend on how many frames were
    predicted before it, so reordering the split would relabel every frame in
    it. Under --jobs > 1 the driver spawns workers that each hold their own
    copy of this object, so a member RNG would additionally make the
    submission depend on which worker happened to take the frame.

    md5 is a fixed, documented bit mixer here and not a security primitive.
    Python's built-in hash() cannot be used: it is salted per process, so the
    same command would emit different labels on every invocation.
    """
    digest = hashlib.md5(f"{seed}:{frame_id}".encode("utf-8")).hexdigest()
    return int(digest[:FRAME_SEED_HEX_CHARS], 16)


@register
class PriorBaseline(Baseline):
    """Answers from a declared prior alone, in both modalities.

    fit() is inherited as the no-op: there is nothing to fit, which is the
    whole claim being made. The driver is passed an empty fit split so the
    inherited no-op is never even called.
    """

    name = "bl_prior"

    def __init__(self, mode: str = DEFAULT_MODE, seed: int = DEFAULT_SEED):
        if mode not in MODES:
            raise ValueError(f"unknown --mode {mode!r}; choose one of {list(MODE_NAMES)}")

        self.mode = MODES[mode]
        self.seed = seed

    def predict(self, frame, T_cam_lidar) -> Prediction:
        """Fill both modalities, every pixel and every point.

        T_cam_lidar is IGNORED DELIBERATELY, and is not a leftover parameter.
        This baseline reads no features, so it has nothing to project and no
        way to be wrong about the rig. The parameter stays because the
        contract in baselines/common/base.py freezes it and the sweep harness
        calls every baseline the same way. The visible consequence is that
        bl_prior's curve is flat across every decalibration and time-offset
        magnitude, by construction: that flat line is the reference the other
        arms' curves are read against, not a sweep that failed to bite.

        Nothing is ever left UNLABELED. A floor that declined to answer where
        it was unsure would be scored on a subset it had selected for itself,
        which is the one way a chance baseline can look better than chance.
        """
        height, width = frame.image.shape[:2]
        n_points = frame.points.shape[0]

        # One RNG per frame, drawn 2D then 3D in that fixed order, the same
        # discipline as the fixture generator. The order is part of what the
        # seed reproduces.
        rng = np.random.default_rng(frame_seed(self.seed, frame.frame_id))

        labels_2d = self._draw(rng, (height, width))
        labels_3d = self._draw(rng, (n_points,))

        # float32 per the Prediction contract, and a full array rather than a
        # scalar because the Accumulator folds it elementwise into the ECE
        # bins alongside the per-pixel correctness.
        conf_2d = np.full((height, width), self.mode.confidence, dtype=np.float32)
        conf_3d = np.full((n_points,), self.mode.confidence, dtype=np.float32)

        return Prediction(labels_2d=labels_2d, labels_3d=labels_3d,
                          conf_2d=conf_2d, conf_3d=conf_3d)

    def _draw(self, rng, shape) -> np.ndarray:
        """One modality's labels. uint8 because UNLABELED is 255 and a
        narrower or signed dtype would alias the sentinel onto a real class."""
        if self.mode.name == MAJORITY_MODE:
            return np.full(shape, MAJORITY_CLASS, dtype=np.uint8)

        return rng.integers(NUM_CLASSES, size=shape, dtype=np.uint8)
