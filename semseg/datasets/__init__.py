"""Dataset port and the fit/score split policy (spec section 9).

Two things live here and nothing else, because both are contracts rather than
implementations. `Dataset` is the only interface a baseline or a sweep sees, so
the GOOSE adapter and the synthetic fixture stay interchangeable and the sweep
harness never learns which one it is driving. `split_frames` is the policy that
keeps a fitted baseline honest.

The split is assigned by a stable hash of the frame id, not by an index, not by
shuffling with a seeded RNG, and never by Python's built-in `hash`, which is
salted per process and would silently move the split between two runs of the
same command. A frame therefore keeps its side of the split when frames are
added, removed or reordered, which is what makes "nothing is ever fitted on the
split it is scored on" checkable after the fact rather than merely intended.
"""

import hashlib
from abc import ABC, abstractmethod

import numpy as np

FIT_FRACTION = 0.5

# Width of the md5 hex prefix turned into the split bucket. Pinned as a
# constant because narrowing it would quietly move every frame's split.
SPLIT_HASH_HEX_CHARS = 16
SPLIT_HASH_DENOM = float(16 ** SPLIT_HASH_HEX_CHARS)


class Dataset(ABC):
    """One bi-modal dataset. `extrinsic` is separate from `load` because the
    extrinsic is a parameter of the prediction call, not a member of the frame:
    eval/sweep.py perturbs it per call and a Frame carrying its own extrinsic
    would invite a baseline to read the unperturbed one."""

    @abstractmethod
    def frame_ids(self) -> list[str]:
        """Every frame id this dataset can load, in a stable order."""

    @abstractmethod
    def load(self, frame_id: str):
        """-> Frame. Raises on a frame id this dataset does not hold."""

    @abstractmethod
    def extrinsic(self, frame_id: str) -> np.ndarray:
        """-> T_cam_lidar, 4x4 float64, lidar frame to camera optical frame.
        Raises when the dataset ships no calibration: a plausible default here
        would turn every projection result into a fiction (spec section 1b)."""


def split_bucket(frame_id: str) -> float:
    """Frame id -> a deterministic bucket in [0, 1).

    md5 is used as a fixed, documented, cross-process bit mixer and not as a
    security primitive. Any stable hash would do; what matters is that it is
    the same on every machine and in every process.
    """
    digest = hashlib.md5(frame_id.encode("utf-8")).hexdigest()
    return int(digest[:SPLIT_HASH_HEX_CHARS], 16) / SPLIT_HASH_DENOM


def split_frames(frame_ids, fit_fraction: float = FIT_FRACTION):
    """-> (fit_ids, score_ids), disjoint, together covering `frame_ids` and
    each in the input order.

    Validated at the boundary: a fraction outside [0, 1] is a caller bug and
    raises rather than being clamped, because a clamped 1.5 would silently fit
    on everything and score on nothing.
    """
    if not 0.0 <= fit_fraction <= 1.0:
        raise ValueError(f"fit_fraction must be in [0, 1], got {fit_fraction}")

    fit_ids, score_ids = [], []
    for frame_id in frame_ids:
        target = fit_ids if split_bucket(frame_id) < fit_fraction else score_ids
        target.append(frame_id)

    return fit_ids, score_ids
