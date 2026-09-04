"""The frozen baseline contract (interface spec section 3) and the tiny
name-to-class registry the CLIs resolve a baseline name through.

Kept apart from the four implementations so the harness can import the
contract on its own. eval/sweep.py only needs to know what a baseline is; it
should not have to import open3d, three feature extractors and a fitted
classifier to find out.
"""

import importlib
from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np

from semseg.types import Frame, Prediction

# name -> class, filled by @register when a bl_*/run.py module is imported
_REGISTRY = {}


class Baseline(ABC):
    """One segmentation method over one frame.

    `name` matches the bl_* directory name, so the source directory, the
    registry key, the submission's `method` field and the result directory
    all spell the method the same way.
    """

    name: str = None

    def fit(self, frames: Iterable[Frame], T_cam_lidar: np.ndarray) -> None:
        """Optional. Called ONLY on the fit split, never on the split that
        will be scored (interface spec section 9).

        `frames` is an iterable and not a list because the driver hands over a
        generator: a GOOSE fit split is hundreds of frames of image plus
        cloud, and materialising them all would cost tens of gigabytes for no
        gain.

        The default is a no-op so an unfitted baseline (bl_prior) does not
        have to carry an empty override.
        """
        return None

    @abstractmethod
    def predict(self, frame: Frame, T_cam_lidar: np.ndarray) -> Prediction:
        """Segment one frame in both modalities.

        T_cam_lidar is a CALL PARAMETER. Never read it from config, never
        stash it on self during __init__, and never cache anything derived
        from it (a uv table, a z-buffer, a painted colour array) across
        calls. eval/sweep.py perturbs this matrix between successive
        predict() calls, so a baseline that caches it returns an identical
        prediction at every perturbation magnitude and silently defeats the
        whole decalibration sweep, which is the scientific core of the
        problem statement section 6.3. The failure mode is the dangerous
        kind: nothing crashes, the curve is simply flat and the published
        crossover never happens.

        T_cam_lidar may be None, meaning no calibration exists for this
        dataset. GOOSE ships none with its val zips (interface spec 1b). A
        baseline that receives None declines whatever needs a projection by
        emitting UNLABELED there. It must never fall back on a plausible
        default extrinsic, because every projection dependent number computed
        from a fabricated rig would be a fiction.
        """


def register(cls):
    """Class decorator. Registration is explicit rather than an
    __init_subclass__ side effect, so that reading a bl_* file tells you
    whether it is reachable from the CLI.
    """
    if not cls.name:
        raise ValueError(f"{cls.__name__} must set a class level name matching its bl_* directory")

    existing = _REGISTRY.get(cls.name)
    if existing is not None and existing is not cls:
        raise ValueError(f"baseline name {cls.name!r} is already registered to {existing.__name__}")

    _REGISTRY[cls.name] = cls
    return cls


def get(name: str) -> type:
    """Look up an already imported baseline class."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown baseline {name!r}; registered: {registered()}")

    return _REGISTRY[name]


def registered() -> list:
    return sorted(_REGISTRY)


def load(name: str) -> type:
    """Import baselines/<name>/run.py so its @register call runs, then return
    the class. The registry is populated by import side effect, so a caller
    holding nothing but an argparse string needs this rather than get().
    """
    importlib.import_module(f"baselines.{name}.run")
    return get(name)
