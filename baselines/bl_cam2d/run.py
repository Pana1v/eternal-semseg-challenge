#!/usr/bin/env python3
"""CLI for bl_cam2d, the camera-only arm (spec section 10).

    python baselines/bl_cam2d/run.py --dataset fixture --root /tmp/semseg-fixture \
        --split score --out submission.json [--limit N] [--jobs N]

Separate from baseline.py so the method is readable without the flags, and so
importing this module is all it takes to register the baseline: the sweep
harness resolves --baseline through baselines.common.base.load, which imports
exactly this file for the @register side effect.
"""

import argparse

from baselines.bl_cam2d.baseline import Cam2dBaseline
from baselines.common import runner
from semseg.datasets import split_frames


def pick_extrinsic_fn(dataset):
    """-> the extrinsic_fn runner.run wants, or the declining one.

    GOOSE ships no calibration in its val zips, so GooseDataset.extrinsic
    raises unless --calib supplied one (spec 13.3). Asking the adapter first,
    rather than catching the raise, keeps the raise meaning what it says.

    The default of True is for the fixture, whose extrinsic is exact by
    construction and which therefore carries no such flag. Getting this wrong
    in the safe direction is cheap: bl_cam2d's 2D half needs no extrinsic at
    all, so a run without one still produces the arm's headline number and
    declines only the 3D resample.
    """
    if getattr(dataset, "calib_available", True):
        return runner.make_extrinsic_fn(dataset)

    return runner.no_extrinsic


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    runner.add_common_args(parser)
    args = parser.parse_args()

    dataset = runner.build_dataset(args)
    fit_ids, score_ids = split_frames(dataset.frame_ids())

    # --split names the split to PREDICT over, so predicting the fit split
    # means fitting on the score split. Swapping rather than reusing one split
    # for both is what keeps runner.run's disjointness check satisfiable:
    # nothing is ever fitted on the split it is scored on (spec section 9).
    if args.split == "fit":
        fit_ids, score_ids = score_ids, fit_ids

    runner.run(Cam2dBaseline(), dataset, fit_ids, score_ids, pick_extrinsic_fn(dataset),
               args.out, limit=args.limit, jobs=args.jobs, split=args.split)


if __name__ == "__main__":
    main()
