#!/usr/bin/env python3
"""bl_prior CLI: write a chance-floor submission in the same format as every
other arm, so the scorer can subtract it.

Usage:
    python -m baselines.bl_prior.run --dataset fixture --split score \\
        --mode uniform --out submission_prior.json

Run it twice, once per --mode, before reading any other number in this repo.
The floor a method has to clear is the BETTER of the two, per metric, and
which mode wins is a property of the split rather than of this file: majority
beats uniform on accuracy-like metrics only to the extent that the declared
class really does dominate. Measured on the synthetic fixture, whose dominant
class is not the declared one, uniform is the higher floor on allAcc (0.11
against 0.05) while majority is far worse calibrated (ECE 0.82 against
0.0005). Both rows belong in the results table for that reason.

Separate from baseline.py so this file holds argparse and split handling and
nothing else. The iterate, fold and write loop is runner.run, shared with the
other three arms, because a floor produced by a different driver would not be
comparable to the method it bounds.
"""

import argparse

from baselines.bl_prior.baseline import DEFAULT_MODE, MODE_NAMES, PriorBaseline
from baselines.common import runner
from semseg.datasets import split_frames


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="chance floor over the goose9 label space, in both modalities")
    runner.add_common_args(parser)
    parser.add_argument("--mode", default=DEFAULT_MODE, choices=MODE_NAMES,
                        help="uniform draws over all nine classes with a fixed seed; "
                             "majority answers one declared constant class everywhere")
    args = parser.parse_args(argv)

    dataset = runner.build_dataset(args)

    # --seed reaches two places on a fixture run, the world generator through
    # build_dataset and the label draw here. Both are deterministic, so the
    # pair is reproducible, but changing --seed moves the world and the draw
    # together and a seed spread reported from it is a spread over both.
    baseline = PriorBaseline(mode=args.mode, seed=args.seed)

    fit_ids, score_ids = split_frames(dataset.frame_ids())
    predict_ids = fit_ids if args.split == "fit" else score_ids

    # An empty fit split, not the real one. Nothing here is fitted, so handing
    # over fit_ids would call the inherited no-op fit() for no reason, and on
    # --split fit it would trip runner.run's disjointness check with the very
    # frames it is about to score.
    #
    # predict() ignores the extrinsic, but the Accumulator projects every cloud
    # itself for the frustum matrices and the consistency counts, so a real rig
    # buys strictly more reporting when one exists.
    #
    # It must not be REQUIRED, though, and this arm is the reason why. bl_prior
    # is the chance floor: whatever else runs, the floor has to run beside it or
    # the other numbers have nothing to be compared against. Demanding --calib
    # here would leave every real GOOSE result floorless, which is the one
    # reading mIoU on a 9-class problem cannot survive without. So it follows
    # the same rule as bl_geom3d and bl_cam2d and declines its
    # projection-dependent output instead of refusing to run.
    extrinsic_fn = (runner.make_extrinsic_fn(dataset)
                    if getattr(dataset, "calib_available", True)
                    else runner.no_extrinsic)

    return runner.run(baseline, dataset, [], predict_ids, extrinsic_fn, args.out,
                      limit=args.limit, jobs=args.jobs, split=args.split)


if __name__ == "__main__":
    main()
