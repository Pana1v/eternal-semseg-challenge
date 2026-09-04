#!/usr/bin/env python3
"""CLI for the LiDAR-only arm (interface spec section 10).

Usage:
    python baselines/bl_geom3d/run.py --dataset fixture --root /tmp/semseg-fixture \\
        --split score --out submission.json [--limit N] [--jobs N]

Thin on purpose. Everything that is identical across the four baselines, the
split handling, the Accumulator folding, the submission write and the compute
sidecar, lives in baselines/common/runner.py, so this file holds only what is
specific to this arm: which class to instantiate and how it behaves when the
dataset ships no calibration.

Importing this module is also what registers the baseline for eval/sweep.py,
by way of baselines/common/base.py's `load`.
"""

import argparse

from baselines.bl_geom3d.baseline import Geom3dBaseline
from baselines.common import runner
from semseg.datasets import split_frames


def resolve_extrinsic_fn(dataset):
    """-> the extrinsic_fn runner.run wants, and a note for the operator.

    This arm is the one that still produces a real 3D answer on a dataset with
    no calibration, which is the actual state of the GOOSE val zips (interface
    spec 13.3). So the no-calib case is a supported run and not an error, and
    it must be announced rather than absorbed: the 2D half of the result is
    then declined, so a reader who sees an empty 2D matrix should have been
    told why.

    The fixture has an exact extrinsic by construction and therefore does not
    carry the flag at all, hence the default.
    """
    if getattr(dataset, "calib_available", True):
        return runner.make_extrinsic_fn(dataset), None

    return runner.no_extrinsic, (
        "no calibration: this dataset ships no T_cam_lidar and none was given with --calib, "
        "so the 3D output is unaffected and the 2D output is declined as UNLABELED. "
        "The numbers are in the GOOSE-DB bags' /tf_static, not in the annotated zips.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    runner.add_common_args(parser)
    args = parser.parse_args(argv)

    dataset = runner.build_dataset(args)
    fit_ids, score_ids = split_frames(dataset.frame_ids())

    # --split names the split to PREDICT over, so the other one is what gets
    # fitted. Swapping here rather than in runner.run keeps the driver's
    # arguments literal: it fits on the first list and scores the second.
    if args.split == "fit":
        fit_ids, score_ids = score_ids, fit_ids

    extrinsic_fn, note = resolve_extrinsic_fn(dataset)
    if note:
        print(f"{Geom3dBaseline.name}: {note}")

    runner.run(Geom3dBaseline(), dataset, fit_ids, score_ids, extrinsic_fn, args.out,
               limit=args.limit, jobs=args.jobs, split=args.split)


if __name__ == "__main__":
    main()
