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

# This arm is the one that still produces a real 3D answer on a dataset with no
# calibration, which is the actual state of the GOOSE val zips (interface spec
# 13.3). So the no-calib case is a supported run and not an error, and it must
# be announced rather than absorbed: the 2D half of the result is then
# declined, so a reader who sees an empty 2D matrix should have been told why.
NO_CALIB_NOTE = (
    "no calibration: this dataset ships no T_cam_lidar and none was given with --calib, "
    "so the 3D output is unaffected and the 2D output is declined as UNLABELED. "
    "The numbers are in the GOOSE-DB bags' /tf_static, not in the annotated zips.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    runner.add_common_args(parser)
    args = parser.parse_args(argv)

    dataset = runner.build_dataset(args)
    fit_ids, score_ids = runner.resolve_splits(dataset, args.split)

    if not runner.has_calibration(dataset):
        print(f"{Geom3dBaseline.name}: {NO_CALIB_NOTE}")

    runner.run(Geom3dBaseline(), dataset, fit_ids, score_ids,
               runner.resolve_extrinsic_fn(dataset), args.out,
               limit=args.limit, jobs=args.jobs, split=args.split)


if __name__ == "__main__":
    main()
