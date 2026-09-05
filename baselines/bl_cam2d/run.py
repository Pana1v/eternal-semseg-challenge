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


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    runner.add_common_args(parser)
    args = parser.parse_args()

    dataset = runner.build_dataset(args)
    fit_ids, score_ids = runner.resolve_splits(dataset, args.split)

    # Running without an extrinsic is cheap for this arm: its 2D half needs
    # none at all, so the headline number survives and only the 3D resample is
    # declined (spec 13.3).
    runner.run(Cam2dBaseline(), dataset, fit_ids, score_ids,
               runner.resolve_extrinsic_fn(dataset),
               args.out, limit=args.limit, jobs=args.jobs, split=args.split)


if __name__ == "__main__":
    main()
