#!/usr/bin/env python3
"""CLI for the fused arm (spec section 10).

Usage:
    python baselines/bl_paint/run.py --dataset fixture --split score \\
        --out submission.json [--limit N] [--jobs N]
    python baselines/bl_paint/run.py --dataset goose --root <path> \\
        --calib rig.json --split score --out submission.json

Separate from baseline.py because everything in here is argument handling and
nothing in here is a method. eval/sweep.py imports the class and never this
file, so the split policy, the submission write and the compute sidecar live
one import away from the feature code.

Two things this file does that the other baselines' CLIs do not have to.

It refuses to run without a calibration. bl_paint's whole content is the
projection, so on a dataset that ships no extrinsic (real GOOSE val, spec
13.3) there is nothing fused left to measure. predict() still degrades
honestly to geometry alone, because the baseline contract requires a baseline
handed None to decline rather than invent a rig, but a submission file whose
`method` field said bl_paint while the numbers inside it were bl_geom3d's
would be the single most misleading artefact this repo could emit. So the
refusal is here, at the point where a file gets written, and it names --calib.

It reports the two model split. The fused arm is one classifier over geometry
plus colour for points that had a pixel and one over geometry alone for the
rest (see baseline.py), and the ratio between those two populations is the
context every number in the submission has to be read in.
"""

import argparse
import os
import sys

# Allows `python baselines/bl_paint/run.py` from a checkout with no PYTHONPATH
# set, which is how a reader will first try it. run_all.sh sets PYTHONPATH
# anyway, so this is a convenience and not the supported path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from baselines.bl_paint.baseline import PaintBaseline          # noqa: E402
from baselines.common import runner                            # noqa: E402
from semseg.datasets import split_frames                       # noqa: E402

# Distinct from 1 so a script driving several baselines can tell "this arm is
# not runnable on this dataset" apart from "this arm crashed".
NO_CALIB_EXIT = 2

NO_CALIB_MESSAGE = (
    "bl_paint needs T_cam_lidar and this dataset ships none.\n"
    "GOOSE distributes its rig calibration through the GOOSE-DB ROS bags' /tf_static, "
    "not through the annotated val zips (spec 13.3).\n"
    "Pass one with --calib, or run bl_geom3d, which needs no extrinsic for its 3D half.\n"
    "Refusing rather than falling back on geometry alone: the submission would be "
    "labelled bl_paint and would contain no fusion.")


def _splits(dataset, split: str):
    """-> (fit_ids, score_ids) with score_ids being the split named on the CLI.

    The stable-hash split of spec section 9 assigns each frame one side, and
    `--split fit` means "predict over the fit side", so the two lists swap
    rather than one being recomputed with a different fraction.
    """
    fit_ids, score_ids = split_frames(dataset.frame_ids())
    if split == "fit":
        return score_ids, fit_ids

    return fit_ids, score_ids


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="bl_paint: geometry features plus painted RGB, the naive early-fusion floor")
    runner.add_common_args(parser)
    args = parser.parse_args(argv)

    dataset = runner.build_dataset(args)

    # getattr, because only the GOOSE adapter has a calibration that can be
    # missing. The fixture's extrinsic is exact by construction.
    if not getattr(dataset, "calib_available", True):
        print(NO_CALIB_MESSAGE, file=sys.stderr)
        return NO_CALIB_EXIT

    fit_ids, score_ids = _splits(dataset, args.split)

    payload = runner.run(PaintBaseline(seed=args.seed), dataset, fit_ids, score_ids,
                         runner.make_extrinsic_fn(dataset), args.out,
                         limit=args.limit, jobs=args.jobs, split=args.split)

    _report_frustum_share(payload)
    return 0


def _report_frustum_share(payload) -> None:
    """Print how much of the cloud the camera could speak for.

    An upper bound, not the exact coloured count: the submission records the
    in-frustum population, and the points among them that lost the z-buffer
    were scored by geometry alone. Printed because a fusion number read
    without it is unreadable (assumption A2, and spec 13.5 measures a 90
    degree camera at 23 percent of a GOOSE cloud).
    """
    counts = payload["consistency"]
    total = counts["total_points"]
    share = counts["in_frustum"] / total if total else float("nan")

    print(f"in frustum: {counts['in_frustum']} of {total} points ({share:.1%}), "
          "an upper bound on the points scored with colour; the rest went to the "
          "geometry-only model")


if __name__ == "__main__":
    sys.exit(main())
