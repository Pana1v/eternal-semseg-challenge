"""The iterate and accumulate driver every bl_*/run.py calls, plus the CLI
arguments the four of them share.

It exists so a baseline file contains feature code and nothing else. The split
handling, the Accumulator folding, the submission write and the compute
sidecar are identical for all four, and four copies of them would drift apart
by the third sweep.
"""

import argparse
import functools
import importlib
import json
import multiprocessing
import platform
import resource
import sys
import time

from eval.io_formats import Accumulator, save_submission
from semseg.datasets import split_frames

# Serial by default. A worker pool is opt in, exactly as in the reference
# repo: the pool costs a pickled Frame per result and only pays off on a full
# split, while most invocations are a --limit smoke run.
DEFAULT_JOBS = 1
DEFAULT_SEED = 0
DEFAULT_SPLIT = "score"

META_SUFFIX = ".meta.json"
KB_PER_MB = 1024

# --dataset choice -> (module, class). Kept as data in one place because the
# two dataset modules are imported lazily and by name.
DATASET_MODULES = {
    "fixture": ("semseg.datasets.fixture", "FixtureDataset"),
    "goose": ("semseg.datasets.goose", "GooseDataset"),
}

_WORKER = {}


def peak_rss_mb() -> float:
    """Peak resident set size of this process and of any worker it reaped.

    ru_maxrss is kilobytes on Linux and bytes on macOS, so the unit has to be
    branched on rather than assumed. RUSAGE_SELF alone would under report a
    --jobs run exactly when memory is worth reporting, because with a pool the
    prediction happens in the children.
    """
    peak = max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    if sys.platform == "darwin":
        return peak / (KB_PER_MB * KB_PER_MB)

    return peak / KB_PER_MB


def write_meta(out_path: str, method_name: str, runtime_sec_total: float, n_frames: int,
                params: dict = None):
    """The compute sidecar, `<submission>.meta.json`.

    Records what the scorer cannot measure for itself: it only ever reads a
    confusion matrix file, so the cost of producing that file has to be
    declared by whoever produced it. Reported next to the score and never
    folded into it, because a self declared figure would be trivially
    gameable as a ranking term.
    """
    with open(out_path, "w") as f:
        json.dump({
            "method_name": method_name,
            "runtime_sec_total": runtime_sec_total,
            "n_frames": n_frames,
            "runtime_sec_per_frame": runtime_sec_total / n_frames if n_frames else None,
            "peak_rss_mb": peak_rss_mb(),
            "machine": {"cpu": platform.processor() or platform.machine(),
                        "platform": platform.platform()},
            "params": params or {},
        }, f, indent=2)


def _new_accumulator(method: str, split: str) -> Accumulator:
    """Interface spec section 6 freezes the Accumulator's add() and to_dict()
    but not its constructor, so the construction lives here, once, and an
    integration mismatch is a one line fix instead of four.
    """
    return Accumulator(method=method, split=split)


def _load_frames(dataset, frame_ids):
    """A generator, so fit() holds one frame at a time rather than a whole
    split of images and clouds in memory."""
    for frame_id in frame_ids:
        yield dataset.load(frame_id)


def _predict_serial(baseline, dataset, score_ids, extrinsic_fn, acc):
    for frame_id in score_ids:
        frame = dataset.load(frame_id)
        T_cam_lidar = extrinsic_fn(frame_id)
        acc.add(frame, baseline.predict(frame, T_cam_lidar), T_cam_lidar)


def _init_worker(baseline, dataset, extrinsic_fn):
    _WORKER["baseline"] = baseline
    _WORKER["dataset"] = dataset
    _WORKER["extrinsic_fn"] = extrinsic_fn


def _predict_one(frame_id):
    frame = _WORKER["dataset"].load(frame_id)
    T_cam_lidar = _WORKER["extrinsic_fn"](frame_id)
    return frame, _WORKER["baseline"].predict(frame, T_cam_lidar), T_cam_lidar


def _predict_pooled(baseline, dataset, score_ids, extrinsic_fn, acc, jobs):
    """Frames are independent, so a pool is free parallelism, with two costs
    worth stating rather than hiding.

    Workers start under "spawn", not "fork". Forking a parent that has already
    used open3d or a BLAS thread pool gives the child locks whose owning
    threads do not exist in it, and the child then completes its work and
    hangs at teardown. That was observed in the reference repo and the fix
    there was the same: spawn, and let the worker build its own state.

    Every result pickles a whole Frame plus Prediction back to the parent, a
    few megabytes per frame. Accepted deliberately: the Accumulator has to
    fold every frame into one object in one process, so the arrays have to
    come home. Note that spawn also requires the baseline, the dataset and
    extrinsic_fn to be picklable, which is why make_extrinsic_fn returns a
    functools.partial and not a closure.
    """
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(jobs, initializer=_init_worker,
                  initargs=(baseline, dataset, extrinsic_fn)) as pool:
        for frame, pred, T_cam_lidar in pool.imap(_predict_one, score_ids, chunksize=1):
            acc.add(frame, pred, T_cam_lidar)


def run(baseline, dataset, fit_ids, score_ids, extrinsic_fn, out_path,
        limit=None, jobs=DEFAULT_JOBS, split=DEFAULT_SPLIT):
    """Fit on fit_ids, predict over score_ids, write `out_path` and
    `out_path + ".meta.json"`. Returns the submission payload.

    extrinsic_fn(frame_id, perturbed=True) -> (4, 4) float64 or None is the
    ONLY source of T_cam_lidar, and two things ride on that signature.

    First, fit() is called with extrinsic_fn(fit_ids[0], perturbed=False).
    Interface spec section 8 requires perturbation to be inference time only:
    perturbing during fit is a different experiment (augmentation) and must
    never happen by accident. Making that call here rather than in each of
    the five callers makes the guarantee structural.

    Second, None means "this dataset ships no calibration", not "use a
    default". GOOSE's val zips carry none (interface spec 1b), so its
    extrinsic() raises, and the lidar-only arm still has to be runnable
    there. The driver therefore never reaches for dataset.extrinsic itself.

    The rig is fixed across a sequence, so fit() takes one matrix for the
    whole fit split rather than one per frame.
    """
    fit_ids, score_ids = list(fit_ids), list(score_ids)
    if limit is not None:
        score_ids = score_ids[:limit]
    if not score_ids:
        raise ValueError("nothing to score: the score split resolved to zero frames")

    # Interface spec section 9: nothing is ever fitted on the split it is
    # scored on. Cheap to check here and impossible to notice in the results.
    overlap = sorted(set(fit_ids) & set(score_ids))
    if overlap:
        raise ValueError(f"{len(overlap)} frames are in both splits, e.g. {overlap[:3]}")

    started = time.perf_counter()

    if fit_ids:
        baseline.fit(_load_frames(dataset, fit_ids), extrinsic_fn(fit_ids[0], perturbed=False))

    acc = _new_accumulator(baseline.name, split)
    if jobs > 1:
        _predict_pooled(baseline, dataset, score_ids, extrinsic_fn, acc, jobs)
    else:
        _predict_serial(baseline, dataset, score_ids, extrinsic_fn, acc)

    runtime_sec_total = time.perf_counter() - started

    # written through io_formats rather than json.dump here: that function
    # validates the payload first, so a baseline that folds a malformed matrix
    # fails at the write and not hours later inside the scorer
    payload = acc.to_dict()
    save_submission(out_path, payload)

    write_meta(out_path + META_SUFFIX, baseline.name, runtime_sec_total, len(score_ids),
                params={"split": split, "jobs": jobs, "limit": limit,
                        "n_fit_frames": len(fit_ids)})

    print(f"{baseline.name}: fitted on {len(fit_ids)} frames, scored {len(score_ids)} "
          f"in {runtime_sec_total:.1f}s")
    print(f"wrote {out_path}")
    print(f"wrote {out_path + META_SUFFIX}")
    return payload


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """The CLI of interface spec section 10, declared once so the four
    baselines cannot drift into four slightly different flag sets.
    """
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_MODULES),
                        help="goose reads the real val split; fixture generates a known world")
    parser.add_argument("--root", help="dataset root; the fixture generates into it")
    parser.add_argument("--split", default=DEFAULT_SPLIT, choices=("fit", "score"),
                        help="which split to predict over; the other one is fitted on")
    parser.add_argument("--out", default="submission.json")
    parser.add_argument("--limit", type=int, help="score only the first N frames")
    parser.add_argument("--jobs", type=int, default=DEFAULT_JOBS,
                        help="worker processes; 1 is serial and is the default")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--calib", help="4x4 T_cam_lidar as .npy or .json. REQUIRED on goose for "
                                        "any projection dependent output: the val zips ship no "
                                        "calibration at all")
    parser.add_argument("--crop-top", type=int,
                        help="goose only: vertical crop offset in pixels, subtracted from the "
                             "published cy. The GOOSE val images are 1000 rows where the "
                             "camera_info declares 1536, and the offset is published nowhere, "
                             "so pass this ONLY if you know it. Left out, cy keeps its "
                             "published value and coverage() reports crop_offset_known False")
    return parser


def build_dataset(args):
    """Resolve --dataset to a Dataset (interface spec section 9).

    Imported lazily and only for the chosen name: a top level import would
    make this module unimportable until both dataset files exist, and it would
    drag open3d into a fixture only run that never opens a GOOSE cloud.
    """
    module_name, class_name = DATASET_MODULES[args.dataset]
    module = importlib.import_module(module_name)
    dataset_class = getattr(module, class_name)

    if args.dataset == "fixture":
        # the fixture's extrinsic is exact by construction, so --calib has
        # nothing to override there
        return dataset_class(root=args.root, seed=args.seed)

    # --calib is handed to the adapter rather than parsed here: the same file
    # carries K as well as T_cam_lidar, and the Frame's intrinsics have to come
    # from the same source as its extrinsic or the projection is a chimera.
    # --crop-top goes only down this branch: the fixture's images are the
    # calibrated images by construction, so it has no crop to correct.
    #
    # getattr, because eval/sweep.py and tools/viewer.py deliberately declare
    # their own flag sets rather than calling add_common_args, so the attribute
    # is genuinely absent there. Absent means "not supplied", which leaves cy
    # at its published value and crop_offset_known False: the honest default,
    # and the same one you get by omitting the flag.
    return dataset_class(root=args.root, calib_path=args.calib,
                         crop_top=getattr(args, "crop_top", None))


def resolve_splits(dataset, split: str):
    """-> (fit_ids, score_ids), with score_ids being the split named on the CLI.

    The stable-hash split of interface spec section 9 assigns each frame one
    side, and `--split fit` means "predict over the fit side", so the two lists
    swap rather than one being recomputed with a different fraction. Swapping
    here rather than inside run() keeps that function's arguments literal: it
    fits on the first list and scores the second.
    """
    fit_ids, score_ids = split_frames(dataset.frame_ids())
    if split == "fit":
        return score_ids, fit_ids

    return fit_ids, score_ids


def _from_dataset(frame_id, perturbed=True, *, dataset):
    return dataset.extrinsic(frame_id)


def no_extrinsic(frame_id, perturbed=True):
    """The extrinsic_fn for a run with no calibration at all.

    Every baseline then declines its projection dependent output. This is the
    only honest way to run anything on real GOOSE without --calib: a
    fabricated default would turn every projection dependent number, and
    every sweep built on them, into a fiction. GooseDataset.calib_available is
    how a caller detects the case before choosing this over make_extrinsic_fn.
    """
    return None


def make_extrinsic_fn(dataset):
    """-> extrinsic_fn(frame_id, perturbed=True) around dataset.extrinsic.

    An adapter and not decoration: Dataset.extrinsic takes a frame id and
    nothing else (interface spec section 9), so handing it to run() directly
    would raise on the fit call's perturbed=False. Nothing here perturbs
    anything; eval/sweep.py wraps this.

    functools.partial rather than a closure because a lambda cannot be
    pickled, and with --jobs > 1 this function object is handed to a spawned
    worker.
    """
    return functools.partial(_from_dataset, dataset=dataset)


def has_calibration(dataset) -> bool:
    """Whether this dataset can supply a T_cam_lidar at all.

    Asking the adapter rather than catching Dataset.extrinsic's raise keeps
    that raise meaning what it says. The default is True for the fixture,
    whose extrinsic is exact by construction and which therefore carries no
    such flag; only the GOOSE adapter has a calibration that can be missing.
    """
    return getattr(dataset, "calib_available", True)


def resolve_extrinsic_fn(dataset):
    """-> the extrinsic_fn run() wants, declining when there is no rig.

    The choice lives here because getting it wrong is invisible: a fabricated
    default extrinsic yields a full set of projection dependent numbers that
    are all fiction. An arm that needs to ANNOUNCE the declining case asks
    has_calibration separately, so the wording stays in the arm whose output
    it describes.
    """
    return make_extrinsic_fn(dataset) if has_calibration(dataset) else no_extrinsic
