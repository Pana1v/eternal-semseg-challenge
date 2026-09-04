#!/usr/bin/env bash
# Drives every tool in eternal-semseg-challenge against the synthetic fixture.
# From a clean checkout, with zero downloads: build the image, then
#
#   FIXTURE=/tmp/semseg-fixture ./run_all.sh
#
# generates a bi-modal world, runs all four baselines, scores them, sweeps the
# calibration and prints the crossover, then writes the HTML report.
#
# FIXTURE defaults outside the repo on purpose. The fixture is derived data,
# exactly reproducible from a seed, so the seed is the artefact worth keeping
# and the megabytes of npz and png it expands into are not. Generating into the
# tree would also mean the next `git status` is noise.
set -euo pipefail

FIXTURE="${FIXTURE:-/tmp/semseg-fixture}"
IMAGE=eternal-semseg-runtime
RESULTS=results

# The dataset root every tool reads is /fixture/frames, one level below the
# mount, so the fixture directory can also hold the generator script and the
# sweep log without FixtureDataset having to ignore them.
#
# 8 frames splits 3 fit / 5 score under FIT_FRACTION 0.5, so every tool sees
# both splits non-empty. The split is md5 of the frame id, so those counts are
# fixed rather than lucky.
FIXTURE_FRAMES="${FIXTURE_FRAMES:-8}"
FIXTURE_SEED="${FIXTURE_SEED:-0}"

# The four arms, in ablation order: floor, lidar-only, camera-only, fused.
BASELINES=(bl_prior bl_geom3d bl_cam2d bl_paint)

# Sweep grid trimming. A sweep re-predicts a whole split once per magnitude,
# so the full DECALIB_ROT_DEG x AXES x SWEEP_SEEDS grid is minutes of work and
# this script is a plumbing demonstration. Two magnitudes bracket the fixture's
# measured crossover region (cross-modal agreement 0.944 at 0.5 deg, 0.830 at
# 2.0 deg), which is the least that can still interpolate a crossover.
# Drop --rot-deg, --trans-cm and --limit for the full published grid.
SWEEP_ROT_DEG="${SWEEP_ROT_DEG:-0.5 2.0}"
SWEEP_TRANS_CM="${SWEEP_TRANS_CM:-2 10}"
SWEEP_FRAMES="${SWEEP_FRAMES:-2}"

mkdir -p "$FIXTURE" "$RESULTS"

# The image has no COPY: it is a runtime environment only, so the repo and the
# fixture both arrive by bind mount. --user keeps outputs owned by you rather
# than root. PYTHONPATH is the repo root because nothing here is pip installed
# and the modules import each other as packages ("import semseg",
# "from eval.io_formats import ...").
dr() {
    docker run --rm --user "$(id -u):$(id -g)" \
        -e PYTHONPATH=/workspace \
        -v "$PWD:/workspace" -v "$FIXTURE:/fixture" \
        -w /workspace "$IMAGE" "$@"
}

cat > "$FIXTURE/make_fixture.py" <<'PYEOF'
"""Materialise the synthetic bi-modal fixture every later step reads.

One generator, two modalities. The cloud is ray cast and the image is
rasterised from the SAME classed primitives through the same exact extrinsic,
so under a correct calibration a point lands on a pixel of its own class and
under a perturbed one it does not. The size of that gap is the only thing the
sweeps in this script measure, and it is measurable here precisely because the
extrinsic is known by construction. The public GOOSE release ships none, so
the fixture is where every projection-dependent number in this repo comes
from.

What the fixture does NOT measure, stated plainly so no result from it gets
overclaimed:

  - real appearance or real lidar physics. No rain, no snow, no dust, no
    retro-reflectors, no exhaust, and Gaussian range noise instead of a real
    beam model.
  - real class balance. The scene has no class 0 (other) at all, and its
    vegetation and ground shares are nothing like GOOSE's measured 61.75
    percent vegetation.
  - anything about generalisation. A method that wins here has shown that its
    plumbing is correct and that it degrades under decalibration. It has shown
    nothing about GOOSE.

Class 8 (sky) is present in the image and absent from the cloud, which makes
assumption A3 of the problem statement structural here rather than incidental.

Writes npz clouds, png images, png label maps and per-frame json once. Every
tool afterwards reads the same bytes, so a baseline and the scorer cannot
disagree about what the world was.
"""
import sys

from semseg.datasets.fixture import write_fixture

OUT = "/fixture/frames"

n_frames = int(sys.argv[1])
seed = int(sys.argv[2])

frame_ids = write_fixture(OUT, n_frames=n_frames, seed=seed)

print(f"fixture: {len(frame_ids)} frames, seed {seed}, written to {OUT}")
PYEOF

echo "== fixture =="
dr python3.12 /fixture/make_fixture.py "$FIXTURE_FRAMES" "$FIXTURE_SEED"

# One invocation per baseline spans BOTH splits: the runner fits on `fit` and
# predicts over `--split score`. The fit split is deliberately never scored.
# A number produced on the frames a method was fitted on is not a result, and
# putting one in the leaderboard next to the honest ones would be the exact
# failure docs/RULES.md is about.
echo "== baselines: fit on the fit split, predict on the disjoint score split =="

# bl_prior needs its mode named. majority is the floor a model that learned
# nothing but the class prior reaches; --mode uniform is the other floor and is
# a one flag change. Without a floor of some kind no mIoU on a 9-class problem
# with a 42 percent class means anything.
echo "-- bl_prior (chance floor)"
dr python3.12 baselines/bl_prior/run.py \
    --dataset fixture --root /fixture/frames --split score \
    --mode majority --out /fixture/sub_bl_prior.json

for bl in bl_geom3d bl_cam2d bl_paint; do
    echo "-- $bl"
    dr python3.12 "baselines/$bl/run.py" \
        --dataset fixture --root /fixture/frames --split score \
        --out "/fixture/sub_$bl.json"
done

# score.py takes no --gt: ground truth is already folded into the submission's
# confusion matrices, which is what keeps the scored artefact kilobytes. It
# also regenerates results/report.html over every method scored so far, so the
# report exists from the first iteration of this loop onward.
echo "== score =="
for bl in "${BASELINES[@]}"; do
    dr python3.12 eval/score.py \
        --submission "/fixture/sub_$bl.json" \
        --split score --method "$bl" --out-dir "$RESULTS"
done

# The headline of the whole repo. The crossover is the calibration accuracy the
# robot has to sustain in production (problem statement section 6.3), so
# sweep.py prints it rather than leaving a reader to interpolate the CSV.
# Both arms are swept in one run because the crossover is defined between two
# curves: bl_paint's RGB features arrive THROUGH the operator being perturbed,
# while bl_geom3d never reads the image and is the flat reference.
# $SWEEP_ROT_DEG and $SWEEP_TRANS_CM are word split on purpose, they are lists.
echo "== decalib sweep: fused arm against the lidar-only arm, trimmed grid =="
dr python3.12 eval/sweep.py \
    --baseline bl_paint --baseline bl_geom3d \
    --dataset fixture --root /fixture/frames --split score \
    --sweep decalib \
    --rot-deg $SWEEP_ROT_DEG --trans-cm $SWEEP_TRANS_CM \
    --limit "$SWEEP_FRAMES" --out-dir "$RESULTS/sweeps" \
    2>&1 | tee "$FIXTURE/decalib_sweep.log"

# These three run on the fixture and only on the fixture. time_offset needs
# per-point times and an ego twist, deskew needs both plus the extrinsic, and
# the GOOSE val zips ship no poses and no calibration at all (spec 13.3), so
# sweep.py refuses them there instead of assuming a constant velocity.
echo "== time_offset, dropout and deskew sweeps =="
for sweep in time_offset dropout deskew; do
    echo "-- $sweep"
    dr python3.12 eval/sweep.py \
        --baseline bl_paint --baseline bl_geom3d \
        --dataset fixture --root /fixture/frames --split score \
        --sweep "$sweep" \
        --limit "$SWEEP_FRAMES" --out-dir "$RESULTS/sweeps"
done

# score.py already rendered the plots and the report on every scoring run, but
# the sweeps ran after the last of those. This final pass is what pulls the
# sweep CSVs and their crossover verdicts into the report.
echo "== plots and report =="
dr python3.12 eval/report.py \
    --results "$RESULTS" --out "$RESULTS/report.html" \
    --title "Eternal SemSeg Challenge, fixture run"

echo "== done =="
echo "fixture and submissions: $FIXTURE"
echo "scores:                  $PWD/$RESULTS"
echo "sweeps:                  $PWD/$RESULTS/sweeps"
echo "decalib sweep log:       $FIXTURE/decalib_sweep.log"
echo "report:                  $PWD/$RESULTS/report.html"
