#!/usr/bin/env bash
# CI smoke test: builds the runtime image, generates a 2-frame synthetic
# fixture, runs the fused baseline against it, scores the submission, and
# verifies nothing crashed.
#
# Deliberately narrow. run_all.sh is the end-to-end demonstration; this one
# only has to catch a repo that cannot start, so it runs one baseline over one
# scored frame and no sweeps.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FIXTURE_DIR="$(mktemp -d)"
trap 'rm -rf "$FIXTURE_DIR"' EXIT
IMAGE=eternal-semseg-runtime

# Same mount pattern as run_all.sh: the image has no COPY, so the repo and the
# fixture both arrive by bind mount. --user avoids root-owned output files, and
# PYTHONPATH is the repo root because the modules import each other as
# packages and nothing here is pip installed.
dr() {
    docker run --rm --user "$(id -u):$(id -g)" \
        -e PYTHONPATH=/workspace \
        -v "$REPO_ROOT:/workspace" -v "$FIXTURE_DIR:/fixture" \
        -w /workspace "$IMAGE" "$@"
}

echo "== building runtime image =="
docker build -f "$REPO_ROOT/docker/runtime.Dockerfile" -t "$IMAGE" "$REPO_ROOT"

# 2 frames is the smallest fixture with a non-empty split on both sides: the
# md5 split sends fixture_0000 to fit and fixture_0001 to score, so the runner
# has something to fit on and something to score.
echo "== writing synthetic fixture =="
dr python3.12 -c "
from semseg.datasets.fixture import write_fixture

frame_ids = write_fixture('/fixture/frames', n_frames=2, seed=0)
print('fixture:', frame_ids)
"

# bl_paint is the fused arm, so it is the one baseline that touches the whole
# stack: the ground plane fit, the projection, the z-buffer, the painting and
# the shared classifier. A crash anywhere in that chain fails here.
echo "== running bl_paint =="
dr python3.12 baselines/bl_paint/run.py \
    --dataset fixture --root /fixture/frames --split score \
    --out /fixture/submission.json

echo "== scoring =="
dr python3.12 eval/score.py \
    --submission /fixture/submission.json \
    --split score --method bl_paint --out-dir /fixture/results

echo "== smoke test passed =="
