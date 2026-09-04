# Rules

## Where your method runs

Inside `docker/runtime.Dockerfile`. The image is a runtime only, with no
`COPY`: the repo and the data both arrive by bind mount, so what you develop
against is what gets scored. No internet while scoring.

`docker/gpu.Dockerfile` exists for anyone reproducing the published PTv3 or 2D
pretrained baselines. Nothing in `run_all.sh` or CI needs it, and a submission
that only runs there has to say so.

## Splits

Frames are assigned to a `fit` split and a `score` split by a stable hash of
the frame id, half each. Stable means md5 of the id string, not Python's
built-in `hash`, which is salted per process and would silently reshuffle your
split between runs.

Fit on the `fit` split. Score on the `score` split. Moving the boundary,
re-hashing with a different seed until the numbers improve, or fitting anything
at all on the scored frames is tuning on the test set. It will not survive
contact with a held-out sequence, and the whole point of the exercise is to
find out what does.

## Pretrained weights

Allowed. Name them, say where they came from, and confirm they run offline. If
your weights were trained on GOOSE train, say that too: it is fine, it is just
a different claim from training on nothing.

## What you have to report

Not a suggestion, this is the graded part.

- **Per-class IoU, in full, both modalities.** Never the mean alone. A mean
  over 9 classes where one class is 42 percent of the pixels tells the reader
  almost nothing.
- **Classes absent from the split excluded from the mean, not scored as zero.**
  An absent class has undefined IoU. In the frame this repo is written around,
  `human` does not appear at all. Score it as zero and your headline number
  stops being comparable to anyone else's.
- **Cross-modal consistency with its coverage.** The ratio alone is
  unreadable. Most lidar points have no pixel; a 90 degree camera sees at most
  about 23 percent of a sweep. Reporting 0.8 consistency without saying it was
  computed over a fifth of the points is misleading by omission.
- **The three arms, if you claim fusion helps.** Lidar-only, camera-only,
  fused. Same classifier, same fit split, only the feature set changing. Three
  different architectures compared against each other measures architecture,
  not fusion.
- **The crossover.** The perturbation at which your fused method falls below
  your lidar-only method. If it never crosses within the swept range, say that
  and say what the range was. "No crossover" and "already below at the smallest
  perturbation" are opposite findings and must not be reported with the same
  words.

## What is not allowed

- Fitting on the scored split, in any form, including early stopping on it.
- Reporting a mean IoU without the per-class table behind it.
- Inventing a calibration for the real data. The extrinsic is not in the public
  release. If you need one, get it from the GOOSE-DB bags' `/tf_static` and
  say which bag, or run on the fixture where it is exact. A guessed extrinsic
  produces confident, wrong, plausible-looking numbers, which is worse than no
  numbers.
- Quoting the published PTv3 figure as though this repo measured it. It is
  0.8096 3D mIoU on the full val split, published by the dataset authors,
  measured in the folded 8-class space. Cite it as theirs.

## Compute

Reported next to your score and never folded into it. It is self-declared, so
ranking on it would be trivially gameable, but it is graded in the write-up:
runtime per frame, peak memory, and an honest answer about whether the method
would hold up on a robot. A model that gains three points of mIoU and costs
200 ms is a worse result than one that gains one point at 30 ms, and saying so
in your write-up counts in your favour rather than against it.

## Honesty

The single rule that matters more than the rest. If something did not work, did
not run, or could not be measured, say so. A partial result with its gap stated
is worth more to us than a complete-looking result we cannot trust, and we will
check.
