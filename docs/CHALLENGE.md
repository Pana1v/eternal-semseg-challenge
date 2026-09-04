# The Challenge

## Setup

You get a time-synchronised camera and 3D lidar looking at the same scene. Your
job is to produce a semantic label for every pixel of the image and every point
of the cloud, using both sensors, and to make the two labellings agree with each
other.

Formally, per frame: given an image `I` of shape `H x W x 3`, a cloud
`P = {(x, y, z, intensity)}`, intrinsics `K` and the extrinsic
`T_cam_lidar` in SE(3), produce `Y_2D` of shape `H x W` and `Y_3D` of shape `N`
over a shared label space, such that a point labelled `vegetation` projects into
a pixel labelled `vegetation`.

That last clause is where the difficulty is. Everything hard about this task
lives in the projection operator

```
pi(p) = K . T_cam_lidar . p
```

and in the fact that `T_cam_lidar` and the timestamps are never exactly right.

## Why this is not concatenating two feature maps

Trace it through, because this is the part most proposals skip.

Fusion improves some classes and degrades others. The usual pattern is a gain
on small or distant objects and a loss on large uniform surfaces. The reason is
that feature association happens through projection, and projection is only
valid if the extrinsic and the timestamps are exact. When they are not, image
features attach to the wrong points, the network learns to distrust the camera
branch, and you converge on a lidar-only model that costs more to run.

Projection is inexact for three independent reasons, and they compound.

**Extrinsic error.** Calibration residual. Half a degree of rotation puts a
10 m point about 9 cm off laterally. Thermal cycling and vibration mean the
calibration you measured in the morning is not the calibration you have in the
afternoon.

**Temporal misalignment.** Camera exposure midpoint and lidar point timestamp
are rarely the same instant. A PTP-clocked lidar and a system-clocked camera can
sit tens to hundreds of milliseconds apart. At 1 m/s that is a decimetre of
translation on top of the extrinsic error.

**Intra-scan skew.** A spinning lidar samples over a whole revolution, so "the
cloud at time t" is a fiction unless you deskew against ego-motion. Some drivers
fabricate per-point timestamps rather than measuring them.

The consequence for evaluation is the part that matters: projection error is
correlated with ego-motion, not iid noise. A benchmark evaluated frame by frame
on well-calibrated slow-moving data will report a fusion gain that evaporates on
a robot turning in place. So the protocol has to include a decalibration and
time-offset sweep, not just a leaderboard number.

## Assumptions this harness makes you confront

Each of these is false in some regime, and the harness surfaces the failure
rather than hiding it.

| | assumption | why it is false |
| --- | --- | --- |
| A1 | extrinsics are constant | thermal cycling, vibration |
| A2 | both sensors see the same scene | the lidar is 360 degrees, the camera is a frustum. At 90 degrees of horizontal field of view, at most about 23 percent of a real GOOSE sweep is in front of the camera at all |
| A3 | the label spaces match | `sky` is 13.08 percent of the split's pixels and **exactly 0** of its 174,891,807 points |
| A4 | 3D ground truth is independent evidence | false for datasets whose 3D labels were made by projecting 2D labels. GOOSE's are not, which is why it was chosen |
| A5 | mIoU reflects task utility | `human` is 1 point in 3,332. mIoU gives it a ninth of the weight; the data gives it 0.03 percent |

None of these are hypothetical here. A2, A3 and A5 are measured over the whole
split in [what the validation split actually contains](#what-the-validation-split-actually-contains),
and A4 is why GOOSE was chosen at all.

## The data

**GOOSE**, the German Outdoor and Offroad Dataset. Chosen over the alternatives
for one reason that decides it: its images and its clouds are each annotated by
hand, per frame, so the 3D ground truth is independent evidence rather than 2D
labels projected onto points. Evaluating an image-to-lidar fusion model on
projected labels is partly circular, which rules out A2D2 and WildScenes for
this purpose, and SemanticKITTI and nuScenes have no densely labelled images on
the same frames.

The validation split, which is what this repo evaluates on:

| | |
| --- | --- |
| frames complete in both modalities | 961 of 962 |
| sequences | 8, across summer, hills, rain, sun, an airfield and a training ground |
| image | 2048 x 1000 RGB |
| cloud | Velodyne VLS-128, roughly 100k to 170k points per frame |
| labels | human, per frame, both modalities |
| label space | 9 classes, mapped from GOOSE's 64 |
| licence | CC BY-SA 4.0 |

Download `goose_2d_val.zip` and `goose_3d_val.zip`, about 6 GB together. The
train split is another 48 GB and this repo does not train, so it is not needed.

### The one honest gap

The annotated release ships no camera-to-lidar extrinsic and no poses. This was
verified rather than assumed; see [`docs/SENSORS.md`](SENSORS.md) for exactly
where it was looked for and where the numbers actually live. Intrinsics are
published, but for a 2048 x 1536 sensor while the shipped images are 2048 x 1000,
so even the principal point's vertical component is ambiguous.

Which means: on real GOOSE you can run the 3D arm and the 2D arm. The fused arm,
cross-modal consistency and every sweep need the extrinsic. Supply one with
`--calib` and they run; otherwise they are skipped with a stated reason and the
projection-dependent results come from the fixture, where the extrinsic is exact
by construction.

Nothing in this repo approximates a missing extrinsic. That is a deliberate
refusal, not an omission.

## What the validation split actually contains

Measured over all 961 bi-modal frames: 1,968,128,000 labelled pixels and
174,891,807 labelled points, with zero unlabelled in either modality. Regenerate
with `docs/goose_val_stats.json` alongside, which this table is read from.

| class | share of pixels | share of points |
| --- | --- | --- |
| `other` | 1.26% | 0.25% |
| `artificial_structures` | 2.20% | 5.73% |
| `artificial_ground` | 12.01% | 3.95% |
| `natural_ground` | 24.51% | 23.45% |
| `obstacle` | 3.58% | 3.60% |
| `vehicle` | 1.08% | 1.24% |
| `vegetation` | 42.26% | 61.75% |
| `human` | 0.019% | 0.030% |
| `sky` | 13.08% | **0.000%** |

Two rows in that table should change how you read every number this repo
produces.

**`sky` is exactly 0 points.** Not approximately, not rounded: zero, out of
174,891,807. And 13.08 percent of the pixels. Assumption A3 is not a caveat to
mention in a limitations paragraph, it is a structural property of the problem,
and whatever you do about it is a design decision you have to state. See
[`docs/ONTOLOGY.md`](ONTOLOGY.md) for what this repo does and why it keeps 9
classes instead of folding to 8.

**`human` is 1 point in 3,332, and 1 pixel in 5,371.** Vegetation is 2,057
times more common. mIoU is an unweighted mean over classes, so it hands `human`
one ninth of the total weight while the data gives it three hundredths of a
percent. A model can post an excellent mIoU and be useless on the only class
where a mistake injures somebody. That is assumption A5 with a number attached,
and it is why the scorer reports every per-class IoU and the report marks
`human`, `vehicle` and `obstacle` explicitly.

It gets sharper with range. Points per bin over the whole split:

| bin | points | share | of which `human` |
| --- | --- | --- | --- |
| 0-5 m | 11,983,717 | 6.9% | 858 (0.0072%) |
| 5-15 m | 57,285,906 | 32.8% | 21,923 (0.0383%) |
| 15-30 m | 53,412,084 | 30.5% | 19,546 (0.0366%) |
| over 30 m | 52,210,100 | 29.9% | 10,156 (0.0195%) |

858 human points inside 5 m, across 961 frames. So a close-range `human` IoU
computed on this split rests on almost nothing, and reporting one without
saying so would be the most misleading number in the whole harness. If your
method's selling point is close-range pedestrian safety, this split cannot
demonstrate it and you should say that rather than quote the figure.

## The fixture

`semseg/datasets/fixture.py` builds a scene from primitives that each carry a
class, then renders the cloud by ray casting and the image by rasterising the
same primitives through a pinhole camera at a known exact extrinsic. Both
modalities come from one source of truth, so a correctly calibrated point lands
on a pixel of its own class and a decalibrated one does not.

That gap is the signal every sweep measures. It is also why the whole harness
runs end to end from a clean checkout with zero downloads:

```
FIXTURE=/tmp/semseg-fixture ./run_all.sh
```

## Splits

Frames go to a `fit` split and a `score` split by md5 of the frame id, half
each. md5 rather than Python's `hash`, which is salted per process and would
reshuffle the split between runs. Fit on `fit`, score on `score`, and see
[`docs/RULES.md`](RULES.md) for why moving that line counts as tuning on the
test set.

## What you submit

Predictions are label maps and point arrays, gigabytes per method, so they are
not the submission. Your method writes them to a gitignored run directory and
emits the statistics they imply:

```json
{
  "submission_version": 1,
  "method": "bl_paint",
  "label_space": "goose9",
  "num_classes": 9,
  "conf_2d": [[...]],
  "conf_2d_boundary": [[...]],
  "conf_3d": [[...]],
  "conf_3d_by_range": {"0-5m": [[...]], "5-15m": [[...]],
                        "15-30m": [[...]], "30m+": [[...]]},
  "conf_3d_in_frustum": [[...]],
  "conf_3d_out_frustum": [[...]],
  "ece_2d": {"counts": [...], "conf_sum": [...], "correct": [...]},
  "ece_3d": {"counts": [...], "conf_sum": [...], "correct": [...]},
  "consistency": {"matched": 0, "scorable": 0,
                   "in_frustum": 0, "total_points": 0}
}
```

Every matrix is 9x9 and rows are ground truth, columns are prediction. mIoU,
per-class IoU and frequency-weighted IoU are all exactly recoverable from a
summed confusion matrix, so the scored artefact is tens of kilobytes and CI can
grade it.

The splits are not decoration:

- **by range**, because point density falls off as one over range squared and a
  single 3D number hides where the model actually fails.
- **in and out of frustum**, because A2 says most points have no pixel, and this
  is what answers "how much of your fusion gain came from points the camera
  never saw".
- **boundary**, because a segmentation model's errors concentrate at class
  edges and interior accuracy flatters it.

Alongside it, `<submission>.meta.json` carries self-declared compute. It is
reported next to your score and never folded into it.

## Scoring

```
python eval/score.py --submission submission.json --split score \
    --method my_method --out-dir results
```

There is no `--gt` flag. Ground truth is already inside the confusion matrices;
that is the whole reason the submission is small.

Reported:

| metric | definition |
| --- | --- |
| mIoU, 2D and 3D | mean over classes **present** in the split. Absent classes are `nan` and excluded, never zero |
| per-class IoU | all 9, both modalities, always. Never the mean alone |
| fwIoU | IoU weighted by ground-truth class frequency, to expose gains that came only from the dominant class |
| boundary mIoU | 2D, within 3 px of a ground-truth class edge |
| range-stratified mIoU | 3D, binned 0-5, 5-15, 15-30, over 30 m |
| in / out of frustum mIoU | 3D, split on whether the camera could see the point |
| cross-modal consistency | fraction of visible in-frustum points whose 3D label matches the 2D label of their own pixel, **always with its coverage** |
| ECE | 15 bins. A head feeding a costmap has to be trustworthy, not just accurate |
| folded 8-class 3D mIoU | sky merged into other, so the number is comparable to the published reference |

A point counts toward consistency only if it is inside the frustum, wins its
pixel in the z-buffer, and has a defined label on both sides. The z-buffer is
not optional: a point behind a wall projects onto the wall's pixel, and without
occlusion handling you score it against the wall's label and report a
disagreement that is an artefact.

Consistency needs no ground truth. You can compute it on unlabelled robot logs,
which is the point of having it.

### Reference

A PTv3 model trained on GOOSE reports **0.8096** 3D mIoU on the full val split,
in the folded 8-class space. That figure is published by the dataset authors in
the GOOSE devkit, not measured here, and every place this repo shows it says so.

## The sweeps

The scientific core, and the reason this repo exists rather than a training
script. The problem statement is explicit that these run *before* architecture
work, not after.

```
python eval/sweep.py --baseline bl_paint --baseline bl_geom3d \
    --dataset fixture --root /tmp/semseg-fixture --sweep decalib
```

| sweep | range |
| --- | --- |
| `decalib` | rotation 0.1, 0.25, 0.5, 1.0, 2.0 degrees, per axis, plus random axis over 3 seeds; translation 1, 2, 5, 10 cm |
| `time_offset` | 0, 10, 25, 50, 100, 200 ms at 0.3, 1.0 and 2.0 m/s |
| `dropout` | camera blacked, camera saturated, points dropped 25, 50, 75 percent |
| `deskew` | with and without per-point motion compensation |

Rotation is swept **per axis** because half a degree about the optical axis is
nearly harmless while half a degree of pitch is a lateral error proportional to
range. One unnamed magnitude would average away the effect the sweep exists to
find.

Perturbation is inference-time only. `fit()` always sees the unperturbed
extrinsic and the unperturbed cloud; perturbing during fitting is an
augmentation experiment and a different question.

### The crossover

The headline output. The perturbation magnitude at which the fused arm's 3D mIoU
falls below the lidar-only arm's, by linear interpolation between the bracketing
magnitudes. That number is the calibration accuracy the robot has to sustain in
production, which is why it is printed rather than left for a reader to derive.

When the curves do not cross, the harness distinguishes two cases, because they
are opposite findings:

- *fusion never falls below lidar-only within 2.0 deg* means fusion is robust
  across the whole swept range.
- *fusion is already below lidar-only at the smallest swept perturbation* means
  fusion was never paying for itself.

## The baselines

Four arms, one classifier. That is the design decision that makes the ablation
mean something: a diagonal Gaussian Naive Bayes shared by all of them, so the
arms differ **only** in which features they receive. Comparing three different
architectures would confound "fusion helps" with "architecture B is better".

| baseline | features | role |
| --- | --- | --- |
| `bl_prior` | none | chance floor, uniform or a declared majority class |
| `bl_geom3d` | height over the fitted ground plane, PCA verticality and planarity, range, intensity | lidar-only arm |
| `bl_cam2d` | pixel colour, normalised row and column | camera-only arm |
| `bl_paint` | `bl_geom3d`'s features concatenated with painted RGB | fused arm |

`bl_prior` is not a joke entry. Without a chance floor no mIoU means anything,
and mIoU on a 9-class problem where one class is 42 percent of the pixels is
exactly the metric that flatters a model which learned nothing.

`bl_paint` is the naive early-fusion floor the problem statement names
explicitly: if a real fusion architecture does not clearly beat appending RGB to
the point feature vector, the added complexity is not justified.

The two single-modality arms produce the other modality by projection, which
gives a useful asymmetry worth knowing about: a decalibration perturbation moves
`bl_geom3d`'s 2D number and not its 3D number, and moves `bl_cam2d`'s 3D number
and not its 2D number. They bracket what the sweep should show.

## Measured baselines, and two numbers you must not read at face value

All from the 12-frame fixture at seed 0, 6 score frames, 164,390 points,
nominal extrinsic. Regenerate with `FIXTURE=/tmp/semseg-fixture ./run_all.sh`.

| arm | 2D mIoU | 3D mIoU | 3D in-frustum | 3D out-of-frustum | consistency |
| --- | --- | --- | --- | --- | --- |
| `bl_prior` | 0.0406 | 0.0339 | 0.0444 | 0.0256 | 0.1101 |
| `bl_geom3d` | 0.5036 | 0.8860 | 0.8793 | 0.7492 | 1.0000 |
| `bl_cam2d` | 1.0000 | 0.8312 | 0.8312 | n/a | 1.0000 |
| `bl_paint` | 0.5581 | **0.9455** | **0.9708** | 0.7492 | 1.0000 |

Coverage is 0.2805 for every arm: 46,108 of 164,390 points are scorable for
consistency, which is assumption A2 in one number.

### The frustum split answers section 6.2 exactly

`bl_paint` out-of-frustum 3D mIoU is 0.7492. So is `bl_geom3d`'s. Identical, to
four decimals, because outside the frustum `bl_paint` *is* `bl_geom3d`: there is
no colour, so it falls through to the geometry-only model.

In-frustum, `bl_paint` scores 0.9708 against `bl_geom3d`'s 0.8793, a gain of
0.0915. That is the answer to "how much of the reported gain comes from points
the camera never saw": none of it. The gain is entirely where the camera
actually contributes, which is what a fusion claim has to demonstrate rather
than assert. `bl_cam2d` reports `n/a` out-of-frustum because it declines to
label points it cannot see, rather than guessing.

### Why every consistency number above is 1.0000

Not because the baselines are well aligned. Because each has a **single head**
and derives the other modality from it by projection. `bl_geom3d` and
`bl_paint` scatter their 3D labels into the image; `bl_cam2d` resamples its 2D
map at each point's pixel. A point is scorable only if it owns its pixel in the
z-buffer, and that is the same pixel the scatter seeded from that point, so it
cannot disagree with itself. `bl_prior`, which predicts the two modalities
independently, scores 0.1101 and is the only informative row.

The metric is correct: occlusion handling, the coverage denominator and
`UNLABELED` exclusion are each unit tested. It is also the only metric here that
needs no ground truth, so it is the one a candidate can compute on unlabelled
robot logs, which is the point of having it.

But it only becomes *informative* for a method with two genuinely independent
heads, which is exactly what a real fusion architecture is and exactly what none
of the shipped baselines is. Read 1.0000 as "single-head arm, 1.0 by
construction", never as a score to beat.

### Why `bl_cam2d` scores 2D mIoU 1.0000

A perfect score is always a finding about the benchmark, not the method. The
fixture paints each class a distinct base colour plus per-pixel noise, so
colour-to-class is very nearly a lookup table and a Gaussian Naive Bayes over
(colour, row, column) recovers it exactly.

The fixture exists to make the projection-dependent sweeps measurable, where its
exactness by construction is the whole point. It is **not** a difficulty
benchmark, and its 2D task in particular is degenerate for an appearance-based
method. Difficulty claims belong to the real GOOSE numbers.

### The same arms on real GOOSE, which is where difficulty claims belong

40 frames of GOOSE val, taken as the first 40 of the md5-stable score split, no
`--calib` and therefore no extrinsic. The frame set is byte identical across
arms (`frame_ids` md5 `aba049dc4ae3`), so these rows are a valid side by side
comparison and not three different subsets.

| arm | 2D mIoU | 3D mIoU | classes in the 3D mean | frustum split | consistency |
| --- | --- | --- | --- | --- | --- |
| `bl_prior` | 0.0420 over 9 | 0.0338 | 9 of 9 | declined | declined |
| `bl_cam2d` | 0.2834 over 7 | declined | | declined | declined |
| `bl_geom3d` | declined | **0.1902** | 8 of 9 | declined | declined |
| `bl_paint` | refuses to run | | | | |

`bl_geom3d` is the arm that still produces a real answer here, which is the
whole reason it exists: no step of its 3D path touches the extrinsic. Its 2D
half is declined because that half *is* a projection of its 3D labels. Against a
uniform chance floor of 0.0381 the margin is 0.1521, about 5x. Compute is
11.35 s/frame at 1518 MB peak, which is graded and is not free.

Folded to the published 8-class space it is also 0.1902, identical to four
decimals, because `sky` has zero lidar points and this arm never predicts it, so
folding sky into `other` moves nothing. Set that against the **0.8096** that a
PTv3 model reports on the full split in the same folded space (published by the
dataset authors, not measured here) and the headroom is the point: the shipped
lidar arm is a hand-crafted-feature Naive Bayes and there is a great deal left
to win.

Read "declined" as the harness refusing to compute a number, not as a zero. With
no extrinsic there is no projection, so the frustum split and the consistency
counts are genuinely unanswerable and `bl_cam2d` has no way to label a point at
all. `bl_paint` refuses outright rather than falling back to geometry, because a
submission labelled `bl_paint` that contained no fusion would be the most
misleading artefact this repo could produce.

#### The fixture's perfect 2D score does not transfer

`bl_cam2d` scores 1.0000 on the fixture and **0.2834** here. That is the
measurement. The interpretation offered above, that the fixture's colour to
class map is very nearly a lookup table, is consistent with it but is not the
only difference between the two runs: the image content, the fit set and the
frame count all differ too. The honest statement is the narrow one. A perfect
score on the fixture predicts nothing about real data, and now there is a number
rather than an argument behind that sentence.

#### The two 2D means are not over the same classes

`vehicle` and `human` appear in exactly zero of the 40 frames' 2D ground truth.
Under the nan-not-zero convention that lands differently on the two arms, and
the asymmetry is worth understanding because it will land on candidate
submissions the same way:

- `bl_prior` samples from the class prior, so it *predicts* both absent classes.
  Union is non-empty, intersection is empty, IoU is a hard 0.0, and that 0.0
  stays in the mean. Its mean is over 9 classes.
- `bl_cam2d` never predicts either one, so ground truth and prediction are both
  empty, IoU is nan, and the class drops out. Its mean is over 7.

So the convention penalises predicting a class that is not there and rewards
staying silent about it. That is the correct behaviour, and it means 0.0420
against 0.2834 is not a ratio. On the matched 7-class basis the chance floor is
**0.0540** and `bl_cam2d` is **0.2834**, a margin of 5.25x over chance. Quote
that one when comparing arms, and always say how many classes a mean covers.

#### What the camera arm actually learned

The mean hides the shape of the result, which is the whole reason this repo
reports every per-class IoU:

| class | 2D IoU |
| --- | --- |
| `sky` | 0.7445 |
| `other` | 0.4549 |
| `vegetation` | 0.4481 |
| `natural_ground` | 0.1760 |
| `artificial_ground` | 0.1604 |
| `artificial_structures` | 0.0000 |
| `obstacle` | 0.0000 |

A diagonal Gaussian Naive Bayes over colour, row and column learns the three
things that are separable by colour and image position, and learns literally
nothing about structures or obstacles. Those are the two classes where an error
matters most for a robot, and they are the two the appearance-only arm fails
completely. A single 0.2834 conceals that; the per-class table is the finding.

#### What the lidar arm actually learned, and where it scores zero

| class | 3D IoU | share of the split |
| --- | --- | --- |
| `vegetation` | 0.5834 | 61.75 percent |
| `natural_ground` | 0.4572 | |
| `artificial_structures` | 0.2798 | |
| `artificial_ground` | 0.1730 | |
| `obstacle` | 0.0284 | |
| `other` | 0.0000 | |
| `vehicle` | 0.0000 | |
| `human` | 0.0000 | 0.030 percent |
| `sky` | nan | 0 points |

Three classes score exactly zero and a fourth is 0.0284. The arm learns the two
classes that dominate the split and the one with distinctive vertical structure,
and learns nothing whatsoever about anything rare or small.

This is the mIoU warning from the README with a number attached. `human` IoU is
**0.0000**, and the unweighted 9-class mean still reads 0.1902, which does not
look like a model that cannot see people at all. `fwIoU`, which weights by class
frequency, reads 0.4703 and looks better still. Any single number that averages
over classes will do this. The per-class column is not an appendix to the score,
it is the score.

#### The range bins are not comparable to each other

Range stratification exists so that a model which is excellent up close and
useless at distance cannot hide behind one mean. It works, but reading the bins
against each other introduces a second problem, and this run demonstrates it:

| bin | 3D mIoU | points | classes in that bin's ground truth |
| --- | --- | --- | --- |
| 0-5 m | 0.130 | 612,393 | 5 |
| 5-15 m | 0.222 | 1,986,296 | 6 |
| 15-30 m | 0.178 | 1,974,283 | 8 |
| 30 m+ | 0.227 | 2,265,253 | 7 |

The curve rises with range, which would be a surprising claim about a sensor
whose angular resolution falls off as one over distance. It is largely not a
claim about the sensor. Each bin's mIoU is a mean over a different set of
classes, from 5 to 8 of them, because the near field simply does not contain
`vehicle`, `human` or `artificial_structures` in these 40 frames while the mid
field contains all eight. The 15-30 m bin is the only one holding `human`, and
`human` scores zero, which is part of why that bin dips.

So compare a bin against the same bin for another method, never against the
neighbouring bin for the same method. The same caution applies to every
partitioned mean in this repo, the frustum split included: partitions with
different class support produce means that are individually correct and
mutually incomparable.

### The crossover, which is the headline

Decalibration sweep over `bl_paint` against `bl_geom3d`, whose 3D mIoU is flat
at 0.8860 by construction since it never reads the extrinsic:

| axis | crossover | at 0.25 deg | 0.5 deg | 1 deg | 2 deg |
| --- | --- | --- | --- | --- | --- |
| yaw | **0.86 deg** | 0.9343 | 0.9144 | 0.8749 | 0.8139 |
| pitch | **1.19 deg** | 0.9371 | 0.9169 | 0.8951 | 0.8480 |
| roll | none within 2 deg | 0.9460 | 0.9430 | 0.9278 | 0.9023 |
| random axis | 0.89, 1.03, 1.83 deg over seeds 0, 1, 2 | | | | |
| translation | none within 10 cm, any axis | | | | |

Read the axis ordering: yaw is tightest, then pitch, and roll never crosses.
That is the per-axis decomposition earning its place. A single unnamed
delta-theta would have averaged 0.86 against no-crossing-at-all and reported
something true of no axis. Measured directly, 0.5 degrees of rotation moves a
centred 10 m point by 8.73 cm in pitch and 0 px in roll, growing off-axis with
distance from the principal point.

So on this fixture the fused arm tolerates about 0.9 degrees of yaw error before
it is worse than ignoring the camera. That number is what a production
calibration has to hold, and it is the sentence the whole harness exists to be
able to write.

## Grading

- 40 percent eval score
- 30 percent write-up, two pages or fewer: what you tried, why, what did not
  work, what you would do next
- 20 percent code quality
- 10 percent experimental hygiene, meaning the ablations are present and the
  defaults are sane

Two to four focused days. A strong entry beats one baseline and explains one
thing we did not already know.
