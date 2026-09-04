# Ontology

The class mapping is a published artefact, not an implementation detail. The
problem statement lists it as deliverable 5 and names ontology drift as a
risk: a mapping that changes quietly between runs inflates or deflates every
score without touching a line of model code. So it lives here, in full, and
the code reads it from a CSV rather than hardcoding it.

## Source

`docs/challenge_label_mapping.csv`, fetched from
<https://goose-dataset.de/docs/resources/challenge_label_mapping.csv> and
committed unmodified. It maps GOOSE's 64 fine classes onto the 9 categories
used by the GOOSE ICRA challenges.

Two things about the file are worth knowing before you parse it:

- The header spells one column `challege_category_id`, missing an `n`. That is
  upstream. `semseg/labels.py` accepts both spellings and does not edit the
  published file.
- The `goose_label_mapping.csv` that ships *inside* the dataset zips has no
  superclass column at all. It gives the 64 fine names only. The mapping to 9
  categories is distributed separately, which is why it is committed here.

## `goose9`, the label space this repo predicts in

| id | name | fine classes | safety relevant |
| --- | --- | --- | --- |
| 0 | `other` | 3 |  |
| 1 | `artificial_structures` | 4 |  |
| 2 | `artificial_ground` | 7 |  |
| 3 | `natural_ground` | 6 |  |
| 4 | `obstacle` | 20 | yes |
| 5 | `vehicle` | 11 | yes |
| 6 | `vegetation` | 10 |  |
| 7 | `human` | 2 | yes |
| 8 | `sky` | 1 |  |

`UNLABELED = 255` is a separate sentinel for pixels and points with no ground
truth, and for predictions a method declines to make. It is dropped from every
confusion matrix rather than scored.

The safety-relevant column is not decoration. Assumption A5 of the problem
statement is that mIoU does not reflect task utility: two points of IoU on
`natural_ground` is worth less than two points on `human`. The report marks
these three classes so a reader cannot skim past which classes a gain came
from.

## The sky problem, and why the space has 9 classes and not 8

A lidar gets no return from sky. In the reference frame this repo is written
around, `sky` is 29.0 percent of the image pixels and 0.0 percent of the
lidar points. Assumption A3 of the problem statement states the case exactly:
"Datasets have sky in 2D and no sky in 3D. Class mapping is lossy and must be
documented."

GOOSE resolves it by folding sky into `other` for its 3D challenge; its
Pointcept configuration maps class 8 to class 0. That gives 8 classes and is
what the published PTv3 numbers are measured in.

This repo does both, deliberately.

- **`goose9` is what methods predict in, and what consistency is measured in.**
  Keeping sky as its own class is what makes cross-modal consistency
  meaningful. A lidar point that projects into a sky pixel is a real signal:
  the camera says the ray is above the horizon and the lidar returned a range,
  so either the calibration is wrong or one of the two labels is. Folding sky
  into `other` absorbs that evidence and you never see it.
- **`fold_sky()` projects 9 down to 8** by mapping class 8 to class 0, exactly
  as GOOSE does. Every 3D number is reported in this view as well, so the
  repo's numbers stay comparable to the published reference.

Both views appear in `summary.json`. Neither is derived from the other after
the fact: the 9-class confusion matrix is accumulated once and the 8-class view
is the same matrix with row 8 and column 8 added into row 0 and column 0.

## Full mapping, all 64 fine classes

### 0 `other`

`undefined` (0), `ego_vehicle` (8), `outlier` (56)

### 1 `artificial_structures`

`building` (38), `wall` (39), `bridge` (43), `tunnel` (44)

### 2 `artificial_ground`

`cobble` (3), `bikeway` (7), `pedestrian_crossing` (9), `road_marking` (11), `sidewalk` (21), `curb` (22), `asphalt` (23)

### 3 `natural_ground`

`snow` (2), `leaves` (5), `gravel` (24), `soil` (31), `low_grass` (50), `water` (54)

### 4 `obstacle`

`traffic_cone` (1), `obstacle` (4), `street_light` (6), `road_block` (10), `traffic_light` (19), `boom_barrier` (25), `rail_track` (26), `debris` (29), `animal` (33), `rock` (40), `fence` (41), `guard_rail` (42), `pole` (45), `traffic_sign` (46), `misc_sign` (47), `barrier_tape` (48), `wire` (55), `container` (58), `barrel` (60), `pipe` (61)

### 5 `vehicle`

`car` (12), `bicycle` (13), `bus` (15), `motorcycle` (20), `truck` (34), `on_rails` (35), `caravan` (36), `trailer` (37), `kick_scooter` (49), `heavy_machinery` (57), `military_vehicle` (63)

### 6 `vegetation`

`forest` (16), `bush` (17), `moss` (18), `tree_crown` (27), `tree_trunk` (28), `crops` (30), `high_grass` (51), `scenery_vegetation` (52), `hedge` (59), `tree_root` (62)

### 7 `human`

`person` (14), `rider` (32)

### 8 `sky`

`sky` (53)

Numbers in brackets are the fine `label_key` as it appears in the label files:
the low 16 bits of each `uint32` in a `.label`, and the raw pixel value in a
`_labelids.png`.

## What this mapping costs

It is lossy in ways that matter for a robot, and the losses should be stated
rather than discovered later:

- `obstacle` absorbs 20 fine classes, from `traffic_cone` to `wire` to
  `animal`. A high `obstacle` IoU says nothing about whether a method can tell
  a wire from a rock. The published PTv3 reference scores 0.4554 on this class
  against 0.9179 on `vegetation`, and the breadth of the class is part of why.
- `natural_ground` holds both `low_grass` and `water`. Those are the same
  category here and very much not the same thing to drive onto.
- `vegetation` holds `tree_trunk` and `low_grass`'s counterpart `high_grass`
  together, so the traversability distinction the problem statement cares about
  in crop rows, whether a green thing is a plant you can push through or a
  trunk you cannot, is inside a single class.

None of that is a reason to invent a different mapping. It is a reason to
report per-class IoU in full, which the scorer does, and to treat a headline
mIoU as a summary rather than a result.
