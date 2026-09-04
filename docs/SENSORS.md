# Sensors

Two rigs matter here. The real one, which produced the GOOSE data, and the
synthetic one, which the fixture generates. They are documented together
because the harness treats them through the same `Frame` and the same
`predict(frame, T_cam_lidar)` call, and because the differences between them
determine which experiments can run on which.

## The real rig: MuCAR-3

GOOSE's vehicle platform is a Volkswagen Touareg with drive-by-wire. The two
sensors this challenge uses:

| | value |
| --- | --- |
| lidar | Velodyne VLS-128 roof, topic `sensor/lidar/vls128_roof` |
| lidar rate | 10 Hz raw, labelled frames at roughly 0.2 Hz |
| camera | prism camera behind the windshield, `vis` and `nir` channels |
| camera rate | 10 Hz raw |
| shipped image | 2048 x 1000 RGB, `_windshield_vis.png` |

Use the `vis` channel. `_windshield_nir.png` is near-infrared, has its own
intrinsics, and is a different modality. Substituting it for RGB silently
would be a category error, not a shortcut.

Measured from one frame of `2023-05-17_neubiberg_sunny`:

| | value |
| --- | --- |
| points per sweep | 111,047 |
| elevation | -25.0 to +15.0 degrees |
| range | 1.12 m minimum, 199.8 m maximum, 16.5 m median |
| intensity | 0 to 252, mean 19.9, not normalised |

Point density falls off with the square of range, which is why the scorer bins
3D IoU by range instead of reporting one number. In that frame, 3.3 percent of
points sit inside 5 m and 24.3 percent beyond 30 m. A model that is excellent
up close and useless at distance and a model that is mediocre everywhere can
report the same 3D mIoU.

### Intrinsics

Published, and committed here as `docs/calib/mucar3_windshield_vis.yaml`:

```
image_width  2048
image_height 1536
K            [1775.62133,          0, 1025.99113,
                       0, 1784.82927,  775.44415,
                       0,          0,          1]
distortion   plumb_bob, [-0.14196, 0.09598, 0.00212, -0.00044, -0.00044]
```

**Read the two heights again.** The intrinsics describe a 2048 x 1536 sensor.
The images in the release are 2048 x 1000. 536 rows are gone, so what ships is
a crop, and the crop's vertical offset is published nowhere. `fx`, `fy` and
`cx` carry over unchanged; `cy` does not. The correct value for the cropped
image is `775.44 - crop_top`, and `crop_top` is unknown.

The adapter therefore warns when it sees the mismatch, reports
`crop_offset_known = False`, and takes `--crop-top` from anyone who knows the
answer. It does not pick a number. A wrong `cy` tilts every projection by a
fixed vertical offset, which is precisely the kind of error that produces
plausible output and wrong conclusions.

### Extrinsics: not published

There is no camera-to-lidar transform in the annotated release. This was
checked, not assumed:

- Neither val zip contains any yaml, json, txt, ini or pose file.
- The published TF tree, committed here as `docs/calib/mucar3_tf.dot`, is an
  `rqt_tf_tree` graph dump. It carries frame topology and broadcaster rates and
  nothing else; it has no translation, rotation or quaternion values anywhere
  in it. It does confirm that `sensor/camera/windshield/vis` is a descendant of
  `sensor/lidar/vls128_roof`, so the calibration exists. It just is not in
  there.
- GOOSE's platform documentation states that every camera is calibrated
  against the VLS-128 roof lidar and that the processed bags publish
  `/tf_static`. That is where the numbers are: in the GOOSE-DB ROS bags.

So `GooseDataset.extrinsic()` raises, names the three places it looked, and
points at `/tf_static`. It never returns a plausible default, because a
plausible default would make every projection-dependent number in this repo a
measurement of a fiction that still looked like a result.

What this costs, concretely:

| experiment | real GOOSE | fixture |
| --- | --- | --- |
| 3D-only arm (`bl_geom3d`) | yes | yes |
| 2D-only arm, 2D half (`bl_cam2d`) | yes | yes |
| fused arm (`bl_paint`) | needs `--calib` | yes |
| cross-modal consistency | needs `--calib` | yes |
| decalibration sweep | needs `--calib` | yes |
| time-offset sweep | needs poses too | yes |

Supply a calibration with `--calib` and the right-hand column's experiments
open up on real data. Until then the projection-dependent results in this repo
come from the fixture, and every place they are reported says so.

The `yes` rows above are measured, not asserted: `bl_geom3d` scores 0.1902 3D
mIoU and `bl_cam2d` 0.2834 2D mIoU on 40 val frames with no calibration at all.
Those numbers, and the three ways they mislead if read carelessly, are in
[`CHALLENGE.md`](CHALLENGE.md#the-same-arms-on-real-goose-which-is-where-difficulty-claims-belong).

## The synthetic rig: the fixture

`semseg/datasets/fixture.py` generates a scene from primitives that each carry
a class, then renders both modalities from those same primitives: the cloud by
ray casting a spinning-lidar pattern, the image by rasterising through a
pinhole camera at an exact known extrinsic.

That construction is the point. Because both modalities come from one source
of truth, a correctly calibrated point lands on a pixel of its own class, and a
decalibrated one does not. The gap between those two is the signal every sweep
measures. The reference GLoc repo is explicit that its own fixture only
exercises plumbing; this one is built to measure, because the crossover number
the problem statement asks for cannot be obtained any other way on public data.

Defaults, all module constants you can change:

| | value |
| --- | --- |
| lidar | configurable beams and horizontal samples, vertical FOV set to span the scene |
| per-point timestamps | derived from azimuth, so deskew and time-offset sweeps have real structure to correct |
| camera | pinhole, no distortion |
| extrinsic | exact, known, and returned by `extrinsic()` |
| ego twist | supplied, so `poses_available` is True |

The fixture is deliberately not photorealistic. It does not need to be. What it
needs is exact agreement between its two modalities under the nominal
extrinsic, which it has by construction, and a class balance varied enough that
per-class IoU means something.

## The frame convention

One convention, stated once, because getting it wrong is silent:

The camera optical frame is x right, y down, z forward. `T_cam_lidar` is 4x4
and maps a point in the lidar frame into that camera frame, so projection is

```
p_cam = T_cam_lidar @ [x, y, z, 1]
uv    = (K @ p_cam[:3])[:2] / (K @ p_cam[:3])[2]
```

Which means, for the per-axis decalibration sweep:

| axis | rotates about | effect |
| --- | --- | --- |
| roll | z, the optical axis | image rotates about its centre, so error grows with distance from the principal point: 0 px at the centre and 2 px at the edge for 0.5 deg. Weakest axis, not a null one |
| pitch | x | vertical image shift, lateral world error grows with range |
| yaw | y | horizontal image shift, same range dependence |

This is why the sweep reports per axis rather than one magnitude, and the
measurements bear it out: the fused arm crosses below lidar-only at 0.86 degrees
of yaw, 1.19 of pitch, and not at all within 2 degrees of roll. Averaging the
three would report a tolerance true of no axis. At 10 m, half a degree of pitch
is 8.73 cm of lateral offset; the same rotation about the optical axis moves a
centred point not at all and an edge point about 2 px.
