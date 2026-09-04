"""Unit tests for the GOOSE val adapter.

These must never need the real dataset: CI has no copy of it, and a test that
skips when the data is absent is a test that never runs. So
`build_synthetic_goose_tree` lays down a miniature tree in exactly the layout
observed on disk (see the goose.py docstring), including both label mapping
CSVs with their real column sets and the upstream header typo.

Nothing here is stubbed. `semseg.types`, `semseg.labels` and the `Dataset` port
are imported for real, so a drift in the ontology parser or in the Frame
contract fails these tests instead of passing against a local mock.
"""

import csv
import io
import json
import os
import warnings

import numpy as np
import pytest
from PIL import Image

from semseg.datasets import Dataset, split_frames
from semseg.datasets.goose import (
    CLOUD_SUFFIX,
    DEFAULT_INTRINSICS_PATH,
    IMAGE_SUFFIX,
    LABEL_2D_SUFFIX,
    LABEL_3D_SUFFIX,
    NIR_SUFFIX,
    CalibrationUnavailable,
    GooseDataset,
)
from semseg.labels import CLASS_NAMES
from semseg.types import NUM_CLASSES, UNLABELED, validate

# Fine ids taken verbatim from the real challenge_label_mapping.csv, so the
# synthetic mapping cannot drift from the ontology it is standing in for.
FINE_UNDEFINED = 0
FINE_TRAFFIC_CONE = 1
FINE_COBBLE = 3
FINE_LEAVES = 5
FINE_CAR = 12
FINE_PERSON = 14
FINE_BUSH = 17
FINE_BUILDING = 38
FINE_SKY = 53

GOOSE9_OTHER = 0
GOOSE9_STRUCTURES = 1
GOOSE9_ARTIFICIAL_GROUND = 2
GOOSE9_NATURAL_GROUND = 3
GOOSE9_OBSTACLE = 4
GOOSE9_VEHICLE = 5
GOOSE9_VEGETATION = 6
GOOSE9_HUMAN = 7
GOOSE9_SKY = 8

# (class_name, label_key, has_instance, hex, challenge_category_id, name)
MAPPING_ROWS = (
    ("undefined", FINE_UNDEFINED, 0, "#000000", GOOSE9_OTHER, "other"),
    ("traffic_cone", FINE_TRAFFIC_CONE, 1, "#ffff00", GOOSE9_OBSTACLE, "obstacle"),
    ("cobble", FINE_COBBLE, 0, "#ff34ff", GOOSE9_ARTIFICIAL_GROUND, "artificial_ground"),
    ("leaves", FINE_LEAVES, 0, "#008941", GOOSE9_NATURAL_GROUND, "natural_ground"),
    ("car", FINE_CAR, 1, "#b79762", GOOSE9_VEHICLE, "vehicle"),
    ("person", FINE_PERSON, 1, "#8fb0ff", GOOSE9_HUMAN, "human"),
    ("bush", FINE_BUSH, 0, "#809693", GOOSE9_VEGETATION, "vegetation"),
    ("building", FINE_BUILDING, 0, "#013349", GOOSE9_STRUCTURES, "artificial_structures"),
    ("sky", FINE_SKY, 0, "#b77b68", GOOSE9_SKY, "sky"),
)

# The real header misspells this column. Reproduce the typo: a parser that only
# accepts the correct spelling breaks on the real file.
TYPO_CATEGORY_COLUMN = "challege_category_id"

SEQ_A = "2022-07-22_flight"
SEQ_B = "2023-03-03_garching_2"
FRAMES = (
    (SEQ_A, "0001", "1658494234334310308"),
    (SEQ_A, "0002", "1658494240334310308"),
    (SEQ_B, "0003", "1677845123456789012"),
)

IMAGE_H = 8
IMAGE_W = 12
N_POINTS = 20

VIS_FILL = 200
NIR_FILL = 37

# An arbitrary but exactly known rig, used only where a test needs projection
# to be possible at all. The real release supplies none.
TEST_K = [[600.0, 0.0, 512.0], [0.0, 600.0, 256.0], [0.0, 0.0, 1.0]]
TEST_T = [
    [0.0, -1.0, 0.0, 0.05],
    [0.0, 0.0, -1.0, 0.20],
    [1.0, 0.0, 0.0, -0.10],
    [0.0, 0.0, 0.0, 1.0],
]


def frame_id_of(sequence, index, stamp) -> str:
    return f"{sequence}__{index}_{stamp}"


def all_frame_ids() -> list:
    return sorted(frame_id_of(*f) for f in FRAMES)


def write_mapping_csvs(root, rows=MAPPING_ROWS) -> str:
    """challenge_label_mapping.csv at the root (six columns, with the typo) and
    goose_label_mapping.csv inside each zip directory (four columns, fine
    ontology only). Both, because that is what ships."""
    challenge_path = os.path.join(root, "challenge_label_mapping.csv")
    with open(challenge_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "class_name",
                "label_key",
                "has_instance",
                "hex",
                TYPO_CATEGORY_COLUMN,
                "challenge_category_name",
            ]
        )
        writer.writerows(rows)

    for zip_dir in ("raw_2d", "raw_3d"):
        fine_path = os.path.join(root, zip_dir, "goose_label_mapping.csv")
        with open(fine_path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["class_name", "label_key", "has_instance", "hex"])
            writer.writerows([r[:4] for r in rows])

    return challenge_path


def label_image_fine() -> np.ndarray:
    """A (IMAGE_H, IMAGE_W) uint8 fine-id map: sky on top, building band,
    vegetation band, cobble ground, one person pixel."""
    labels = np.full((IMAGE_H, IMAGE_W), FINE_COBBLE, dtype=np.uint8)
    labels[:2, :] = FINE_SKY
    labels[2:4, :] = FINE_BUILDING
    labels[4:6, :] = FINE_BUSH
    labels[6, 3] = FINE_PERSON
    return labels


def cloud_fine_ids() -> np.ndarray:
    ids = np.full(N_POINTS, FINE_LEAVES, dtype=np.uint32)
    ids[:4] = FINE_CAR
    ids[4:8] = FINE_BUSH
    ids[8] = FINE_TRAFFIC_CONE
    ids[9] = FINE_UNDEFINED
    return ids


def build_synthetic_goose_tree(tmp_path, frames=FRAMES) -> str:
    """Lay down the real GOOSE val layout with `frames` frames and return the
    root. Deliberately includes the decoy files the adapter must ignore
    (`_nir`, `_color`, `_instanceids`) and the LICENSE/CHANGELOG pair, and
    deliberately includes no calibration and no poses, because the release
    includes none.
    """
    root = str(tmp_path)
    sequences = sorted({f[0] for f in frames})

    for base in ("raw_2d/images", "raw_2d/labels", "raw_3d/lidar", "raw_3d/labels"):
        for sequence in sequences:
            os.makedirs(os.path.join(root, base, "val", sequence), exist_ok=True)

    for zip_dir in ("raw_2d", "raw_3d"):
        for extra in ("LICENSE", "CHANGELOG"):
            with open(os.path.join(root, zip_dir, extra), "w") as handle:
                handle.write("synthetic\n")

    write_mapping_csvs(root)

    fine_2d = label_image_fine()
    fine_3d = cloud_fine_ids()

    for sequence, index, stamp in frames:
        fid = frame_id_of(sequence, index, stamp)
        img_dir = os.path.join(root, "raw_2d/images/val", sequence)
        lab_dir = os.path.join(root, "raw_2d/labels/val", sequence)

        vis = np.full((IMAGE_H, IMAGE_W, 3), VIS_FILL, dtype=np.uint8)
        Image.fromarray(vis).save(os.path.join(img_dir, fid + IMAGE_SUFFIX))

        nir = np.full((IMAGE_H, IMAGE_W, 3), NIR_FILL, dtype=np.uint8)
        Image.fromarray(nir).save(os.path.join(img_dir, fid + NIR_SUFFIX))

        Image.fromarray(fine_2d).save(os.path.join(lab_dir, fid + LABEL_2D_SUFFIX))
        Image.fromarray(np.zeros((IMAGE_H, IMAGE_W, 3), np.uint8)).save(
            os.path.join(lab_dir, fid + "_color.png")
        )
        Image.fromarray(np.zeros((IMAGE_H, IMAGE_W), np.uint8)).save(
            os.path.join(lab_dir, fid + "_instanceids.png")
        )

        write_cloud(root, sequence, fid, fine_3d)

    return root


def write_cloud(root, sequence, fid, fine_3d, instance_ids=None) -> None:
    """One .bin of float32 x/y/z/intensity plus its uint32 .label, with the
    instance id packed into the high 16 bits exactly as GOOSE does."""
    fields = np.zeros((N_POINTS, 4), dtype=np.float32)
    fields[:, 0] = np.arange(N_POINTS, dtype=np.float32)
    fields[:, 1] = np.arange(N_POINTS, dtype=np.float32) * 0.5
    fields[:, 2] = -1.5
    fields[:, 3] = np.linspace(0.0, 254.0, N_POINTS, dtype=np.float32)
    fields.tofile(os.path.join(root, "raw_3d/lidar/val", sequence, fid + CLOUD_SUFFIX))

    packed = fine_3d.astype(np.uint32)
    if instance_ids is not None:
        packed = packed | (instance_ids.astype(np.uint32) << 16)
    packed.tofile(os.path.join(root, "raw_3d/labels/val", sequence, fid + LABEL_3D_SUFFIX))


# The published windshield intrinsics, in the numbers of
# docs/calib/mucar3_windshield_vis.yaml. Written into a tmp_path yaml per test
# rather than read from docs/calib, so these tests pin the PARSER: they keep
# passing if the committed calibration is ever revised, and they still fail if
# the parser stops honouring what it is handed.
CI_FX = 1775.62133
CI_FY = 1784.82927
CI_CX = 1025.99113
CI_CY = 775.44415
CI_DIST = (-0.14196, 0.09598, 0.00212, -0.00044, -0.00044)
CI_DECLARED_W = 2048

# The value written into projection_matrix, deliberately unlike anything in
# camera_matrix. Four sections of a real camera_info carry a `data:` key and on
# an unrectified camera projection_matrix repeats the same focal lengths, so a
# parser that matched the bare key would pass every length and shape check
# while reading the wrong section. A distinct decoy makes that visible.
CI_DECOY = 999.0


def write_camera_info(path, height, width=CI_DECLARED_W, camera_matrix=None,
                      distortion=CI_DIST) -> str:
    """A flat ROS camera_info dump in the exact layout of the committed file,
    decoy sections included.

    `roi:` keeps its own indented `height:` on purpose. Together with
    projection_matrix it means a parser that matches keys without tracking
    which section it is inside gets BOTH the image height and the intrinsics
    wrong. `camera_matrix` takes a raw list so a malformed one can be written.
    """
    if camera_matrix is None:
        camera_matrix = [CI_FX, 0, CI_CX, 0, CI_FY, CI_CY, 0, 0, 1]

    projection = [CI_DECOY, 0, CI_DECOY, 0, 0, CI_DECOY, CI_DECOY, 0, 0, 0, 1, 0]

    def data_line(values):
        return "  data: [" + ", ".join(str(v) for v in values) + "]"

    lines = [
        f"image_width: {width}",
        f"image_height: {height}",
        "camera_name: sensor/camera/windshield/vis",
        "camera_matrix:",
        "  rows: 3",
        "  cols: 3",
        data_line(camera_matrix),
        "distortion_model: plumb_bob",
        "distortion_coefficients:",
        "  rows: 1",
        "  cols: 5",
        data_line(distortion),
        "rectification_matrix:",
        "  rows: 3",
        "  cols: 3",
        data_line([1, 0, 0, 0, 1, 0, 0, 0, 1]),
        "projection_matrix:",
        "  rows: 3",
        "  cols: 4",
        data_line(projection),
        "binning_x: 0",
        "binning_y: 0",
        "roi:",
        "  x_offset: 0",
        "  y_offset: 0",
        "  height: 0",
        "  width: 0",
        "  do_rectify: false",
    ]

    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")

    return str(path)


def write_calib(path, K=TEST_K, T=TEST_T) -> str:
    payload = {}
    if K is not None:
        payload["K"] = K
    if T is not None:
        payload["T_cam_lidar"] = T

    with open(path, "w") as handle:
        json.dump(payload, handle)

    return str(path)


@pytest.fixture
def root(tmp_path):
    return build_synthetic_goose_tree(tmp_path)


def test_discovery_finds_every_modality(root):
    ds = GooseDataset(root)
    cov = ds.coverage()

    assert ds.frame_ids() == all_frame_ids()
    assert cov["n_images"] == len(FRAMES)
    assert cov["n_labels_2d"] == len(FRAMES)
    assert cov["n_clouds"] == len(FRAMES)
    assert cov["n_labels_3d"] == len(FRAMES)
    assert cov["n_intersection"] == len(FRAMES)
    assert cov["sequences"] == [SEQ_A, SEQ_B]
    assert cov["per_sequence"][SEQ_A]["intersection"] == 2
    assert cov["per_sequence"][SEQ_B]["intersection"] == 1


def test_nir_counted_but_not_used(root):
    """1924 files in images/val against 962 used is a number a reader will
    call a bug unless the ignored half is reported."""
    ds = GooseDataset(root)
    assert ds.coverage()["n_nir_ignored"] == len(FRAMES)

    frame = ds.load(all_frame_ids()[0])
    assert int(frame.image[0, 0, 0]) == VIS_FILL
    assert int(frame.image[0, 0, 0]) != NIR_FILL


def test_intersection_drops_when_cloud_missing(root):
    """PERTURBATION: remove exactly one 3D file and the intersection must fall
    by exactly one, and name the orphaned frame."""
    ds_before = GooseDataset(root)
    dropped = all_frame_ids()[0]
    sequence = dropped.split("__")[0]

    os.remove(os.path.join(root, "raw_3d/lidar/val", sequence, dropped + CLOUD_SUFFIX))

    ds_after = GooseDataset(root)
    before = ds_before.coverage()
    after = ds_after.coverage()

    assert after["n_intersection"] == before["n_intersection"] - 1
    assert dropped not in ds_after.frame_ids()
    assert after["only_2d"] == [dropped]
    assert after["only_3d"] == []


def test_intersection_drops_when_image_missing(root):
    ds_before = GooseDataset(root)
    dropped = all_frame_ids()[1]
    sequence = dropped.split("__")[0]

    os.remove(os.path.join(root, "raw_2d/images/val", sequence, dropped + IMAGE_SUFFIX))

    after = GooseDataset(root).coverage()
    assert after["n_intersection"] == ds_before.coverage()["n_intersection"] - 1
    assert after["only_3d"] == [dropped]


def test_remap_2d_to_goose9(root):
    ds = GooseDataset(root)
    frame = ds.load(all_frame_ids()[0])

    assert frame.labels_2d_gt.shape == (IMAGE_H, IMAGE_W)
    assert frame.labels_2d_gt.dtype == np.uint8
    assert frame.labels_2d_gt[0, 0] == GOOSE9_SKY
    assert frame.labels_2d_gt[2, 0] == GOOSE9_STRUCTURES
    assert frame.labels_2d_gt[4, 0] == GOOSE9_VEGETATION
    assert frame.labels_2d_gt[6, 3] == GOOSE9_HUMAN
    assert frame.labels_2d_gt[7, 0] == GOOSE9_ARTIFICIAL_GROUND

    present = set(np.unique(frame.labels_2d_gt).tolist())
    assert present <= set(range(NUM_CLASSES)) | {UNLABELED}


def test_remap_3d_to_goose9(root):
    ds = GooseDataset(root)
    frame = ds.load(all_frame_ids()[0])

    assert frame.labels_3d_gt.shape == (N_POINTS,)
    assert frame.labels_3d_gt.dtype == np.uint8
    assert frame.labels_3d_gt[0] == GOOSE9_VEHICLE
    assert frame.labels_3d_gt[4] == GOOSE9_VEGETATION
    assert frame.labels_3d_gt[8] == GOOSE9_OBSTACLE

    # fine id 0 is `undefined`, which the ontology assigns to goose9 `other`.
    # It must NOT become UNLABELED, or its mass leaves every confusion matrix.
    assert frame.labels_3d_gt[9] == GOOSE9_OTHER
    assert frame.labels_3d_gt[10] == GOOSE9_NATURAL_GROUND


def test_remap_moves_when_mapping_row_changes(root):
    """PERTURBATION: reassign one fine class in the CSV and only the pixels of
    that class may change."""
    ds = GooseDataset(root)
    before = ds.load(all_frame_ids()[0]).labels_2d_gt.copy()

    perturbed = []
    for row in MAPPING_ROWS:
        if row[1] == FINE_BUILDING:
            perturbed.append(row[:4] + (GOOSE9_OBSTACLE, "obstacle"))
            continue
        perturbed.append(row)

    write_mapping_csvs(root, rows=tuple(perturbed))
    after = GooseDataset(root).load(all_frame_ids()[0]).labels_2d_gt

    changed = before != after
    assert changed.any()
    assert np.array_equal(changed, label_image_fine() == FINE_BUILDING)
    assert after[2, 0] == GOOSE9_OBSTACLE


def test_unmapped_fine_id_becomes_unlabeled(root):
    """PERTURBATION: a fine id the CSV never mentions must land on UNLABELED
    rather than on class 0, which would inflate `other`."""
    unmapped = 61
    assert unmapped not in {row[1] for row in MAPPING_ROWS}

    fine = cloud_fine_ids()
    fine[11] = unmapped
    fid = all_frame_ids()[0]
    write_cloud(root, fid.split("__")[0], fid, fine)

    labels = GooseDataset(root).load(fid).labels_3d_gt
    assert labels[11] == UNLABELED
    assert labels[10] == GOOSE9_NATURAL_GROUND


def test_instance_bits_do_not_reach_semantics(root):
    """PERTURBATION on the bit mask: flipping a HIGH bit of the .label word
    must not move the semantic output, flipping a LOW bit must. This is the
    only test that proves `& 0xFFFF` is applied rather than the uint32 read
    wholesale."""
    fid = all_frame_ids()[0]
    sequence = fid.split("__")[0]
    fine = cloud_fine_ids()

    write_cloud(root, sequence, fid, fine)
    baseline = GooseDataset(root).load(fid).labels_3d_gt.copy()

    instances = np.zeros(N_POINTS, dtype=np.uint32)
    instances[3] = 7
    write_cloud(root, sequence, fid, fine, instance_ids=instances)
    with_instances = GooseDataset(root).load(fid).labels_3d_gt
    assert np.array_equal(baseline, with_instances)

    low_bit_changed = fine.copy()
    low_bit_changed[3] = FINE_PERSON
    write_cloud(root, sequence, fid, low_bit_changed, instance_ids=instances)
    moved = GooseDataset(root).load(fid).labels_3d_gt
    assert moved[3] == GOOSE9_HUMAN
    assert moved[3] != baseline[3]


def test_cloud_geometry_and_intensity(root):
    ds = GooseDataset(root)
    frame = ds.load(all_frame_ids()[0])

    assert frame.points.shape == (N_POINTS, 3)
    assert frame.points.dtype == np.float32
    assert frame.intensity.shape == (N_POINTS,)

    # intensity stays raw (0 to 254 on the real data), it is not normalised
    assert frame.intensity.max() > 1.0


def test_missing_calibration_raises(root):
    """The whole point of section 1b: no default EXTRINSIC, ever.

    The intrinsics half of this test changed under it, because spec section
    13.2 publishes the camera_info, and the extrinsic half did not. Asserting
    both at once is stronger than the old version: K present AND T refused, in
    one dataset, is exactly the distinction the adapter has to hold.
    """
    with pytest.warns(UserWarning):
        ds = GooseDataset(root)

    assert ds.calib_available is False

    with pytest.raises(CalibrationUnavailable) as excinfo:
        ds.extrinsic(all_frame_ids()[0])

    message = str(excinfo.value)
    assert os.path.join(root, "calib.json") in message
    assert os.path.join(root, "calibration.yaml") in message
    assert "fiction" in message

    # the intrinsics ARE published, and that must not leak into the extrinsic
    assert ds.intrinsics_available is True
    assert ds.load(all_frame_ids()[0]).K is not None


def test_calib_override_supplies_extrinsic(tmp_path, root):
    calib = write_calib(tmp_path / "calib.json")
    ds = GooseDataset(root, calib_path=calib)

    assert ds.calib_available is True
    T = ds.extrinsic(all_frame_ids()[0])
    assert T.shape == (4, 4)
    assert np.allclose(T, np.asarray(TEST_T, dtype=np.float64))
    assert np.allclose(ds.load(all_frame_ids()[0]).K, np.asarray(TEST_K, dtype=np.float64))


def test_extrinsic_moves_when_calib_perturbed(tmp_path, root):
    """PERTURBATION on the geometric transform: nudge one translation entry by
    1e-9 and the returned extrinsic must report it."""
    delta = 1e-9
    base_path = write_calib(tmp_path / "calib.json")
    base_T = GooseDataset(root, calib_path=base_path).extrinsic(all_frame_ids()[0])

    nudged = [list(row) for row in TEST_T]
    nudged[0][3] += delta
    nudged_path = write_calib(tmp_path / "calib_nudged.json", T=nudged)
    moved_T = GooseDataset(root, calib_path=nudged_path).extrinsic(all_frame_ids()[0])

    assert not np.array_equal(base_T, moved_T)
    assert moved_T[0, 3] - base_T[0, 3] == pytest.approx(delta, rel=1e-6)


def test_extrinsic_is_a_copy(tmp_path, root):
    """A sweep mutating the matrix it was handed must not corrupt the source."""
    calib = write_calib(tmp_path / "calib.json")
    ds = GooseDataset(root, calib_path=calib)

    handed = ds.extrinsic(all_frame_ids()[0])
    handed[0, 3] = 999.0
    assert ds.extrinsic(all_frame_ids()[0])[0, 3] != 999.0


def test_half_calibration_rejected(tmp_path, root):
    only_K = write_calib(tmp_path / "half.json", T=None)
    with pytest.raises(ValueError, match="T_cam_lidar"):
        GooseDataset(root, calib_path=only_K)

    only_T = write_calib(tmp_path / "half2.json", K=None)
    with pytest.raises(ValueError, match="'K'"):
        GooseDataset(root, calib_path=only_T)


def test_discovered_calib_is_used(root):
    """A tree that DOES carry a calibration must use it, so the refusal is a
    statement about this release and not a hardcoded no."""
    write_calib(os.path.join(root, "calib.json"))
    ds = GooseDataset(root)

    assert ds.calib_available is True
    assert np.allclose(ds.extrinsic(all_frame_ids()[0]), np.asarray(TEST_T, dtype=np.float64))


def test_poses_and_point_times_absent(root):
    ds = GooseDataset(root)
    frame = ds.load(all_frame_ids()[0])

    assert ds.poses_available is False
    assert frame.ego_twist is None
    assert frame.point_times is None
    assert ds.coverage()["point_times_available"] is False


def test_missing_mapping_raises_on_remap(root):
    """coverage() and discover_report() still work without the challenge CSV,
    because a user diagnosing a tree should not need it. load() must not."""
    os.remove(os.path.join(root, "challenge_label_mapping.csv"))
    ds = GooseDataset(root)

    assert ds.coverage()["n_intersection"] == len(FRAMES)

    with pytest.raises(FileNotFoundError, match="challenge_label_mapping.csv"):
        ds.load(all_frame_ids()[0])


def test_mapping_override(tmp_path, root):
    moved = str(tmp_path / "elsewhere.csv")
    os.rename(os.path.join(root, "challenge_label_mapping.csv"), moved)

    ds = GooseDataset(root, mapping_path=moved)
    assert ds.load(all_frame_ids()[0]).labels_2d_gt[0, 0] == GOOSE9_SKY


def test_unknown_frame_raises_with_diagnosis(root):
    ds = GooseDataset(root)
    with pytest.raises(KeyError, match="discover_report"):
        ds.load("not-a-frame")


def test_bad_cloud_size_raises(root):
    """A cloud whose float count does not divide by 4 is not an x/y/z/i cloud
    and must not be reshaped into a plausible one."""
    fid = all_frame_ids()[0]
    path = os.path.join(root, "raw_3d/lidar/val", fid.split("__")[0], fid + CLOUD_SUFFIX)
    np.arange(9, dtype=np.float32).tofile(path)

    with pytest.raises(ValueError, match="do not divide by 4"):
        GooseDataset(root).load(fid)


def test_label_count_mismatch_raises(root):
    fid = all_frame_ids()[0]
    path = os.path.join(root, "raw_3d/labels/val", fid.split("__")[0], fid + LABEL_3D_SUFFIX)
    np.zeros(N_POINTS - 1, dtype=np.uint32).tofile(path)

    with pytest.raises(ValueError, match="labels for"):
        GooseDataset(root).load(fid)


def test_missing_split_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Searched"):
        GooseDataset(build_synthetic_goose_tree(tmp_path), split="train")


def test_velodyne_dir_name_accepted(root):
    """GOOSE's own docs call the cloud directory `velodyne`; the val zip calls
    it `lidar`. Both must discover."""
    os.rename(os.path.join(root, "raw_3d/lidar"), os.path.join(root, "raw_3d/velodyne"))
    assert GooseDataset(root).frame_ids() == all_frame_ids()


def test_discover_report_names_the_real_constraint(root):
    stream = io.StringIO()
    GooseDataset(root).discover_report(stream=stream)
    text = stream.getvalue()

    assert f"INTERSECTION={len(FRAMES)}" in text
    assert "NOT what caps a fusion claim" in text
    assert "calibration: NOT FOUND" in text
    assert "pose file: NOT FOUND" in text
    assert "poses_available=False" in text
    assert "point_times_available=False" in text
    assert NIR_SUFFIX in text


def test_report_flags_thin_overlap(root):
    """Wording must flip when the overlap really is the binding constraint."""
    for _, index, stamp in FRAMES[1:]:
        for sequence in {f[0] for f in FRAMES}:
            fid = frame_id_of(sequence, index, stamp)
            path = os.path.join(root, "raw_3d/lidar/val", sequence, fid + CLOUD_SUFFIX)
            if os.path.exists(path):
                os.remove(path)

    stream = io.StringIO()
    GooseDataset(root).discover_report(stream=stream)
    text = stream.getvalue()

    assert "caps directly how much" in text
    assert "NOT what caps a fusion claim" not in text


def test_calib_error_lists_files_it_did_see(root):
    """When the named candidates miss, the error must report what the sweep
    DID find, so a user can spot a calibration under an unexpected name."""
    decoy = os.path.join(root, "raw_3d", "sensor_setup.yaml")
    with open(decoy, "w") as handle:
        handle.write("not a calibration\n")

    with pytest.raises(CalibrationUnavailable) as excinfo:
        GooseDataset(root).extrinsic(all_frame_ids()[0])

    message = str(excinfo.value)
    assert decoy in message
    assert "unrelated file" in message


def test_finding_a_pose_file_does_not_promise_ego_twist(root):
    """poses_available means "Frame.ego_twist is populated", so merely finding
    a pose file must not flip it: eval/sweep.py would start the time-offset
    sweep and then measure nothing, which is worse than refusing."""
    ds = GooseDataset(root)
    assert ds.pose_file_found is False
    assert ds.poses_available is False

    with open(os.path.join(root, "poses.txt"), "w") as handle:
        handle.write("0 0 0\n")

    ds = GooseDataset(root)
    assert ds.pose_file_found is True
    assert ds.coverage()["pose_file_found"] is True

    # the flag the sweep reads stays False, because load() still has no twist
    assert ds.poses_available is False
    assert ds.coverage()["poses_available"] is False
    assert ds.load(all_frame_ids()[0]).ego_twist is None


def test_implements_the_dataset_port(root):
    """The ABC is the only thing eval/sweep.py sees, so the adapter must be a
    real subclass and not merely a lookalike."""
    ds = GooseDataset(root)
    assert isinstance(ds, Dataset)

    fit_ids, score_ids = split_frames(ds.frame_ids())
    assert sorted(fit_ids + score_ids) == all_frame_ids()
    assert not set(fit_ids) & set(score_ids)


def test_validate_accepts_a_goose_frame(root):
    """Was a DEFECT MARKER: K was None, semseg/types.py lists K in
    REQUIRED_ARRAYS, so validate() rejected every real GOOSE frame. Populating
    K from the published camera_info closes that gap through this adapter, and
    K was the only field the validator was missing.

    The types.py asymmetry itself is untouched and still live: an adapter with
    genuinely no intrinsics still cannot produce a validatable Frame. That fix
    belongs in semseg/types.py, not here, and fabricating a K to satisfy the
    validator remains the one thing spec section 1b forbids.
    """
    with pytest.warns(UserWarning):
        ds = GooseDataset(root)

    validate(ds.load(all_frame_ids()[0]))


def test_validate_passes_once_a_calib_is_supplied(root):
    """The rest of the Frame really is contract-clean; K is the only gap."""
    write_calib(os.path.join(root, "calib.json"))
    validate(GooseDataset(root).load(all_frame_ids()[0]))


def test_goose9_constants_match_the_ontology():
    """Pins this file's GOOSE9_* constants to semseg.labels. Without it the
    synthetic mapping could drift from the real ontology and every remap
    assertion above would keep passing while testing the wrong thing."""
    assert len(CLASS_NAMES) == NUM_CLASSES
    assert CLASS_NAMES[GOOSE9_OTHER] == "other"
    assert CLASS_NAMES[GOOSE9_STRUCTURES] == "artificial_structures"
    assert CLASS_NAMES[GOOSE9_ARTIFICIAL_GROUND] == "artificial_ground"
    assert CLASS_NAMES[GOOSE9_NATURAL_GROUND] == "natural_ground"
    assert CLASS_NAMES[GOOSE9_OBSTACLE] == "obstacle"
    assert CLASS_NAMES[GOOSE9_VEHICLE] == "vehicle"
    assert CLASS_NAMES[GOOSE9_VEGETATION] == "vegetation"
    assert CLASS_NAMES[GOOSE9_HUMAN] == "human"
    assert CLASS_NAMES[GOOSE9_SKY] == "sky"

    # every row of the synthetic mapping names the category its id selects
    for row in MAPPING_ROWS:
        assert CLASS_NAMES[row[4]] == row[5]


# --- published intrinsics and the unpublished crop (spec sections 13.2, 13.3) --


def test_published_intrinsics_populate_K(tmp_path, root):
    """Section 13.2: the camera_info IS published, so K must not be None. This
    is the gap the adapter was written before and did not know about."""
    camera_info = write_camera_info(tmp_path / "ci.yaml", height=IMAGE_H)
    ds = GooseDataset(root, intrinsics_path=camera_info)

    K = ds.load(all_frame_ids()[0]).K
    assert K is not None
    assert K.shape == (3, 3)
    assert K.dtype == np.float64
    assert K[0, 0] == pytest.approx(CI_FX)
    assert K[1, 1] == pytest.approx(CI_FY)
    assert K[0, 2] == pytest.approx(CI_CX)
    assert K[1, 2] == pytest.approx(CI_CY)
    assert K[2, 2] == pytest.approx(1.0)

    assert ds.intrinsics_available is True
    assert np.allclose(ds.intrinsics(all_frame_ids()[0]), K)


def test_default_intrinsics_resolve_from_the_repo_not_the_cwd(root, tmp_path, monkeypatch):
    """The default is a file committed to this repo, so it has to resolve off
    the module's own location. Resolved off the cwd it would vanish for any
    sweep launched from elsewhere, and K would silently go back to None."""
    monkeypatch.chdir(tmp_path)

    with pytest.warns(UserWarning):
        ds = GooseDataset(root)

    assert os.path.isfile(DEFAULT_INTRINSICS_PATH)
    assert ds.coverage()["intrinsics_path"] == DEFAULT_INTRINSICS_PATH
    assert ds.load(all_frame_ids()[0]).K is not None


def test_camera_matrix_is_read_not_projection_matrix(tmp_path, root):
    """The single most likely parse bug. camera_matrix, rectification_matrix,
    projection_matrix and distortion_coefficients ALL carry a `data:` key, and
    on an unrectified camera projection_matrix repeats the focal lengths, so
    reading the wrong section passes every length check."""
    camera_info = write_camera_info(tmp_path / "ci.yaml", height=IMAGE_H)
    K = GooseDataset(root, intrinsics_path=camera_info).intrinsics(all_frame_ids()[0])

    assert K[0, 0] == pytest.approx(CI_FX)
    assert CI_DECOY not in set(K.reshape(-1).tolist())


def test_nested_roi_height_is_not_the_image_height(tmp_path, root):
    """`roi:` carries its own indented `height: 0`. A parser that matched keys
    without tracking sections would read the declared height as 0, find it
    unequal to everything, and warn about the wrong number forever."""
    camera_info = write_camera_info(tmp_path / "ci.yaml", height=1536)

    with pytest.warns(UserWarning, match="1536"):
        ds = GooseDataset(root, intrinsics_path=camera_info)

    assert ds.coverage()["intrinsics_declared_size"] == [CI_DECLARED_W, 1536]


def test_height_mismatch_warns_and_flags_the_crop_unknown(tmp_path, root):
    """THE CATCH of section 13.3. The yaml declares 1536 rows, the release
    ships fewer, and the vertical crop offset is published nowhere. The
    adapter must say so and must not guess."""
    camera_info = write_camera_info(tmp_path / "ci.yaml", height=1536)

    with pytest.warns(UserWarning) as record:
        ds = GooseDataset(root, intrinsics_path=camera_info)

    assert len(record) == 1
    message = str(record[0].message)
    assert "1536" in message
    assert str(IMAGE_H) in message
    assert str(camera_info) in message

    assert ds.crop_offset_known is False
    assert ds.coverage()["crop_offset_known"] is False

    # and cy is LEFT ALONE rather than nudged towards something plausible
    assert ds.intrinsics(all_frame_ids()[0])[1, 2] == pytest.approx(CI_CY)


def test_matching_height_does_not_warn(tmp_path, root):
    """The warning has to be a statement about this release, not an
    unconditional one, or it stops carrying information."""
    camera_info = write_camera_info(tmp_path / "ci.yaml", height=IMAGE_H)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ds = GooseDataset(root, intrinsics_path=camera_info)

    assert ds.crop_offset_known is True
    assert ds.coverage()["crop_top_used"] == 0
    assert ds.coverage()["crop_top_source"] == "none"


def test_warning_is_once_per_dataset_not_once_per_frame(root, tmp_path):
    """"Warn once" is the requirement, and 962 identical warnings is
    operationally the same as none. Every load() here turns any warning into
    an error, so a per-frame warning fails this test."""
    camera_info = write_camera_info(tmp_path / "ci.yaml", height=1536)

    with pytest.warns(UserWarning):
        ds = GooseDataset(root, intrinsics_path=camera_info)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for frame_id in ds.frame_ids():
            assert ds.load(frame_id).K is not None


def test_crop_top_shifts_cy_by_exactly_that_many_pixels(tmp_path, root):
    """PERTURBATION on the one entry a vertical crop invalidates. cy must move
    by exactly crop_top, and fx, fy and cx must not move at all."""
    crop_top = 536
    camera_info = write_camera_info(tmp_path / "ci.yaml", height=1536)

    with pytest.warns(UserWarning):
        base = GooseDataset(root, intrinsics_path=camera_info)
    with pytest.warns(UserWarning):
        shifted = GooseDataset(root, intrinsics_path=camera_info, crop_top=crop_top)

    K_base = base.intrinsics(all_frame_ids()[0])
    K_shifted = shifted.intrinsics(all_frame_ids()[0])

    assert K_base[1, 2] - K_shifted[1, 2] == pytest.approx(crop_top)
    assert K_shifted[1, 2] == pytest.approx(CI_CY - crop_top)

    moved = K_base != K_shifted
    assert moved.sum() == 1
    assert moved[1, 2]

    cov = shifted.coverage()
    assert cov["crop_top_used"] == crop_top
    assert cov["crop_top_source"] == "user"
    assert cov["cy_published"] == pytest.approx(CI_CY)
    assert cov["cy_used"] == pytest.approx(CI_CY - crop_top)


def test_crop_top_does_not_make_the_offset_known(tmp_path, root):
    """A supplied crop_top is ASSERTED, not published. Same rule as
    poses_available: a flag downstream code gates on must mean "this is known",
    never "somebody said so", or a submission cannot tell the two apart. The
    asserted number still travels, in coverage()."""
    camera_info = write_camera_info(tmp_path / "ci.yaml", height=1536)

    with pytest.warns(UserWarning):
        ds = GooseDataset(root, intrinsics_path=camera_info, crop_top=536)

    assert ds.crop_offset_known is False
    assert ds.coverage()["crop_offset_known"] is False
    assert ds.coverage()["crop_top_used"] == 536
    assert ds.coverage()["crop_top_source"] == "user"


def test_impossible_crop_top_raises(tmp_path, root):
    """Validated at the boundary and not clamped: a clamped value would shift
    cy by a different amount than coverage() reports."""
    camera_info = write_camera_info(tmp_path / "ci.yaml", height=IMAGE_H)

    with pytest.raises(ValueError, match="crop_top must be >= 0"):
        GooseDataset(root, intrinsics_path=camera_info, crop_top=-1)

    with pytest.raises(ValueError, match="not inside the declared image height"):
        GooseDataset(root, intrinsics_path=camera_info, crop_top=IMAGE_H)


def test_malformed_camera_matrix_raises_naming_the_file(tmp_path, root):
    """Nine wrong numbers reshape into a perfectly plausible K, so the count is
    checked rather than reshaped into whatever fits, and the error names the
    file so a user knows which of several calibrations is broken."""
    short = tmp_path / "short.yaml"
    write_camera_info(short, height=IMAGE_H, camera_matrix=[1, 0, 2, 0, 3, 4, 0, 0])

    with pytest.raises(ValueError) as excinfo:
        GooseDataset(root, intrinsics_path=str(short))

    message = str(excinfo.value)
    assert str(short) in message
    assert "8 numbers" in message
    assert "9" in message

    text = tmp_path / "text.yaml"
    write_camera_info(text, height=IMAGE_H, camera_matrix=["fx"] * 9)

    with pytest.raises(ValueError, match="not a list of numbers"):
        GooseDataset(root, intrinsics_path=str(text))


def test_camera_info_without_a_camera_matrix_raises(tmp_path, root):
    not_camera_info = tmp_path / "other.yaml"
    with open(not_camera_info, "w") as handle:
        handle.write("image_width: 2048\nimage_height: 8\nsomething_else: 3\n")

    with pytest.raises(ValueError, match="camera_matrix"):
        GooseDataset(root, intrinsics_path=str(not_camera_info))


def test_missing_intrinsics_override_raises(root, tmp_path):
    with pytest.raises(FileNotFoundError):
        GooseDataset(root, intrinsics_path=str(tmp_path / "nope.yaml"))


def test_distortion_is_reported_but_not_applied(tmp_path, root):
    """plumb_bob distortion is real and is parsed, but Frame carries no
    distortion field. Undistorting the pixels here would change what the image
    arm is scored on without saying so, hence the explicit False."""
    camera_info = write_camera_info(tmp_path / "ci.yaml", height=IMAGE_H)
    cov = GooseDataset(root, intrinsics_path=camera_info).coverage()

    assert len(cov["distortion_coefficients"]) == 5
    assert cov["distortion_coefficients"] == pytest.approx(list(CI_DIST))
    assert cov["distortion_applied"] is False


def test_published_intrinsics_do_not_supply_the_extrinsic(tmp_path, root):
    """THE TRAP, stated as a test. K being published must not make T_cam_lidar
    available: the two halves are distributed separately and only one of them
    ships. extrinsic() has to keep refusing with the message that names every
    path it searched."""
    camera_info = write_camera_info(tmp_path / "ci.yaml", height=IMAGE_H)
    ds = GooseDataset(root, intrinsics_path=camera_info)

    assert ds.intrinsics_available is True
    assert ds.load(all_frame_ids()[0]).K is not None

    # and yet, with K in hand
    assert ds.calib_available is False
    with pytest.raises(CalibrationUnavailable) as excinfo:
        ds.extrinsic(all_frame_ids()[0])

    message = str(excinfo.value)
    assert os.path.join(root, "calib.json") in message
    assert os.path.join(root, "calibration.yaml") in message
    assert "fiction" in message


def test_full_rig_K_wins_over_published_intrinsics(tmp_path, root):
    """A K and a T_cam_lidar measured together must not be split up. Pairing
    the published K with a separately supplied extrinsic would assemble a
    chimera out of two real calibrations, so --calib overrides the yaml
    outright. Combining it with a crop_top is refused outright; see
    test_crop_top_with_a_full_rig_is_rejected.

    The rig also has to SILENCE the crop warning, even though the yaml still
    declares a height the images do not have: the declared height describes a K
    that is no longer in use, so warning about it would state something false.
    """
    camera_info = write_camera_info(tmp_path / "ci.yaml", height=1536)
    rig = write_calib(tmp_path / "calib.json")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ds = GooseDataset(root, calib_path=rig, intrinsics_path=camera_info)

    K = ds.load(all_frame_ids()[0]).K
    assert np.allclose(K, np.asarray(TEST_K, dtype=np.float64))
    assert K[1, 2] == pytest.approx(TEST_K[1][2])
    assert ds.coverage()["K_source"] == rig


def test_crop_metadata_survives_json(tmp_path, root):
    """coverage() has to reach submission metadata unchanged, and a numpy
    float64 does not survive json.dumps. Without this the crop provenance
    would be lost exactly where it matters most."""
    camera_info = write_camera_info(tmp_path / "ci.yaml", height=1536)

    with pytest.warns(UserWarning):
        cov = GooseDataset(root, intrinsics_path=camera_info, crop_top=536).coverage()

    reloaded = json.loads(json.dumps(cov))
    assert reloaded["crop_offset_known"] is False
    assert reloaded["crop_top_used"] == 536
    assert reloaded["crop_top_source"] == "user"
    assert reloaded["cy_used"] == pytest.approx(CI_CY - 536)
    assert reloaded["image_size_actual"] == [IMAGE_W, IMAGE_H]


def test_discover_report_states_the_intrinsic_extrinsic_split(root):
    """A reader who sees K working and extrinsic() raising must be told why,
    or they will file it as a bug in one of the two."""
    stream = io.StringIO()
    with pytest.warns(UserWarning):
        GooseDataset(root).discover_report(stream=stream)

    text = stream.getvalue()
    assert "crop_offset_known=False" in text
    assert "cy is NOT trustworthy" in text
    assert "calib_available means BOTH" in text
    assert "calibration: NOT FOUND" in text


def test_crop_top_with_a_full_rig_is_rejected(tmp_path, root):
    """Contradictory request, refused at the boundary rather than half honoured.

    A rig REPLACES the published intrinsics, so a crop offset measured against
    those intrinsics has nothing to apply to. Honouring it would record
    crop_top_used=N in the metadata next to a cy that moved by a completely
    different amount, which is the submission describing a projection that
    never happened.
    """
    camera_info = write_camera_info(tmp_path / "ci.yaml", height=1536)
    rig = write_calib(tmp_path / "calib.json")

    with pytest.raises(ValueError, match="crop_top"):
        GooseDataset(root, calib_path=rig, intrinsics_path=camera_info, crop_top=100)


def test_reported_cy_delta_always_equals_crop_top_used(tmp_path, root):
    """The invariant the three coverage fields have to satisfy together:
    cy_published minus cy_used IS crop_top_used. Asserted rather than assumed,
    because a reader of the submission metadata will do this subtraction."""
    camera_info = write_camera_info(tmp_path / "ci.yaml", height=1536)

    for crop_top in (None, 0, 1, 536):
        kwargs = {} if crop_top is None else {"crop_top": crop_top}
        with pytest.warns(UserWarning):
            cov = GooseDataset(root, intrinsics_path=camera_info, **kwargs).coverage()

        assert cov["cy_published"] - cov["cy_used"] == pytest.approx(cov["crop_top_used"])
