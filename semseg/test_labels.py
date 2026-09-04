"""Tests for the goose9 ontology and the Frame boundary validator.

Every case that checks a transform also perturbs one input by the smallest
meaningful amount and asserts the output moves. A comparator that cannot fail
proves nothing, and a label mapping that quietly returns all-UNLABELED would
pass any test that only asserts "no exception".
"""

import os
from dataclasses import fields

import numpy as np
import pytest
from PIL import Image

from semseg.labels import (
    CATEGORY_ID_COLUMNS, CLASS_COLORS, CLASS_NAMES, DEVKIT_COLORS_BGR,
    FOLDED_CLASS_NAMES, FOLDED_NUM_CLASSES, IGNORE, LUT_SIZE, OTHER_CLASS,
    SKY_CLASS, fold_sky, load_label_mapping, remap, save_label_png,
)
from semseg.types import NUM_CLASSES, UNLABELED, Frame, Prediction, validate

# The real mapping, used only by tests that are skipped when it is absent: a
# clean checkout must be able to run this suite with zero downloads.
REAL_MAPPING = "/home/pan-navigator/datasets/goose/challenge_label_mapping.csv"

# A CSV shaped exactly like the real one, small enough to reason about. Fine id
# 200 is deliberately absent so the unmapped path is exercised.
SMALL_CSV = """class_name,label_key,has_instance,hex,challege_category_id,challenge_category_name
undefined,0,0,#000000,0,other
traffic_cone,1,1,#ffff00,4,obstacle
building,38,0,#013349,1,artificial_structures
car,12,1,#b79762,5,vehicle
sky,53,0,#b77b68,8,sky
"""

EXPECTED_SMALL = {0: 0, 1: 4, 38: 1, 12: 5, 53: 8}


def _write_csv(tmp_path, text, name="mapping.csv"):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def _frame(n_points=7, height=4, width=6):
    """A minimal Frame that validate() accepts, for the mismatch cases to break
    one field at a time."""
    return Frame(
        frame_id="fixture__0000",
        image=np.zeros((height, width, 3), dtype=np.uint8),
        points=np.zeros((n_points, 3), dtype=np.float32),
        intensity=np.zeros(n_points, dtype=np.float32),
        K=np.eye(3, dtype=np.float64),
        point_times=np.zeros(n_points, dtype=np.float32),
        ego_twist=np.zeros(6, dtype=np.float32),
        labels_2d_gt=np.zeros((height, width), dtype=np.uint8),
        labels_3d_gt=np.zeros(n_points, dtype=np.uint8),
    )


# ----- the class table itself -----

def test_label_space_is_nine_classes():
    assert NUM_CLASSES == 9
    assert len(CLASS_NAMES) == NUM_CLASSES
    assert CLASS_NAMES[OTHER_CLASS] == "other"
    assert CLASS_NAMES[SKY_CLASS] == "sky"
    assert FOLDED_CLASS_NAMES == CLASS_NAMES[:FOLDED_NUM_CLASSES]
    assert "sky" not in FOLDED_CLASS_NAMES


def test_ignore_is_the_same_byte_as_unlabeled():
    """Two names for one sentinel; if they ever drift, every confusion matrix
    silently starts scoring unlabelled pixels."""
    assert IGNORE == UNLABELED == 255


def test_class_colors_are_rgb_not_bgr():
    """The devkit table is BGR. Reversed it lands on the Material Design
    palette the devkit screenshots show, which is the evidence that the
    reversal is correct and must not be undone."""
    assert CLASS_COLORS.shape == (NUM_CLASSES, 3)
    assert CLASS_COLORS.dtype == np.uint8

    assert tuple(CLASS_COLORS[4]) == (0xFF, 0xC1, 0x07)   # amber 500
    assert tuple(CLASS_COLORS[5]) == (0xF4, 0x43, 0x36)   # red 500
    assert tuple(CLASS_COLORS[6]) == (0x4C, 0xAF, 0x50)   # green 500
    assert tuple(CLASS_COLORS[SKY_CLASS]) == (0xB7, 0x7B, 0x68)

    # human is light blue in the screenshots, not skin-tone orange
    assert CLASS_COLORS[7][2] > CLASS_COLORS[7][0]
    assert tuple(DEVKIT_COLORS_BGR[7]) == (255, 176, 143)


def test_no_two_classes_share_a_colour():
    assert len({tuple(row) for row in CLASS_COLORS}) == NUM_CLASSES


# ----- load_label_mapping -----

def test_load_mapping_small_csv(tmp_path):
    lut = load_label_mapping(_write_csv(tmp_path, SMALL_CSV))

    assert lut.shape == (LUT_SIZE,)
    assert lut.dtype == np.uint8

    for fine_id, expected in EXPECTED_SMALL.items():
        assert lut[fine_id] == expected

    # an id the CSV never mentions is missing ground truth, not `other`
    assert lut[200] == UNLABELED
    assert lut[UNLABELED] == UNLABELED


def test_load_mapping_perturbation(tmp_path):
    """Move one row's superclass id by one and the table must follow. Without
    this the parser could be returning a constant."""
    baseline = load_label_mapping(_write_csv(tmp_path, SMALL_CSV, "a.csv"))

    perturbed_csv = SMALL_CSV.replace("car,12,1,#b79762,5,vehicle", "car,12,1,#b79762,4,obstacle")
    perturbed = load_label_mapping(_write_csv(tmp_path, perturbed_csv, "b.csv"))

    assert baseline[12] == 5
    assert perturbed[12] == 4

    changed = np.flatnonzero(baseline != perturbed)
    assert changed.tolist() == [12]


def test_load_mapping_header_variants(tmp_path):
    """Same content, three plausible headers: the upstream typo, the corrected
    spelling, and a revision carrying only the superclass name."""
    typo = load_label_mapping(_write_csv(tmp_path, SMALL_CSV, "typo.csv"))

    fixed_csv = SMALL_CSV.replace("challege_category_id", "challenge_category_id")
    fixed = load_label_mapping(_write_csv(tmp_path, fixed_csv, "fixed.csv"))

    name_only_csv = "\n".join(
        ",".join(part for i, part in enumerate(line.split(",")) if i != 4)
        for line in SMALL_CSV.strip().splitlines()
    ) + "\n"
    name_only = load_label_mapping(_write_csv(tmp_path, name_only_csv, "names.csv"))

    assert np.array_equal(typo, fixed)
    assert np.array_equal(typo, name_only)
    assert "challege_category_id" in CATEGORY_ID_COLUMNS


def test_load_mapping_reordered_columns(tmp_path):
    """Columns are found by header, not by position, so shuffling them changes
    nothing."""
    reordered = "\n".join(
        ",".join(line.split(",")[::-1]) for line in SMALL_CSV.strip().splitlines()
    ) + "\n"

    assert np.array_equal(
        load_label_mapping(_write_csv(tmp_path, SMALL_CSV, "plain.csv")),
        load_label_mapping(_write_csv(tmp_path, reordered, "rev.csv")),
    )


def test_load_mapping_missing_file_names_path(tmp_path):
    missing = str(tmp_path / "not_here.csv")

    with pytest.raises(FileNotFoundError) as excinfo:
        load_label_mapping(missing)

    assert missing in str(excinfo.value)


def test_load_mapping_without_superclass_column(tmp_path):
    """This is the shape of the goose_label_mapping.csv that ships inside the
    dataset zips. It must fail loudly rather than build an all-UNLABELED table
    that would score as a silent zero."""
    fine_only = "\n".join(
        ",".join(line.split(",")[:4]) for line in SMALL_CSV.strip().splitlines()
    ) + "\n"

    with pytest.raises(ValueError, match="no superclass column"):
        load_label_mapping(_write_csv(tmp_path, fine_only))


def test_load_mapping_rejects_non_goose9_id(tmp_path):
    bad = SMALL_CSV.replace("sky,53,0,#b77b68,8,sky", "sky,53,0,#b77b68,11,sky")

    with pytest.raises(ValueError, match="outside 0..8"):
        load_label_mapping(_write_csv(tmp_path, bad))


@pytest.mark.skipif(not os.path.isfile(REAL_MAPPING), reason="GOOSE mapping CSV not on this machine")
def test_real_mapping_covers_64_fine_classes():
    lut = load_label_mapping(REAL_MAPPING)

    mapped = np.flatnonzero(lut != UNLABELED)
    assert mapped.tolist() == list(range(64))

    assert lut[0] == 0     # undefined -> other
    assert lut[12] == 5    # car -> vehicle
    assert lut[14] == 7    # person -> human
    assert lut[53] == 8    # sky -> sky
    assert lut[50] == 3    # low_grass -> natural_ground

    # every goose9 class must actually be reachable, or the ontology is wrong
    assert set(lut[mapped].tolist()) == set(range(NUM_CLASSES))


# ----- remap -----

def test_remap_round_trip(tmp_path):
    lut = load_label_mapping(_write_csv(tmp_path, SMALL_CSV))
    raw = np.array([[0, 1, 38], [12, 53, 200]], dtype=np.uint8)

    out = remap(raw, lut)

    assert out.dtype == np.uint8
    assert out.shape == raw.shape
    assert out.tolist() == [[0, 4, 1], [5, 8, UNLABELED]]

    # the mapping is many-to-one, so the round trip is fine id -> goose9 -> name
    assert [CLASS_NAMES[c] for c in out[0]] == ["other", "obstacle", "artificial_structures"]


def test_remap_perturbation(tmp_path):
    """Move one LUT entry and only the pixels holding that fine id may change."""
    lut = load_label_mapping(_write_csv(tmp_path, SMALL_CSV))
    raw = np.array([0, 1, 38, 12, 1, 53], dtype=np.uint8)

    before = remap(raw, lut)

    bumped = lut.copy()
    bumped[1] = 6
    after = remap(raw, bumped)

    assert before.tolist() == [0, 4, 1, 5, 4, 8]
    assert after.tolist() == [0, 6, 1, 5, 6, 8]
    assert np.flatnonzero(before != after).tolist() == [1, 4]


def test_remap_handles_wide_dtype(tmp_path):
    """A .label file's semantic field is uint32; ids past the table become the
    sentinel instead of raising an IndexError mid-run."""
    lut = load_label_mapping(_write_csv(tmp_path, SMALL_CSV))
    raw = np.array([1, 53, 300, 70000], dtype=np.uint32)

    out = remap(raw, lut)

    assert out.dtype == np.uint8
    assert out.tolist() == [4, 8, UNLABELED, UNLABELED]


def test_remap_rejects_wrong_lut_length():
    with pytest.raises(ValueError, match="lut must be"):
        remap(np.zeros(3, dtype=np.uint8), np.zeros(NUM_CLASSES, dtype=np.uint8))


# ----- fold_sky -----

def test_fold_sky_labels_moves_sky_into_other():
    labels = np.array([[SKY_CLASS, 6, 7], [UNLABELED, SKY_CLASS, OTHER_CLASS]], dtype=np.uint8)

    folded = fold_sky(labels)

    assert folded.tolist() == [[0, 6, 7], [UNLABELED, 0, 0]]
    assert not np.array_equal(folded, labels)          # the input really held sky
    assert SKY_CLASS not in folded[folded != UNLABELED]
    assert labels[0, 0] == SKY_CLASS                   # input not mutated


def test_fold_sky_labels_perturbation():
    """Turn one non-sky pixel into sky and exactly that pixel must move."""
    labels = np.array([6, 6, 6, 7], dtype=np.uint8)
    before = fold_sky(labels)

    perturbed = labels.copy()
    perturbed[2] = SKY_CLASS
    after = fold_sky(perturbed)

    assert np.array_equal(before, labels)
    assert np.flatnonzero(before != after).tolist() == [2]
    assert after[2] == OTHER_CLASS


def test_fold_sky_confusion_merges_row_and_column():
    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    conf[OTHER_CLASS, OTHER_CLASS] = 5
    conf[SKY_CLASS, SKY_CLASS] = 11
    conf[SKY_CLASS, OTHER_CLASS] = 2
    conf[OTHER_CLASS, SKY_CLASS] = 3
    conf[SKY_CLASS, 6] = 7
    conf[6, SKY_CLASS] = 4
    conf[3, 3] = 100

    folded = fold_sky(conf)

    assert folded.shape == (FOLDED_NUM_CLASSES, FOLDED_NUM_CLASSES)
    assert folded.sum() == conf.sum()                       # no counts lost
    assert folded[OTHER_CLASS, OTHER_CLASS] == 5 + 11 + 2 + 3
    assert folded[OTHER_CLASS, 6] == 7
    assert folded[6, OTHER_CLASS] == 4
    assert folded[3, 3] == 100
    assert conf[SKY_CLASS, SKY_CLASS] == 11                 # input not mutated


def test_fold_sky_confusion_perturbation():
    """One extra sky-on-sky count has to show up in the folded `other`
    diagonal, or the fold is dropping the class instead of merging it."""
    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    conf[3, 3] = 40

    before = fold_sky(conf)

    perturbed = conf.copy()
    perturbed[SKY_CLASS, SKY_CLASS] += 1
    after = fold_sky(perturbed)

    assert after[OTHER_CLASS, OTHER_CLASS] == before[OTHER_CLASS, OTHER_CLASS] + 1
    assert np.flatnonzero((before != after).ravel()).tolist() == [0]


def test_fold_sky_rejects_a_confidence_map():
    """A float (H, W) max-probability map has no class axis to fold, so it must
    raise rather than be silently mangled."""
    with pytest.raises(ValueError, match="confusion"):
        fold_sky(np.zeros((4, 6), dtype=np.float32))


# ----- save_label_png -----

def test_save_label_png_is_a_readable_palette(tmp_path):
    labels = np.array([
        [0, 1, 2, UNLABELED],
        [5, 6, SKY_CLASS, 200],
    ], dtype=np.uint8)
    path = tmp_path / "labels.png"

    save_label_png(labels, path)

    with Image.open(path) as image:
        assert image.mode == "P"
        assert np.array_equal(np.array(image), labels)      # ids survive the round trip

        palette = image.getpalette()
        for class_id in range(NUM_CLASSES):
            entry = palette[3 * class_id:3 * class_id + 3]
            assert entry == CLASS_COLORS[class_id].tolist()

        # UNLABELED and any id outside the ontology render black
        assert palette[3 * UNLABELED:3 * UNLABELED + 3] == [0, 0, 0]
        assert palette[3 * 200:3 * 200 + 3] == [0, 0, 0]

        rgb = np.array(image.convert("RGB"))

    assert rgb[0, 2].tolist() == CLASS_COLORS[2].tolist()
    assert rgb[1, 2].tolist() == CLASS_COLORS[SKY_CLASS].tolist()
    assert rgb[0, 3].tolist() == [0, 0, 0]


def test_save_label_png_perturbation(tmp_path):
    """Change one pixel's class and the read-back colour at that pixel must
    move, and nowhere else."""
    labels = np.full((3, 3), 6, dtype=np.uint8)
    before_path = tmp_path / "before.png"
    save_label_png(labels, before_path)

    perturbed = labels.copy()
    perturbed[1, 1] = 7
    after_path = tmp_path / "after.png"
    save_label_png(perturbed, after_path)

    with Image.open(before_path) as image:
        before = np.array(image.convert("RGB"))
    with Image.open(after_path) as image:
        after = np.array(image.convert("RGB"))

    assert before[1, 1].tolist() == CLASS_COLORS[6].tolist()
    assert after[1, 1].tolist() == CLASS_COLORS[7].tolist()

    differing = np.argwhere(np.any(before != after, axis=-1))
    assert differing.tolist() == [[1, 1]]


def test_save_label_png_rejects_bad_input(tmp_path):
    with pytest.raises(ValueError, match=r"\(H, W\)"):
        save_label_png(np.zeros((2, 2, 3), dtype=np.uint8), tmp_path / "a.png")

    with pytest.raises(ValueError, match="uint8"):
        save_label_png(np.zeros((2, 2), dtype=np.int32), tmp_path / "b.png")


# ----- the frozen dataclasses -----

def test_dataclass_fields_match_the_spec():
    """Spec section 2 freezes these names and this order. Adapters and
    baselines construct both, some positionally, and a reorder would be
    invisible to every keyword-only test in this file."""
    assert [f.name for f in fields(Frame)] == [
        "frame_id", "image", "points", "intensity", "K",
        "point_times", "ego_twist", "labels_2d_gt", "labels_3d_gt",
    ]
    assert [f.name for f in fields(Prediction)] == [
        "labels_2d", "labels_3d", "conf_2d", "conf_3d",
    ]


def test_prediction_confidences_default_to_none():
    """None means ECE is skipped for that modality, not zero confidence."""
    pred = Prediction(
        labels_2d=np.zeros((2, 2), dtype=np.uint8),
        labels_3d=np.zeros(3, dtype=np.uint8),
    )

    assert pred.conf_2d is None
    assert pred.conf_3d is None


# ----- validate -----

def test_validate_accepts_a_well_formed_frame():
    validate(_frame())


def test_validate_accepts_missing_optionals():
    frame = _frame()
    frame.point_times = None
    frame.ego_twist = None
    frame.labels_2d_gt = None
    frame.labels_3d_gt = None

    validate(frame)


def test_validate_accepts_an_empty_cloud():
    """The dropout sweep hands baselines a near-empty cloud on purpose, so zero
    points is valid, not a mismatch."""
    validate(_frame(n_points=0))


@pytest.mark.parametrize("field,broken", [
    ("image", np.zeros((4, 6), dtype=np.uint8)),                  # missing channel axis
    ("image", np.zeros((4, 6, 4), dtype=np.uint8)),               # RGBA, not RGB
    ("image", np.zeros((4, 6, 3), dtype=np.float32)),             # not uint8
    ("points", np.zeros((7, 4), dtype=np.float32)),               # x,y,z,intensity left in
    ("points", np.zeros(7, dtype=np.float32)),                    # flattened
    ("points", np.zeros((7, 3), dtype=np.int16)),                  # quantised
    ("intensity", np.zeros(6, dtype=np.float32)),                 # one short of the cloud
    ("intensity", np.zeros((7, 1), dtype=np.float32)),            # column vector
    ("K", np.eye(4, dtype=np.float64)),                            # 4x4 extrinsic by mistake
    ("point_times", np.zeros(8, dtype=np.float32)),               # one long
    ("ego_twist", np.zeros(3, dtype=np.float32)),                 # linear only
    ("labels_2d_gt", np.zeros((5, 6), dtype=np.uint8)),           # off-by-one row
    ("labels_2d_gt", np.zeros((6, 4), dtype=np.uint8)),           # transposed
    ("labels_2d_gt", np.zeros((4, 6), dtype=np.int32)),           # sentinel aliasing dtype
    ("labels_3d_gt", np.zeros((7, 1), dtype=np.uint8)),           # column vector
    ("labels_3d_gt", np.zeros(7, dtype=np.uint16)),               # sentinel aliasing dtype
    ("labels_3d_gt", np.zeros(6, dtype=np.uint8)),                # one short of the cloud
])
def test_validate_names_the_mismatched_field(field, broken):
    frame = _frame()
    setattr(frame, field, broken)

    with pytest.raises(ValueError) as excinfo:
        validate(frame)

    message = str(excinfo.value)
    assert field in message
    assert frame.frame_id in message


def test_validate_rejects_a_list_for_an_array():
    frame = _frame()
    frame.points = [[0.0, 0.0, 0.0]]

    with pytest.raises(ValueError, match="points must be a numpy array"):
        validate(frame)
