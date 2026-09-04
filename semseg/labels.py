"""The goose9 label space: the nine class names, the devkit colour map, the
parser that builds the raw-to-goose9 lookup table, and the sky fold.

Separate from types.py because types.py is the data contract (what a Frame
holds) while this file is the ontology (what a label value means). Assumption
A3 of the problem statement says the label spaces are not compatible and the
class mapping is lossy and must be documented, so the mapping is read from the
dataset's own CSV instead of being transcribed into code, and the one lossy
decision (sky) lives in exactly one function.
"""

import csv
import os

import numpy as np
from PIL import Image

from semseg.types import NUM_CLASSES, UNLABELED

# Superclass ids and names, verbatim from the GOOSE challenge mapping. Index is
# the goose9 id, so CLASS_NAMES[label] is always the right name.
CLASS_NAMES = (
    "other",
    "artificial_structures",
    "artificial_ground",
    "natural_ground",
    "obstacle",
    "vehicle",
    "vegetation",
    "human",
    "sky",
)

# The two classes the sky fold merges, named so the fold reads as intent.
OTHER_CLASS = 0
SKY_CLASS = 8

# The folded 8-class view GOOSE's own 3D config uses (see fold_sky).
FOLDED_NUM_CLASSES = 8
FOLDED_CLASS_NAMES = CLASS_NAMES[:FOLDED_NUM_CLASSES]

# The devkit's word for the no-ground-truth sentinel. Same value as
# types.UNLABELED, aliased rather than redefined so the two cannot drift: code
# reading GOOSE files says IGNORE, code talking about our own predictions says
# UNLABELED, and they must always be the same byte.
IGNORE = UNLABELED

# Devkit superclass colours, exactly as the devkit ships them, which is BGR
# because the devkit was written against cv2. Do NOT "fix" this back: reversed,
# row 5 is #F44336 (Material Red 500), row 6 is #4CAF50 (Material Green 500)
# and row 4 is #FFC107 (Material Amber 500), which is what the devkit's
# screenshots show. Read as BGR those are blue, and the screenshots are not.
DEVKIT_COLORS_BGR = np.array([
    [169, 169, 169],   # 0 other
    [222, 136, 222],   # 1 artificial_structures
    [59, 255, 235],    # 2 artificial_ground
    [127, 136, 161],   # 3 natural_ground
    [7, 193, 255],     # 4 obstacle
    [54, 67, 244],     # 5 vehicle
    [80, 175, 76],     # 6 vegetation
    [255, 176, 143],   # 7 human
], dtype=np.uint8)

# Class 8 has no devkit superclass colour because the devkit's own 3D config
# folds it away. Borrow the fine class `sky` hex from the mapping CSV.
SKY_COLOR_RGB = (0xB7, 0x7B, 0x68)

CLASS_COLORS = np.vstack([
    DEVKIT_COLORS_BGR[:, ::-1],
    np.array([SKY_COLOR_RGB], dtype=np.uint8),
])

# A raw GOOSE fine label id is one byte, so the lookup table is a flat 256-entry
# array and remapping is a single fancy-index rather than a dict lookup per
# pixel. 962 val frames at 1000x2048 makes that difference measurable.
LUT_SIZE = 256

# PNG palette entries, fixed by the format at 256 RGB triples.
PALETTE_SIZE = 256

# The file that actually carries the 64-to-9 mapping. The per-modality
# goose_label_mapping.csv inside the dataset zips has the 64 fine classes but no
# superclass column, so it cannot build this table (spec section 1b).
MAPPING_FILENAME = "challenge_label_mapping.csv"

# Header spellings accepted for the fine label index column, in priority order.
FINE_ID_COLUMNS = (
    "label_key",
    "label_id",
    "labelid",
    "fine_label_id",
    "fine_id",
    "label_index",
    "index",
    "id",
    "key",
)

# Header spellings accepted for the superclass id column. `challege_category_id`
# is first because it is what upstream actually ships, missing the n. The typo
# is left in the data on purpose: silently rewriting a published resource makes
# the next fetch look like a regression.
CATEGORY_ID_COLUMNS = (
    "challege_category_id",
    "challenge_category_id",
    "category_id",
    "superclass_id",
    "super_class_id",
    "challege_id",
    "challenge_id",
)

# Fallback for a revision that ships only the superclass name.
CATEGORY_NAME_COLUMNS = (
    "challenge_category_name",
    "challege_category_name",
    "category_name",
    "superclass_name",
    "superclass",
    "super_class",
    "category",
)


def _norm(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def _pick_column(headers, candidates):
    """The raw header matching the first candidate present, or None. Matching is
    exact after normalisation, so `category_id` never captures
    `challege_category_id` and the priority order stays meaningful."""
    available = {_norm(h): h for h in headers if h}

    for candidate in candidates:
        if candidate in available:
            return available[candidate]

    return None


def _category_id(path, lineno, row, id_col, name_col):
    """The goose9 id for one CSV row. The id column wins when present because it
    is unambiguous; the name column is the fallback."""
    if id_col is not None:
        raw = (row.get(id_col) or "").strip()
        if not raw.isdigit():
            raise ValueError(f"{path}:{lineno}: superclass id {raw!r} is not a non-negative integer")

        category = int(raw)
        if category >= NUM_CLASSES:
            raise ValueError(
                f"{path}:{lineno}: superclass id {category} outside 0..{NUM_CLASSES - 1}, "
                f"the mapping is not goose9"
            )

        return category

    name = _norm(row.get(name_col) or "")
    if name not in CLASS_NAMES:
        raise ValueError(f"{path}:{lineno}: superclass name {name!r} is not one of {CLASS_NAMES}")

    return CLASS_NAMES.index(name)


def load_label_mapping(csv_path) -> np.ndarray:
    """Read the challenge mapping CSV into a length-256 uint8 lookup table from
    a raw GOOSE fine label id to a goose9 id.

    Anything the CSV does not mention maps to UNLABELED, not to `other`: an id
    the ontology has never heard of is missing ground truth, and UNLABELED is
    excluded from every confusion matrix whereas `other` would be scored.

    Columns are discovered from the header rather than taken by position,
    because the mapping is a separately fetched web resource whose column order
    is not ours to depend on.
    """
    path = str(csv_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"label mapping CSV not found at {path}. Expected {MAPPING_FILENAME} from the "
            f"GOOSE challenge resources, saved next to the raw_2d and raw_3d trees. The "
            f"goose_label_mapping.csv inside the zips is not a substitute: it lists the 64 "
            f"fine classes with no superclass column."
        )

    lut = np.full(LUT_SIZE, UNLABELED, dtype=np.uint8)

    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []

        fine_col = _pick_column(headers, FINE_ID_COLUMNS)
        id_col = _pick_column(headers, CATEGORY_ID_COLUMNS)
        name_col = _pick_column(headers, CATEGORY_NAME_COLUMNS)

        if fine_col is None:
            raise ValueError(
                f"{path}: no fine label id column, headers are {headers}. "
                f"Accepted spellings: {FINE_ID_COLUMNS}"
            )

        if id_col is None and name_col is None:
            raise ValueError(
                f"{path}: no superclass column, headers are {headers}. This is what the "
                f"per-modality goose_label_mapping.csv looks like; fetch {MAPPING_FILENAME}."
            )

        # DictReader consumed the header, so data rows start at line 2
        for lineno, row in enumerate(reader, start=2):
            fine_raw = (row.get(fine_col) or "").strip()
            if not fine_raw:
                continue

            if not fine_raw.isdigit():
                raise ValueError(f"{path}:{lineno}: fine label id {fine_raw!r} is not a non-negative integer")

            fine_id = int(fine_raw)
            if fine_id >= LUT_SIZE:
                raise ValueError(
                    f"{path}:{lineno}: fine label id {fine_id} outside 0..{LUT_SIZE - 1}, "
                    f"it does not fit a uint8 label image"
                )

            lut[fine_id] = _category_id(path, lineno, row, id_col, name_col)

    return lut


def remap(raw_labels, lut) -> np.ndarray:
    """Raw fine label ids -> goose9, vectorised.

    A .label file's semantic field is uint32, so an id past the table's domain
    is possible; those become UNLABELED rather than raising, because that is
    exactly what they mean.
    """
    raw = np.asarray(raw_labels)
    lut = np.asarray(lut)

    if lut.shape != (LUT_SIZE,):
        raise ValueError(f"lut must be ({LUT_SIZE},), got {lut.shape}")

    if raw.dtype == np.uint8:
        return lut[raw]

    out = np.full(raw.shape, UNLABELED, dtype=np.uint8)
    inside = (raw >= 0) & (raw < LUT_SIZE)
    out[inside] = lut[raw[inside]]

    return out


def fold_sky(labels_or_conf) -> np.ndarray:
    """Project goose9 down to 8 classes by merging sky into `other`, which is
    what GOOSE's own Pointcept 3D config does (`get_learning_map` maps 8 to 0).

    A lidar gets no return from sky, so class 8 is essentially empty in 3D
    ground truth while it is one of the largest classes in 2D. We keep it as a
    real class in goose9 because that is what makes cross-modal consistency
    definable, and report every 3D number in this folded view as well so it
    stays comparable to the published 8-class PTv3 reference.

    Takes either a label array, remapped elementwise with UNLABELED untouched,
    or a (9, 9) confusion matrix, whose sky row and sky column are added into
    `other` before both are dropped. Shape is the discriminator: a label array
    is an (H, W) image or an (N,) vector, never exactly 9 by 9.
    """
    array = np.asarray(labels_or_conf)

    if array.shape == (NUM_CLASSES, NUM_CLASSES):
        folded = array.copy()
        folded[OTHER_CLASS, :] += folded[SKY_CLASS, :]
        folded[:, OTHER_CLASS] += folded[:, SKY_CLASS]
        return folded[:FOLDED_NUM_CLASSES, :FOLDED_NUM_CLASSES]

    if array.dtype.kind not in "ui":
        raise ValueError(
            f"fold_sky takes integer labels or a ({NUM_CLASSES}, {NUM_CLASSES}) confusion "
            f"matrix, got dtype {array.dtype} shape {array.shape}"
        )

    folded = array.copy()
    folded[array == SKY_CLASS] = OTHER_CLASS

    return folded


def save_label_png(labels, path) -> None:
    """Write a goose9 label map as an 8-bit palette PNG.

    One byte per pixel plus a palette, so the file is both small and still
    readable back as class ids, unlike an RGB render which has to be matched
    against a colour table to recover a label. Every id outside the ontology,
    UNLABELED included, is black, so an unlabelled region is visibly absent
    instead of being mistaken for `other`.
    """
    labels = np.asarray(labels)

    if labels.ndim != 2:
        raise ValueError(f"labels must be (H, W), got {labels.shape}")

    if labels.dtype != np.uint8:
        raise ValueError(f"labels must be uint8 to index a PNG palette, got {labels.dtype}")

    # zero-filled, so every entry we do not set stays black
    palette = bytearray(PALETTE_SIZE * 3)
    palette[:CLASS_COLORS.nbytes] = CLASS_COLORS.tobytes()

    height, width = labels.shape

    # frombytes rather than fromarray: fromarray's explicit-mode form is
    # deprecated and would drop the palette semantics on a newer Pillow
    image = Image.frombytes("P", (width, height), labels.tobytes())
    image.putpalette(bytes(palette))
    image.save(str(path))
