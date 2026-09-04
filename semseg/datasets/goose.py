"""GOOSE val adapter: turns the two separately distributed GOOSE zips into
`Frame` objects in the `goose9` label space.

Kept separate from the fixture adapter (`semseg/datasets/fixture.py`) because
this file is entirely about coping with a real release that is missing things,
while the fixture is about having everything by construction. Mixing the two
would hide which numbers came from real data and which came from a generator.

WHAT IS ACTUALLY ON DISK (observed 2026-09-04 under
/home/pan-navigator/datasets/goose, and identical to spec section 1b):

    challenge_label_mapping.csv                the 64 to 9 mapping, fetched
                                               separately, NOT in either zip
    raw_2d/goose_label_mapping.csv             64 fine classes, no superclass
    raw_2d/LICENSE  raw_2d/CHANGELOG
    raw_2d/images/val/<seq>/<fid>_windshield_vis.png     962
    raw_2d/images/val/<seq>/<fid>_windshield_nir.png     962  (ignored)
    raw_2d/labels/val/<seq>/<fid>_labelids.png           962
    raw_2d/labels/val/<seq>/<fid>_color.png              962  (ignored)
    raw_2d/labels/val/<seq>/<fid>_instanceids.png        962  (ignored)
    raw_3d/goose_label_mapping.csv  raw_3d/LICENSE  raw_3d/CHANGELOG
    raw_3d/lidar/val/<seq>/<fid>_vls128.bin              961
    raw_3d/labels/val/<seq>/<fid>_goose.label            961

    <fid> is `<sequence>__<index>_<nanosecond stamp>`, unique and sortable.
    8 sequences in both modalities.

Formats verified by reading the bytes, not by trusting the docs:
    .bin    float32 x, y, z, intensity, so element count divides by 4.
            Intensity ranges 0 to 254, it is not normalised.
    .label  uint32, low 16 bits are the FINE semantic id, high 16 bits are the
            instance id (values up to 23 seen). Label count equals point count
            exactly.
    _labelids.png  uint8 in the FINE 64-class space, 1000x2048 in the frames
            sampled. The resolution is read per frame, never hardcoded.

WHAT IS MISSING, AND WHY THAT DECIDES THE SHAPE OF THIS FILE:

No calibration and no poses ship in the val zips. Confirmed three ways: an
exhaustive on-disk search for yaml, json, txt and ini files across both
extracted trees found only LICENSE and CHANGELOG; the GOOSE docs at
https://goose-dataset.de/docs/dataset-structure/ list neither in the annotated
downloads and place `metadata.yml` only inside the ROS bag `setups/` tree; and
spec section 1b records the same finding.

The two halves of the rig calibration are NOT equally missing, and this file is
careful to keep them apart:

  INTRINSICS ARE PUBLISHED. GOOSE publishes the windshield camera_info outside
  the annotated zips, and it is committed here as
  docs/calib/mucar3_windshield_vis.yaml (spec section 13.2). So `Frame.K` is
  populated from a real published measurement, and the camera arm needs no
  override to project.

  THE EXTRINSIC IS NOT. Nothing published gives T_cam_lidar for this release,
  so `extrinsic()` RAISES `CalibrationUnavailable` naming every path it
  searched. Having K does NOT make T available. Treating "the calibration" as
  one indivisible thing is exactly how a fabricated extrinsic would get in:
  problem statement section 6.3 makes the extrinsic the independent variable of
  the headline sweep, so a made-up value would not be a small error, it would
  be the whole result. `calib_available` therefore still means "both halves",
  which is what baselines/bl_paint gates on, and `intrinsics_available` is
  tracked separately.

THE CROP, WHICH IS THE ONE TRAP INSIDE THE PUBLISHED INTRINSICS:

The yaml declares image_height 1536. The shipped val images are 1000 rows
(measured, 2048x1000 on every frame sampled). 536 rows are missing, so the
release is a CROP of the calibrated sensor, and the vertical crop offset is
published NOWHERE.

Under a pure vertical crop fx, fy and cx carry over unchanged and cy does not:
the correct value is the published 775.44415 minus crop_top. crop_top is
unknown, so cy is LEFT AT THE PUBLISHED VALUE and `crop_offset_known` is False.

What earns this a warning, a flag and this many lines of comment is that it
does not look like a failure. 775.44 still lands inside a 1000-row image, so
every projection still produces pixels, still fills a z-buffer and still
scores. A wrong cy tilts every projection by a constant vertical offset, which
is precisely the error that yields plausible output and wrong conclusions. So
the adapter warns once, exposes the flag, accepts a `crop_top` from an operator
who knows something the release does not state, records whatever was used in
`coverage()`, and never guesses a value itself.

`Frame.ego_twist` is None and `poses_available` is False, so `eval/sweep.py`
must refuse the `time_offset` sweep here rather than assume a constant
velocity. `pose_file_found` is kept separate on purpose: a flag the sweep gates
on must mean "the twist is populated", never merely "a file was seen".

`Frame.point_times` is None as well, because the .bin carries exactly four
float32 fields per point and no per-point timestamp. That makes the `deskew`
sweep fixture-only for the same reason `time_offset` is: there is nothing real
to deskew against.

The modality overlap is NOT the limiting factor here. 961 of 962 frames are
present in both modalities, so fusion on val is capped by the missing
calibration and by nothing else. `coverage()` and `discover_report()` state
that in those words so a reader does not blame the wrong constraint.
"""

import glob
import json
import os
import sys
import warnings
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from semseg.datasets import Dataset
from semseg.labels import MAPPING_FILENAME, load_label_mapping, remap
from semseg.types import Frame

# Relative directories per role. GOOSE's own docs call the cloud directory
# `velodyne` while the val zip we extracted calls it `lidar`, so both are
# accepted: a differently shaped tree should produce a diagnosis, not a
# KeyError halfway through a sweep.
IMAGE_DIRS = ("raw_2d/images",)
LABEL_2D_DIRS = ("raw_2d/labels",)
CLOUD_DIRS = ("raw_3d/lidar", "raw_3d/velodyne")
LABEL_3D_DIRS = ("raw_3d/labels",)

# `_windshield_vis` is the RGB camera. `_nir` is near-infrared, a different
# modality with different statistics, and feeding it to an RGB baseline would
# quietly change what the camera arm is measuring.
IMAGE_SUFFIX = "_windshield_vis.png"
NIR_SUFFIX = "_windshield_nir.png"
LABEL_2D_SUFFIX = "_labelids.png"
CLOUD_SUFFIX = "_vls128.bin"
LABEL_3D_SUFFIX = "_goose.label"

# SemanticKITTI convention, verified against the real bytes.
CLOUD_FIELDS = 4
SEMANTIC_ID_MASK = 0xFFFF
INSTANCE_ID_SHIFT = 16

# The only file in the release that carries the 64 to 9 superclass column. The
# `goose_label_mapping.csv` shipped inside each zip has four columns and stops
# at the fine ontology, so it cannot produce goose9 on its own (spec section
# 1). Searched in order. The filename comes from semseg.labels so the adapter
# and the parser cannot disagree about what they are looking for.
LABEL_MAPPING_CANDIDATES = (
    MAPPING_FILENAME,
    os.path.join("raw_2d", MAPPING_FILENAME),
    os.path.join("raw_3d", MAPPING_FILENAME),
)

# Everything a rig calibration could plausibly be called in a GOOSE-shaped
# tree. None of these exist in the val release; the list exists so the error
# message can prove the adapter looked.
CALIB_CANDIDATES = (
    "calib.json",
    "calibration.json",
    "calib.yaml",
    "calib.yml",
    "calibration.yaml",
    "calibration.yml",
    "calib.txt",
    "metadata.yml",
    "metadata.yaml",
    "raw_2d/calib.json",
    "raw_2d/calibration.yaml",
    "raw_3d/calib.json",
    "raw_3d/calibration.yaml",
)

# Extensions swept recursively when the named candidates all miss, so a tree
# that hides its calibration under an unexpected name still gets found.
CALIB_GLOB_EXTENSIONS = ("yaml", "yml", "json", "txt", "ini")

POSE_CANDIDATES = (
    "poses.txt",
    "poses.json",
    "poses.csv",
    "raw_3d/poses.txt",
    "raw_3d/poses.json",
)
POSE_GLOB_PATTERNS = ("**/poses*.txt", "**/poses*.json", "**/*.pose", "**/odometry*.txt")

# Cap on how many discovered candidate files an error message lists, so a
# pathological tree cannot turn one exception into thousands of lines.
MAX_REPORTED_CANDIDATES = 20

# The published windshield intrinsics. They are committed to this repo rather
# than read from the dataset root because they are NOT in the val zips: GOOSE
# distributes the camera_info separately from the annotated downloads. Resolved
# from this file's own location, never from the cwd, so a sweep launched from
# anywhere resolves the same file.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_INTRINSICS_PATH = os.path.join(REPO_ROOT, "docs", "calib", "mucar3_windshield_vis.yaml")

# The default names the `_vis` file for the same reason IMAGE_SUFFIX does: the
# sibling mucar3_windshield_nir.yaml is a DIFFERENT camera with a different K,
# and pairing it with the RGB images would offset every projection by the
# difference between two real calibrations, which is harder to spot than a
# fabricated one. The camera_name is parsed and reported rather than enforced,
# because an explicit --calib style override is allowed to name any camera.

# Keys read out of the ROS camera_info dump. `camera_matrix` and emphatically
# NOT `projection_matrix`: on this unrectified plumb_bob camera both sections
# exist and BOTH parse as 9 numbers, so a reader that matched a bare `data:`
# key would pick up whichever section came last, pass every length check, and
# be wrong only in the entries that matter.
CAMERA_MATRIX_KEY = "camera_matrix"
DISTORTION_KEY = "distortion_coefficients"
CAMERA_NAME_KEY = "camera_name"
DECLARED_WIDTH_KEY = "image_width"
DECLARED_HEIGHT_KEY = "image_height"
DATA_KEY = "data"

CAMERA_MATRIX_VALUES = 9
# plumb_bob is [k1, k2, p1, p2, k3].
DISTORTION_VALUES = 5

# Position of cy in K. Called out as a constant because it is the ONLY entry a
# vertical crop invalidates, and the whole crop_top mechanism is one
# subtraction at this index.
CY_ROW = 1
CY_COL = 2

# Provenance of the crop offset that was actually applied, reported by
# coverage() so a submission records WHO supplied the number rather than just
# what it was. There is deliberately no third value meaning "derived": nothing
# in the release lets it be derived.
CROP_TOP_ABSENT = "none"
CROP_TOP_FROM_USER = "user"


class CalibrationUnavailable(RuntimeError):
    """Raised when the half of the calibration being asked for is not
    available and none was supplied. A distinct type so `eval/sweep.py` can
    refuse to run a projection-dependent sweep with a precise reason instead of
    catching a bare RuntimeError or, worse, receiving a default matrix.

    On this release `extrinsic()` ALWAYS raises it and `intrinsics()` normally
    does not, because the camera_info is published and T_cam_lidar is not. One
    exception type, two independently tracked availabilities.
    """


@dataclass(frozen=True)
class Discovery:
    """What the on-disk scan actually found. A value object rather than loose
    locals so `coverage()`, `discover_report()` and the error messages all read
    from one snapshot and cannot disagree with each other.
    """
    root: str
    split: str
    sequences: tuple = ()
    images: dict = field(default_factory=dict)
    labels_2d: dict = field(default_factory=dict)
    clouds: dict = field(default_factory=dict)
    labels_3d: dict = field(default_factory=dict)
    nir_count: int = 0
    mapping_path: str = None
    mapping_searched: tuple = ()
    calib_path: str = None
    calib_searched: tuple = ()
    calib_found_nearby: tuple = ()
    pose_path: str = None
    pose_searched: tuple = ()


@dataclass(frozen=True)
class Calibration:
    """A supplied rig calibration. K and T_cam_lidar are two halves of one
    artefact, so they travel together and a file carrying only one is rejected:
    a half-populated Frame is the same fiction risk as an invented extrinsic,
    only harder to notice.
    """
    K: np.ndarray
    T_cam_lidar: np.ndarray
    source: str


@dataclass(frozen=True)
class Intrinsics:
    """The published camera_info, on its own.

    Deliberately NOT the `Calibration` above, which insists on both halves and
    rejects a file carrying one. That rule is right for a rig file and wrong
    for this release: the intrinsics genuinely are published while the
    extrinsic genuinely is not, so the adapter has to be able to hold one
    without the other. Folding them into one type would force a fabricated
    T_cam_lidar into existence just to carry a real K.

    `K` is as published, before any crop correction. `declared_height` is what
    the yaml claims, which on this release is not what ships; see the module
    docstring.
    """
    K: np.ndarray
    distortion: np.ndarray
    declared_width: int
    declared_height: int
    camera_name: str
    source: str


def _sequences_under(base: str, split: str) -> list:
    split_dir = os.path.join(base, split)
    if not os.path.isdir(split_dir):
        return []

    return sorted(d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d)))


def _resolve_dir(root: str, candidates, split: str):
    """First candidate that exists and holds at least one sequence directory.
    Returns (absolute path or None, list of absolute paths tried)."""
    tried = [os.path.join(root, c) for c in candidates]
    for base in tried:
        if _sequences_under(base, split):
            return base, tried

    return None, tried


def _scan(base: str, split: str, suffix: str) -> dict:
    """frame_id -> absolute path, for every file under base/split/*/ ending in
    suffix. The frame id is the basename with the suffix stripped."""
    if base is None:
        return {}

    found = {}
    for path in glob.glob(os.path.join(base, split, "*", "*" + suffix)):
        found[os.path.basename(path)[: -len(suffix)]] = path

    return found


def _first_existing(root: str, candidates):
    tried = [os.path.join(root, c) for c in candidates]
    for path in tried:
        if os.path.isfile(path):
            return path, tried

    return None, tried


def _sweep_for_calib_files(root: str) -> tuple:
    """Every file under root whose extension could hold a calibration. Run only
    when the named candidates miss, so the raised error can say what it did see
    rather than just what it wanted."""
    hits = []
    for ext in CALIB_GLOB_EXTENSIONS:
        hits.extend(glob.glob(os.path.join(root, "**", "*." + ext), recursive=True))

    # the label mapping CSVs are not calibration, and neither is anything we
    # already name in the candidate list
    named = {os.path.join(root, c) for c in CALIB_CANDIDATES}
    return tuple(sorted(h for h in hits if h not in named))


def _sweep_for_pose_files(root: str) -> tuple:
    hits = []
    for pattern in POSE_GLOB_PATTERNS:
        hits.extend(glob.glob(os.path.join(root, pattern), recursive=True))

    return tuple(sorted(set(hits)))


def _cy_consequence(cy: float, actual_height: int) -> str:
    """The tail of the crop warning, which differs in kind rather than degree.

    An uncorrected cy that still falls inside the image is the dangerous case:
    every projection lands somewhere plausible while being tilted by a constant
    vertical offset. One that falls outside is the loud case, and saying it
    "still lands inside" there would be false, which is how the sentence read
    before this split.
    """
    if cy < actual_height:
        return (f"still lands inside a {actual_height}-row image, so projections will look "
                "entirely plausible while being tilted by a constant vertical offset.")

    return (f"falls OUTSIDE a {actual_height}-row image, so every projection will land off "
            "the frame and nothing will be scorable at all.")


def _load_calib_file(path: str) -> Calibration:
    """Parse an explicit --calib override, which must carry the FULL rig.

    JSON only. The published camera_info yaml is read by `_read_camera_info`
    instead and cannot serve here, because it holds no extrinsic at all.
    """
    with open(path) as handle:
        text = handle.read()

    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        # The only calibration file this repo ships is a ROS camera_info YAML, so
        # pointing --calib at it is the first thing anyone tries. Spending that
        # moment on a parser traceback wastes the one chance to explain the actual
        # constraint.
        if CAMERA_MATRIX_KEY in text:
            raise ValueError(
                f"{path} looks like a ROS camera_info dump: it carries "
                f"'{CAMERA_MATRIX_KEY}' and no camera-to-lidar transform. Pass it as "
                "--intrinsics instead, which is where the published GOOSE intrinsics "
                "belong. The extrinsic is not in any published file; it is in the "
                "GOOSE-DB bags as /tf_static, and --calib wants JSON carrying both 'K' "
                "and 'T_cam_lidar'."
            ) from exc

        raise ValueError(
            f"{path}: --calib expects JSON with 'K' and 'T_cam_lidar' and this file is "
            f"not valid JSON ({exc.msg} at line {exc.lineno})"
        ) from exc

    for key in ("K", "T_cam_lidar"):
        if key not in raw:
            raise ValueError(
                f"{path}: calibration is missing '{key}'. Both the intrinsics and the "
                "extrinsic are required; a Frame populated from half a calibration would "
                "project as confidently as a correct one"
            )

    K = np.asarray(raw["K"], dtype=np.float64).reshape(3, 3)
    T = np.asarray(raw["T_cam_lidar"], dtype=np.float64).reshape(4, 4)
    return Calibration(K=K, T_cam_lidar=T, source=path)


def _parse_flat_yaml(path: str) -> tuple:
    """-> (top level scalars, {section name: {key: text}}), all values as raw
    text.

    A few lines of key parsing rather than pyyaml, which the runtime image does
    not carry. That is safe here only because a ROS camera_info dump is flat:
    top level scalars, and one level of section holding scalars and one-line
    lists. Nothing nested deeper, no anchors, no multi-line strings.

    Sections are tracked by INDENTATION, and that is the load-bearing detail.
    Four sections of this file carry a `data:` key, so a reader that matched
    `data:` alone would silently read the wrong matrix (see CAMERA_MATRIX_KEY).
    """
    scalars = {}
    sections = {}
    section = None

    with open(path) as handle:
        for line in handle:
            if not line.strip() or line.lstrip().startswith("#"):
                continue

            key, separator, value = line.partition(":")
            if not separator:
                continue

            name = key.strip()
            text = value.strip()

            # an unindented line always closes the previous section, and only
            # opens a new one if it has no value of its own
            if not key[:1].isspace():
                section = None if text else name
                if text:
                    scalars[name] = text
                continue

            if section is not None:
                sections.setdefault(section, {})[name] = text

    return scalars, sections


def _numbers(text: str, path: str, label: str, expected: int) -> list:
    """One-line yaml list of numbers -> list of float, or raise naming `path`.

    The count is checked rather than reshaped into whatever fits: 9 numbers
    that are not a camera matrix reshape to (3, 3) just as happily as the real
    thing, and the error would then surface as a tilted projection rather than
    as a parse failure.
    """
    items = [item for item in text.strip().strip("[]").split(",") if item.strip()]

    try:
        values = [float(item) for item in items]
    except ValueError as error:
        raise ValueError(
            f"{path}: {label} is not a list of numbers ({error}). Expected a ROS "
            f"camera_info dump with {label}: [{expected} numbers]"
        ) from error

    if len(values) != expected:
        raise ValueError(
            f"{path}: {label} holds {len(values)} numbers, expected exactly {expected}. "
            "Refusing to reshape a partial matrix into a plausible one"
        )

    return values


def _read_camera_info(path: str) -> Intrinsics:
    """Parse a ROS camera_info yaml into `Intrinsics`, or raise naming `path`.

    The intrinsics are read from `camera_matrix`, which is the raw pinhole. On
    an unrectified camera `projection_matrix` carries the same numbers padded
    with a zero column, so reading either would work TODAY and diverge the
    moment a rectified calibration is dropped in. `camera_matrix` is the one
    that means "these are the intrinsics of the image as shipped".
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"camera_info {path} does not exist")

    scalars, sections = _parse_flat_yaml(path)

    for key, holder, label in (
        (CAMERA_MATRIX_KEY, sections, "section"),
        (DECLARED_WIDTH_KEY, scalars, "key"),
        (DECLARED_HEIGHT_KEY, scalars, "key"),
    ):
        if key not in holder:
            raise ValueError(
                f"{path}: no '{key}' {label}. This does not look like a ROS camera_info "
                "dump; a camera_matrix, an image_width and an image_height are all required"
            )

    if DATA_KEY not in sections[CAMERA_MATRIX_KEY]:
        raise ValueError(
            f"{path}: the '{CAMERA_MATRIX_KEY}' section has no '{DATA_KEY}' key"
        )

    K = np.asarray(
        _numbers(
            sections[CAMERA_MATRIX_KEY][DATA_KEY],
            path,
            f"{CAMERA_MATRIX_KEY}.{DATA_KEY}",
            CAMERA_MATRIX_VALUES,
        ),
        dtype=np.float64,
    ).reshape(3, 3)

    # distortion is metadata here: it is reported in coverage() but never
    # applied, because Frame carries no distortion field and silently
    # undistorting the pixels would change what the image arm is scored on
    distortion = None
    if DATA_KEY in sections.get(DISTORTION_KEY, {}):
        distortion = np.asarray(
            _numbers(
                sections[DISTORTION_KEY][DATA_KEY],
                path,
                f"{DISTORTION_KEY}.{DATA_KEY}",
                DISTORTION_VALUES,
            ),
            dtype=np.float64,
        )

    return Intrinsics(
        K=K,
        distortion=distortion,
        declared_width=int(float(scalars[DECLARED_WIDTH_KEY])),
        declared_height=int(float(scalars[DECLARED_HEIGHT_KEY])),
        camera_name=scalars.get(CAMERA_NAME_KEY),
        source=path,
    )


def _validated_crop_top(crop_top, intrinsics) -> int:
    """Boundary validation for an operator-supplied vertical crop offset.

    Raises rather than clamping. A clamped value would shift cy by a different
    amount than the number `coverage()` records, so the submission metadata
    would describe a projection that never happened.
    """
    value = int(crop_top)

    if value < 0:
        raise ValueError(f"crop_top must be >= 0, got {crop_top}")

    if intrinsics is not None and value >= intrinsics.declared_height:
        raise ValueError(
            f"crop_top {value} is not inside the declared image height "
            f"{intrinsics.declared_height} of {intrinsics.source}"
        )

    return value


def _probe_image_size(images: dict):
    """-> (width, height) of the first discovered image, or None if there is
    none.

    PIL fills `.size` from the PNG header, so this costs one open and does not
    decode a 2048x1000 frame just to learn its shape. It has to be measured
    rather than assumed: the declared height is exactly the thing under
    suspicion.
    """
    for frame_id in sorted(images):
        with Image.open(images[frame_id]) as handle:
            return handle.size

    return None


class GooseDataset(Dataset):
    """GOOSE val, discovered rather than assumed.

    `split` names the on-disk split directory, which is `val` for the released
    annotations. The fit/score partition of spec section 9 is `split_frames`
    from this package, applied to `frame_ids()` by the caller, so the policy is
    shared with the fixture adapter instead of copied into it.

    `intrinsics_path` overrides the committed camera_info; `calib_path`
    overrides the whole rig and its K then wins over the camera_info, because
    a K and a T_cam_lidar measured together must not be split up.
    `crop_top` is the vertical crop offset the release does not publish: pass
    it only if you actually know it, and see `crop_offset_known`.
    """

    def __init__(self, root, split="val", calib_path=None, mapping_path=None,
                 intrinsics_path=None, crop_top=None):
        self.root = os.path.abspath(str(root))
        self.split = split

        self._discovery = self._discover(calib_path=calib_path, mapping_path=mapping_path)
        self._lut = None

        self._calib = None
        if self._discovery.calib_path is not None:
            self._calib = _load_calib_file(self._discovery.calib_path)

        # "BOTH halves are present", which is what a projection needs and what
        # baselines/bl_paint gates on. The published intrinsics must not flip
        # it: bl_paint would then start painting through a fabricated
        # extrinsic, which is the failure this whole adapter is shaped to
        # prevent.
        self.calib_available = self._calib is not None

        self._intrinsics = self._resolve_intrinsics(intrinsics_path)
        self.intrinsics_available = self._calib is not None or self._intrinsics is not None

        # measured, not assumed: the declared height is the thing in doubt
        self._image_size = _probe_image_size(self._discovery.images)

        # False whenever the release and the calibration disagree about the
        # image height, INCLUDING when the caller supplied a crop_top. Same
        # rule as poses_available below: a flag a sweep gates on must mean
        # "this is known", never "somebody asserted it". The asserted value
        # travels separately, in coverage().
        self.crop_offset_known = self._check_declared_height()

        # A rig and a crop offset together are a contradictory request, and it
        # is refused rather than half honoured. The rig REPLACES the published
        # intrinsics, so the offset has nothing to correct, and applying it to
        # nothing would still record crop_top_used=N in coverage() next to a cy
        # that moved by a different amount. A reader who subtracts cy_used from
        # cy_published would then get a number that describes no projection.
        if crop_top is not None and self._calib is not None:
            raise ValueError(
                f"crop_top {crop_top} was supplied together with the full rig "
                f"{self._discovery.calib_path}. That rig replaces the published intrinsics, "
                "so there is no published cy left for a crop offset to correct. Supply the "
                "rig with the cy its own images need, or drop crop_top"
            )

        self.crop_top_used = 0 if crop_top is None else _validated_crop_top(
            crop_top, self._intrinsics)
        self.crop_top_source = CROP_TOP_ABSENT if crop_top is None else CROP_TOP_FROM_USER

        self._K = self._effective_K()

        self.pose_file_found = self._discovery.pose_path is not None

        # eval/sweep.py gates the time-offset sweep on this flag, so it means
        # exactly "Frame.ego_twist is populated" and nothing weaker. It is
        # always False here: no pose file ships with the val annotations, and
        # this adapter has no parser for the format one would arrive in, so
        # `load()` never fills ego_twist. A flag that said True on the strength
        # of merely FINDING a file would let the sweep start and then measure
        # nothing, which is worse than refusing to run.
        self.poses_available = False

        self._frame_ids = sorted(
            set(self._discovery.images)
            & set(self._discovery.labels_2d)
            & set(self._discovery.clouds)
            & set(self._discovery.labels_3d)
        )

    def frame_ids(self) -> list:
        """Sorted intersection of the frames complete in BOTH modalities. The
        2D and 3D annotations ship as separate zips, so their frame sets are
        not guaranteed to agree and anything outside the intersection cannot
        support a fusion claim."""
        return list(self._frame_ids)

    def coverage(self) -> dict:
        """Per-modality counts, the intersection, and which frames are missing
        from which side. Returned as data (not printed) so a submission or a
        sweep can record it next to its score."""
        d = self._discovery
        complete_2d = set(d.images) & set(d.labels_2d)
        complete_3d = set(d.clouds) & set(d.labels_3d)

        per_sequence = {}
        for seq in d.sequences:
            per_sequence[seq] = {
                "images": _count_in_seq(d.images, seq),
                "labels_2d": _count_in_seq(d.labels_2d, seq),
                "clouds": _count_in_seq(d.clouds, seq),
                "labels_3d": _count_in_seq(d.labels_3d, seq),
                "intersection": _count_in_seq({f: 1 for f in self._frame_ids}, seq),
            }

        # The crop story travels WITH the numbers. A cy left at the published
        # value and a cy shifted on an operator's say-so are two different
        # measurements, and a submission that reports a projection score has to
        # be able to say which one produced it. Everything here stays
        # JSON-native so it can be written into submission metadata unchanged.
        declared_size = None
        distortion = None
        cy_published = None
        if self._intrinsics is not None:
            declared_size = [self._intrinsics.declared_width, self._intrinsics.declared_height]
            cy_published = float(self._intrinsics.K[CY_ROW, CY_COL])
            if self._intrinsics.distortion is not None:
                distortion = [float(v) for v in self._intrinsics.distortion]

        return {
            "root": self.root,
            "split": self.split,
            "sequences": list(d.sequences),
            "n_images": len(d.images),
            "n_labels_2d": len(d.labels_2d),
            "n_clouds": len(d.clouds),
            "n_labels_3d": len(d.labels_3d),
            "n_nir_ignored": d.nir_count,
            "n_complete_2d": len(complete_2d),
            "n_complete_3d": len(complete_3d),
            "n_intersection": len(self._frame_ids),
            "only_2d": sorted(complete_2d - complete_3d),
            "only_3d": sorted(complete_3d - complete_2d),
            "per_sequence": per_sequence,
            "label_mapping": d.mapping_path,
            "calib_path": d.calib_path,
            "calib_available": self.calib_available,
            "intrinsics_path": None if self._intrinsics is None else self._intrinsics.source,
            "intrinsics_available": self.intrinsics_available,
            # which of the two the Frame's K actually came from, so cy_used
            # below is never read against the wrong cy_published
            "K_source": d.calib_path if self._calib is not None else (
                None if self._intrinsics is None else self._intrinsics.source),
            "intrinsics_declared_size": declared_size,
            "image_size_actual": None if self._image_size is None else list(self._image_size),
            "distortion_coefficients": distortion,
            # reported, never applied: Frame carries no distortion field, and
            # undistorting the pixels here would change what the image arm is
            # scored on without saying so
            "distortion_applied": False,
            "crop_offset_known": self.crop_offset_known,
            "crop_top_used": self.crop_top_used,
            "crop_top_source": self.crop_top_source,
            "cy_published": cy_published,
            "cy_used": None if self._K is None else float(self._K[CY_ROW, CY_COL]),
            "pose_file": d.pose_path,
            "pose_file_found": self.pose_file_found,
            "poses_available": self.poses_available,
            "point_times_available": False,
        }

    def load(self, frame_id: str) -> Frame:
        """A fully populated Frame, except for the two members the release
        genuinely cannot supply: ego_twist and point_times are both None.

        K IS populated, from the published camera_info, with the crop caveat in
        the module docstring: read `crop_offset_known` before trusting cy. What
        is still absent is the extrinsic, and `extrinsic()` raises for it.

        `semseg.types.validate` is still not called here. It no longer rejects
        these frames (K was the only field it was missing), but making load()
        validate is a change to the 3D-only path as well, and it belongs in one
        deliberate commit rather than as a side effect of parsing a yaml.
        """
        d = self._discovery
        for role, table in (
            ("image", d.images),
            ("2D label", d.labels_2d),
            ("cloud", d.clouds),
            ("3D label", d.labels_3d),
        ):
            if frame_id not in table:
                raise KeyError(
                    f"frame {frame_id!r} has no {role} under {self.root} split {self.split!r}. "
                    f"{len(self._frame_ids)} frames are complete in both modalities; "
                    "call discover_report() for the full diagnosis"
                )

        image = np.asarray(Image.open(d.images[frame_id]).convert("RGB"), dtype=np.uint8)
        labels_2d = self._remap(np.asarray(Image.open(d.labels_2d[frame_id]), dtype=np.uint8))

        points, intensity = _read_cloud(d.clouds[frame_id])
        labels_3d = self._remap(_read_point_labels(d.labels_3d[frame_id], points.shape[0]))

        return Frame(
            frame_id=frame_id,
            image=image,
            points=points,
            intensity=intensity,
            K=None if self._K is None else self._K.copy(),
            point_times=None,
            ego_twist=None,
            labels_2d_gt=labels_2d,
            labels_3d_gt=labels_3d,
        )

    def extrinsic(self, frame_id: str) -> np.ndarray:
        """T_cam_lidar, 4x4 float64. Raises unless a calibration was found on
        disk or supplied via `calib_path`."""
        if self._calib is None:
            raise CalibrationUnavailable(self._calib_error_text())

        return self._calib.T_cam_lidar.copy()

    def intrinsics(self, frame_id: str) -> np.ndarray:
        """K, 3x3 float64. NOT the same availability as `extrinsic`.

        This normally succeeds where `extrinsic` always raises, because GOOSE
        publishes the windshield camera_info and publishes no T_cam_lidar. The
        asymmetry is the point: a caller that needs both must check both, and
        `calib_available` is the flag that means both.

        cy carries the crop caveat in the module docstring. `crop_offset_known`
        says whether it can be trusted, and it is False on the real release.
        """
        if self._K is None:
            raise CalibrationUnavailable(self._intrinsics_error_text())

        return self._K.copy()

    def discover_report(self, stream=None) -> None:
        """Print what was found and what was missing. A user whose tree is
        shaped differently should get a diagnosis here rather than a KeyError
        several minutes into a sweep."""
        out = sys.stdout if stream is None else stream
        cov = self.coverage()
        d = self._discovery

        print(f"GOOSE discovery: root={self.root} split={self.split!r}", file=out)
        print(f"  sequences: {len(cov['sequences'])}", file=out)
        for seq in cov["sequences"]:
            counts = cov["per_sequence"][seq]
            print(
                f"    {seq}: images={counts['images']} labels_2d={counts['labels_2d']} "
                f"clouds={counts['clouds']} labels_3d={counts['labels_3d']} "
                f"both={counts['intersection']}",
                file=out,
            )

        print(
            f"  images={cov['n_images']} labels_2d={cov['n_labels_2d']} "
            f"clouds={cov['n_clouds']} labels_3d={cov['n_labels_3d']}",
            file=out,
        )
        print(
            f"  ignored {cov['n_nir_ignored']} {NIR_SUFFIX} files: near-infrared is a "
            "different modality and must not stand in for RGB",
            file=out,
        )
        print(
            f"  complete_2d={cov['n_complete_2d']} complete_3d={cov['n_complete_3d']} "
            f"INTERSECTION={cov['n_intersection']}",
            file=out,
        )
        _report_overlap_reading(cov, out)

        print(f"  label mapping: {d.mapping_path or 'NOT FOUND'}", file=out)
        if d.mapping_path is None:
            for path in d.mapping_searched:
                print(f"    searched {path}", file=out)

        print(f"  intrinsics: {cov['intrinsics_path'] or 'NOT FOUND'}", file=out)
        print(
            f"    declared size {cov['intrinsics_declared_size']} vs shipped "
            f"{cov['image_size_actual']}; crop_offset_known={cov['crop_offset_known']}; "
            f"crop_top_used={cov['crop_top_used']} (source: {cov['crop_top_source']}); "
            f"cy {cov['cy_published']} -> {cov['cy_used']}",
            file=out,
        )
        if not self.crop_offset_known:
            print(
                "    cy is NOT trustworthy: the release is a crop of the calibrated sensor "
                "and publishes no vertical offset, so every projection is tilted by an "
                "unknown constant that will not look like an error",
                file=out,
            )

        print(f"  calibration: {d.calib_path or 'NOT FOUND'}", file=out)
        if d.calib_path is None:
            print(
                "    the INTRINSICS above are published and usable; what is missing here is "
                "the extrinsic. calib_available means BOTH halves, which is why it is False",
                file=out,
            )
            print("    " + self._calib_error_text().replace("\n", "\n    "), file=out)

        print(f"  pose file: {d.pose_path or 'NOT FOUND'}", file=out)
        print(
            "    poses_available=False, so ego motion is unknown and the time_offset "
            "sweep must refuse to run rather than assume a constant velocity",
            file=out,
        )
        if d.pose_path is not None:
            print(
                "    a pose file WAS found but this adapter has no parser for it, so "
                "ego_twist stays None and poses_available stays False",
                file=out,
            )
        print(
            "  point_times_available=False: the .bin holds exactly x, y, z, intensity, "
            "so the deskew sweep is fixture-only for the same reason",
            file=out,
        )

    def _discover(self, calib_path, mapping_path) -> Discovery:
        image_base, image_tried = _resolve_dir(self.root, IMAGE_DIRS, self.split)
        label_2d_base, label_2d_tried = _resolve_dir(self.root, LABEL_2D_DIRS, self.split)
        cloud_base, cloud_tried = _resolve_dir(self.root, CLOUD_DIRS, self.split)
        label_3d_base, label_3d_tried = _resolve_dir(self.root, LABEL_3D_DIRS, self.split)

        if image_base is None and cloud_base is None:
            searched = image_tried + cloud_tried
            raise FileNotFoundError(
                f"no GOOSE split {self.split!r} under {self.root}. Searched: "
                + ", ".join(searched)
            )

        images = _scan(image_base, self.split, IMAGE_SUFFIX)
        nir = _scan(image_base, self.split, NIR_SUFFIX)

        sequences = sorted(
            set(_sequences_under(image_base or "", self.split))
            | set(_sequences_under(cloud_base or "", self.split))
        )

        found_mapping, mapping_tried = self._resolve_mapping(mapping_path)
        found_calib, calib_tried, calib_nearby = self._resolve_calib(calib_path)
        found_pose, pose_tried = self._resolve_poses()

        return Discovery(
            root=self.root,
            split=self.split,
            sequences=tuple(sequences),
            images=images,
            labels_2d=_scan(label_2d_base, self.split, LABEL_2D_SUFFIX),
            clouds=_scan(cloud_base, self.split, CLOUD_SUFFIX),
            labels_3d=_scan(label_3d_base, self.split, LABEL_3D_SUFFIX),
            nir_count=len(nir),
            mapping_path=found_mapping,
            mapping_searched=tuple(mapping_tried),
            calib_path=found_calib,
            calib_searched=tuple(calib_tried),
            calib_found_nearby=calib_nearby,
            pose_path=found_pose,
            pose_searched=tuple(pose_tried),
        )

    def _resolve_mapping(self, mapping_path):
        if mapping_path is not None:
            path = os.path.abspath(str(mapping_path))
            if not os.path.isfile(path):
                raise FileNotFoundError(f"label mapping override {path} does not exist")
            return path, [path]

        return _first_existing(self.root, LABEL_MAPPING_CANDIDATES)

    def _resolve_calib(self, calib_path):
        if calib_path is not None:
            path = os.path.abspath(str(calib_path))
            if not os.path.isfile(path):
                raise FileNotFoundError(f"calibration override {path} does not exist")
            return path, [path], ()

        found, tried = _first_existing(self.root, CALIB_CANDIDATES)
        if found is not None:
            return found, tried, ()

        return None, tried, _sweep_for_calib_files(self.root)

    def _resolve_intrinsics(self, intrinsics_path):
        """-> `Intrinsics` from the published camera_info, or None.

        An explicit override that does not exist raises, matching --calib and
        --mapping. A missing DEFAULT degrades to None instead, so someone
        diagnosing an incomplete checkout can still call coverage() and
        discover_report() rather than being stopped at construction.
        """
        if intrinsics_path is not None:
            return _read_camera_info(os.path.abspath(str(intrinsics_path)))

        if not os.path.isfile(DEFAULT_INTRINSICS_PATH):
            return None

        return _read_camera_info(DEFAULT_INTRINSICS_PATH)

    def _check_declared_height(self) -> bool:
        """-> crop_offset_known, warning ONCE if the calibration and the
        release disagree about how tall the image is.

        Warned at construction and not in `load()`, because the disagreement is
        a property of the release rather than of any one frame. Per-frame it
        would fire 962 times and be scrolled past, which for a defect that
        produces plausible output is the same as not warning at all.
        """
        # a supplied rig REPLACES the published intrinsics, so the declared
        # height in the yaml describes a K that is not in use and warning about
        # it would state something false. Not applicable reports as False,
        # which declines to vouch for the rig's cy rather than over-claiming.
        if self._calib is not None:
            return False

        if self._intrinsics is None or self._image_size is None:
            return False

        declared = self._intrinsics.declared_height
        actual = self._image_size[1]
        if declared == actual:
            return True

        cy = float(self._intrinsics.K[CY_ROW, CY_COL])
        warnings.warn(
            f"{self._intrinsics.source} declares {DECLARED_HEIGHT_KEY} {declared} but the "
            f"shipped {self.split!r} images are {actual} rows, a disagreement of "
            f"{abs(declared - actual)} rows. The shipped image is therefore not the "
            "calibrated image (this release is a vertical crop of it) and the crop offset "
            "is published NOWHERE. fx, fy and cx carry over unchanged, cy does not: its "
            f"correct value is {cy} minus crop_top. crop_offset_known is False and cy is "
            f"left at {cy}, which {_cy_consequence(cy, actual)} Pass crop_top=N "
            "(--crop-top N) if you know the offset. Nothing here will guess it.",
            stacklevel=3,
        )
        return False

    def _effective_K(self):
        """-> the K a Frame gets, or None if no intrinsics exist at all.

        A --calib rig wins outright over the published yaml, and crop_top is
        NOT applied to it: that rig's K and T_cam_lidar were measured together
        against whatever images the operator had, and mixing the published K
        with a foreign extrinsic would assemble a chimera out of two real
        calibrations.
        """
        if self._calib is not None:
            return self._calib.K.copy()

        if self._intrinsics is None:
            return None

        K = self._intrinsics.K.copy()
        K[CY_ROW, CY_COL] -= self.crop_top_used
        return K

    def _resolve_poses(self):
        found, tried = _first_existing(self.root, POSE_CANDIDATES)
        if found is not None:
            return found, tried

        swept = _sweep_for_pose_files(self.root)
        if swept:
            return swept[0], tried

        return None, tried

    def _remap(self, fine: np.ndarray) -> np.ndarray:
        """Fine 64-class ids to goose9, via semseg.labels so the ontology lives
        in exactly one place. Loaded lazily so a user diagnosing a tree that is
        missing the challenge CSV can still call coverage() and
        discover_report()."""
        if self._lut is None:
            self._lut = self._load_lut()

        return remap(fine, self._lut)

    def _load_lut(self) -> np.ndarray:
        path = self._discovery.mapping_path
        if path is None:
            raise FileNotFoundError(
                f"no {MAPPING_FILENAME} found. Searched: "
                + ", ".join(self._discovery.mapping_searched)
                + ". The goose_label_mapping.csv inside each zip carries only the fine "
                "ontology; fetch the challenge mapping from "
                "https://goose-dataset.de/docs/resources/" + MAPPING_FILENAME
            )

        return load_label_mapping(path)

    def _calib_error_text(self) -> str:
        d = self._discovery
        lines = [
            "no camera-to-lidar calibration available for GOOSE "
            f"{self.split!r} under {self.root}.",
            "The GOOSE val zips ship no calibration and no poses: the rig calibration is "
            "distributed through the metadata database and the ROS bags, not the annotated "
            "downloads (https://goose-dataset.de/docs/dataset-structure/).",
            "Refusing to return a default, because every projection, consistency and sweep "
            "number computed from a fabricated extrinsic would be measuring a fiction.",
            "Searched these exact paths:",
        ]
        lines.extend(f"  {p}" for p in d.calib_searched)

        if d.calib_found_nearby:
            shown = d.calib_found_nearby[:MAX_REPORTED_CANDIDATES]
            lines.append(
                f"Also swept the tree for {', '.join('*.' + e for e in CALIB_GLOB_EXTENSIONS)} "
                f"and found {len(d.calib_found_nearby)} unrelated file(s):"
            )
            lines.extend(f"  {p}" for p in shown)
            if len(d.calib_found_nearby) > len(shown):
                lines.append(f"  ... {len(d.calib_found_nearby) - len(shown)} more")
        else:
            lines.append(
                "Also swept the tree for "
                + ", ".join("*." + e for e in CALIB_GLOB_EXTENSIONS)
                + " and found nothing."
            )

        lines.append("Supply one with --calib <file.json> holding both 'K' and 'T_cam_lidar'.")
        return "\n".join(lines)

    def _intrinsics_error_text(self) -> str:
        """Only reachable when the committed camera_info is missing from the
        checkout, which is a broken repo rather than a limitation of the
        release. Says so, so the two causes do not get confused."""
        return (
            f"no camera intrinsics available. GOOSE publishes the windshield camera_info, "
            f"and this repo commits it at {DEFAULT_INTRINSICS_PATH}, but that file is not "
            "there. This is a broken checkout, not a gap in the release. Restore it, or "
            "pass intrinsics_path=<camera_info.yaml>."
        )


def _count_in_seq(table, sequence: str) -> int:
    prefix = sequence + "__"
    return sum(1 for fid in table if fid.startswith(prefix))


def _report_overlap_reading(cov: dict, out) -> None:
    """Say out loud which constraint the intersection actually implies. On the
    real release the overlap is near total, so a reader who sees the number
    alone would blame the wrong thing for the missing fusion results."""
    total = max(cov["n_complete_2d"], cov["n_complete_3d"])
    if total == 0:
        print("    intersection is EMPTY: no frame is complete in both modalities", file=out)
        return

    fraction = cov["n_intersection"] / total
    if fraction >= 0.95:
        print(
            "    modality overlap is essentially total, so it is NOT what caps a fusion "
            "claim on this split. The missing calibration is.",
            file=out,
        )
        return

    print(
        f"    only {fraction:.1%} of frames are bi-modal, which caps directly how much "
        "any fusion claim on this split can mean",
        file=out,
    )


def _read_cloud(path: str):
    """-> (points (N, 3) float32 in the lidar frame, intensity (N,) float32).

    Intensity is returned raw (0 to 254 on the real data). Normalising it here
    would hide a scale the baselines need to declare for themselves.
    """
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size == 0 or raw.size % CLOUD_FIELDS != 0:
        raise ValueError(
            f"{path}: {raw.size} float32 values do not divide by {CLOUD_FIELDS}, "
            "so this is not a GOOSE x/y/z/intensity cloud"
        )

    fields = raw.reshape(-1, CLOUD_FIELDS)
    points = np.ascontiguousarray(fields[:, :3], dtype=np.float32)
    intensity = np.ascontiguousarray(fields[:, 3], dtype=np.float32)
    return points, intensity


def _read_point_labels(path: str, n_points: int) -> np.ndarray:
    """-> (N,) uint16 FINE semantic ids.

    The .label file is uint32 with the instance id in the high 16 bits, so the
    mask is what makes this a semantic read. Without it every instance beyond
    the first would carry a semantic id in the tens of thousands.
    """
    raw = np.fromfile(path, dtype=np.uint32)
    if raw.size != n_points:
        raise ValueError(
            f"{path}: {raw.size} labels for {n_points} points. The GOOSE label file must "
            "be one uint32 per point"
        )

    return (raw & SEMANTIC_ID_MASK).astype(np.uint16)
