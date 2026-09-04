"""Unit tests for the frame viewer.

Every claim the viewer makes visually is asserted numerically here, with a
perturbation case behind each one: the agreement fraction has to fall when the
extrinsic is perturbed, the projected pixels have to move by the amount the
pinhole geometry predicts, and flipping one ground truth label has to move the
matched count by exactly one. A viewer whose agreement panel cannot go red
would draw a reassuring picture of a broken rig, which is the failure this file
exists to prevent.

Headless throughout. The Agg backend is selected here, before anything imports
pyplot, so the tests pass in the runtime container which has no display.
"""

import os
import sys

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from semseg.datasets.fixture import (  # noqa: E402
    FixtureConfig, HUMAN, VEGETATION, make_frame, nominal_extrinsic,
)
from semseg.labels import CLASS_COLORS  # noqa: E402
from semseg.projection import perturb_extrinsic, project, zbuffer  # noqa: E402
from semseg.types import NUM_CLASSES, UNLABELED  # noqa: E402
from tools import viewer  # noqa: E402

# The perturbation the whole repo argues about (problem statement section 6.3
# tops its rotation sweep out here), and a floor for the drop it has to cause.
# The measured drop on this fixture is about 0.15, so 0.10 has margin without
# being vacuous. The nominal floor is likewise below the measured 0.98: pinning
# the measured values would turn any change to the fixture's noise constants
# into a failure of this file.
DECALIB_DEG = 2.0
MIN_NOMINAL_AGREEMENT = 0.95
MIN_AGREEMENT_DROP = 0.10

# The ladder, to check the curve is monotone rather than only its endpoints.
DECALIB_LADDER_DEG = (0.25, 0.5, 1.0, 2.0)

# Big enough to swing a chunk of the cloud out of the frustum, which is what
# moves coverage. The 2 degree case barely touches it: the points shift within
# the image rather than off it.
FRUSTUM_YAW_DEG = 30.0

# A 4-panel figure with a 20k point scatter. Anything near this floor is an
# empty canvas, which is the failure mode a bare os.path.exists misses.
MIN_PNG_BYTES = 20_000


@pytest.fixture(scope="module")
def config():
    return FixtureConfig()


@pytest.fixture(scope="module")
def frame(config):
    return make_frame("fixture_0000", 0, 0, config)


@pytest.fixture(scope="module")
def T_nominal():
    return nominal_extrinsic()


@pytest.fixture(scope="module")
def nominal(frame, T_nominal):
    return viewer.agreement(frame, T_nominal)


def test_nominal_agreement_is_high(nominal):
    """The fixture renders both modalities from the same primitives through the
    same extrinsic, so at the nominal calibration nearly every point lands on a
    pixel of its own class. The residual is class boundaries and range noise."""
    assert nominal.fraction > MIN_NOMINAL_AGREEMENT
    assert nominal.matched <= nominal.scorable <= nominal.in_frustum <= nominal.total_points


def test_coverage_is_reported_and_partial(nominal):
    """Assumption A2 quantified: the camera is a frustum and the lidar is 360
    degrees, so most points have no pixel. A coverage of 1.0 here would mean
    the projection had stopped rejecting anything."""
    assert 0.0 < nominal.coverage < 1.0
    assert nominal.coverage == pytest.approx(nominal.scorable / nominal.total_points)


def test_agreement_drops_under_pitch(frame, T_nominal, nominal):
    """The headline perturbation. Two degrees of pitch is a vertical image
    shift of about six pixels on this rig, and it has to show up as red."""
    decalibrated = viewer.agreement(frame, viewer.decalibrate(T_nominal, DECALIB_DEG))

    assert decalibrated.fraction < nominal.fraction - MIN_AGREEMENT_DROP
    assert (~decalibrated.scorable_match).sum() > (~nominal.scorable_match).sum()


def test_agreement_falls_along_the_ladder(frame, T_nominal, nominal):
    """Monotone, not just lower at the far end. A metric that only moves at 2
    degrees would hide the small residuals a real rig actually carries."""
    fractions = [viewer.agreement(frame, viewer.decalibrate(T_nominal, deg)).fraction
                 for deg in DECALIB_LADDER_DEG]

    assert nominal.fraction > fractions[0]
    assert all(a > b for a, b in zip(fractions, fractions[1:]))


def test_one_flipped_label_moves_matched(frame, T_nominal, nominal):
    """The exact-arithmetic perturbation: relabel one agreeing point and the
    matched count has to fall by exactly one. This is the only case here whose
    output is predictable to the integer, so it is the one that proves the
    counter is counting and not estimating."""
    victim = _first_matched_point(frame, T_nominal)

    labels_3d = frame.labels_3d_gt.copy()
    original = labels_3d[victim]
    labels_3d[victim] = VEGETATION if original != VEGETATION else HUMAN

    perturbed = viewer.agreement(_with_labels_3d(frame, labels_3d), T_nominal)

    assert perturbed.scorable == nominal.scorable
    assert perturbed.matched == nominal.matched - 1
    assert perturbed.fraction < nominal.fraction


def test_coverage_moves_when_the_frustum_moves(frame, T_nominal, nominal):
    """Coverage answers a different question from agreement (how many points
    the camera saw at all), so it needs its own perturbation. A large yaw
    rotates points out of the frustum entirely."""
    swung = viewer.agreement(frame, perturb_extrinsic(T_nominal, "yaw", FRUSTUM_YAW_DEG, 0.0))

    assert swung.coverage < nominal.coverage
    assert swung.in_frustum < nominal.in_frustum


def test_decalibrate_at_zero_changes_nothing(T_nominal):
    assert np.array_equal(viewer.decalibrate(T_nominal, 0.0), T_nominal)


def test_decalibrate_shifts_pixels_upwards(frame, T_nominal):
    """The geometric transform's own perturbation case, checked against the
    pinhole prediction rather than against itself.

    A positive rotation about the camera x axis maps a point at (0, 0, z) to
    (0, -z sin, z cos), so its row moves up by f tan(theta), and further for a
    point already below the principal row. Asserting the direction and a lower
    bound on the magnitude catches both a sign error and a degrees/radians
    error, which are the two ways this could be silently wrong.
    """
    height, width = frame.image.shape[:2]
    focal = frame.K[1, 1]

    uv_before, _d, in_before = project(frame.points, frame.K, T_nominal, width, height)
    uv_after, _d, in_after = project(frame.points, frame.K,
                                     viewer.decalibrate(T_nominal, DECALIB_DEG), width, height)

    both = in_before & in_after
    shift = uv_after[both, 1].astype(np.int64) - uv_before[both, 1].astype(np.int64)
    expected = focal * np.tan(np.radians(DECALIB_DEG))

    assert both.sum() > 0
    assert (shift <= 0).all()
    assert abs(np.median(shift)) >= expected - 1.0


def test_colorise_uses_the_devkit_colours():
    labels = np.array([[0, 1, 2], [6, 7, 8]], dtype=np.uint8)

    rgb = viewer.colorise(labels)

    assert rgb.shape == (2, 3, 3)
    assert np.array_equal(rgb[0, 1], CLASS_COLORS[1])
    assert np.array_equal(rgb[1, 2], CLASS_COLORS[8])


def test_colorise_blacks_out_unlabeled():
    """An unlabelled pixel must look absent. Painting it as class 0 `other`
    would make a region with no ground truth look like a scored one."""
    labels = np.array([[UNLABELED, NUM_CLASSES]], dtype=np.uint8)

    assert not viewer.colorise(labels).any()


def test_colorise_moves_with_one_pixel():
    labels = np.full((2, 2), VEGETATION, dtype=np.uint8)
    before = viewer.colorise(labels)

    labels[0, 0] = HUMAN

    assert not np.array_equal(viewer.colorise(labels)[0, 0], before[0, 0])


def test_backdrop_fades_towards_white(frame):
    """The scatter panels need the photo out of the way. Brighter and grey
    everywhere, and still tracking the image it came from."""
    faded = viewer.backdrop(frame.image)

    assert faded.shape == frame.image.shape
    assert faded.mean() > frame.image.mean()
    assert np.array_equal(faded[:, :, 0], faded[:, :, 2])


def test_backdrop_tracks_a_brightened_pixel(frame):
    image = frame.image.copy()
    before = viewer.backdrop(image)[0, 0, 0]

    image[0, 0] = 255

    assert viewer.backdrop(image)[0, 0, 0] > before


def test_resolve_frame_id_forms():
    frame_ids = ["fixture_0000", "fixture_0001", "fixture_0002"]

    assert viewer.resolve_frame_id(frame_ids, "fixture_0001") == "fixture_0001"
    assert viewer.resolve_frame_id(frame_ids, "0002") == "fixture_0002"
    assert viewer.resolve_frame_id(frame_ids, "000000") == "fixture_0000"
    assert viewer.resolve_frame_id(frame_ids, "1") == "fixture_0001"


def test_resolve_frame_id_rejects_ambiguity():
    frame_ids = ["seq_a__0001", "seq_b__0001"]

    with pytest.raises(ValueError, match="matches 2 frames"):
        viewer.resolve_frame_id(frame_ids, "__0001")


def test_resolve_frame_id_rejects_unknown():
    with pytest.raises(ValueError, match="no frame matching"):
        viewer.resolve_frame_id(["fixture_0000"], "nope")


def test_agreement_needs_both_ground_truths(frame, T_nominal):
    with pytest.raises(ValueError, match="missing one of them"):
        viewer.agreement(_with_labels_3d(frame, None), T_nominal)


def test_render_writes_a_nontrivial_png(frame, nominal, tmp_path):
    out_path = str(tmp_path / "nominal.png")

    viewer.show_or_save(viewer.render(frame, nominal), out_path)

    assert os.path.getsize(out_path) > MIN_PNG_BYTES


def test_render_with_a_reference_panel(frame, T_nominal, nominal, tmp_path):
    """The decalibrated figure carries the nominal number in its panel 4 title,
    so the argument survives being looked at on its own."""
    active = viewer.agreement(frame, viewer.decalibrate(T_nominal, DECALIB_DEG))
    out_path = str(tmp_path / "decalib.png")

    viewer.show_or_save(viewer.render(frame, active, nominal, DECALIB_DEG), out_path)

    assert os.path.getsize(out_path) > MIN_PNG_BYTES
    assert f"{nominal.fraction:.3f}" in viewer._agreement_title(active, nominal)


def test_main_saves_a_figure(tmp_path):
    """End to end through the CLI, on the generated fixture, with no display."""
    out_path = str(tmp_path / "cli.png")

    code = viewer.main(["--dataset", "fixture", "--frame", "000000", "--out", out_path])

    assert code == 0
    assert os.path.getsize(out_path) > MIN_PNG_BYTES


def test_main_accepts_a_decalib(tmp_path):
    out_path = str(tmp_path / "cli_decalib.png")

    code = viewer.main(["--dataset", "fixture", "--frame", "0",
                        "--decalib-pitch-deg", str(DECALIB_DEG), "--out", out_path])

    assert code == 0
    assert os.path.getsize(out_path) > MIN_PNG_BYTES


def test_main_reports_a_missing_calibration(monkeypatch, capsys, tmp_path):
    """Real GOOSE val ships no extrinsic (interface spec 13.3), so the viewer
    has to say which flag fixes it and stop. Never a fabricated default, and
    never a bare traceback out of a viewer."""
    monkeypatch.setattr(viewer, "build_dataset", lambda args: _UncalibratedDataset())

    code = viewer.main(["--dataset", "goose", "--root", str(tmp_path),
                        "--frame", "0", "--out", str(tmp_path / "unused.png")])

    assert code == viewer.EXIT_NO_CALIBRATION
    assert "--calib" in capsys.readouterr().err
    assert not os.path.exists(str(tmp_path / "unused.png"))


class _UncalibratedDataset:
    """A dataset that ships no extrinsic, which is what GooseDataset is without
    --calib. Stubbed here rather than importing the GOOSE adapter, because that
    adapter needs a real val tree on disk to construct at all."""

    def frame_ids(self):
        return ["seq__0000_1"]

    def load(self, frame_id):
        raise AssertionError("nothing should load a frame it cannot project")

    def extrinsic(self, frame_id):
        raise RuntimeError("no calibration found; pass --calib with a 4x4 T_cam_lidar")


def _with_labels_3d(frame, labels_3d):
    """A shallow copy of `frame` carrying different 3D labels. dataclasses.replace
    would do, but naming the intent keeps the perturbation tests readable."""
    import dataclasses

    return dataclasses.replace(frame, labels_3d_gt=labels_3d)


def _first_matched_point(frame, T_cam_lidar) -> int:
    """The index of one point that is scorable AND agrees with its pixel.

    Found rather than assumed, and found through the z-buffer rather than by
    pixel coordinates. Point 0 is whatever azimuth the sweep started at, which
    is behind the camera, and the nearest point sharing a pixel with a matched
    one is not necessarily the point that owns it. Flipping either of those
    would move nothing and the test would pass while asserting nothing.
    """
    height, width = frame.image.shape[:2]
    uv, depth, in_frustum = project(frame.points, frame.K, T_cam_lidar, width, height)
    _owner, visible = zbuffer(uv, depth, in_frustum, width, height)

    label_3d = frame.labels_3d_gt
    label_2d = frame.labels_2d_gt[uv[:, 1], uv[:, 0]]

    matched = (visible & (label_3d != UNLABELED) & (label_2d != UNLABELED)
               & (label_3d == label_2d))
    candidates = np.flatnonzero(matched)

    assert candidates.size >= 1
    return int(candidates[0])
