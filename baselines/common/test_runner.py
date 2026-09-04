"""Tests for how `build_dataset` translates CLI flags into adapter kwargs.

Kept separate from the four bl_*/test_run.py files because those test what a
baseline computes, while this tests only the flag-to-constructor wiring. That
wiring has its own failure mode: three callers in this repo build their own
argparse parsers instead of calling `add_common_args`, so a flag added here is
not present in every namespace that reaches `build_dataset`.

No dataset is touched. `GooseDataset` is replaced by a recorder, because what
is under test is which kwargs arrive, not what the adapter then does with them.
"""

import argparse

import pytest

import semseg.datasets.goose as goose_module
from baselines.common import runner

GOOSE_ROOT = "/nonexistent-root"


class _Recorder:
    """Stands in for a Dataset constructor and keeps the kwargs it was given."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.fixture
def recorded(monkeypatch):
    monkeypatch.setattr(goose_module, "GooseDataset", _Recorder)
    return None


def goose_args(parser, extra=()):
    return parser.parse_args(["--dataset", "goose", "--root", GOOSE_ROOT, *extra])


def test_crop_top_reaches_the_adapter(recorded):
    """The flag is useless unless it arrives as the constructor kwarg: the
    adapter is the only place that knows cy is the entry a crop invalidates."""
    parser = runner.add_common_args(argparse.ArgumentParser())
    dataset = runner.build_dataset(goose_args(parser, ["--crop-top", "536"]))

    assert dataset.kwargs["crop_top"] == 536


def test_crop_top_defaults_to_none(recorded):
    """Omitted must mean "not supplied", which leaves cy published and
    crop_offset_known False, rather than defaulting to a plausible 0 offset
    that would read as a measurement."""
    parser = runner.add_common_args(argparse.ArgumentParser())
    dataset = runner.build_dataset(goose_args(parser))

    assert dataset.kwargs["crop_top"] is None


def test_a_parser_without_crop_top_still_builds(recorded):
    """REGRESSION. eval/sweep.py and tools/viewer.py declare their own flag
    sets and never call add_common_args, so `args.crop_top` is genuinely
    absent there. Reading it directly turned every sweep and every viewer
    launch into an AttributeError."""
    bare = argparse.ArgumentParser()
    bare.add_argument("--dataset")
    bare.add_argument("--root")
    bare.add_argument("--calib")

    args = bare.parse_args(["--dataset", "goose", "--root", GOOSE_ROOT])
    assert not hasattr(args, "crop_top")

    dataset = runner.build_dataset(args)
    assert dataset.kwargs["crop_top"] is None


def test_fixture_branch_is_not_given_crop_top(monkeypatch):
    """The fixture's images ARE the calibrated images by construction, so it
    has no crop to correct and its constructor takes no such kwarg."""
    import semseg.datasets.fixture as fixture_module

    monkeypatch.setattr(fixture_module, "FixtureDataset", _Recorder)
    parser = runner.add_common_args(argparse.ArgumentParser())
    args = parser.parse_args(["--dataset", "fixture", "--crop-top", "536"])

    dataset = runner.build_dataset(args)
    assert "crop_top" not in dataset.kwargs
