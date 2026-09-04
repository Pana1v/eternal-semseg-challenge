"""bl_cam2d, the camera-only arm of the section 6.2 ablation.

Deliberately empty of imports, like the other bl_* packages. baselines.common
must stay importable by the harness without dragging in a fitted classifier,
and importing this package is not what registers the baseline: importing
run.py is (see baselines.common.base.load).
"""
