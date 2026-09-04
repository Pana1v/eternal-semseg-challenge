"""Baseline package root.

Deliberately empty of imports. `baselines.common` must stay importable by the
harness (eval/sweep.py resolves --baseline through baselines.common.base)
without pulling in four run.py files and their dataset dependencies.
"""
