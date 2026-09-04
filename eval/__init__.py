"""Scoring side of the harness: metrics, submission I/O, the scorer CLI and the
robustness sweeps.

A package rather than a bag of scripts so `eval.metrics` resolves to the same
module whichever directory a caller runs from, and so the tests under
`eval/tests/` import exactly what the scorer imports.
"""
