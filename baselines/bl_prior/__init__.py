"""bl_prior, tier 0: the chance floor.

A package of its own, alongside the three real arms, because it is not a
courtesy comparison that can live in a test file. Every mIoU in this repo is
reported as a margin over this baseline, so the floor has to be produced by
the same driver, folded by the same Accumulator and scored by the same scorer
as the method it is the floor for. A floor computed a different way is not
comparable to the number it is supposed to bound.
"""
