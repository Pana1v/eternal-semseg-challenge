"""Everything the four baselines share: the contract (base), the one
classifier (nb), the feature extractors (features) and the driver (runner).

No re-exports here on purpose. A baseline imports the module it needs, so a
reader of bl_paint/run.py can see that it uses geom_features and rgb_features
and nothing else.
"""
