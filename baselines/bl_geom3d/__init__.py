"""The LiDAR-only arm of the section 6.2 ablation.

Deliberately empty of imports, exactly as baselines/__init__.py is. Importing
the class here would make `import baselines.bl_geom3d` pull in scipy's KD-tree
and the whole projection stack, and baselines/common/base.py's registry is
populated by importing run.py, not this file.
"""
