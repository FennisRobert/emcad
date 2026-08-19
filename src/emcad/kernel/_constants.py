"""
Shared numerical tolerances for fragmentation, the boolean ops, and
join. One source of truth so `_fragment.py`, `_boolean_ops.py`, and
`_join.py` don't each carry their own (potentially drifting) copy.

No dependencies -- safe to import from anywhere, including `api.py`'s
module top level, without any circular-import risk.
"""

DEFAULT_MERGE_TOL = 1e-7   # absolute distance below which two points merge
DEFAULT_T_TOL = 1e-7       # parametric (0..1) tolerance for "at the endpoint"
DEFAULT_AREA_TOL = 1e-10   # faces with |signed area| below this are noise
