"""
Shared timing-format helper for the `logger.trace` instrumentation used
by several wrapper functions in this package (`_primitives.py`,
`_ch.py`). No dependencies -- safe to import from anywhere.
"""


def format_runtime(t0: float, t1: float) -> str:
    """Format an elapsed time with an adaptively-chosen unit (s/ms/us),
    e.g. '[123.456ms]'.
    """
    dt = t1 - t0
    prefix = ""
    if dt < 1e-3:
        prefix = "μ"
        dt *= 1_000_000
    elif dt < 1.0:
        prefix = "m"
        dt *= 1_000
    return f"[{dt:.3f}{prefix}s]"
