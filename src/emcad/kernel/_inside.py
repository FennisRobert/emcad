from numba import njit, f8, bool
import numpy as np

@njit(cache=True)
def _is_inside(xs: np.ndarray, ys: np.ndarray, x: float, y: float, include_boundary: bool):
    n = len(xs)
    inside = False

    x0, y0 = xs[-1], ys[-1]
    for i in range(n):
        x1, y1 = xs[i], ys[i]

        # boundary check: is (x, y) exactly on segment (x0,y0)-(x1,y1)?
        dx, dy = x1 - x0, y1 - y0
        cross = (x - x0) * dy - (y - y0) * dx
        if cross == 0.0:
            dot = (x - x0) * dx + (y - y0) * dy
            if 0.0 <= dot <= dx * dx + dy * dy:
                return include_boundary

        # standard ray-casting crossing test (ray shoots in +x direction)
        if (y0 > y) != (y1 > y):
            x_cross = x0 + (y - y0) / (y1 - y0) * (x1 - x0)
            if x_cross > x:
                inside = not inside

        x0, y0 = x1, y1

    return inside
