"""
Numba edge-intersection kernel. Pure array in/out, no dependency on
anything else in the package -- `_primitives.py` is the only thing
that should call into this file.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def compute_intersections(ps1s, ps1e, ps2s, ps2e):
    # TODO: Update to the sweep line algorithm
    xss1 = ps1s[0, :]
    yss1 = ps1s[1, :]
    xse1 = ps1e[0, :]
    yse1 = ps1e[1, :]
    xss2 = ps2s[0, :]
    yss2 = ps2s[1, :]
    xse2 = ps2e[0, :]
    yse2 = ps2e[1, :]

    n1 = xss1.shape[0]
    n2 = xss2.shape[0]

    idspair = np.empty((2, (n1 * n2) // 2 + 1), dtype=np.int64)
    coords = np.empty((2, (n1 * n2) // 2 + 1), dtype=np.float64)
    ctr = 0

    for i1 in range(n1):
        x1_1 = xss1[i1]
        y1_1 = yss1[i1]
        x2_1 = xse1[i1]
        y2_1 = yse1[i1]
        ymin = min(y1_1, y2_1)
        ymax = max(y1_1, y2_1)
        dx0 = x2_1 - x1_1
        dy0 = y2_1 - y1_1

        for i2 in range(n2):
            x1_2 = xss2[i2]
            y1_2 = yss2[i2]
            x2_2 = xse2[i2]
            y2_2 = yse2[i2]
            # cannot intersect
            if (y1_2 < ymin) & (y2_2 < ymin):
                continue
            if (y1_2 > ymax) & (y2_2 > ymax):
                continue

            dx1 = x2_2 - x1_2
            dy1 = y2_2 - y1_2

            denom = dx0 * dy1 - dx1 * dy0
            if denom == 0.0:
                continue

            dxq = x1_2 - x1_1
            dyq = y1_2 - y1_1

            s = (dxq * dy1 - dyq * dx1) / denom   # param along segment 1
            u = (dxq * dy0 - dyq * dx0) / denom   # param along segment 2

            if 0.0 <= s <= 1.0 and 0.0 <= u <= 1.0:
                idspair[0, ctr] = i1
                idspair[1, ctr] = i2
                coords[0, ctr] = x1_1 + s * dx0
                coords[1, ctr] = y1_1 + s * dy0
                ctr += 1

    return idspair[:, :ctr], coords[:, :ctr]
