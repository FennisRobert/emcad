//! Via proximity graph kernels for `viaconnect.py` (`_proximity_edges`,
//! `_prune_redundant_edges`), backed by a uniform grid instead of an
//! all-pairs scan. Output order matches an (i, j)-nested loop.

use std::collections::HashMap;

use rayon::prelude::*;

struct Grid {
    cell: f64,
    buckets: HashMap<(i64, i64), Vec<usize>>,
}

impl Grid {
    fn new(xs: &[f64], ys: &[f64], cell: f64) -> Self {
        let mut buckets: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
        for i in 0..xs.len() {
            buckets.entry(Self::key(xs[i], ys[i], cell)).or_default().push(i);
        }
        Grid { cell, buckets }
    }
    #[inline]
    fn key(x: f64, y: f64, cell: f64) -> (i64, i64) {
        ((x / cell).floor() as i64, (y / cell).floor() as i64)
    }
    /// every point in the 3x3 cell block around (x, y)
    fn around(&self, x: f64, y: f64, mut f: impl FnMut(usize)) {
        let (cx, cy) = Self::key(x, y, self.cell);
        for dx in -1..=1 {
            for dy in -1..=1 {
                if let Some(b) = self.buckets.get(&(cx + dx, cy + dy)) {
                    for &k in b {
                        f(k);
                    }
                }
            }
        }
    }
}

/// All pairs (i < j) with distance <= max_dist, sorted by (i, j).
pub fn proximity_edges(xs: &[f64], ys: &[f64], max_dist: f64) -> (Vec<usize>, Vec<usize>) {
    let n = xs.len();
    let max_d2 = max_dist * max_dist;
    let mut pairs: Vec<(usize, usize)> = if max_dist > 0.0 && max_dist.is_finite() {
        let grid = Grid::new(xs, ys, max_dist * (1.0 + 1e-9));
        (0..n)
            .into_par_iter()
            .flat_map_iter(|i| {
                let mut row = Vec::new();
                grid.around(xs[i], ys[i], |j| {
                    if j > i {
                        let (dx, dy) = (xs[j] - xs[i], ys[j] - ys[i]);
                        if dx * dx + dy * dy <= max_d2 {
                            row.push((i, j));
                        }
                    }
                });
                row.sort_unstable();
                row
            })
            .collect()
    } else {
        let mut v = Vec::new();
        for i in 0..n {
            for j in i + 1..n {
                let (dx, dy) = (xs[j] - xs[i], ys[j] - ys[i]);
                if dx * dx + dy * dy <= max_d2 {
                    v.push((i, j));
                }
            }
        }
        v
    };
    pairs.sort_unstable();
    pairs.into_iter().unzip()
}

/// Relative-neighborhood pruning: drop (i, j) if some k is within
/// max_dist of both and strictly closer to each than they are to each
/// other. Any such k lies within max_dist of i, so only i's grid
/// neighborhood needs scanning.
pub fn prune_redundant_edges(xs: &[f64], ys: &[f64], ei: &[usize], ej: &[usize], max_dist: f64) -> Vec<bool> {
    let max_d2 = max_dist * max_dist;
    let test = |e: usize, k: usize| -> bool {
        let (i, j) = (ei[e], ej[e]);
        if k == i || k == j {
            return false;
        }
        let dij2 = (xs[j] - xs[i]).powi(2) + (ys[j] - ys[i]).powi(2);
        let dik2 = (xs[k] - xs[i]).powi(2) + (ys[k] - ys[i]).powi(2);
        if dik2 > max_d2 || dik2 >= dij2 {
            return false;
        }
        let djk2 = (xs[k] - xs[j]).powi(2) + (ys[k] - ys[j]).powi(2);
        !(djk2 > max_d2 || djk2 >= dij2)
    };
    if max_dist > 0.0 && max_dist.is_finite() {
        let grid = Grid::new(xs, ys, max_dist * (1.0 + 1e-9));
        (0..ei.len())
            .into_par_iter()
            .map(|e| {
                let mut redundant = false;
                grid.around(xs[ei[e]], ys[ei[e]], |k| {
                    if !redundant && test(e, k) {
                        redundant = true;
                    }
                });
                !redundant
            })
            .collect()
    } else {
        (0..ei.len()).map(|e| !(0..xs.len()).any(|k| test(e, k))).collect()
    }
}
