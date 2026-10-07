//! Exact grid-integer predicates and the split finder -- port of
//! `kernel/_exact.py` + `kernel/_arrangement_kernels.find_all_splits_i64`.
//!
//! The split finder replaces the original all-pairs O(E^2) loop with an
//! x-sorted sweep (only pairs whose x-ranges overlap are ever visited),
//! run across threads for large inputs. Results are re-sorted into
//! exactly the order the original (i, j)-nested loop produced them, so
//! everything downstream (vertex numbering, face order, output order)
//! is bit-identical to the original (numba-era) implementation.

use rayon::prelude::*;

#[inline(always)]
pub fn orient(ax: i64, ay: i64, bx: i64, by: i64, cx: i64, cy: i64) -> i64 {
    (bx.wrapping_sub(ax))
        .wrapping_mul(cy.wrapping_sub(ay))
        .wrapping_sub((by.wrapping_sub(ay)).wrapping_mul(cx.wrapping_sub(ax)))
}

#[inline(always)]
pub fn on_segment(ax: i64, ay: i64, bx: i64, by: i64, px: i64, py: i64) -> bool {
    ax.min(bx) <= px && px <= ax.max(bx) && ay.min(by) <= py && py <= ay.max(by)
}

#[inline(always)]
pub fn strictly_interior(ax: i64, ay: i64, bx: i64, by: i64, px: i64, py: i64) -> bool {
    if orient(ax, ay, bx, by, px, py) != 0 {
        return false;
    }
    if px == ax && py == ay {
        return false;
    }
    if px == bx && py == by {
        return false;
    }
    on_segment(ax, ay, bx, by, px, py)
}

/// `classify_segment_pair_i64`, reduced to what the split finder needs:
/// `Some((x, y))` for a proper transversal crossing (kind 1), `None`
/// otherwise (no relation, touch, T-junction, or collinear overlap --
/// none of which the caller uses the coordinates of).
#[inline(always)]
pub fn proper_crossing(
    ax: i64, ay: i64, bx: i64, by: i64, cx: i64, cy: i64, dx: i64, dy: i64,
) -> Option<(i64, i64)> {
    let d1 = orient(cx, cy, dx, dy, ax, ay);
    let d2 = orient(cx, cy, dx, dy, bx, by);
    let d3 = orient(ax, ay, bx, by, cx, cy);
    let d4 = orient(ax, ay, bx, by, dx, dy);

    if d1 == 0 && d2 == 0 && d3 == 0 && d4 == 0 {
        return None;
    }
    if d3 == 0 && on_segment(ax, ay, bx, by, cx, cy) {
        return None;
    }
    if d4 == 0 && on_segment(ax, ay, bx, by, dx, dy) {
        return None;
    }
    if d1 == 0 && on_segment(cx, cy, dx, dy, ax, ay) {
        return None;
    }
    if d2 == 0 && on_segment(cx, cy, dx, dy, bx, by) {
        return None;
    }
    if ((d1 > 0) != (d2 > 0)) && ((d3 > 0) != (d4 > 0)) {
        let denom = (bx - ax)
            .wrapping_mul(dy - cy)
            .wrapping_sub((by - ay).wrapping_mul(dx - cx));
        let t_num = (cx - ax)
            .wrapping_mul(dy - cy)
            .wrapping_sub((cy - ay).wrapping_mul(dx - cx));
        let t = t_num as f64 / denom as f64;
        let x = (ax as f64 + t * (bx - ax) as f64).round_ties_even() as i64;
        let y = (ay as f64 + t * (by - ay) as f64).round_ties_even() as i64;
        return Some((x, y));
    }
    None
}

/// One split point: edge `edge` needs grid point (x, y) inserted. The
/// remaining fields reproduce the original discovery order: pair (i, j)
/// with i < j, then `slot` (0..=3 interior checks, 4/5 crossing).
#[derive(Clone, Copy, Debug)]
pub struct Split {
    pub edge: u32,
    pub x: i64,
    pub y: i64,
    pub i: u32,
    pub j: u32,
    pub slot: u8,
}

#[inline(always)]
fn pair_splits(
    i: usize, j: usize, ex0: &[i64], ey0: &[i64], ex1: &[i64], ey1: &[i64], out: &mut Vec<Split>,
) {
    let (ax, ay, bx, by) = (ex0[i], ey0[i], ex1[i], ey1[i]);
    let (cx, cy, dx, dy) = (ex0[j], ey0[j], ex1[j], ey1[j]);
    if ax.max(bx) < cx.min(dx) || ax.min(bx) > cx.max(dx) {
        return;
    }
    if ay.max(by) < cy.min(dy) || ay.min(by) > cy.max(dy) {
        return;
    }
    let (iu, ju) = (i as u32, j as u32);
    let mk = |edge: u32, x: i64, y: i64, slot: u8| Split { edge, x, y, i: iu, j: ju, slot };
    if strictly_interior(ax, ay, bx, by, cx, cy) {
        out.push(mk(iu, cx, cy, 0));
    }
    if strictly_interior(ax, ay, bx, by, dx, dy) {
        out.push(mk(iu, dx, dy, 1));
    }
    if strictly_interior(cx, cy, dx, dy, ax, ay) {
        out.push(mk(ju, ax, ay, 2));
    }
    if strictly_interior(cx, cy, dx, dy, bx, by) {
        out.push(mk(ju, bx, by, 3));
    }
    if let Some((x, y)) = proper_crossing(ax, ay, bx, by, cx, cy, dx, dy) {
        out.push(mk(iu, x, y, 4));
        out.push(mk(ju, x, y, 5));
    }
}

const PAR_THRESHOLD: usize = 2048;

/// Every split point every edge needs (see module docs), in the same
/// order `find_all_splits_i64`'s nested loop emits them.
pub fn find_all_splits(ex0: &[i64], ey0: &[i64], ex1: &[i64], ey1: &[i64]) -> Vec<Split> {
    let e = ex0.len();
    if e < 2 {
        return Vec::new();
    }
    let xmin: Vec<i64> = (0..e).map(|k| ex0[k].min(ex1[k])).collect();
    let xmax: Vec<i64> = (0..e).map(|k| ex0[k].max(ex1[k])).collect();
    let mut order: Vec<u32> = (0..e as u32).collect();
    order.sort_unstable_by_key(|&k| (xmin[k as usize], k));

    let scan = |p: usize, out: &mut Vec<Split>| {
        let a = order[p] as usize;
        let ax_hi = xmax[a];
        for &bq in &order[p + 1..] {
            let b = bq as usize;
            if xmin[b] > ax_hi {
                break;
            }
            let (i, j) = if a < b { (a, b) } else { (b, a) };
            pair_splits(i, j, ex0, ey0, ex1, ey1, out);
        }
    };

    let mut splits: Vec<Split> = if e >= PAR_THRESHOLD {
        (0..e)
            .into_par_iter()
            .fold(Vec::new, |mut acc, p| {
                scan(p, &mut acc);
                acc
            })
            .reduce(Vec::new, |mut a, mut b| {
                a.append(&mut b);
                a
            })
    } else {
        let mut acc = Vec::new();
        for p in 0..e {
            scan(p, &mut acc);
        }
        acc
    };
    splits.sort_unstable_by_key(|s| (s.i, s.j, s.slot));
    splits
}

/// `to_grid`: np.round (half-to-even) of x * scale.
#[inline(always)]
pub fn to_grid(v: f64, scale: f64) -> i64 {
    (v * scale).round_ties_even() as i64
}
