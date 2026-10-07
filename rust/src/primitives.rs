//! Port of `_intersect_kernels.py`, `_ch.py` and `_simplify.py`.

use rayon::prelude::*;

use crate::geom::{KErr, KResult};

/// `compute_intersections`: every (i1, i2) pair whose closed segments
/// cross (parallel pairs skipped), in row-major (i1, i2) order.
#[allow(clippy::too_many_arguments)]
pub fn compute_intersections(
    s1x: &[f64], s1y: &[f64], e1x: &[f64], e1y: &[f64], s2x: &[f64], s2y: &[f64], e2x: &[f64], e2y: &[f64],
) -> (Vec<usize>, Vec<usize>, Vec<f64>, Vec<f64>) {
    let n1 = s1x.len();
    let n2 = s2x.len();
    let row = |i1: usize| -> Vec<(usize, usize, f64, f64)> {
        let mut out = Vec::new();
        let (x1_1, y1_1, x2_1, y2_1) = (s1x[i1], s1y[i1], e1x[i1], e1y[i1]);
        let ymin = y1_1.min(y2_1);
        let ymax = y1_1.max(y2_1);
        let dx0 = x2_1 - x1_1;
        let dy0 = y2_1 - y1_1;
        for i2 in 0..n2 {
            let (x1_2, y1_2, x2_2, y2_2) = (s2x[i2], s2y[i2], e2x[i2], e2y[i2]);
            if y1_2 < ymin && y2_2 < ymin {
                continue;
            }
            if y1_2 > ymax && y2_2 > ymax {
                continue;
            }
            let dx1 = x2_2 - x1_2;
            let dy1 = y2_2 - y1_2;
            let denom = dx0 * dy1 - dx1 * dy0;
            if denom == 0.0 {
                continue;
            }
            let dxq = x1_2 - x1_1;
            let dyq = y1_2 - y1_1;
            let s = (dxq * dy1 - dyq * dx1) / denom;
            let u = (dxq * dy0 - dyq * dx0) / denom;
            if (0.0..=1.0).contains(&s) && (0.0..=1.0).contains(&u) {
                out.push((i1, i2, x1_1 + s * dx0, y1_1 + s * dy0));
            }
        }
        out
    };
    let rows: Vec<Vec<(usize, usize, f64, f64)>> = if n1.saturating_mul(n2) >= 1 << 18 {
        (0..n1).into_par_iter().map(row).collect()
    } else {
        (0..n1).map(row).collect()
    };
    let total: usize = rows.iter().map(|r| r.len()).sum();
    let mut a = Vec::with_capacity(total);
    let mut b = Vec::with_capacity(total);
    let mut cx = Vec::with_capacity(total);
    let mut cy = Vec::with_capacity(total);
    for r in rows {
        for (i, j, x, y) in r {
            a.push(i);
            b.push(j);
            cx.push(x);
            cy.push(y);
        }
    }
    (a, b, cx, cy)
}

// --------------------------------------------------------------------------
// convex hull (monotone chain, same construction as `_ch._convex_hull`)
// --------------------------------------------------------------------------

fn top_hull(xs: &[f64], ys: &[f64]) -> Vec<usize> {
    let n = xs.len();
    let mut ids: Vec<usize> = (0..n).collect();
    ids.sort_by(|&a, &b| {
        xs[a]
            .partial_cmp(&xs[b])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(ys[a].partial_cmp(&ys[b]).unwrap_or(std::cmp::Ordering::Equal))
    });
    let cross = |i1: usize, i2: usize, i3: usize| {
        (xs[i2] - xs[i1]) * (ys[i3] - ys[i2]) - (ys[i2] - ys[i1]) * (xs[i3] - xs[i2])
    };
    let mut stack: Vec<usize> = vec![ids[0], ids[1]];
    for &idx in &ids[2..] {
        while stack.len() >= 2 && cross(stack[stack.len() - 2], stack[stack.len() - 1], idx) <= 0.0 {
            stack.pop();
        }
        stack.push(idx);
    }
    stack
}

pub fn convex_hull(xs: &[f64], ys: &[f64]) -> Vec<usize> {
    let n = xs.len();
    if n < 3 {
        return (0..n).collect();
    }
    let nx: Vec<f64> = xs.iter().map(|v| -v).collect();
    let ny: Vec<f64> = ys.iter().map(|v| -v).collect();
    let mut ids = top_hull(xs, ys);
    ids.extend(top_hull(&nx, &ny));
    // `_clean_loop`: drop consecutive duplicates (the two chains share
    // their end points), then an explicit closing duplicate.
    ids.dedup();
    if ids.len() > 1 && ids[0] == ids[ids.len() - 1] {
        ids.pop();
    }
    ids
}

// --------------------------------------------------------------------------
// Ramer-Douglas-Peucker + sanitize
// --------------------------------------------------------------------------

#[inline]
fn point_segment_distance(px: f64, py: f64, ax: f64, ay: f64, bx: f64, by: f64) -> f64 {
    let dx = bx - ax;
    let dy = by - ay;
    let seg_len_sq = dx * dx + dy * dy;
    if seg_len_sq < 1e-30 {
        let (ddx, ddy) = (px - ax, py - ay);
        return (ddx * ddx + ddy * ddy).sqrt();
    }
    let mut t = ((px - ax) * dx + (py - ay) * dy) / seg_len_sq;
    if t < 0.0 {
        t = 0.0;
    } else if t > 1.0 {
        t = 1.0;
    }
    let (ddx, ddy) = (px - (ax + t * dx), py - (ay + t * dy));
    (ddx * ddx + ddy * ddy).sqrt()
}

pub fn rdp_mask(xs: &[f64], ys: &[f64], delta: f64) -> Vec<bool> {
    let n = xs.len();
    let mut keep = vec![false; n];
    keep[0] = true;
    keep[n - 1] = true;
    let mut stack: Vec<(usize, usize)> = vec![(0, n - 1)];
    while let Some((i0, i1)) = stack.pop() {
        if i1 - i0 < 2 {
            continue;
        }
        let (ax, ay, bx, by) = (xs[i0], ys[i0], xs[i1], ys[i1]);
        let mut max_dist = -1.0;
        let mut max_idx = 0;
        // zipped slices: no per-element bounds checks in the hot loop
        for (k, (&px, &py)) in xs[i0 + 1..i1].iter().zip(&ys[i0 + 1..i1]).enumerate() {
            let d = point_segment_distance(px, py, ax, ay, bx, by);
            if d > max_dist {
                max_dist = d;
                max_idx = i0 + 1 + k;
            }
        }
        if max_dist > delta {
            keep[max_idx] = true;
            stack.push((i0, max_idx));
            stack.push((max_idx, i1));
        }
    }
    keep
}

#[inline]
fn dist(a: (f64, f64), b: (f64, f64)) -> f64 {
    ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
}

pub fn sanitize_polygon(xs: &[f64], ys: &[f64], tol: f64, closed: bool) -> KResult<(Vec<f64>, Vec<f64>)> {
    if xs.len() != ys.len() {
        return Err(KErr::Value("xs and ys must be 1D arrays of the same length".into()));
    }
    if xs.iter().chain(ys.iter()).any(|v| !v.is_finite()) {
        return Err(KErr::Value("xs/ys contain NaN or infinite values".into()));
    }
    if tol < 0.0 {
        return Err(KErr::Value(format!("tol must be >= 0, got {}", tol)));
    }
    if xs.is_empty() {
        return Err(KErr::Value("empty polygon".into()));
    }
    let mut pts: Vec<(f64, f64)> = xs.iter().copied().zip(ys.iter().copied()).collect();
    if closed && pts.len() > 1 && dist(pts[0], pts[pts.len() - 1]) <= tol {
        pts.pop();
    }
    let mut changed = true;
    while changed && pts.len() > 1 {
        changed = false;
        let mut cleaned: Vec<(f64, f64)> = Vec::with_capacity(pts.len());
        cleaned.push(pts[0]);
        for &p in &pts[1..] {
            if dist(p, *cleaned.last().unwrap()) > tol {
                cleaned.push(p);
            } else {
                changed = true;
            }
        }
        if closed && cleaned.len() > 1 && dist(cleaned[0], cleaned[cleaned.len() - 1]) <= tol {
            cleaned.pop();
            changed = true;
        }
        pts = cleaned;
    }
    let min_pts = if closed { 3 } else { 2 };
    if pts.len() < min_pts {
        return Err(KErr::Value(format!(
            "Polygon degenerated to {} distinct point(s) after sanitizing (tol={}); not enough left for a valid {}.",
            pts.len(),
            tol,
            if closed { "ring" } else { "polyline" }
        )));
    }
    if closed {
        pts.push(pts[0]);
    }
    Ok(pts.into_iter().unzip())
}

pub fn simplify_polyline(xs: &[f64], ys: &[f64], delta: f64) -> KResult<(Vec<f64>, Vec<f64>)> {
    if xs.len() != ys.len() {
        return Err(KErr::Value(format!(
            "xs and ys must be the same shape, got ({},) vs ({},)",
            xs.len(),
            ys.len()
        )));
    }
    if delta < 0.0 {
        return Err(KErr::Value(format!("delta must be >= 0, got {}", delta)));
    }
    if xs.len() < 3 {
        return Ok((xs.to_vec(), ys.to_vec()));
    }
    let keep = rdp_mask(xs, ys, delta);
    let kx: Vec<f64> = xs.iter().zip(&keep).filter(|(_, &k)| k).map(|(&v, _)| v).collect();
    let ky: Vec<f64> = ys.iter().zip(&keep).filter(|(_, &k)| k).map(|(&v, _)| v).collect();
    sanitize_polygon(&kx, &ky, 1e-9, false)
}

// --------------------------------------------------------------------------
// dezigzag
// --------------------------------------------------------------------------

fn angle_between_deg(d1: (f64, f64), d2: (f64, f64)) -> f64 {
    let len1 = d1.0.hypot(d1.1);
    let len2 = d2.0.hypot(d2.1);
    if len1 < 1e-300 || len2 < 1e-300 {
        return 180.0;
    }
    let cos_a = ((d1.0 * d2.0 + d1.1 * d2.1) / (len1 * len2)).clamp(-1.0, 1.0);
    cos_a.acos().to_degrees()
}

fn line_intersect(a1: (f64, f64), a2: (f64, f64), b1: (f64, f64), b2: (f64, f64)) -> Option<(f64, f64)> {
    let (x1, y1) = a1;
    let (x2, y2) = a2;
    let (x3, y3) = b1;
    let (x4, y4) = b2;
    let d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
    if d.abs() < 1e-12 {
        return None;
    }
    let t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / d;
    Some((x1 + t * (x2 - x1), y1 + t * (y2 - y1)))
}

pub fn dezigzag_polyline(
    xs: &[f64], ys: &[f64], max_kink_length: f64, max_angle_deg: f64, min_neighbor_factor: f64,
) -> KResult<(Vec<f64>, Vec<f64>)> {
    if xs.len() != ys.len() {
        return Err(KErr::Value(format!(
            "xs and ys must be the same shape, got ({},) vs ({},)",
            xs.len(),
            ys.len()
        )));
    }
    if max_kink_length < 0.0 {
        return Err(KErr::Value(format!("max_kink_length must be >= 0, got {}", max_kink_length)));
    }
    if xs.len() < 5 {
        return Ok((xs.to_vec(), ys.to_vec()));
    }
    let mut pts: Vec<(f64, f64)> = xs.iter().copied().zip(ys.iter().copied()).collect();
    let min_neighbor = max_kink_length * min_neighbor_factor;
    let mut changed = true;
    while changed && pts.len() >= 5 {
        changed = false;
        let n = pts.len();
        let mut out: Vec<(f64, f64)> = Vec::with_capacity(n);
        out.push(pts[0]);
        let mut i = 1;
        while i < n - 2 {
            let p_prev = *out.last().unwrap();
            let (p_i, p_next, p_next2) = (pts[i], pts[i + 1], pts[i + 2]);
            let kink_len = (p_next.0 - p_i.0).hypot(p_next.1 - p_i.1);
            if 0.0 < kink_len && kink_len <= max_kink_length {
                let d1 = (p_i.0 - p_prev.0, p_i.1 - p_prev.1);
                let d2 = (p_next2.0 - p_next.0, p_next2.1 - p_next.1);
                let len1 = d1.0.hypot(d1.1);
                let len2 = d2.0.hypot(d2.1);
                if len1 >= min_neighbor && len2 >= min_neighbor && angle_between_deg(d1, d2) <= max_angle_deg {
                    let midpoint = ((p_i.0 + p_next.0) / 2.0, (p_i.1 + p_next.1) / 2.0);
                    let ipt = match line_intersect(p_prev, p_i, p_next, p_next2) {
                        Some(p) if (p.0 - p_i.0).hypot(p.1 - p_i.1) <= 2.0 * len1.max(len2) => p,
                        _ => midpoint,
                    };
                    out.push(ipt);
                    i += 2;
                    changed = true;
                    continue;
                }
            }
            out.push(p_i);
            i += 1;
        }
        out.extend_from_slice(&pts[i..]);
        pts = out;
    }
    Ok(pts.into_iter().unzip())
}
