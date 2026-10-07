//! Map-making style outline regularization (building-footprint
//! regularization) of one closed ring -- see `kernel.api.regularize_polyline`
//! for the contract.
//!
//!   1. anchors -- edges whose direction is (within `angle_tol`) a
//!      multiple of `dangle`; an edge counts once the total length of all
//!      collinear candidates on its snapped line (same direction index,
//!      offsets within `offset_tol`) reaches `min_anchor_len`.
//!   2. merge   -- consecutive anchors on the same snapped line fuse when
//!      every vertex between them lies within `tol` of the line and the
//!      walk progresses forward along it.
//!   3. corners -- consecutive non-parallel line groups are joined at their
//!      intersection when the detour between them stays within `tol` of the
//!      resulting two-segment corner (and the intersection lies inside the
//!      detour's bbox grown by `tol`).
//!   4. the rest -- Visvalingam-Whyatt (area threshold `vw_area`) with its
//!      two end points projected exactly onto the neighbouring lines.

use crate::geom::{KErr, KResult};

type Pt = (f64, f64);

pub struct Params {
    pub tol: f64,
    pub dangle_deg: f64,
    pub angle_tol_deg: f64,
    pub min_anchor_len: f64,
    pub offset_tol: f64,
    pub vw_area: f64,
}

#[derive(Clone, Copy)]
struct Line {
    k: i64,
    d: Pt,
    c: f64,
}

/// Exact 0 / +-1 for direction components on the axes (cos(pi/2) is
/// 6e-17, not 0) -- keeps axis-aligned output edges exactly axis-aligned,
/// with corner coordinates bit-identical to the input lines.
#[inline]
fn snap_unit(v: f64) -> f64 {
    if v.abs() < 1e-12 {
        0.0
    } else if (v.abs() - 1.0).abs() < 1e-12 {
        v.signum()
    } else {
        v
    }
}

#[inline]
fn dist_to_line(p: Pt, l: &Line) -> f64 {
    (-l.d.1 * p.0 + l.d.0 * p.1 - l.c).abs()
}

#[inline]
fn proj(p: Pt, l: &Line) -> Pt {
    let (nx, ny) = (-l.d.1, l.d.0);
    let s = nx * p.0 + ny * p.1 - l.c;
    (p.0 - s * nx, p.1 - s * ny)
}

fn intersect(a: &Line, b: &Line) -> Option<Pt> {
    let n1 = (-a.d.1, a.d.0);
    let n2 = (-b.d.1, b.d.0);
    let det = n1.0 * n2.1 - n1.1 * n2.0;
    if det.abs() < 1e-9 {
        return None;
    }
    let mut x = (a.c * n2.1 - n1.1 * b.c) / det;
    let mut y = (n1.0 * b.c - a.c * n2.0) / det;
    // a corner on an axis-aligned line takes that line's coordinate
    // exactly, so the edge stays bit-exact horizontal/vertical
    for l in [a, b] {
        if l.d.1 == 0.0 {
            y = l.c * l.d.0; // horizontal: n = (0, d.0), n.p = c
        } else if l.d.0 == 0.0 {
            x = -l.c * l.d.1; // vertical: n = (-d.1, 0), n.p = c
        }
    }
    Some((x, y))
}

fn seg_dist(p: Pt, a: Pt, b: Pt) -> f64 {
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    let l2 = dx * dx + dy * dy;
    let t = if l2 == 0.0 { 0.0 } else { (((p.0 - a.0) * dx + (p.1 - a.1) * dy) / l2).clamp(0.0, 1.0) };
    (p.0 - (a.0 + t * dx)).hypot(p.1 - (a.1 + t * dy))
}

/// Visvalingam-Whyatt on an open polyline, end points fixed.
fn vw(mut pts: Vec<Pt>, area_min: f64) -> Vec<Pt> {
    if pts.len() <= 2 || area_min <= 0.0 {
        return pts;
    }
    let tri = |a: Pt, b: Pt, c: Pt| ((b.0 - a.0) * (c.1 - a.1) - (c.0 - a.0) * (b.1 - a.1)).abs() / 2.0;
    while pts.len() > 2 {
        let mut best = 1usize;
        let mut best_a = f64::INFINITY;
        for i in 1..pts.len() - 1 {
            let a = tri(pts[i - 1], pts[i], pts[i + 1]);
            if a < best_a {
                best_a = a;
                best = i;
            }
        }
        if best_a >= area_min {
            break;
        }
        pts.remove(best);
    }
    pts
}

struct Anchor {
    edge: usize,
    line: Line,
    len: f64,
}

struct Group {
    line: Line,
    w: f64,
    first_edge: usize,
    last_edge: usize,
}

/// Regularize one ring. Accepts open or closed (first == last) input
/// and returns the same convention.
pub fn regularize_ring(xs: &[f64], ys: &[f64], prm: &Params) -> KResult<(Vec<f64>, Vec<f64>)> {
    if xs.len() != ys.len() {
        return Err(KErr::Value(format!(
            "xs and ys must be the same shape, got ({},) vs ({},)",
            xs.len(),
            ys.len()
        )));
    }
    if !(prm.tol > 0.0) {
        return Err(KErr::Value(format!("tol must be > 0, got {}", prm.tol)));
    }
    if !(prm.dangle_deg > 0.0 && prm.dangle_deg <= 180.0) {
        return Err(KErr::Value(format!("dangle_deg must be in (0, 180], got {}", prm.dangle_deg)));
    }
    let closed_in = xs.len() > 1 && xs[0] == xs[xs.len() - 1] && ys[0] == ys[ys.len() - 1];
    let m = if closed_in { xs.len() - 1 } else { xs.len() };
    let p: Vec<Pt> = (0..m).map(|i| (xs[i], ys[i])).collect();
    let n = p.len();
    let finish = |out: Vec<Pt>| -> (Vec<f64>, Vec<f64>) {
        let (mut ox, mut oy): (Vec<f64>, Vec<f64>) = out.into_iter().unzip();
        if closed_in && !ox.is_empty() {
            ox.push(ox[0]);
            oy.push(oy[0]);
        }
        (ox, oy)
    };
    if n < 3 {
        return Ok(finish(p));
    }

    // -- 1. candidates + support-based anchors --
    let step = prm.dangle_deg.to_radians();
    let atol = prm.angle_tol_deg.to_radians();
    let n_dirs = (2.0 * std::f64::consts::PI / step).round() as i64;
    let mut cands: Vec<Anchor> = Vec::new();
    for i in 0..n {
        let (a, b) = (p[i], p[(i + 1) % n]);
        let len = (b.0 - a.0).hypot(b.1 - a.1);
        if len <= prm.offset_tol {
            continue;
        }
        let ang = (b.1 - a.1).atan2(b.0 - a.0);
        let k = (ang / step).round();
        if (ang - k * step).abs() > atol {
            continue;
        }
        let th = k * step;
        let d = (snap_unit(th.cos()), snap_unit(th.sin()));
        let c = 0.5 * ((-d.1 * a.0 + d.0 * a.1) + (-d.1 * b.0 + d.0 * b.1));
        cands.push(Anchor { edge: i, line: Line { k: (k as i64).rem_euclid(n_dirs), d, c }, len });
    }
    // total collinear support per candidate: sort by (k, c), sweep runs
    // whose consecutive offsets differ by <= offset_tol
    let mut order: Vec<usize> = (0..cands.len()).collect();
    order.sort_by(|&x, &y| {
        cands[x].line.k.cmp(&cands[y].line.k).then(cands[x].line.c.partial_cmp(&cands[y].line.c).unwrap())
    });
    let mut support = vec![0.0f64; cands.len()];
    let mut s = 0;
    while s < order.len() {
        let mut e = s + 1;
        while e < order.len()
            && cands[order[e]].line.k == cands[order[s]].line.k
            && cands[order[e]].line.c - cands[order[e - 1]].line.c <= prm.offset_tol
        {
            e += 1;
        }
        let total: f64 = order[s..e].iter().map(|&ci| cands[ci].len).sum();
        for &ci in &order[s..e] {
            support[ci] = total;
        }
        s = e;
    }
    let anchors: Vec<&Anchor> = cands.iter().zip(&support).filter(|(_, &sp)| sp >= prm.min_anchor_len).map(|(a, _)| a).collect();
    if anchors.is_empty() {
        let mut ring = p.clone();
        ring.push(p[0]);
        let mut out = vw(ring, prm.vw_area);
        out.pop();
        return Ok(finish(if out.len() >= 3 { out } else { p }));
    }

    // vertices strictly between end vertex a and start vertex b (cyclic)
    let between = |a: usize, b: usize| -> Vec<usize> {
        let mut out = Vec::new();
        if a == b {
            return out;
        }
        let mut v = (a + 1) % n;
        while v != b {
            out.push(v);
            v = (v + 1) % n;
        }
        out
    };
    let mergeable = |g: &Group, a: &Anchor| -> bool {
        if a.line.k != g.line.k || (a.line.c - g.line.c).abs() > prm.offset_tol {
            return false;
        }
        let end_v = (g.last_edge + 1) % n;
        if between(end_v, a.edge).iter().any(|&v| dist_to_line(p[v], &g.line) > prm.tol) {
            return false;
        }
        let t0 = g.line.d.0 * p[end_v].0 + g.line.d.1 * p[end_v].1;
        let t1 = g.line.d.0 * p[a.edge].0 + g.line.d.1 * p[a.edge].1;
        t1 >= t0 - prm.offset_tol
    };
    let single = |a: &Anchor| Group { line: a.line, w: a.len, first_edge: a.edge, last_edge: a.edge };

    // -- 2. group consecutive anchors on the same line --
    let m_a = anchors.len();
    let mut start = 0;
    for s in 0..m_a {
        let prev = anchors[(s + m_a - 1) % m_a];
        if m_a == 1 || !mergeable(&single(prev), anchors[s]) {
            start = s;
            break;
        }
    }
    let mut groups: Vec<Group> = Vec::new();
    for t in 0..m_a {
        let a = anchors[(start + t) % m_a];
        if let Some(g) = groups.last_mut() {
            if mergeable(g, a) {
                // incremental length-weighted mean: equal offsets stay
                // bit-identical (no (c*w1 + c*w2)/(w1+w2) rounding)
                g.w += a.len;
                g.line.c += (a.line.c - g.line.c) * (a.len / g.w);
                g.last_edge = a.edge;
                continue;
            }
        }
        groups.push(single(a));
    }

    // -- 3/4. junctions between consecutive groups --
    let n_g = groups.len();
    let mut out: Vec<Pt> = Vec::new();
    for gi in 0..n_g {
        let (g, h) = (&groups[gi], &groups[(gi + 1) % n_g]);
        let end_v = (g.last_edge + 1) % n;
        let start_v = h.first_edge;
        let run: Vec<Pt> = between(end_v, start_v).into_iter().map(|v| p[v]).collect();
        let pe = proj(p[end_v], &g.line);
        let ps = proj(p[start_v], &h.line);
        if n_g > 1 {
            if let Some(x) = intersect(&g.line, &h.line) {
                let mut pts = Vec::with_capacity(run.len() + 2);
                pts.push(p[end_v]);
                pts.extend_from_slice(&run);
                pts.push(p[start_v]);
                let (mut bx0, mut bx1, mut by0, mut by1) = (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
                for q in &pts {
                    bx0 = bx0.min(q.0);
                    bx1 = bx1.max(q.0);
                    by0 = by0.min(q.1);
                    by1 = by1.max(q.1);
                }
                let in_box = bx0 - prm.tol <= x.0 && x.0 <= bx1 + prm.tol && by0 - prm.tol <= x.1 && x.1 <= by1 + prm.tol;
                if in_box && pts.iter().all(|&q| seg_dist(q, pe, x).min(seg_dist(q, x, ps)) <= prm.tol) {
                    out.push(x);
                    continue;
                }
            }
        }
        let mut chain = Vec::with_capacity(run.len() + 2);
        chain.push(pe);
        chain.extend(run);
        chain.push(ps);
        out.extend(vw(chain, prm.vw_area));
    }

    // drop exact duplicates
    let mut clean: Vec<Pt> = Vec::with_capacity(out.len());
    for q in out {
        if clean.last().map_or(true, |l: &Pt| (q.0 - l.0).hypot(q.1 - l.1) > 1e-9) {
            clean.push(q);
        }
    }
    if clean.len() > 1 {
        let (f, l) = (clean[0], clean[clean.len() - 1]);
        if (f.0 - l.0).hypot(f.1 - l.1) <= 1e-9 {
            clean.pop();
        }
    }
    if clean.len() < 3 {
        return Ok(finish(p));
    }
    Ok(finish(clean))
}
