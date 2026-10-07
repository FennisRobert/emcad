//! Port of `_cluster.py`, `_boolean_ops.py` and `_join.py`.

use crate::arrangement::{build_arrangement, label_and_assemble, Ring};
use crate::geom::{KErr, KResult, PolyTree};
use crate::halfedge::{group_in_order, UnionFind};

// --------------------------------------------------------------------------
// bbox clustering
// --------------------------------------------------------------------------

fn cluster_by_bbox_overlap(xmin: &[f64], xmax: &[f64], ymin: &[f64], ymax: &[f64], pad: f64) -> Vec<Vec<usize>> {
    let n = xmin.len();
    if n == 0 {
        return Vec::new();
    }
    let mut uf = UnionFind::new(n);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| xmin[a].partial_cmp(&xmin[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut active: Vec<usize> = Vec::new();
    for &i in &order {
        let xi = xmin[i];
        active.retain(|&j| xmax[j] + pad >= xi);
        let (yi0, yi1) = (ymin[i] - pad, ymax[i] + pad);
        for &j in &active {
            if yi0 <= ymax[j] && ymin[j] <= yi1 {
                uf.union(i, j);
            }
        }
        active.push(i);
    }
    let roots: Vec<usize> = (0..n).map(|i| uf.find(i)).collect();
    group_in_order(&roots)
}

const MAX_HULL_ROUNDS: usize = 64;

pub fn cluster_operands(rings: &[Ring], ring_operand: &[u32], n_operands: usize, merge_tol: f64) -> Vec<Vec<usize>> {
    if n_operands <= 1 {
        return if n_operands == 1 { vec![vec![0]] } else { vec![] };
    }
    let mut xmin = vec![f64::INFINITY; n_operands];
    let mut xmax = vec![f64::NEG_INFINITY; n_operands];
    let mut ymin = vec![f64::INFINITY; n_operands];
    let mut ymax = vec![f64::NEG_INFINITY; n_operands];
    for ((xs, ys), &op) in rings.iter().zip(ring_operand) {
        let op = op as usize;
        for &x in xs.iter() {
            xmin[op] = xmin[op].min(x);
            xmax[op] = xmax[op].max(x);
        }
        for &y in ys.iter() {
            ymin[op] = ymin[op].min(y);
            ymax[op] = ymax[op].max(y);
        }
    }
    let mut groups: Vec<Vec<usize>> = (0..n_operands).map(|i| vec![i]).collect();
    for _ in 0..MAX_HULL_ROUNDS {
        let hull = |v: &[f64], g: &[usize], lo: bool| {
            g.iter().map(|&i| v[i]).fold(if lo { f64::INFINITY } else { f64::NEG_INFINITY }, |a, b| {
                if lo { a.min(b) } else { a.max(b) }
            })
        };
        let hxmin: Vec<f64> = groups.iter().map(|g| hull(&xmin, g, true)).collect();
        let hxmax: Vec<f64> = groups.iter().map(|g| hull(&xmax, g, false)).collect();
        let hymin: Vec<f64> = groups.iter().map(|g| hull(&ymin, g, true)).collect();
        let hymax: Vec<f64> = groups.iter().map(|g| hull(&ymax, g, false)).collect();
        let merge = cluster_by_bbox_overlap(&hxmin, &hxmax, &hymin, &hymax, merge_tol);
        if merge.len() == groups.len() {
            return groups;
        }
        groups = merge
            .iter()
            .map(|mg| mg.iter().flat_map(|&gi| groups[gi].iter().copied()).collect())
            .collect();
    }
    vec![groups.into_iter().flatten().collect()]
}

// --------------------------------------------------------------------------
// boolean ops
// --------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub enum Op {
    Add,
    Intersect,
    Subtract { n_add: u32 },
    /// add + "no face may belong to two operands"
    Join,
}

fn flatten_operands(polys: &[PolyTree]) -> (Vec<Ring<'_>>, Vec<u32>) {
    let mut rings = Vec::new();
    let mut ops = Vec::new();
    for (i, p) in polys.iter().enumerate() {
        let start = rings.len();
        p.all_loops(&mut rings);
        ops.extend(std::iter::repeat(i as u32).take(rings.len() - start));
    }
    (rings, ops)
}

const JOIN_OVERLAP_MSG: &str = "join_polygons: found inputs whose interiors genuinely overlap -- whether via a \
transversal crossing, a same-side collinear overlap, or one input sitting fully inside another with no shared \
boundary of its own anywhere. join only fuses polygons that meet as complementary (touching, non-overlapping) \
neighbors -- use a boolean operation (add_polygons / intersect_polygons / subtract_polygons) instead if the \
inputs' interiors actually overlap.";

fn run_one(rings: &[Ring], ring_op: &[u32], gids: Option<&[usize]>, n_total: usize, op: Op, merge_tol: f64,
           area_tol: f64, eps: f64) -> KResult<Vec<PolyTree>> {
    let arr = build_arrangement(rings, ring_op, merge_tol)?;
    let g = |k: u32| -> usize { gids.map_or(k as usize, |ids| ids[k as usize]) };
    match op {
        Op::Add => label_and_assemble(&arr, |m| !m.is_empty(), area_tol, eps),
        Op::Intersect => label_and_assemble(&arr, |m| m.len() == n_total, area_tol, eps),
        Op::Subtract { n_add } => label_and_assemble(
            &arr,
            |m| {
                let in_add = m.iter().any(|&k| g(k) < n_add as usize);
                let in_sub = m.iter().any(|&k| g(k) >= n_add as usize);
                in_add && !in_sub
            },
            area_tol,
            eps,
        ),
        Op::Join => {
            if arr.membership.iter().any(|m| m.len() >= 2) {
                return Err(KErr::Geometry(JOIN_OVERLAP_MSG.into()));
            }
            label_and_assemble(&arr, |m| !m.is_empty(), area_tol, eps)
        }
    }
}

pub fn run_boolean(polys: &[PolyTree], op: Op, merge_tol: f64, area_tol: f64, eps: f64) -> KResult<Vec<PolyTree>> {
    if polys.is_empty() {
        return Ok(Vec::new());
    }
    let (rings, ring_op) = flatten_operands(polys);
    let n = polys.len();
    let clusters = cluster_operands(&rings, &ring_op, n, merge_tol);
    if clusters.len() <= 1 {
        return run_one(&rings, &ring_op, None, n, op, merge_tol, area_tol, eps);
    }
    let mut rings_by_op: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (r, &o) in ring_op.iter().enumerate() {
        rings_by_op[o as usize].push(r);
    }
    let run_cluster = |gids: &Vec<usize>| -> KResult<Vec<PolyTree>> {
        let mut lr: Vec<Ring> = Vec::new();
        let mut lo: Vec<u32> = Vec::new();
        for (li, &g) in gids.iter().enumerate() {
            for &r in &rings_by_op[g] {
                lr.push(rings[r]);
                lo.push(li as u32);
            }
        }
        run_one(&lr, &lo, Some(gids), n, op, merge_tol, area_tol, eps)
    };
    // clusters are independent: run them in parallel, concatenate in order
    use rayon::prelude::*;
    let parts: Vec<KResult<Vec<PolyTree>>> = if clusters.len() >= 8 {
        clusters.par_iter().map(run_cluster).collect()
    } else {
        clusters.iter().map(run_cluster).collect()
    };
    let mut out = Vec::new();
    for p in parts {
        out.extend(p?);
    }
    Ok(out)
}
