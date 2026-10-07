//! Port of `_keyhole.py`.

use std::collections::HashMap;

use crate::geom::{signed_area, KErr, KResult, PolyTree};

type Pt = (f64, f64);

fn key(p: Pt, tol: f64) -> (i64, i64) {
    ((p.0 / tol).round_ties_even() as i64, (p.1 / tol).round_ties_even() as i64)
}

fn dekeyhole_ring(pts: &[Pt], tol: f64) -> (Vec<Pt>, Vec<Vec<Pt>>) {
    let mut stack: Vec<Pt> = Vec::new();
    let mut pos_of: HashMap<(i64, i64), usize> = HashMap::new();
    let mut extracted: Vec<Vec<Pt>> = Vec::new();
    for &p in pts {
        let k = key(p, tol);
        if let Some(&pos) = pos_of.get(&k) {
            let sub: Vec<Pt> = stack[pos..].to_vec();
            for sv in &stack[pos + 1..] {
                pos_of.remove(&key(*sv, tol));
            }
            stack.truncate(pos + 1);
            if sub.len() >= 3 {
                extracted.push(sub);
            }
        } else {
            pos_of.insert(k, stack.len());
            stack.push(p);
        }
    }
    (stack, extracted)
}

fn dekeyhole_recursive(pts: &[Pt], tol: f64) -> (Vec<Pt>, Vec<Vec<Pt>>) {
    let (outer, extracted) = dekeyhole_ring(pts, tol);
    let mut flat = Vec::new();
    for sub in extracted {
        let (sub_outer, sub_flat) = dekeyhole_recursive(&sub, tol);
        flat.push(sub_outer);
        flat.extend(sub_flat);
    }
    (outer, flat)
}

fn to_poly(pts: Vec<Pt>, eps: f64) -> KResult<PolyTree> {
    let (xs, ys): (Vec<f64>, Vec<f64>) = pts.into_iter().unzip();
    PolyTree::verified(xs, ys, Vec::new(), eps)
}

fn assemble_holes(outer: PolyTree, subs: Vec<PolyTree>, eps: f64) -> KResult<PolyTree> {
    if subs.is_empty() {
        return Ok(outer);
    }
    let mut all = vec![outer];
    all.extend(subs);
    let n = all.len();
    let areas: Vec<f64> = all.iter().map(|p| signed_area(&p.xs, &p.ys).abs()).collect();
    let points: Vec<Pt> = all.iter().map(|p| p.point_inside()).collect::<KResult<_>>()?;
    let mut parent: Vec<Option<usize>> = vec![None; n];
    for i in 1..n {
        let (xi, yi) = points[i];
        let mut best: Option<usize> = None;
        for j in 0..n {
            if j == i || areas[j] <= areas[i] {
                continue;
            }
            if all[j].is_inside(xi, yi, true) && best.map_or(true, |b| areas[j] < areas[b]) {
                best = Some(j);
            }
        }
        if best.is_none() {
            return Err(KErr::Geometry(
                "dekeyhole_polygon: an extracted sub-ring isn't nested inside the outer boundary (or any other \
                 extracted ring) -- this doesn't look like a valid keyhole polygon."
                    .into(),
            ));
        }
        parent[i] = best;
    }
    let mut children_of: Vec<Vec<usize>> = vec![Vec::new(); n];
    for i in 0..n {
        if let Some(p) = parent[i] {
            children_of[p].push(i);
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| areas[a].partial_cmp(&areas[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut built: Vec<Option<PolyTree>> = vec![None; n];
    for &i in &order {
        let holes: Vec<PolyTree> = children_of[i].iter().map(|&c| built[c].take().unwrap()).collect();
        built[i] = Some(PolyTree::verified(all[i].xs.clone(), all[i].ys.clone(), holes, eps)?);
    }
    Ok(built[0].take().unwrap())
}

pub fn dekeyhole_polygon(poly: &PolyTree, tol: f64, eps: f64) -> KResult<PolyTree> {
    let pts: Vec<Pt> = poly.xs.iter().copied().zip(poly.ys.iter().copied()).collect();
    let (outer_pts, sub_pts) = dekeyhole_recursive(&pts, tol);
    let outer = to_poly(outer_pts, eps)?;
    let subs: Vec<PolyTree> = sub_pts.into_iter().map(|r| to_poly(r, eps)).collect::<KResult<_>>()?;
    let mut result = assemble_holes(outer, subs, eps)?;
    if !poly.holes.is_empty() {
        let extra: Vec<PolyTree> = poly.holes.iter().map(|h| dekeyhole_polygon(h, tol, eps)).collect::<KResult<_>>()?;
        let mut holes = std::mem::take(&mut result.holes);
        holes.extend(extra);
        result = PolyTree::verified(result.xs, result.ys, holes, eps)?;
    }
    Ok(result)
}
