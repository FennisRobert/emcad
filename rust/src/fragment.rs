//! Port of `_fragment.poly_fragment` and its float-tolerance helpers in
//! `fragment_tools.py` / `fragment_kernels.py`.

use std::collections::HashMap;

use crate::geom::{KResult, PolyTree};
use crate::halfedge::{group_in_order, HalfEdges, UnionFind};
use crate::primitives::compute_intersections;

#[derive(Clone, Copy, PartialEq)]
pub enum Keep {
    Positive,
    Negative,
    All,
}

pub fn poly_fragment(
    polys: &[PolyTree], merge_tol: f64, t_tol: f64, area_tol: f64, keep: Keep, filter_to_originals: bool, eps: f64,
) -> KResult<Vec<PolyTree>> {
    if polys.is_empty() {
        return Ok(Vec::new());
    }
    // collect_edges
    let mut rings = Vec::new();
    for p in polys {
        p.all_loops(&mut rings);
    }
    let n_edges: usize = rings.iter().map(|(xs, _)| xs.len()).sum();
    let mut sx = Vec::with_capacity(n_edges);
    let mut sy = Vec::with_capacity(n_edges);
    let mut ex = Vec::with_capacity(n_edges);
    let mut ey = Vec::with_capacity(n_edges);
    let mut edge_end_vid = Vec::with_capacity(n_edges);
    let mut base = 0usize;
    for (xs, ys) in &rings {
        let n = xs.len();
        for i in 0..n {
            let j = (i + 1) % n;
            sx.push(xs[i]);
            sy.push(ys[i]);
            ex.push(xs[j]);
            ey.push(ys[j]);
            edge_end_vid.push(base + j);
        }
        base += n;
    }

    // raw crossings (all ordered pairs, like edge_self_intersections)
    let (ids_i, ids_j, cx, cy) = compute_intersections(&sx, &sy, &ex, &ey, &sx, &sy, &ex, &ey);

    // classify_crossings
    let mut split_edge: Vec<usize> = Vec::new();
    let mut split_t: Vec<f64> = Vec::new();
    let mut split_x: Vec<f64> = Vec::new();
    let mut split_y: Vec<f64> = Vec::new();
    for k in 0..ids_i.len() {
        let (i, j) = (ids_i[k], ids_j[k]);
        let (px, py) = (cx[k], cy[k]);
        let (dxi, dyi) = (ex[i] - sx[i], ey[i] - sy[i]);
        let len2_i = dxi * dxi + dyi * dyi;
        if len2_i <= 0.0 {
            continue;
        }
        let ti = ((px - sx[i]) * dxi + (py - sy[i]) * dyi) / len2_i;
        let (dxj, dyj) = (ex[j] - sx[j], ey[j] - sy[j]);
        let len2_j = dxj * dxj + dyj * dyj;
        if len2_j <= 0.0 {
            continue;
        }
        let tj = ((px - sx[j]) * dxj + (py - sy[j]) * dyj) / len2_j;
        let i_at_end = ti <= t_tol || ti >= 1.0 - t_tol;
        let j_at_end = tj <= t_tol || tj >= 1.0 - t_tol;
        if i_at_end && j_at_end {
            continue;
        }
        if !i_at_end {
            split_edge.push(i);
            split_t.push(ti);
            split_x.push(px);
            split_y.push(py);
        }
        if !j_at_end {
            split_edge.push(j);
            split_t.push(tj);
            split_x.push(px);
            split_y.push(py);
        }
    }

    // merge_points: grid keys -> lexicographically sorted unique ids,
    // merged coordinate = mean of the group (summed in input order)
    let n_new = split_x.len();
    let all_x: Vec<f64> = sx.iter().chain(split_x.iter()).copied().collect();
    let all_y: Vec<f64> = sy.iter().chain(split_y.iter()).copied().collect();
    let n_all = all_x.len();
    let keys: Vec<(i64, i64)> = (0..n_all)
        .map(|k| ((all_x[k] / merge_tol).round_ties_even() as i64, (all_y[k] / merge_tol).round_ties_even() as i64))
        .collect();
    let mut uniq: Vec<(i64, i64)> = keys.clone();
    uniq.sort_unstable();
    uniq.dedup();
    let key_id: HashMap<(i64, i64), usize> = uniq.iter().enumerate().map(|(i, &k)| (k, i)).collect();
    let inverse: Vec<usize> = keys.iter().map(|k| key_id[k]).collect();
    let n_verts = uniq.len();
    let mut mx = vec![0.0f64; n_verts];
    let mut my = vec![0.0f64; n_verts];
    let mut cnt = vec![0.0f64; n_verts];
    for k in 0..n_all {
        mx[inverse[k]] += all_x[k];
        my[inverse[k]] += all_y[k];
        cnt[inverse[k]] += 1.0;
    }
    for v in 0..n_verts {
        mx[v] /= cnt[v];
        my[v] /= cnt[v];
    }

    // chain_edges + build_segments (dedupe keeps first occurrence + its direction)
    let mut order: Vec<usize> = (0..n_new).collect();
    order.sort_by(|&a, &b| {
        split_edge[a]
            .cmp(&split_edge[b])
            .then(split_t[a].partial_cmp(&split_t[b]).unwrap_or(std::cmp::Ordering::Equal))
    });
    let mut seg_u: Vec<usize> = Vec::new();
    let mut seg_v: Vec<usize> = Vec::new();
    let mut seen: HashMap<(usize, usize), ()> = HashMap::new();
    let mut ptr = 0usize;
    let mut chain: Vec<usize> = Vec::new();
    for e in 0..n_edges {
        chain.clear();
        chain.push(inverse[e]);
        while ptr < n_new && split_edge[order[ptr]] == e {
            chain.push(inverse[n_edges + order[ptr]]);
            ptr += 1;
        }
        chain.push(inverse[edge_end_vid[e]]);
        for w in chain.windows(2) {
            if w[0] == w[1] {
                continue;
            }
            let key = (w[0].min(w[1]), w[0].max(w[1]));
            if seen.insert(key, ()).is_none() {
                seg_u.push(w[0]);
                seg_v.push(w[1]);
            }
        }
    }
    if seg_u.is_empty() {
        return Ok(Vec::new());
    }

    let he = HalfEdges::new(&seg_u, &seg_v, &mx, &my, n_verts);
    let face_starts = he.trace_faces();
    let areas = he.face_areas(&face_starts, &mx, &my);

    let reconstruct = |h0: usize| -> KResult<PolyTree> {
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        let mut h = h0;
        loop {
            xs.push(mx[he.start[h]]);
            ys.push(my[he.start[h]]);
            let hn = he.next(h);
            if hn == h0 {
                break;
            }
            h = hn;
        }
        PolyTree::verified(xs, ys, Vec::new(), eps)
    };

    let mut out: Vec<PolyTree> = match keep {
        Keep::Positive => {
            assemble_nested(&he, &face_starts, &areas, &seg_u, &seg_v, n_verts, area_tol, eps, &reconstruct)?
        }
        Keep::Negative => face_starts
            .iter()
            .zip(&areas)
            .filter(|(_, &a)| a < -area_tol)
            .map(|(&h0, _)| reconstruct(h0))
            .collect::<KResult<_>>()?,
        Keep::All => face_starts
            .iter()
            .zip(&areas)
            .filter(|(_, &a)| a.abs() > area_tol)
            .map(|(&h0, _)| reconstruct(h0))
            .collect::<KResult<_>>()?,
    };

    if filter_to_originals {
        out.retain(|frag| match frag.point_inside() {
            Err(_) => false,
            Ok((px, py)) => polys.iter().any(|o| o.is_inside(px, py, true)),
        });
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn assemble_nested(
    he: &HalfEdges, face_starts: &[usize], areas: &[f64], seg_u: &[usize], seg_v: &[usize], n_verts: usize,
    area_tol: f64, eps: f64, reconstruct: &dyn Fn(usize) -> KResult<PolyTree>,
) -> KResult<Vec<PolyTree>> {
    let mut uf = UnionFind::new(n_verts);
    for k in 0..seg_u.len() {
        uf.union(seg_u[k], seg_v[k]);
    }
    let comp_all: Vec<usize> = face_starts.iter().map(|&h| uf.find(he.start[h])).collect();
    let mut keep_idx = Vec::new();
    for idxs in group_in_order(&comp_all) {
        if idxs.len() == 2 {
            let (a0, a1) = (areas[idxs[0]].abs(), areas[idxs[1]].abs());
            if (a0 - a1).abs() / a0.max(a1).max(1e-300) < 1e-6 {
                keep_idx.push(idxs[0]);
                continue;
            }
        }
        keep_idx.extend(idxs);
    }

    let mut faces: Vec<PolyTree> = Vec::new();
    let mut signed: Vec<f64> = Vec::new();
    let mut comps: Vec<usize> = Vec::new();
    let mut points: Vec<(f64, f64)> = Vec::new();
    for &k in &keep_idx {
        if areas[k].abs() <= area_tol {
            continue;
        }
        let poly = reconstruct(face_starts[k])?;
        // a face point_inside can't classify is dropped (degenerate sliver)
        if let Ok(pt) = poly.point_inside() {
            points.push(pt);
            faces.push(poly);
            signed.push(areas[k]);
            comps.push(comp_all[k]);
        }
    }
    let n = faces.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let abs_a: Vec<f64> = signed.iter().map(|a| a.abs()).collect();
    let parent: Vec<Option<usize>> = (0..n)
        .map(|i| {
            let (xi, yi) = points[i];
            let mut best: Option<usize> = None;
            for j in 0..n {
                if j == i || abs_a[j] <= abs_a[i] || comps[j] == comps[i] {
                    continue;
                }
                if faces[j].is_inside(xi, yi, true) && best.map_or(true, |b| abs_a[j] < abs_a[b]) {
                    best = Some(j);
                }
            }
            best
        })
        .collect();
    let mut children_of: Vec<Vec<usize>> = vec![Vec::new(); n];
    for i in 0..n {
        if let Some(p) = parent[i] {
            children_of[p].push(i);
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| abs_a[a].partial_cmp(&abs_a[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut built: Vec<Option<PolyTree>> = vec![None; n];
    for &i in &order {
        let holes: Vec<PolyTree> = children_of[i].iter().map(|&c| built[c].take().unwrap()).collect();
        let f = &faces[i];
        built[i] = Some(PolyTree::verified(f.xs.clone(), f.ys.clone(), holes, eps)?);
    }
    Ok((0..n).filter(|&i| parent[i].is_none() && signed[i] > 0.0).filter_map(|i| built[i].take()).collect())
}
