//! Port of `kernel/_arrangement.py`: exact-predicate segment splitting,
//! half-edge tracing, winding-parity membership propagation, and
//! boolean-op face labeling/assembly.
//!
//! Membership sets (Python `frozenset[int]`) are sorted `Vec<u32>`;
//! `^` on frozensets becomes `sym_diff`.

use std::collections::HashMap;

use crate::exact::{find_all_splits, to_grid};
use crate::geom::{KErr, KResult, PolyTree};
use crate::halfedge::{
    face_bboxes, group_in_order, nearest_container_batch, ContainerCands, ContainerQuery, HalfEdges, UnionFind,
};

pub type Membership = Vec<u32>;

/// Symmetric difference of two sorted, duplicate-free sets.
pub fn sym_diff(a: &[u32], b: &[u32]) -> Membership {
    if b.is_empty() {
        return a.to_vec();
    }
    if a.is_empty() {
        return b.to_vec();
    }
    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        if a[i] < b[j] {
            out.push(a[i]);
            i += 1;
        } else if a[i] > b[j] {
            out.push(b[j]);
            j += 1;
        } else {
            i += 1;
            j += 1;
        }
    }
    out.extend_from_slice(&a[i..]);
    out.extend_from_slice(&b[j..]);
    out
}

pub fn make_scale(merge_tol: f64) -> f64 {
    1.0 / merge_tol
}

/// `is_simple_ring`: no split points at all among a ring's own edges.
pub fn is_simple_ring(xs: &[f64], ys: &[f64], merge_tol: f64) -> bool {
    let n = xs.len();
    if n < 3 {
        return false;
    }
    let scale = make_scale(merge_tol);
    let gx: Vec<i64> = xs.iter().map(|&v| to_grid(v, scale)).collect();
    let gy: Vec<i64> = ys.iter().map(|&v| to_grid(v, scale)).collect();
    let ex1: Vec<i64> = (0..n).map(|i| gx[(i + 1) % n]).collect();
    let ey1: Vec<i64> = (0..n).map(|i| gy[(i + 1) % n]).collect();
    find_all_splits(&gx, &gy, &ex1, &ey1).is_empty()
}

pub struct Arrangement {
    pub he: HalfEdges,
    pub he_face_id: Vec<usize>,
    pub face_starts: Vec<usize>,
    pub membership: Vec<Membership>,
    pub vx: Vec<i64>,
    pub vy: Vec<i64>,
    pub scale: f64,
}

impl Arrangement {
    fn empty(scale: f64) -> Self {
        Arrangement {
            he: HalfEdges::new(&[], &[], &[], &[], 0),
            he_face_id: vec![],
            face_starts: vec![],
            membership: vec![],
            vx: vec![],
            vy: vec![],
            scale,
        }
    }
    pub fn n_faces(&self) -> usize {
        self.face_starts.len()
    }
}

pub type Ring<'a> = (&'a [f64], &'a [f64]);

pub fn build_arrangement(rings: &[Ring], ring_operand: &[u32], merge_tol: f64) -> KResult<Arrangement> {
    let scale = make_scale(merge_tol);
    if rings.is_empty() {
        return Ok(Arrangement::empty(scale));
    }

    // -- step 1: flatten + exact vertex merge (first-appearance ids) --
    let mut key_to_id: HashMap<(i64, i64), usize> = HashMap::new();
    let mut vx: Vec<i64> = Vec::new();
    let mut vy: Vec<i64> = Vec::new();
    let mut edge_u: Vec<usize> = Vec::new();
    let mut edge_v: Vec<usize> = Vec::new();
    let mut edge_op: Vec<u32> = Vec::new();
    let mut intern = |x: i64, y: i64, vx: &mut Vec<i64>, vy: &mut Vec<i64>| -> usize {
        *key_to_id.entry((x, y)).or_insert_with(|| {
            vx.push(x);
            vy.push(y);
            vx.len() - 1
        })
    };
    for (r, (xs, ys)) in rings.iter().enumerate() {
        let n = xs.len();
        let ids: Vec<usize> = (0..n)
            .map(|i| intern(to_grid(xs[i], scale), to_grid(ys[i], scale), &mut vx, &mut vy))
            .collect();
        for i in 0..n {
            edge_u.push(ids[i]);
            edge_v.push(ids[(i + 1) % n]);
            edge_op.push(ring_operand[r]);
        }
    }

    // -- step 2: split + chain --
    let ex0: Vec<i64> = edge_u.iter().map(|&u| vx[u]).collect();
    let ey0: Vec<i64> = edge_u.iter().map(|&u| vy[u]).collect();
    let ex1: Vec<i64> = edge_v.iter().map(|&v| vx[v]).collect();
    let ey1: Vec<i64> = edge_v.iter().map(|&v| vy[v]).collect();
    let splits = find_all_splits(&ex0, &ey0, &ex1, &ey1);

    // bucket split indices per edge (discovery order), then stable-sort
    // each bucket along its edge with the same integer key as Python
    let n_e = edge_u.len();
    let mut bucket_off = vec![0usize; n_e + 1];
    for s in &splits {
        bucket_off[s.edge as usize + 1] += 1;
    }
    for e in 0..n_e {
        bucket_off[e + 1] += bucket_off[e];
    }
    let mut fill = bucket_off.clone();
    let mut by_edge = vec![0usize; splits.len()];
    for (k, s) in splits.iter().enumerate() {
        let e = s.edge as usize;
        by_edge[fill[e]] = k;
        fill[e] += 1;
    }

    let mut seg_u: Vec<usize> = Vec::with_capacity(n_e + splits.len());
    let mut seg_v: Vec<usize> = Vec::with_capacity(n_e + splits.len());
    let mut seg_op: Vec<u32> = Vec::with_capacity(n_e + splits.len());
    let mut chain: Vec<usize> = Vec::new();
    for e in 0..n_e {
        let (u, v) = (edge_u[e], edge_v[e]);
        chain.clear();
        chain.push(u);
        let ks = &mut by_edge[bucket_off[e]..bucket_off[e + 1]];
        if !ks.is_empty() {
            let (ax, ay, bx, by) = (vx[u], vy[u], vx[v], vy[v]);
            let (dx, dy) = (bx - ax, by - ay);
            if dx.abs() >= dy.abs() {
                let sg = if dx >= 0 { 1 } else { -1 };
                ks.sort_by_key(|&k| (splits[k].x - ax) * sg);
            } else {
                let sg = if dy >= 0 { 1 } else { -1 };
                ks.sort_by_key(|&k| (splits[k].y - ay) * sg);
            }
            for &k in ks.iter() {
                let vid = intern(splits[k].x, splits[k].y, &mut vx, &mut vy);
                if vid != *chain.last().unwrap() {
                    chain.push(vid);
                }
            }
        }
        if *chain.last().unwrap() != v {
            chain.push(v);
        }
        for w in chain.windows(2) {
            if w[0] == w[1] {
                continue;
            }
            seg_u.push(w[0]);
            seg_v.push(w[1]);
            seg_op.push(edge_op[e]);
        }
    }
    let n_verts = vx.len();
    if seg_u.is_empty() {
        return Ok(Arrangement::empty(scale));
    }

    // -- step 3: dedupe by undirected key; flip set = odd-count operands --
    let mut seg_index: HashMap<(usize, usize), usize> = HashMap::new();
    let mut uniq_u: Vec<usize> = Vec::new();
    let mut uniq_v: Vec<usize> = Vec::new();
    let mut counts: Vec<Vec<(u32, u32)>> = Vec::new();
    for k in 0..seg_u.len() {
        let (a, b) = (seg_u[k], seg_v[k]);
        let key = if a < b { (a, b) } else { (b, a) };
        let idx = *seg_index.entry(key).or_insert_with(|| {
            uniq_u.push(key.0);
            uniq_v.push(key.1);
            counts.push(Vec::new());
            counts.len() - 1
        });
        let c = &mut counts[idx];
        match c.iter_mut().find(|(op, _)| *op == seg_op[k]) {
            Some(entry) => entry.1 += 1,
            None => c.push((seg_op[k], 1)),
        }
    }
    let flip_sets: Vec<Membership> = counts
        .into_iter()
        .map(|c| {
            let mut f: Vec<u32> = c.into_iter().filter(|(_, n)| n % 2 == 1).map(|(op, _)| op).collect();
            f.sort_unstable();
            f
        })
        .collect();

    // -- step 4: half-edges + faces --
    let fx: Vec<f64> = vx.iter().map(|&v| v as f64).collect();
    let fy: Vec<f64> = vy.iter().map(|&v| v as f64).collect();
    let he = HalfEdges::new(&uniq_u, &uniq_v, &fx, &fy, n_verts);
    let face_starts = he.trace_faces();
    let areas = he.face_areas(&face_starts, &fx, &fy);
    let he_face_id = he.face_id_per_halfedge(&face_starts);

    let mut uf = UnionFind::new(n_verts);
    for k in 0..uniq_u.len() {
        uf.union(uniq_u[k], uniq_v[k]);
    }

    // -- step 5: relative membership by BFS parity propagation --
    let n_faces = face_starts.len();
    let component_of: Vec<usize> = face_starts.iter().map(|&h| uf.find(he.start[h])).collect();
    let mut relative: Vec<Option<Membership>> = vec![None; n_faces];
    let comp_groups = group_in_order(&component_of);
    let mut outer_face: Vec<usize> = Vec::with_capacity(comp_groups.len());
    for faces in &comp_groups {
        let mut outer = faces[0];
        for &k in &faces[1..] {
            if areas[k] < areas[outer] {
                outer = k;
            }
        }
        outer_face.push(outer);
        relative[outer] = Some(Vec::new());
        let mut frontier = vec![outer];
        while !frontier.is_empty() {
            let mut nxt = Vec::new();
            for &fk in &frontier {
                let h0 = face_starts[fk];
                let mut h = h0;
                loop {
                    let nb_face = he_face_id[h ^ 1];
                    if relative[nb_face].is_none() {
                        let m = sym_diff(relative[fk].as_ref().unwrap(), &flip_sets[h / 2]);
                        relative[nb_face] = Some(m);
                        nxt.push(nb_face);
                    }
                    let hn = he.next(h);
                    if hn == h0 {
                        break;
                    }
                    h = hn;
                }
            }
            frontier = nxt;
        }
    }
    let relative: Vec<Membership> = relative.into_iter().map(|m| m.unwrap_or_default()).collect();

    // -- cross-component containment forest over each component's outer face --
    let (verts, offsets) = he.face_vertex_loops(&face_starts);
    let bb = face_bboxes(&offsets, &verts, &vx, &vy);
    let comp_root: Vec<usize> = comp_groups.iter().map(|g| component_of[g[0]]).collect();
    let qx: Vec<i64> = outer_face.iter().map(|&f| vx[verts[offsets[f]]]).collect();
    let qy: Vec<i64> = outer_face.iter().map(|&f| vy[verts[offsets[f]]]).collect();
    let q_min_area = vec![-1.0f64; comp_groups.len()];
    let cand_area: Vec<f64> = areas.iter().map(|a| a.abs()).collect();
    let best = nearest_container_batch(
        &ContainerQuery { qx: &qx, qy: &qy, q_exclude: &comp_root, q_min_area: &q_min_area },
        &ContainerCands { group: &component_of, area: &cand_area, bb: &bb, offsets: &offsets, verts: &verts },
        &vx,
        &vy,
    );

    // resolve each component's absolute outer membership (order-independent)
    let n_comp = comp_groups.len();
    let comp_idx_of_root: HashMap<usize, usize> = comp_root.iter().enumerate().map(|(i, &r)| (r, i)).collect();
    let mut resolved: Vec<Option<Membership>> = vec![None; n_comp];
    let mut remaining: Vec<usize> = (0..n_comp).collect();
    while !remaining.is_empty() {
        let mut progressed = false;
        let mut still = Vec::new();
        for &ci in &remaining {
            let pf = best[ci];
            if pf == usize::MAX {
                resolved[ci] = Some(Vec::new());
                progressed = true;
            } else {
                let pci = comp_idx_of_root[&component_of[pf]];
                if let Some(pm) = &resolved[pci] {
                    resolved[ci] = Some(sym_diff(pm, &relative[pf]));
                    progressed = true;
                } else {
                    still.push(ci);
                }
            }
        }
        if !progressed {
            return Err(KErr::Arrangement(
                "containment forest over components did not resolve -- unexpected cycle".into(),
            ));
        }
        remaining = still;
    }
    let membership: Vec<Membership> = (0..n_faces)
        .map(|k| {
            let ci = comp_idx_of_root[&component_of[k]];
            sym_diff(resolved[ci].as_ref().unwrap(), &relative[k])
        })
        .collect();

    Ok(Arrangement { he, he_face_id, face_starts, membership, vx, vy, scale })
}

/// `label_and_assemble`: keep faces by `keep_fn(membership)`, re-trace
/// only the boundary between kept and dropped faces, then nest the
/// resulting loops into polygons-with-holes.
pub fn label_and_assemble<F: Fn(&[u32]) -> bool>(
    arr: &Arrangement,
    keep_fn: F,
    area_tol: f64,
    eps: f64,
) -> KResult<Vec<PolyTree>> {
    if arr.n_faces() == 0 {
        return Ok(Vec::new());
    }
    let keep: Vec<bool> = arr.membership.iter().map(|m| keep_fn(m)).collect();
    let (vx, vy, scale) = (&arr.vx, &arr.vy, arr.scale);
    let n_verts = vx.len();

    let m_count = arr.he.len() / 2;
    let mut red_u: Vec<usize> = Vec::new();
    let mut red_v: Vec<usize> = Vec::new();
    let mut orig_face_of_dir: HashMap<(usize, usize), usize> = HashMap::new();
    for m in 0..m_count {
        let (h, ht) = (2 * m, 2 * m + 1);
        let (fa, fb) = (arr.he_face_id[h], arr.he_face_id[ht]);
        if keep[fa] == keep[fb] {
            continue;
        }
        let (u, v) = (arr.he.start[h], arr.he.end[h]);
        red_u.push(u);
        red_v.push(v);
        orig_face_of_dir.insert((u, v), fa);
        orig_face_of_dir.insert((v, u), fb);
    }
    if red_u.is_empty() {
        return Ok(Vec::new());
    }

    let fx: Vec<f64> = vx.iter().map(|&v| v as f64).collect();
    let fy: Vec<f64> = vy.iter().map(|&v| v as f64).collect();
    let rhe = HalfEdges::new(&red_u, &red_v, &fx, &fy, n_verts);
    let r_face_starts_all = rhe.trace_faces();
    let r_areas_all = rhe.face_areas(&r_face_starts_all, &fx, &fy);

    // drop the CW twin of every isolated loop
    let mut uf = UnionFind::new(n_verts);
    for k in 0..red_u.len() {
        uf.union(red_u[k], red_v[k]);
    }
    let comp: Vec<usize> = r_face_starts_all.iter().map(|&h| uf.find(rhe.start[h])).collect();
    let mut keep_idx: Vec<usize> = Vec::new();
    for idxs in group_in_order(&comp) {
        if idxs.len() == 2 {
            let (a0, a1) = (r_areas_all[idxs[0]], r_areas_all[idxs[1]]);
            if (a0.abs() - a1.abs()).abs() <= 1e-6 * a0.abs().max(a1.abs()).max(1.0) {
                keep_idx.push(if a0 > a1 { idxs[0] } else { idxs[1] });
                continue;
            }
        }
        keep_idx.extend(idxs);
    }
    let r_face_starts: Vec<usize> = keep_idx.iter().map(|&k| r_face_starts_all[k]).collect();
    let r_areas: Vec<f64> = keep_idx.iter().map(|&k| r_areas_all[k]).collect();
    let n = r_face_starts.len();

    let (r_verts, r_offsets) = rhe.face_vertex_loops(&r_face_starts);
    let r_keep: Vec<bool> = (0..n)
        .map(|i| {
            let lp = &r_verts[r_offsets[i]..r_offsets[i + 1]];
            let u = lp[0];
            let v = if lp.len() > 1 { lp[1] } else { u };
            keep[orig_face_of_dir[&(u, v)]]
        })
        .collect();

    let bb = face_bboxes(&r_offsets, &r_verts, vx, vy);
    let qx: Vec<i64> = (0..n).map(|i| vx[r_verts[r_offsets[i]]]).collect();
    let qy: Vec<i64> = (0..n).map(|i| vy[r_verts[r_offsets[i]]]).collect();
    let face_id: Vec<usize> = (0..n).collect();
    let cand_area: Vec<f64> = r_areas.iter().map(|a| a.abs()).collect();
    let best = nearest_container_batch(
        &ContainerQuery { qx: &qx, qy: &qy, q_exclude: &face_id, q_min_area: &cand_area },
        &ContainerCands { group: &face_id, area: &cand_area, bb: &bb, offsets: &r_offsets, verts: &r_verts },
        vx,
        vy,
    );
    let parent: Vec<Option<usize>> = best.iter().map(|&b| if b == usize::MAX { None } else { Some(b) }).collect();

    let mut order_desc: Vec<usize> = (0..n).collect();
    order_desc.sort_by(|&a, &b| cand_area[b].partial_cmp(&cand_area[a]).unwrap_or(std::cmp::Ordering::Equal));
    let mut rendered_ancestor_keep = vec![false; n];
    let mut is_rendered = vec![false; n];
    let mut render_target: Vec<Option<usize>> = vec![None; n];
    for &i in &order_desc {
        let p = parent[i];
        let ancestor_keep = match p {
            None => false,
            Some(p) => rendered_ancestor_keep[p],
        };
        is_rendered[i] = r_keep[i] != ancestor_keep;
        rendered_ancestor_keep[i] = if is_rendered[i] { r_keep[i] } else { ancestor_keep };
        render_target[i] = p.and_then(|p| if is_rendered[p] { Some(p) } else { render_target[p] });
    }

    // children_of[target] in index order; None -> top level
    let mut children_of: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut top: Vec<usize> = Vec::new();
    for i in 0..n {
        if is_rendered[i] {
            match render_target[i] {
                None => top.push(i),
                Some(t) => children_of[t].push(i),
            }
        }
    }

    let mut order_asc: Vec<usize> = (0..n).collect();
    order_asc.sort_by(|&a, &b| cand_area[a].partial_cmp(&cand_area[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut built: Vec<Option<PolyTree>> = vec![None; n];
    for &i in &order_asc {
        if !is_rendered[i] || cand_area[i] <= area_tol {
            continue;
        }
        let holes: Vec<PolyTree> = children_of[i].iter().filter_map(|&c| built[c].take()).collect();
        let lp = &r_verts[r_offsets[i]..r_offsets[i + 1]];
        let xs: Vec<f64> = lp.iter().map(|&v| vx[v] as f64 / scale).collect();
        let ys: Vec<f64> = lp.iter().map(|&v| vy[v] as f64 / scale).collect();
        built[i] = Some(PolyTree::verified(xs, ys, holes, eps)?);
    }
    Ok(top.into_iter().filter_map(|i| built[i].take()).collect())
}
