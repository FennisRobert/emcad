//! Half-edge rotation system, face tracing, and containment queries --
//! port of `fragment_tools.build_rotation_system`, `fragment_kernels`
//! (`trace_faces`, `face_areas`) and `_arrangement_kernels`
//! (`face_id_per_halfedge`, `face_vertex_loops`, `face_bboxes_i64`,
//! `nearest_container_batch`).
//!
//! Half-edges always come in twin pairs (2m, 2m+1), so the twin of `h`
//! is `h ^ 1` -- no separate twin array is needed.

use rayon::prelude::*;

pub struct HalfEdges {
    pub start: Vec<usize>,
    pub end: Vec<usize>,
    pub sorted_he: Vec<usize>,
    pub v_offset: Vec<usize>,
    pub pos_in_block: Vec<usize>,
}

impl HalfEdges {
    /// Build twin-paired half-edges from undirected segments (u[m], v[m])
    /// plus their angle-sorted rotation system over `n_verts` vertices.
    pub fn new(seg_u: &[usize], seg_v: &[usize], xs: &[f64], ys: &[f64], n_verts: usize) -> Self {
        let m = seg_u.len();
        let mut start = Vec::with_capacity(2 * m);
        let mut end = Vec::with_capacity(2 * m);
        for k in 0..m {
            start.push(seg_u[k]);
            end.push(seg_v[k]);
            start.push(seg_v[k]);
            end.push(seg_u[k]);
        }
        let h = start.len();
        let angle: Vec<f64> = (0..h)
            .map(|k| (ys[end[k]] - ys[start[k]]).atan2(xs[end[k]] - xs[start[k]]))
            .collect();
        // np.lexsort((angle, he_start)): stable, primary he_start, then angle
        let mut sorted_he: Vec<usize> = (0..h).collect();
        sorted_he.sort_by(|&a, &b| {
            start[a]
                .cmp(&start[b])
                .then(angle[a].partial_cmp(&angle[b]).unwrap_or(std::cmp::Ordering::Equal))
        });
        let mut v_offset = vec![0usize; n_verts + 1];
        for &s in &start {
            v_offset[s + 1] += 1;
        }
        for v in 0..n_verts {
            v_offset[v + 1] += v_offset[v];
        }
        let mut pos_in_block = vec![0usize; h];
        for (k, &he) in sorted_he.iter().enumerate() {
            pos_in_block[he] = k - v_offset[start[he]];
        }
        HalfEdges { start, end, sorted_he, v_offset, pos_in_block }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.start.len()
    }

    /// Next half-edge of the same face: clockwise predecessor of the twin
    /// in the end vertex's rotation.
    #[inline(always)]
    pub fn next(&self, h: usize) -> usize {
        let vv = self.end[h];
        let t = h ^ 1;
        let off = self.v_offset[vv];
        let deg = self.v_offset[vv + 1] - off;
        let p = self.pos_in_block[t];
        let next_pos = if p == 0 { deg - 1 } else { p - 1 };
        self.sorted_he[off + next_pos]
    }

    pub fn trace_faces(&self) -> Vec<usize> {
        let h_n = self.len();
        let mut visited = vec![false; h_n];
        let mut face_starts = Vec::new();
        for h0 in 0..h_n {
            if visited[h0] {
                continue;
            }
            face_starts.push(h0);
            let mut h = h0;
            loop {
                visited[h] = true;
                let hn = self.next(h);
                if hn == h0 {
                    break;
                }
                h = hn;
            }
        }
        face_starts
    }

    pub fn face_areas(&self, face_starts: &[usize], xs: &[f64], ys: &[f64]) -> Vec<f64> {
        face_starts
            .iter()
            .map(|&h0| {
                let mut h = h0;
                let mut area2 = 0.0f64;
                loop {
                    let u = self.start[h];
                    let v = self.end[h];
                    area2 += xs[u] * ys[v] - xs[v] * ys[u];
                    let hn = self.next(h);
                    if hn == h0 {
                        break;
                    }
                    h = hn;
                }
                0.5 * area2
            })
            .collect()
    }

    pub fn face_id_per_halfedge(&self, face_starts: &[usize]) -> Vec<usize> {
        let mut id = vec![usize::MAX; self.len()];
        for (fi, &h0) in face_starts.iter().enumerate() {
            let mut h = h0;
            loop {
                id[h] = fi;
                let hn = self.next(h);
                if hn == h0 {
                    break;
                }
                h = hn;
            }
        }
        id
    }

    /// CSR flattening of every face's vertex loop.
    pub fn face_vertex_loops(&self, face_starts: &[usize]) -> (Vec<usize>, Vec<usize>) {
        let mut verts = Vec::with_capacity(self.len());
        let mut offsets = Vec::with_capacity(face_starts.len() + 1);
        offsets.push(0);
        for &h0 in face_starts {
            let mut h = h0;
            loop {
                verts.push(self.start[h]);
                let hn = self.next(h);
                if hn == h0 {
                    break;
                }
                h = hn;
            }
            offsets.push(verts.len());
        }
        (verts, offsets)
    }
}

pub struct BBoxes {
    pub xmin: Vec<i64>,
    pub xmax: Vec<i64>,
    pub ymin: Vec<i64>,
    pub ymax: Vec<i64>,
}

pub fn face_bboxes(offsets: &[usize], verts: &[usize], vx: &[i64], vy: &[i64]) -> BBoxes {
    let n = offsets.len() - 1;
    let mut b = BBoxes {
        xmin: Vec::with_capacity(n),
        xmax: Vec::with_capacity(n),
        ymin: Vec::with_capacity(n),
        ymax: Vec::with_capacity(n),
    };
    for i in 0..n {
        let (s, e) = (offsets[i], offsets[i + 1]);
        let (mut x0, mut y0) = (vx[verts[s]], vy[verts[s]]);
        let (mut x1, mut y1) = (x0, y0);
        for &v in &verts[s + 1..e] {
            let (xv, yv) = (vx[v], vy[v]);
            x0 = x0.min(xv);
            x1 = x1.max(xv);
            y0 = y0.min(yv);
            y1 = y1.max(yv);
        }
        b.xmin.push(x0);
        b.xmax.push(x1);
        b.ymin.push(y0);
        b.ymax.push(y1);
    }
    b
}

/// Exact even-odd test of grid point (px, py) against the CSR ring
/// `verts[start..end]` (`_point_in_ring_idx_i64`).
#[inline]
fn point_in_ring_idx(vx: &[i64], vy: &[i64], verts: &[usize], px: i64, py: i64) -> bool {
    let n = verts.len();
    let mut inside = false;
    for ii in 0..n {
        let jj = if ii == 0 { n - 1 } else { ii - 1 };
        let yi = vy[verts[ii]];
        let yj = vy[verts[jj]];
        if (yi > py) != (yj > py) {
            let xi = vx[verts[ii]];
            let xj = vx[verts[jj]];
            let lhs = (xi - xj).wrapping_mul(py - yj);
            let rhs = (px - xj).wrapping_mul(yi - yj);
            let crosses = if yi > yj { lhs > rhs } else { lhs < rhs };
            if crosses {
                inside = !inside;
            }
        }
    }
    inside
}

pub struct ContainerQuery<'a> {
    pub qx: &'a [i64],
    pub qy: &'a [i64],
    pub q_exclude: &'a [usize],
    pub q_min_area: &'a [f64],
}

pub struct ContainerCands<'a> {
    pub group: &'a [usize],
    pub area: &'a [f64],
    pub bb: &'a BBoxes,
    pub offsets: &'a [usize],
    pub verts: &'a [usize],
}

/// `nearest_container_batch`: smallest-area candidate ring (strictly
/// larger than the query's min area, different group) containing each
/// query point; `usize::MAX` when none does. Ties resolve to the first
/// candidate, exactly like the original.
pub fn nearest_container_batch(q: &ContainerQuery, c: &ContainerCands, vx: &[i64], vy: &[i64]) -> Vec<usize> {
    let n_c = c.group.len();
    let one = |qi: usize| -> usize {
        let (qx, qy) = (q.qx[qi], q.qy[qi]);
        let excl = q.q_exclude[qi];
        let min_area = q.q_min_area[qi];
        let mut best_area = 1.0e300;
        let mut best = usize::MAX;
        for ci in 0..n_c {
            if c.group[ci] == excl {
                continue;
            }
            let a = c.area[ci];
            if a <= min_area || a >= best_area {
                continue;
            }
            if qx < c.bb.xmin[ci] || qx > c.bb.xmax[ci] || qy < c.bb.ymin[ci] || qy > c.bb.ymax[ci] {
                continue;
            }
            if point_in_ring_idx(vx, vy, &c.verts[c.offsets[ci]..c.offsets[ci + 1]], qx, qy) {
                best_area = a;
                best = ci;
            }
        }
        best
    };
    let n_q = q.qx.len();
    if n_q.saturating_mul(n_c) >= 1 << 16 {
        (0..n_q).into_par_iter().map(one).collect()
    } else {
        (0..n_q).map(one).collect()
    }
}

/// Union-find with path halving; `union(a, b)` sets parent[ra] = rb,
/// same as the Python helper (so roots, and therefore any root-keyed
/// grouping, match).
pub struct UnionFind {
    parent: Vec<usize>,
}

impl UnionFind {
    pub fn new(n: usize) -> Self {
        UnionFind { parent: (0..n).collect() }
    }
    #[inline]
    pub fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }
    #[inline]
    pub fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            self.parent[ra] = rb;
        }
    }
}

/// Group `keys` by value, preserving first-appearance order of groups
/// and index order within each -- a Python `dict.setdefault(k, []).append(i)`.
pub fn group_in_order(keys: &[usize]) -> Vec<Vec<usize>> {
    let mut slot: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for (i, &k) in keys.iter().enumerate() {
        let g = *slot.entry(k).or_insert_with(|| {
            groups.push(Vec::new());
            groups.len() - 1
        });
        groups[g].push(i);
    }
    groups
}
