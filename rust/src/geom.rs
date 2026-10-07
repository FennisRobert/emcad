//! Polygon tree (Rust mirror of `emcad.poly.Polygon`) plus the float
//! geometry helpers `Polygon` itself uses: `point_inside`, even-odd
//! `is_inside`, and the constructor's own validation.

#[derive(Clone, Debug)]
pub struct PolyTree {
    pub xs: Vec<f64>,
    pub ys: Vec<f64>,
    pub holes: Vec<PolyTree>,
}

#[derive(Debug)]
pub enum KErr {
    /// -> emcad.poly.GeometryException
    Geometry(String),
    /// -> ValueError
    Value(String),
    /// -> emcad.kernel._arrangement.ArrangementError
    Arrangement(String),
}

pub type KResult<T> = Result<T, KErr>;

impl PolyTree {
    /// `Polygon._construct_verified`: `_validate` (drop an explicit
    /// closing point) + the length checks, no hole-nesting check.
    pub fn verified(mut xs: Vec<f64>, mut ys: Vec<f64>, holes: Vec<PolyTree>, eps: f64) -> KResult<Self> {
        if !xs.is_empty()
            && !ys.is_empty()
            && (xs[xs.len() - 1] - xs[0]).abs() < eps
            && (ys[ys.len() - 1] - ys[0]).abs() < eps
        {
            xs.pop();
            ys.pop();
        }
        if xs.len() != ys.len() {
            return Err(KErr::Geometry(format!(
                "Length of xs({}) and ys({}) is not equal.",
                xs.len(),
                ys.len()
            )));
        }
        if xs.len() < 3 {
            return Err(KErr::Geometry(format!("Number of points must be 3 or larger. Not {}.", xs.len())));
        }
        Ok(PolyTree { xs, ys, holes })
    }

    /// Pre-order flatten of every ring (own boundary, then each hole
    /// recursively) -- `Polygon._all_loops` / `_flatten_rings` order.
    pub fn all_loops<'a>(&'a self, out: &mut Vec<(&'a [f64], &'a [f64])>) {
        out.push((&self.xs, &self.ys));
        for h in &self.holes {
            h.all_loops(out);
        }
    }

    pub fn loops(&self) -> Vec<(&[f64], &[f64])> {
        let mut v = Vec::new();
        self.all_loops(&mut v);
        v
    }

    /// `Polygon.point_inside`.
    pub fn point_inside(&self) -> KResult<(f64, f64)> {
        point_inside_loops(&self.loops())
    }

    /// `Polygon.is_inside`: even-odd across every ring, any exact
    /// boundary hit decides via `include_boundary`.
    pub fn is_inside(&self, x: f64, y: f64, include_boundary: bool) -> bool {
        if self.holes.is_empty() {
            return is_inside_ring(&self.xs, &self.ys, x, y, include_boundary);
        }
        let mut parity = false;
        let mut on_boundary = false;
        for (xs, ys) in self.loops() {
            let inclusive = is_inside_ring(xs, ys, x, y, true);
            let strict = is_inside_ring(xs, ys, x, y, false);
            if inclusive != strict {
                on_boundary = true;
            }
            parity ^= strict;
        }
        if on_boundary {
            include_boundary
        } else {
            parity
        }
    }
}

pub fn point_inside_loops(loops: &[(&[f64], &[f64])]) -> KResult<(f64, f64)> {
    let mut y_top = f64::NEG_INFINITY;
    for (_, ys) in loops {
        for &y in ys.iter() {
            if y > y_top {
                y_top = y;
            }
        }
    }
    let mut y_second = f64::NEG_INFINITY;
    let mut any_lower = false;
    for (_, ys) in loops {
        for &y in ys.iter() {
            if y < y_top {
                any_lower = true;
                if y > y_second {
                    y_second = y;
                }
            }
        }
    }
    if !any_lower {
        return Err(KErr::Value("Polygon is degenerate (zero height) -- no interior point exists".into()));
    }
    let y_test = 0.5 * (y_top + y_second);
    let mut xi_all: Vec<f64> = Vec::new();
    for (xs, ys) in loops {
        let n = xs.len();
        for i in 0..n {
            let (x0, y0) = (xs[i], ys[i]);
            let j = if i + 1 == n { 0 } else { i + 1 };
            let (x1, y1) = (xs[j], ys[j]);
            if (y0 <= y_test && y1 > y_test) || (y1 <= y_test && y0 > y_test) {
                xi_all.push(x0 + (y_test - y0) / (y1 - y0) * (x1 - x0));
            }
        }
    }
    if xi_all.len() < 2 {
        return Err(KErr::Value(
            "Could not find an interior point (degenerate or near-degenerate polygon)".into(),
        ));
    }
    // only the two smallest are needed
    let (mut a, mut b) = (f64::INFINITY, f64::INFINITY);
    for &v in &xi_all {
        if v < a {
            b = a;
            a = v;
        } else if v < b {
            b = v;
        }
    }
    Ok((0.5 * (a + b), y_test))
}

/// `_inside._is_inside`: float ray casting with an exact-zero boundary test.
pub fn is_inside_ring(xs: &[f64], ys: &[f64], x: f64, y: f64, include_boundary: bool) -> bool {
    let n = xs.len();
    if n == 0 {
        return false;
    }
    let mut inside = false;
    let (mut x0, mut y0) = (xs[n - 1], ys[n - 1]);
    for i in 0..n {
        let (x1, y1) = (xs[i], ys[i]);
        let (dx, dy) = (x1 - x0, y1 - y0);
        let cross = (x - x0) * dy - (y - y0) * dx;
        if cross == 0.0 {
            let dot = (x - x0) * dx + (y - y0) * dy;
            if 0.0 <= dot && dot <= dx * dx + dy * dy {
                return include_boundary;
            }
        }
        if (y0 > y) != (y1 > y) {
            let x_cross = x0 + (y - y0) / (y1 - y0) * (x1 - x0);
            if x_cross > x {
                inside = !inside;
            }
        }
        x0 = x1;
        y0 = y1;
    }
    inside
}

/// 0.5 * sum(xs * roll(ys, -1) - roll(xs, -1) * ys)
pub fn signed_area(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len();
    let mut s = 0.0;
    for i in 0..n {
        let j = if i + 1 == n { 0 } else { i + 1 };
        s += xs[i] * ys[j] - xs[j] * ys[i];
    }
    0.5 * s
}
