//! `emcad._rs`: Rust implementation of the emcad geometry kernel.
//!
//! Boundary design: polygon-level entry points take the Python
//! `Polygon` objects themselves (reading `.xs`/`.ys`/`.holes` directly,
//! no per-ring numpy temporaries) and build the result `Polygon`s
//! directly (`Polygon.__new__` + attribute assignment, i.e. exactly
//! what `Polygon._construct_verified` produces, with its validation
//! already done in Rust). Array-level entry points take/return numpy
//! arrays.

mod arrangement;
mod boolean;
mod exact;
mod fragment;
mod geom;
mod halfedge;
mod keyhole;
mod primitives;
mod regularize;
mod via;

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;

use geom::{KErr, PolyTree};

// --------------------------------------------------------------------------
// conversions
// --------------------------------------------------------------------------

fn to_pyerr(py: Python<'_>, e: KErr) -> PyErr {
    let raise = |module: &str, cls: &str, msg: String| -> PyErr {
        match py.import(module).and_then(|m| m.getattr(cls)).and_then(|c| c.call1((msg.clone(),))) {
            Ok(inst) => PyErr::from_value(inst),
            Err(err) => err,
        }
    };
    match e {
        KErr::Geometry(m) => raise("emcad.poly", "GeometryException", m),
        KErr::Arrangement(m) => raise("emcad.kernel._errors", "ArrangementError", m),
        KErr::Value(m) => PyValueError::new_err(m),
    }
}

/// Any 1-D float-like sequence -> Vec<f64> (fast path for float64 arrays).
fn vec_f64(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    if let Ok(a) = obj.extract::<PyReadonlyArray1<f64>>() {
        return Ok(a.as_array().iter().copied().collect());
    }
    obj.extract::<Vec<f64>>()
}

fn vec_usize(obj: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    if let Ok(a) = obj.extract::<PyReadonlyArray1<i64>>() {
        return Ok(a.as_array().iter().map(|&v| v as usize).collect());
    }
    obj.extract::<Vec<usize>>()
}

fn extract_tree(obj: &Bound<'_, PyAny>) -> PyResult<PolyTree> {
    let xs = vec_f64(&obj.getattr("xs")?)?;
    let ys = vec_f64(&obj.getattr("ys")?)?;
    let holes_obj = obj.getattr("holes")?;
    let mut holes = Vec::new();
    for h in holes_obj.try_iter()? {
        holes.push(extract_tree(&h?)?);
    }
    Ok(PolyTree { xs, ys, holes })
}

fn extract_trees(objs: &Bound<'_, PyAny>) -> PyResult<Vec<PolyTree>> {
    objs.try_iter()?.map(|o| extract_tree(&o?)).collect()
}

struct Ctx<'py> {
    py: Python<'py>,
    polygon_cls: Bound<'py, PyAny>,
    eps: f64,
}

impl<'py> Ctx<'py> {
    fn new(py: Python<'py>) -> PyResult<Self> {
        let polygon_cls = py.import("emcad.poly")?.getattr("Polygon")?;
        let eps: f64 = py.import("emcad.glob")?.getattr("GlobalSettings")?.getattr("EPS")?.extract()?;
        Ok(Ctx { py, polygon_cls, eps })
    }

    fn build(&self, t: PolyTree) -> PyResult<Bound<'py, PyAny>> {
        let obj = self.polygon_cls.call_method1("__new__", (&self.polygon_cls,))?;
        obj.setattr("xs", PyList::new(self.py, t.xs)?)?;
        obj.setattr("ys", PyList::new(self.py, t.ys)?)?;
        let holes: Vec<Bound<'py, PyAny>> = t.holes.into_iter().map(|h| self.build(h)).collect::<PyResult<_>>()?;
        obj.setattr("holes", PyList::new(self.py, holes)?)?;
        Ok(obj)
    }

    fn build_list(&self, ts: Vec<PolyTree>) -> PyResult<Bound<'py, PyList>> {
        let objs: Vec<Bound<'py, PyAny>> = ts.into_iter().map(|t| self.build(t)).collect::<PyResult<_>>()?;
        PyList::new(self.py, objs)
    }

    fn err(&self, e: KErr) -> PyErr {
        to_pyerr(self.py, e)
    }
}

fn rows2<'py>(arr: &PyReadonlyArray2<'py, f64>) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let a = arr.as_array();
    if a.shape()[0] != 2 {
        return Err(PyValueError::new_err("expected a (2, N) coordinate array"));
    }
    Ok((a.row(0).to_vec(), a.row(1).to_vec()))
}

fn arr2<'py, T: numpy::Element>(py: Python<'py>, r0: Vec<T>, r1: Vec<T>) -> Bound<'py, PyArray2<T>> {
    let n = r0.len();
    let mut data = r0;
    data.extend(r1);
    Array2::from_shape_vec((2, n), data).unwrap().into_pyarray(py)
}

type Arr1<'py, T> = Bound<'py, PyArray1<T>>;

// --------------------------------------------------------------------------
// array-level primitives
// --------------------------------------------------------------------------

#[pyfunction]
fn edge_intersections<'py>(
    py: Python<'py>, s1: PyReadonlyArray2<'py, f64>, e1: PyReadonlyArray2<'py, f64>, s2: PyReadonlyArray2<'py, f64>,
    e2: PyReadonlyArray2<'py, f64>,
) -> PyResult<(Bound<'py, PyArray2<i64>>, Bound<'py, PyArray2<f64>>)> {
    let (s1x, s1y) = rows2(&s1)?;
    let (e1x, e1y) = rows2(&e1)?;
    let (s2x, s2y) = rows2(&s2)?;
    let (e2x, e2y) = rows2(&e2)?;
    let (i, j, cx, cy) = primitives::compute_intersections(&s1x, &s1y, &e1x, &e1y, &s2x, &s2y, &e2x, &e2y);
    let i: Vec<i64> = i.into_iter().map(|v| v as i64).collect();
    let j: Vec<i64> = j.into_iter().map(|v| v as i64).collect();
    Ok((arr2(py, i, j), arr2(py, cx, cy)))
}

#[pyfunction]
fn is_inside(xs: &Bound<'_, PyAny>, ys: &Bound<'_, PyAny>, x: f64, y: f64, include_boundary: bool) -> PyResult<bool> {
    Ok(geom::is_inside_ring(&vec_f64(xs)?, &vec_f64(ys)?, x, y, include_boundary))
}

#[pyfunction]
fn convex_hull<'py>(py: Python<'py>, xs: &Bound<'py, PyAny>, ys: &Bound<'py, PyAny>) -> PyResult<Arr1<'py, i64>> {
    let ids = primitives::convex_hull(&vec_f64(xs)?, &vec_f64(ys)?);
    Ok(PyArray1::from_vec(py, ids.into_iter().map(|v| v as i64).collect()))
}

type XY<'py> = (Arr1<'py, f64>, Arr1<'py, f64>);

fn xy_out<'py>(py: Python<'py>, r: Result<(Vec<f64>, Vec<f64>), KErr>) -> PyResult<XY<'py>> {
    let (a, b) = r.map_err(|e| to_pyerr(py, e))?;
    Ok((PyArray1::from_vec(py, a), PyArray1::from_vec(py, b)))
}

#[pyfunction]
#[pyo3(signature = (xs, ys, tol=1e-9, closed=true))]
fn sanitize_polygon<'py>(py: Python<'py>, xs: &Bound<'py, PyAny>, ys: &Bound<'py, PyAny>, tol: f64, closed: bool) -> PyResult<XY<'py>> {
    xy_out(py, primitives::sanitize_polygon(&vec_f64(xs)?, &vec_f64(ys)?, tol, closed))
}

#[pyfunction]
fn simplify_polyline<'py>(py: Python<'py>, xs: &Bound<'py, PyAny>, ys: &Bound<'py, PyAny>, delta: f64) -> PyResult<XY<'py>> {
    xy_out(py, primitives::simplify_polyline(&vec_f64(xs)?, &vec_f64(ys)?, delta))
}

#[pyfunction]
#[pyo3(signature = (xs, ys, max_kink_length, max_angle_deg=20.0, min_neighbor_factor=3.0))]
fn dezigzag_polyline<'py>(
    py: Python<'py>, xs: &Bound<'py, PyAny>, ys: &Bound<'py, PyAny>, max_kink_length: f64, max_angle_deg: f64,
    min_neighbor_factor: f64,
) -> PyResult<XY<'py>> {
    xy_out(
        py,
        primitives::dezigzag_polyline(&vec_f64(xs)?, &vec_f64(ys)?, max_kink_length, max_angle_deg, min_neighbor_factor),
    )
}

#[pyfunction]
#[pyo3(signature = (xs, ys, tol, dangle_deg=5.0, angle_tol_deg=0.05, min_anchor_len=None, offset_tol=5e-6, vw_area=None))]
#[allow(clippy::too_many_arguments)]
fn regularize_polyline<'py>(
    py: Python<'py>, xs: &Bound<'py, PyAny>, ys: &Bound<'py, PyAny>, tol: f64, dangle_deg: f64, angle_tol_deg: f64,
    min_anchor_len: Option<f64>, offset_tol: f64, vw_area: Option<f64>,
) -> PyResult<XY<'py>> {
    let prm = regularize::Params {
        tol,
        dangle_deg,
        angle_tol_deg,
        min_anchor_len: min_anchor_len.unwrap_or(2.0 * tol),
        offset_tol,
        vw_area: vw_area.unwrap_or(tol * tol),
    };
    xy_out(py, regularize::regularize_ring(&vec_f64(xs)?, &vec_f64(ys)?, &prm))
}

#[pyfunction]
fn is_simple_ring(xs: &Bound<'_, PyAny>, ys: &Bound<'_, PyAny>, merge_tol: f64) -> PyResult<bool> {
    Ok(arrangement::is_simple_ring(&vec_f64(xs)?, &vec_f64(ys)?, merge_tol))
}

// --------------------------------------------------------------------------
// polygon-level operations
// --------------------------------------------------------------------------

#[pyfunction]
fn boolean_op<'py>(
    py: Python<'py>, polys: &Bound<'py, PyAny>, op: &str, n_add: u32, merge_tol: f64, area_tol: f64,
) -> PyResult<Bound<'py, PyList>> {
    let ctx = Ctx::new(py)?;
    let trees = extract_trees(polys)?;
    let op = match op {
        "add" => boolean::Op::Add,
        "intersect" => boolean::Op::Intersect,
        "subtract" => boolean::Op::Subtract { n_add },
        "join" => boolean::Op::Join,
        other => return Err(PyValueError::new_err(format!("unknown boolean op {other:?}"))),
    };
    let eps = ctx.eps;
    let out = py.detach(|| boolean::run_boolean(&trees, op, merge_tol, area_tol, eps)).map_err(|e| ctx.err(e))?;
    ctx.build_list(out)
}

/// Benchmark helper: `boolean_op`, but returns the wall time of each
/// phase -- (extract Python -> Rust, compute, build Rust -> Python, n_out)
/// -- to quantify the cost of crossing the language boundary.
#[pyfunction]
fn profile_boolean_op<'py>(
    py: Python<'py>, polys: &Bound<'py, PyAny>, op: &str, n_add: u32, merge_tol: f64, area_tol: f64,
) -> PyResult<(f64, f64, f64, usize)> {
    use std::time::Instant;
    let ctx = Ctx::new(py)?;
    let t0 = Instant::now();
    let trees = extract_trees(polys)?;
    let t1 = Instant::now();
    let op = match op {
        "add" => boolean::Op::Add,
        "intersect" => boolean::Op::Intersect,
        "subtract" => boolean::Op::Subtract { n_add },
        _ => boolean::Op::Join,
    };
    let out = boolean::run_boolean(&trees, op, merge_tol, area_tol, ctx.eps).map_err(|e| ctx.err(e))?;
    let t2 = Instant::now();
    let n = out.len();
    let list = ctx.build_list(out)?;
    let t3 = Instant::now();
    drop(list);
    Ok(((t1 - t0).as_secs_f64(), (t2 - t1).as_secs_f64(), (t3 - t2).as_secs_f64(), n))
}

#[pyfunction]
fn poly_fragment<'py>(
    py: Python<'py>, polys: &Bound<'py, PyAny>, merge_tol: f64, t_tol: f64, area_tol: f64, keep: &str,
    filter_to_originals: bool,
) -> PyResult<Bound<'py, PyList>> {
    let ctx = Ctx::new(py)?;
    let trees = extract_trees(polys)?;
    let keep = match keep {
        "positive" => fragment::Keep::Positive,
        "negative" => fragment::Keep::Negative,
        "all" => fragment::Keep::All,
        other => {
            return Err(PyValueError::new_err(format!(
                "Unknown keep={other:?}, expected 'positive', 'negative', or 'all'"
            )))
        }
    };
    let eps = ctx.eps;
    let out = py
        .detach(|| fragment::poly_fragment(&trees, merge_tol, t_tol, area_tol, keep, filter_to_originals, eps))
        .map_err(|e| ctx.err(e))?;
    ctx.build_list(out)
}

#[pyfunction]
fn dekeyhole_polygon<'py>(py: Python<'py>, poly: &Bound<'py, PyAny>, tol: f64) -> PyResult<Bound<'py, PyAny>> {
    let ctx = Ctx::new(py)?;
    let t = extract_tree(poly)?;
    let out = keyhole::dekeyhole_polygon(&t, tol, ctx.eps).map_err(|e| ctx.err(e))?;
    ctx.build(out)
}

#[pyfunction]
fn dekeyhole_polygons<'py>(py: Python<'py>, polys: &Bound<'py, PyAny>, tol: f64) -> PyResult<Bound<'py, PyList>> {
    let ctx = Ctx::new(py)?;
    let trees = extract_trees(polys)?;
    let eps = ctx.eps;
    let out: Result<Vec<PolyTree>, KErr> = py.detach(|| {
        use rayon::prelude::*;
        trees.par_iter().map(|t| keyhole::dekeyhole_polygon(t, tol, eps)).collect()
    });
    ctx.build_list(out.map_err(|e| ctx.err(e))?)
}

#[pyfunction]
fn polygon_point_inside(py: Python<'_>, poly: &Bound<'_, PyAny>) -> PyResult<(f64, f64)> {
    extract_tree(poly)?.point_inside().map_err(|e| to_pyerr(py, e))
}

#[pyfunction]
fn polygon_is_inside(poly: &Bound<'_, PyAny>, x: f64, y: f64, include_boundary: bool) -> PyResult<bool> {
    Ok(extract_tree(poly)?.is_inside(x, y, include_boundary))
}

// --------------------------------------------------------------------------
// via kernels
// --------------------------------------------------------------------------

#[pyfunction]
fn proximity_edges<'py>(py: Python<'py>, xs: &Bound<'py, PyAny>, ys: &Bound<'py, PyAny>, max_dist: f64) -> PyResult<Bound<'py, PyArray2<i64>>> {
    let (xs, ys) = (vec_f64(xs)?, vec_f64(ys)?);
    let (i, j) = py.detach(|| via::proximity_edges(&xs, &ys, max_dist));
    Ok(arr2(py, i.into_iter().map(|v| v as i64).collect(), j.into_iter().map(|v| v as i64).collect()))
}

#[pyfunction]
fn prune_redundant_edges<'py>(
    py: Python<'py>, xs: &Bound<'py, PyAny>, ys: &Bound<'py, PyAny>, ei: &Bound<'py, PyAny>, ej: &Bound<'py, PyAny>,
    max_dist: f64,
) -> PyResult<Arr1<'py, bool>> {
    let (xs, ys, ei, ej) = (vec_f64(xs)?, vec_f64(ys)?, vec_usize(ei)?, vec_usize(ej)?);
    let keep = py.detach(|| via::prune_redundant_edges(&xs, &ys, &ei, &ej, max_dist));
    Ok(PyArray1::from_vec(py, keep))
}

/// Faces of an undirected planar graph (edge m -> half-edges 2m, 2m+1):
/// returns (face_starts, areas, he_face_id, verts, offsets).
#[pyfunction]
#[allow(clippy::type_complexity)]
fn trace_graph_faces<'py>(
    py: Python<'py>, edge_u: &Bound<'py, PyAny>, edge_v: &Bound<'py, PyAny>, xs: &Bound<'py, PyAny>,
    ys: &Bound<'py, PyAny>,
) -> PyResult<(Arr1<'py, i64>, Arr1<'py, f64>, Arr1<'py, i64>, Arr1<'py, i64>, Arr1<'py, i64>)> {
    let (u, v, xs, ys) = (vec_usize(edge_u)?, vec_usize(edge_v)?, vec_f64(xs)?, vec_f64(ys)?);
    let he = halfedge::HalfEdges::new(&u, &v, &xs, &ys, xs.len());
    let fs = he.trace_faces();
    let areas = he.face_areas(&fs, &xs, &ys);
    let fid = he.face_id_per_halfedge(&fs);
    let (verts, offsets) = he.face_vertex_loops(&fs);
    let i64v = |v: Vec<usize>| PyArray1::from_vec(py, v.into_iter().map(|x| x as i64).collect());
    Ok((i64v(fs), PyArray1::from_vec(py, areas), i64v(fid), i64v(verts), i64v(offsets)))
}

#[pymodule]
fn _rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(edge_intersections, m)?)?;
    m.add_function(wrap_pyfunction!(is_inside, m)?)?;
    m.add_function(wrap_pyfunction!(convex_hull, m)?)?;
    m.add_function(wrap_pyfunction!(sanitize_polygon, m)?)?;
    m.add_function(wrap_pyfunction!(simplify_polyline, m)?)?;
    m.add_function(wrap_pyfunction!(dezigzag_polyline, m)?)?;
    m.add_function(wrap_pyfunction!(is_simple_ring, m)?)?;
    m.add_function(wrap_pyfunction!(regularize_polyline, m)?)?;
    m.add_function(wrap_pyfunction!(boolean_op, m)?)?;
    m.add_function(wrap_pyfunction!(poly_fragment, m)?)?;
    m.add_function(wrap_pyfunction!(profile_boolean_op, m)?)?;
    m.add_function(wrap_pyfunction!(dekeyhole_polygon, m)?)?;
    m.add_function(wrap_pyfunction!(dekeyhole_polygons, m)?)?;
    m.add_function(wrap_pyfunction!(polygon_point_inside, m)?)?;
    m.add_function(wrap_pyfunction!(polygon_is_inside, m)?)?;
    m.add_function(wrap_pyfunction!(proximity_edges, m)?)?;
    m.add_function(wrap_pyfunction!(prune_redundant_edges, m)?)?;
    m.add_function(wrap_pyfunction!(trace_graph_faces, m)?)?;
    Ok(())
}
