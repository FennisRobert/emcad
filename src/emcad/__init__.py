from .poly import Polygon, Edge, Vertex
from .plot import GeometryPlotter
from .glob import GlobalSettings
from .kernel.api import simplify_polyline, convex_hull, edge_self_intersections, edge_cross_intersections, join_polygons, add_polygons, subtract_polygons, intersect_polygons, poly_fragment
from .viaconnect import via_wall_polygons