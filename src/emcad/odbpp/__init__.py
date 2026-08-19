"""odbpp — a native-Python, SI-normalized object model for ODB++ PCB data.

Quick start -- raw object graph
--------------------------------
    import odbpp

    pm = odbpp.ProductModel.from_path("/path/to/MyBoard.odb")
    step = pm.steps["pcb"]
    top_copper = step.layers["top"].features        # FeaturesFile
    for feature in top_copper.features:
        if isinstance(feature, odbpp.Stamp):
            island = feature.to_island()             # geometry, in meters

Quick start -- resolved, Z-stacked geometry
---------------------------------------------
    pcb = odbpp.PCBView("/path/to/MyBoard.odb", thickness_stack=[35e-6, 508e-6, 35e-6])
    xs, ys = pcb.get_board_polygon()
    for layer in pcb.iter_geo_layers():
        polygons = pcb.resolve_layer_polygons(layer)   # boolean-unified

``PCBView`` parses the board itself (via ``ProductModel.from_path``) when
given a path -- or pass it an already-parsed ``ProductModel`` if you built
one yourself for the raw-object-graph walk above and want to reuse it rather
than parsing twice.

Every coordinate everywhere in this package is a plain float in **meters**.
All curve/arc discretization resolution is controlled from one place —
``odbpp.config.TESSELLATION`` — see that module's docstring; set
``odbpp.config.TESSELLATION.max_angle_deg = ...`` once to change curve
resolution everywhere, or pass ``max_angle_deg=`` to an individual call to
override it just there.

See each submodule's docstring for the ODB++ spec section it implements:

    odbpp.config     shared tessellation/discretization defaults
    odbpp.units      unit conversion (mm/inch/mil/micron -> meters)
    odbpp.geometry   Point / Polygon / Island / Surface + shape builders
    odbpp.symbols    standard + user-defined symbol classes and name parser
    odbpp.features   Stamp / Line / Arc / TextFeature / BarcodeFeature /
                      SurfaceFeature + the features-file parser
    odbpp.layers     DrillTool / ToolsTable / Layer + their parsers
    odbpp.matrix     MatrixStep / MatrixLayer / Matrix (layer stack) + parser
    odbpp.model      Step / ProductModel (ties everything together) --
                      ProductModel.from_path() is the main entry point
    odbpp.parse      parse_product_model() — the underlying parse function
                      ProductModel.from_path() wraps
    odbpp.design     PCBView — high-level polygon/Z-stack modeling helper
    odbpp.cache      save_pcb_cache / load_pcb_cache — msgpack snapshot of a
                      PCBView's resolved geometry, skips re-parsing +
                      re-resolving on repeat runs against the same board
"""
from . import config
from .config import TessellationConfig
from .geometry import Island, Point, Polygon, Surface
from .symbols import (
    ButterflySymbol,
    ChamferedRectangleSymbol,
    DiamondSymbol,
    DonutSymbol,
    HalfOvalSymbol,
    HexagonSymbol,
    OctagonSymbol,
    OvalDonutSymbol,
    OvalSymbol,
    RectDonutSymbol,
    RectangleSymbol,
    RoundedRectangleSymbol,
    RoundedSquareDonutSymbol,
    RoundSymbol,
    SquareSymbol,
    Symbol,
    ThermalSymbol,
    TriangleSymbol,
    UserDefinedSymbol,
    parse_symbol_name,
)
from .features import (
    Arc,
    BarcodeFeature,
    Feature,
    FeaturesFile,
    Line,
    Stamp,
    SurfaceFeature,
    TextFeature,
    parse_features_file,
)
from .layers import DrillTool, Layer, ToolsTable, parse_layer, parse_tools_file
from .matrix import Matrix, MatrixLayer, MatrixStep, parse_matrix_file
from .model import ProductModel, Step
from .parse import parse_product_model
from .design import DrillHole, GeoLayer, LayerPolygon, PCBView, buffer_segment
from .cache import CachedPCBView, save_pcb_cache, load_pcb_cache
from .units import (
    coord_to_meters,
    default_symbol_size_units,
    symbol_size_to_meters,
)

__all__ = [
    # entry point
    "parse_product_model",
    # model
    "ProductModel", "Step",
    # designer (high-level FEM-geometry helper)
    "PCBView", "GeoLayer", "LayerPolygon",
    # layers / tools / matrix
    "Layer", "parse_layer", "ToolsTable", "DrillTool", "parse_tools_file",
    "Matrix", "MatrixStep", "MatrixLayer", "parse_matrix_file",
    # features
    "FeaturesFile", "parse_features_file", "Feature",
    "Stamp", "Line", "Arc", "TextFeature", "BarcodeFeature", "SurfaceFeature",
    # symbols
    "Symbol", "parse_symbol_name", "UserDefinedSymbol",
    "RoundSymbol", "SquareSymbol", "RectangleSymbol",
    "RoundedRectangleSymbol", "ChamferedRectangleSymbol",
    "OvalSymbol", "DiamondSymbol", "OctagonSymbol", "HexagonSymbol",
    "ButterflySymbol", "TriangleSymbol", "HalfOvalSymbol",
    "DonutSymbol", "RoundedSquareDonutSymbol", "RectDonutSymbol", "OvalDonutSymbol",
    "ThermalSymbol",
    # geometry
    "Point", "Polygon", "Island", "Surface",
    # design (high-level polygon/Z-stack modeling helper)
    "PCBView", "GeoLayer", "LayerPolygon", "DrillHole", "buffer_segment",
    # cache (save/load a PCBView's resolved geometry)
    "CachedPCBView", "save_pcb_cache", "load_pcb_cache",
    # units
    "coord_to_meters", "symbol_size_to_meters", "default_symbol_size_units",
    # config (global tessellation/discretization settings)
    "config", "TessellationConfig",
]

__version__ = "0.1.0"
