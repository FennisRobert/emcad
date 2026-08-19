"""Main entry point: read an ODB++ product model directory tree into a
:class:`odbpp.model.ProductModel`.

Expected directory layout (see ODB++Design Format Specification, Chapter 2
"Product Model Tree", p.43)::

    <product_model>/
      matrix/matrix
      symbols/<symbol_name>/features
      steps/<step_name>/
        profile
        layers/<layer_name>/
          features
          profile
          tools
          attrlist

Individual pieces can also be parsed on their own via the re-exported
``parse_features_file``, ``parse_matrix_file``, ``parse_tools_file``, and
``parse_layer`` functions, if you don't have (or don't want to load) a full
product model tree.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from loguru import logger

from .features import parse_features_file
from .layers import Layer, parse_layer, parse_tools_file
from .matrix import Matrix, parse_matrix_file
from .model import ProductModel, Step


def parse_product_model(root: Union[str, Path], max_angle_deg: Optional[float] = None) -> ProductModel:
    """Parse a full ODB++ product model directory tree.

    ``root`` is the ``<product_model>`` directory itself (the one that
    directly contains ``matrix/``, ``steps/``, and ``symbols/``).

    ``max_angle_deg`` controls curve tessellation resolution for every
    curved outline/surface in the tree (board profile, layer profiles,
    layer features, symbol features): each arc is split into enough equal
    segments that no segment sweeps more than this many degrees. ``None``
    (the default) uses the shared :mod:`odbpp.config` setting — see that
    module to change the default once, globally, instead of passing this
    everywhere. Pass a smaller value here if this specific board has large
    sweeping curves you want rendered smoothly, or a larger one to keep
    point counts down on boards with many small fillets/cutouts.
    """
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a product model directory: {root}")

    logger.debug(f"parse_product_model: reading '{root}'")

    matrix_path = root / "matrix" / "matrix"
    matrix = parse_matrix_file(matrix_path) if matrix_path.exists() else Matrix()

    symbol_library = {}
    symbols_dir = root / "symbols"
    if symbols_dir.is_dir():
        for sym_dir in symbols_dir.iterdir():
            if not sym_dir.is_dir():
                continue
            features_path = sym_dir / "features"
            if features_path.exists() or features_path.with_name("features.gz").exists():
                symbol_library[sym_dir.name] = parse_features_file(features_path, max_angle_deg=max_angle_deg)
    logger.debug(f"parse_product_model: {len(symbol_library)} symbol(s) in library")

    steps: dict = {}
    steps_dir = root / "steps"
    if steps_dir.is_dir():
        for step_dir in steps_dir.iterdir():
            if not step_dir.is_dir():
                continue
            logger.debug(f"parse_product_model: parsing step '{step_dir.name}'")
            steps[step_dir.name] = _parse_step(step_dir, max_angle_deg)

    logger.debug(f"parse_product_model: done -- {len(steps)} step(s), {len(matrix.layers)} matrix layer(s)")
    return ProductModel(name=root.name, matrix=matrix, steps=steps, symbol_library=symbol_library)


def _parse_step(step_dir: Path, max_angle_deg: Optional[float] = None) -> Step:
    profile = None
    profile_path = step_dir / "profile"
    if profile_path.exists() or profile_path.with_name("profile.gz").exists():
        profile = parse_features_file(profile_path, max_angle_deg=max_angle_deg)

    layers: dict = {}
    layers_dir = step_dir / "layers"
    if layers_dir.is_dir():
        for layer_dir in layers_dir.iterdir():
            if layer_dir.is_dir():
                logger.debug(f"parse_product_model: parsing layer '{layer_dir.name}' (step '{step_dir.name}')")
                layers[layer_dir.name] = parse_layer(layer_dir, max_angle_deg=max_angle_deg)

    return Step(name=step_dir.name, profile=profile, layers=layers)
