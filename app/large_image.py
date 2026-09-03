"""Tiled inference helpers for images much larger than the training crops."""

from __future__ import annotations

from collections.abc import Mapping

from PIL import Image

from postprocess import apply_category_thresholds, detections_to_objects

LARGE_IMAGE_THRESHOLD = 1280
TILE_SIZE = 800
TILE_OVERLAP = 200
TILE_STRIDE = TILE_SIZE - TILE_OVERLAP
TILE_BATCH_SIZE = 8
INTERNAL_EDGE_MARGIN = 2.0


def needs_tiling(image: Image.Image) -> bool:
    """Keep all existing <=1280 px images on the original inference path."""
    return image.width > LARGE_IMAGE_THRESHOLD or image.height > LARGE_IMAGE_THRESHOLD


def _axis_starts(length: int) -> list[int]:
    """Cover an axis fully, aligning the last tile with the far image edge."""
    if length <= TILE_SIZE:
        return [0]
    last = length - TILE_SIZE
    starts = list(range(0, last + 1, TILE_STRIDE))
    if starts[-1] != last:
        starts.append(last)
    return starts


def _tile_bounds(image: Image.Image) -> list[tuple[int, int, int, int]]:
    return [
        (left, top, min(left + TILE_SIZE, image.width), min(top + TILE_SIZE, image.height))
        for top in _axis_starts(image.height)
        for left in _axis_starts(image.width)
    ]


def _touches_internal_edge(
    bbox: list[float],
    bounds: tuple[int, int, int, int],
    image_size: tuple[int, int],
) -> bool:
    """Reject boxes clipped by a tile seam; an overlapping tile sees them whole."""
    x1, y1, x2, y2 = bbox
    left, top, right, bottom = bounds
    image_width, image_height = image_size
    tile_width, tile_height = right - left, bottom - top
    return (
        (left > 0 and x1 <= INTERNAL_EDGE_MARGIN)
        or (top > 0 and y1 <= INTERNAL_EDGE_MARGIN)
        or (right < image_width and x2 >= tile_width - INTERNAL_EDGE_MARGIN)
        or (bottom < image_height and y2 >= tile_height - INTERNAL_EDGE_MARGIN)
    )


def predict_tiled(model, image: Image.Image, predict_options: Mapping, category_thresholds) -> list[dict]:
    """Predict overlapping tiles, restore full-image coordinates, and merge duplicates."""
    bounds = _tile_bounds(image)
    full_objects: list[dict] = []
    tile_options = dict(predict_options)

    for start in range(0, len(bounds), TILE_BATCH_SIZE):
        batch_bounds = bounds[start : start + TILE_BATCH_SIZE]
        tiles = [image.crop(tile_bounds) for tile_bounds in batch_bounds]
        tile_options["batch"] = len(tiles)
        results = model.predict(source=tiles, **tile_options)
        if len(results) != len(tiles):
            raise RuntimeError(f"Expected {len(tiles)} tile results, received {len(results)}")

        for result, tile, tile_bounds in zip(results, tiles, batch_bounds, strict=True):
            left, top, _, _ = tile_bounds
            for obj in detections_to_objects(result, tile.size):
                if _touches_internal_edge(obj["bbox"], tile_bounds, image.size):
                    continue
                x1, y1, x2, y2 = obj["bbox"]
                obj["bbox"] = [
                    x1 + left,
                    y1 + top,
                    x2 + left,
                    y2 + top,
                ]
                full_objects.append(obj)

    model_device = next(model.model.parameters()).device
    return apply_category_thresholds(full_objects, category_thresholds, model_device)
