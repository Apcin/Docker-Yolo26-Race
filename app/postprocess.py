
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def _category_name(names: Mapping | Sequence, category_id: int) -> str:
    try:
        return str(names[category_id])
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError(f"No category name for class index {category_id}") from error


def detections_to_objects(result, image_size: tuple[int, int]) -> list[dict]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    width, height = image_size
    xyxy = boxes.xyxy.detach().cpu().tolist()
    scores = boxes.conf.detach().cpu().tolist()
    classes = boxes.cls.detach().cpu().tolist()

    objects = []
    for coordinates, score, class_value in zip(xyxy, scores, classes, strict=True):
        values = [float(value) for value in coordinates]
        score = float(score)
        if not all(math.isfinite(value) for value in [*values, score]):
            raise RuntimeError("Model produced a non-finite bounding box or confidence score")
        x1, y1, x2, y2 = values
        bbox = [
            min(max(x1, 0.0), float(width)),
            min(max(y1, 0.0), float(height)),
            min(max(x2, 0.0), float(width)),
            min(max(y2, 0.0), float(height)),
        ]
        category_id = int(class_value)
        objects.append(
            {
                "category_id": category_id,
                "category_name": _category_name(result.names, category_id),
                "score": min(max(score, 0.0), 1.0),
                "bbox": bbox,
            }
        )
    return objects


def apply_category_thresholds(objects: list[dict], category_thresholds, device) -> list[dict]:
    """Apply a confidence threshold and same-class NMS IoU threshold to each category group."""
    import torch
    from torchvision.ops import batched_nms

    kept_objects = []
    for group in category_thresholds:
        class_ids = set(group["class_ids"])
        candidates = [
            obj for obj in objects if obj["category_id"] in class_ids and obj["score"] >= group["conf"]
        ]
        if not candidates:
            continue

        boxes = torch.tensor([obj["bbox"] for obj in candidates], dtype=torch.float32, device=device)
        scores = torch.tensor([obj["score"] for obj in candidates], dtype=torch.float32, device=device)
        classes = torch.tensor([obj["category_id"] for obj in candidates], dtype=torch.int64, device=device)
        keep = batched_nms(boxes, scores, classes, float(group["iou"])).cpu().tolist()
        kept_objects.extend(candidates[index] for index in keep)

    return sorted(kept_objects, key=lambda obj: obj["score"], reverse=True)
