
from __future__ import annotations

import os
from pathlib import Path


os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics-config")
os.environ.setdefault("YOLO_OFFLINE", "true")

from PIL import Image
from ultralytics import YOLO

from large_image import needs_tiling, predict_tiled
from postprocess import apply_category_thresholds, detections_to_objects
from preprocess import prepare_image

CONTAINER_MODEL_PATH = Path("/app/models/best.pt")
LOCAL_MODEL_PATH = Path(__file__).resolve().parent / "models" / "best.pt"
CATEGORY_THRESHOLDS = (
    {"class_ids": range(0, 4), "conf": 0.67, "iou": 0.75},  # ships
    {"class_ids": range(4, 24), "conf": 0.80, "iou": 0.70},  # aircraft
    {"class_ids": (24,), "conf": 0.72, "iou": 0.70},  # vehicles
)



class Detector:

    def __init__(self) -> None:
        model_path = CONTAINER_MODEL_PATH if CONTAINER_MODEL_PATH.is_file() else LOCAL_MODEL_PATH
        if not model_path.is_file():
            raise FileNotFoundError(f"Model weights not found: {model_path}")

        self.model = YOLO(str(model_path), task="detect")
        self.predict_options = {
            "imgsz": 640,
            "batch": 1,
            "device": 0,
            # Keep every candidate needed by the category-specific postprocess below.
            "conf": min(group["conf"] for group in CATEGORY_THRESHOLDS),
            "iou": max(group["iou"] for group in CATEGORY_THRESHOLDS),
            "end2end": False,
            "max_det": 300,
            "save": False,
            "verbose": False,
        }

    def predict(self, image: Image.Image) -> list[dict]:
        prepared = prepare_image(image)
        if needs_tiling(prepared):
            return predict_tiled(self.model, prepared, self.predict_options, CATEGORY_THRESHOLDS)
        results = self.model.predict(source=prepared, **self.predict_options)
        if len(results) != 1:
            raise RuntimeError(f"Expected one prediction result, received {len(results)}")
        objects = detections_to_objects(results[0], prepared.size)
        model_device = next(self.model.model.parameters()).device
        return apply_category_thresholds(objects, CATEGORY_THRESHOLDS, model_device)
