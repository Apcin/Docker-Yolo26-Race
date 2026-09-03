from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from detector import Detector
from preprocess import discover_images, load_images


def check_gpu() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required, but torch.cuda.is_available() is False. "
            "Start the container with Docker's --gpus option."
        )
    if torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA is available but no GPU is visible to the container.")

    torch.cuda.set_device(0)
    torch.empty(1, device="cuda:0")
    print(f"Using cuda:0 ({torch.cuda.get_device_name(0)})", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO26 detection on an image directory.")
    parser.add_argument("--input", required=True, help="Directory containing input images (first level only).")
    parser.add_argument("--output", required=True, help="Directory in which result.json will be written.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    check_gpu()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    paths = discover_images(input_dir)


    images = load_images(paths)

    detector = Detector()
    output_dir.mkdir(parents=True, exist_ok=True)

    final_results = []
    for path, image in images:
        objects = detector.predict(image)
        run_end_timestamp = time.time_ns() // 1_000_000
        final_results.append(
            {
                "image_id": path.stem,
                "file_name": path.name,
                "width": image.width,
                "height": image.height,
                "run_end_timestamp": run_end_timestamp,
                "objects": objects,
            }
        )

    with (output_dir / "result.json").open("w", encoding="utf-8") as file:
        json.dump(
            {"status": "success", "images": final_results},
            file,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
