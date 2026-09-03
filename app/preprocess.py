from __future__ import annotations

from pathlib import Path

from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def discover_images(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def load_images(paths: list[Path]) -> list[tuple[Path, Image.Image]]:
    loaded = []
    for path in paths:
        with Image.open(path) as image:
            loaded.append((path, image.convert("RGB").copy()))
    return loaded


def prepare_image(image: Image.Image) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL.Image.Image, received {type(image).__name__}")
    return image if image.mode == "RGB" else image.convert("RGB")
