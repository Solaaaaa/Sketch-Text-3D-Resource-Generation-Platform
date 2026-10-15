#!/usr/bin/env python3
"""Prepare VisDrone images and hand-drawn Canny controls for FLUX ControlNet.

This script combines the final square-padding and hand-drawn edge ideas used
for the VisDrone-to-FLUX training set. It never removes or moves an existing
output directory; use ``--overwrite`` to replace files deliberately.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def pad_to_square(image: Image.Image, size: int) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    scale = size / max(width, height)
    resized = image.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    return canvas


class HandDrawnCanny:
    """Canny plus the contour jitter, gaps, and sparse marks used in training."""

    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)

    def generate(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        canvas = np.zeros_like(edges)

        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            if perimeter < 10:
                continue
            contour = cv2.approxPolyDP(contour, 0.8 * 0.01 * perimeter, True)
            for index, point in enumerate(contour):
                next_point = contour[(index + 1) % len(contour)]
                x1, y1 = (int(value) for value in point[0])
                x2, y2 = (int(value) for value in next_point[0])
                distance = float(np.hypot(x2 - x1, y2 - y1))
                width = int(1 + (5 - 1) * min(1.0, distance / 50.0) * self.rng.uniform(0, 0.7))
                jitter = 3
                p1 = (x1 + self.rng.randint(-jitter, jitter), y1 + self.rng.randint(-jitter, jitter))
                p2 = (x2 + self.rng.randint(-jitter, jitter), y2 + self.rng.randint(-jitter, jitter))
                if self.rng.random() >= 0.1:
                    cv2.line(canvas, p1, p2, 255, max(1, min(5, width)))

        height, width = canvas.shape
        for _ in range(int(height * width * 2 * 0.00005)):
            x1 = self.rng.randrange(width)
            y1 = self.rng.randrange(height)
            length = self.rng.randint(3, 15)
            angle = self.rng.uniform(0, 2 * np.pi)
            x2 = max(0, min(width - 1, int(x1 + length * np.cos(angle))))
            y2 = max(0, min(height - 1, int(y1 + length * np.sin(angle))))
            cv2.line(canvas, (x1, y1), (x2, y2), 255, self.rng.randint(1, 3))

        for _ in range(int(height * width * 2 * 0.00001)):
            cv2.circle(canvas, (self.rng.randrange(width), self.rng.randrange(height)), self.rng.randint(1, 3), 255, -1)

        edge_pixels = np.where(canvas > 0)
        remove_count = int(len(edge_pixels[0]) * 0.1)
        for _ in range(remove_count):
            if not len(edge_pixels[0]):
                break
            index = self.rng.randrange(len(edge_pixels[0]))
            canvas[edge_pixels[0][index], edge_pixels[1][index]] = 0

        canvas = cv2.GaussianBlur(canvas, (3, 3), 0.5)
        _, canvas = cv2.threshold(canvas, 100, 255, cv2.THRESH_BINARY)
        return cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)


def image_files(root: Path) -> list[Path]:
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    names = [path.name for path in files]
    duplicates = {name for name in names if names.count(name) > 1}
    if duplicates:
        raise ValueError(f"duplicate image basenames in input: {sorted(duplicates)[:5]}")
    return files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Original VisDrone image directory")
    parser.add_argument("--target-dir", type=Path, required=True, help="Output directory for square target images")
    parser.add_argument("--condition-dir", type=Path, required=True, help="Output directory for Canny/sketch controls")
    parser.add_argument("--size", type=int, default=512, help="Square output resolution (default: 512)")
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic sketch perturbations")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing files in existing output directories")
    return parser.parse_args()


def check_output_dirs(args: argparse.Namespace) -> None:
    for directory in (args.target_dir, args.condition_dir):
        if directory.exists() and not args.overwrite:
            raise FileExistsError(f"output directory exists; pass --overwrite explicitly: {directory}")
        directory.mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()
    if args.size < 16:
        raise ValueError("--size must be at least 16")
    if not args.input_dir.is_dir():
        raise NotADirectoryError(args.input_dir)
    check_output_dirs(args)
    files = image_files(args.input_dir)
    if not files:
        raise FileNotFoundError(f"no supported images found in {args.input_dir}")

    processed = 0
    failed = 0
    for source in files:
        target_path = args.target_dir / f"{source.stem}.jpg"
        condition_path = args.condition_dir / f"{source.stem}.jpg"
        try:
            with Image.open(source) as image:
                square = pad_to_square(image, args.size)
            square.save(target_path, format="JPEG", quality=95)
            condition = HandDrawnCanny(args.seed + processed).generate(np.asarray(square))
            Image.fromarray(condition).save(condition_path, format="JPEG", quality=95)
            processed += 1
        except Exception as exc:  # keep the batch moving while reporting failures
            failed += 1
            print(f"failed: {source}: {exc}")

    print(f"processed={processed} failed={failed} total={len(files)} size={args.size}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
