#!/usr/bin/env python3
"""Build a JSONL manifest from paired target/condition images and captions."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--condition-dir", type=Path, required=True)
    parser.add_argument("--captions", type=Path, required=True, help="Caption JSONL")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--caption-field", default="text")
    parser.add_argument("--path-mode", choices=("as-given", "absolute", "relative"), default="as-given")
    parser.add_argument("--path-root", type=Path, help="Root used by --path-mode relative")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def files_by_stem(directory: Path) -> tuple[dict[str, Path], list[str]]:
    files = sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    grouped: dict[str, list[Path]] = {}
    for path in files:
        grouped.setdefault(path.stem, []).append(path)
    duplicates = sorted(stem for stem, paths in grouped.items() if len(paths) > 1)
    return {stem: paths[0] for stem, paths in grouped.items()}, duplicates


def format_path(path: Path, args: argparse.Namespace) -> str:
    if args.path_mode == "absolute":
        return str(path.resolve())
    if args.path_mode == "relative":
        if args.path_root is None:
            raise ValueError("--path-root is required with --path-mode relative")
        return str(path.resolve().relative_to(args.path_root.resolve()))
    return str(path)


def load_captions(path: Path, field: str) -> tuple[dict[str, str], list[str]]:
    captions: dict[str, str] = {}
    duplicate_keys: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            image_value = record.get("image") or record.get("filename")
            text = record.get(field)
            if not image_value or not isinstance(text, str) or not text.strip():
                raise ValueError(f"caption line {line_number} needs image and non-empty {field}")
            key = Path(str(image_value)).stem
            if key in captions:
                duplicate_keys.append(key)
            captions[key] = text.strip()
    return captions, duplicate_keys


def main() -> int:
    args = parse_args()
    for directory in (args.target_dir, args.condition_dir):
        if not directory.is_dir():
            raise NotADirectoryError(directory)
    if not args.captions.is_file():
        raise FileNotFoundError(args.captions)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists; pass --overwrite explicitly: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    targets, target_duplicates = files_by_stem(args.target_dir)
    conditions, condition_duplicates = files_by_stem(args.condition_dir)
    captions, caption_duplicates = load_captions(args.captions, args.caption_field)

    target_keys = set(targets)
    condition_keys = set(conditions)
    caption_keys = set(captions)
    missing_conditions = sorted(target_keys - condition_keys)
    missing_targets = sorted(condition_keys - target_keys)
    missing_captions = sorted(target_keys - caption_keys)
    extra_captions = sorted(caption_keys - target_keys)
    usable = sorted(target_keys & condition_keys & caption_keys)

    with args.output.open("w", encoding="utf-8") as handle:
        for key in usable:
            handle.write(json.dumps({
                "image": format_path(targets[key], args),
                "conditioning_image": format_path(conditions[key], args),
                "text": captions[key],
            }, ensure_ascii=False) + "\n")

    print(f"targets={len(targets)} conditions={len(conditions)} captions={len(captions)}")
    print(f"usable={len(usable)} missing_conditions={len(missing_conditions)} missing_targets={len(missing_targets)} missing_captions={len(missing_captions)}")
    print(f"duplicate_targets={len(target_duplicates)} duplicate_conditions={len(condition_duplicates)} duplicate_captions={len(caption_duplicates)} extra_captions={len(extra_captions)}")
    for label, values in (("missing_condition", missing_conditions), ("missing_target", missing_targets), ("missing_caption", missing_captions)):
        if values:
            print(f"{label}_examples={values[:10]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
