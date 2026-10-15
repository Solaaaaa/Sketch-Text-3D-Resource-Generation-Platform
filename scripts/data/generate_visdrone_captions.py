#!/usr/bin/env python3
"""Generate VisDrone captions through a user-supplied vision-language API.

The script is intentionally provider-neutral. It expects an OpenAI-compatible
chat-completions endpoint, but the endpoint, model, and API-key environment
variable are all supplied by the user. No dataset, key, token, or private URL
is embedded in this repository.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
from pathlib import Path

import requests


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_PROMPT = (
    "Describe this aerial urban image in one concise English caption. Mention "
    "buildings, roads, vehicles, people, and viewpoint only when visible."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--conditioning-dir", type=Path, help="Optional paired sketch directory")
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--endpoint", default=os.environ.get("VISION_API_ENDPOINT"))
    parser.add_argument("--model", default=os.environ.get("VISION_API_MODEL"))
    parser.add_argument("--api-key-env", default="VISION_API_KEY")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def encode_image(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def extract_text(response: requests.Response) -> str:
    payload = response.json()
    choices = payload.get("choices") or []
    if not choices:
        raise ValueError("vision API response did not contain choices")
    content = choices[0].get("message", {}).get("content", "")
    if isinstance(content, list):
        content = " ".join(item.get("text", "") for item in content if isinstance(item, dict))
    text = str(content).strip()
    if not text:
        raise ValueError("vision API returned an empty caption")
    return text


def request_caption(session: requests.Session, endpoint: str, model: str, key: str, image: Path, prompt: str, timeout: float) -> str:
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": encode_image(image)}},
            ],
        }],
        "temperature": 0.2,
    }
    response = session.post(
        endpoint,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return extract_text(response)


def main() -> int:
    args = parse_args()
    if not args.endpoint or not args.model:
        raise SystemExit("provide --endpoint/--model or VISION_API_ENDPOINT/VISION_API_MODEL")
    key = os.environ.get(args.api_key_env)
    if not key:
        raise SystemExit(f"missing API key environment variable: {args.api_key_env}")
    if not args.input_dir.is_dir():
        raise NotADirectoryError(args.input_dir)
    if args.output_jsonl.exists() and not args.overwrite:
        raise FileExistsError(f"output exists; pass --overwrite explicitly: {args.output_jsonl}")
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    images = sorted(path for path in args.input_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if args.limit is not None:
        images = images[: args.limit]
    if not images:
        raise FileNotFoundError(f"no supported images found in {args.input_dir}")

    session = requests.Session()
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for image in images:
            caption = request_caption(session, args.endpoint, args.model, key, image, args.prompt, args.timeout)
            record = {"image": str(image), "text": caption}
            if args.conditioning_dir:
                condition = args.conditioning_dir / image.name
                record["conditioning_image"] = str(condition)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"captioned {image.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
