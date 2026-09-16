#!/usr/bin/env python3
"""Download MediaPipe AI models into backend/models/.

Run once before first launch (or when upgrading models)::

    cd backend
    python scripts/download_models.py

The script is idempotent: existing files are skipped unless ``--force`` is set.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

MODEL_URLS = {
    "face_detector.tflite": (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
    ),
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    ),
    "efficientdet_lite0.tflite": (
        "https://storage.googleapis.com/mediapipe-models/"
        "object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite"
    ),
}


def download(url: str, dest: Path) -> None:
    """Download *url* to *dest*, showing progress."""
    print(f"  Downloading {dest.name} …")
    urllib.request.urlretrieve(url, str(dest))
    print(f"  ✓ saved to {dest}  ({dest.stat().st_size:,} bytes)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="re-download even if files exist")
    args = parser.parse_args(argv)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    ok = True
    for name, url in MODEL_URLS.items():
        dest = MODELS_DIR / name
        if dest.exists() and not args.force:
            print(f"  {dest.name} already exists — skipping")
            continue
        try:
            download(url, dest)
        except Exception as exc:
            print(f"  ✗ FAILED to download {name}: {exc}", file=sys.stderr)
            ok = False

    if ok:
        print("\nAll models ready.")
    else:
        print("\nSome downloads failed — the app will degrade gracefully.", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())