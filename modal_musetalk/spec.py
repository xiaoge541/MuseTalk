"""Pinned constants for the single MuseTalk 1.5 Modal task.

Prices are the public list rates read from https://modal.com/pricing on
2026-11-17. They are not a quote, an invoice, or a measurement of this job.
"""

from __future__ import annotations

from pathlib import Path

APP_NAME = "musetalk15-starter-once"
FUNCTION_NAME = "infer_once"

# SDK 1.6.4 proto enum is GPU_TYPE_T4 / GPU_TYPE_L4 / GPU_TYPE_A10G.
# The pricing page labels the third card "Nvidia A10" at the A10G rate below.
GPU = "T4"
ALLOWED_GPUS = ("T4", "L4", "A10G")
GPU_USD_PER_SEC = {
    "T4": "0.000164",
    "L4": "0.000222",
    "A10G": "0.000306",
}
CPU_USD_PER_CORE_SEC = "0.0000131"
MEMORY_USD_PER_GIB_SEC = "0.00000222"
VOLUME_USD_PER_GIB_MONTH = "0.09"
EGRESS_USD_PER_GIB = "0.04"

# Modal documents scaledown_window as 2 seconds through 20 minutes.
# Default idle billing is 60 seconds. This task sets the documented minimum.
SCALEDOWN_WINDOW_S = 2
FUNCTION_TIMEOUT_S = 3600
STARTUP_TIMEOUT_S = 600
CLIENT_WAIT_S = 1800
RETRIES = 0
MAX_CONTAINERS = 1

MODAL_SDK_PIN = "1.6.4"
PYTHON_IMAGE = "3.10"
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu118"
CODE_BASE_SHA = "0a89dec45a0192b824e3cf4daf96c239440c5ed8"
REMOTE_REPO = "/root/musetalk"

# Pricing page fetch date. Not a guarantee the rate is still in force.
PRICE_SOURCE = "https://modal.com/pricing"
PRICE_FETCHED_ON = "2026-11-17"

MEDIA_SUFFIXES = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".gif",
    ".bmp",
}
SKIP_DIR_NAMES = {
    "assets",
    "data",
    "models",
    "results",
    "dataset",
    ".git",
    ".musetalk_modal_calls",
    "__pycache__",
    "out",
}
WEIGHT_SUFFIXES = {".pth", ".pt", ".bin", ".safetensors", ".ckpt", ".onnx"}


def modal_image_ignore(path: Path) -> bool:
    """Return True when a local path must not be uploaded into the Modal image."""
    parts = set(path.parts)
    if parts & SKIP_DIR_NAMES:
        return True
    suffix = path.suffix.lower()
    if suffix in MEDIA_SUFFIXES or suffix in WEIGHT_SUFFIXES:
        return True
    name = path.name.lower()
    if "attestation" in name or name.endswith(".pyc"):
        return True
    return False
