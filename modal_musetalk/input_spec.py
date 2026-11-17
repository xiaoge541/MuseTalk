"""Declared input contract for the single technical check.

The numbers below are the operator's measured spec for the files that stay
outside this repository. This module does not contain those files.
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

PORTRAIT_WIDTH = 1086
PORTRAIT_HEIGHT = 1448
PORTRAIT_CODEC = "png"

AUDIO_SAMPLE_RATE_HZ = 48000
AUDIO_CHANNELS = 2
AUDIO_CODEC = "pcm_s16le"
AUDIO_DURATION_SEC = 3.377708
AUDIO_DURATION_TOLERANCE_SEC = 0.001

OUTPUT_FPS = 25
STILL_CONTAINER_FPS = 25
STILL_CONTAINER_FRAMES = 85
STILL_CONTAINER_DURATION_SEC = 3.4
STILL_CONTAINER_CODEC = "h264"
STILL_CONTAINER_PIX_FMT = "yuv420p"
STILL_CONTAINER_AUDIO = "none"

# scripts/inference.py counts frames as floor((n_samples / 16000) * fps)
# after librosa.load(..., sr=16000). This uses the measured duration as a
# stand-in for n_samples/16000. The resampled sample count was not measured.
EXPECTED_INFERENCE_FRAMES_FROM_DURATION = math.floor(AUDIO_DURATION_SEC * OUTPUT_FPS)


def expected_inference_frames(duration_sec: float, fps: int = OUTPUT_FPS) -> int:
    return math.floor(float(duration_sec) * int(fps))


def ffprobe_json(path: Path) -> dict:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(path),
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path.name}: {proc.stderr.strip()}")
    return json.loads(proc.stdout)


def validate_portrait_png(path: Path) -> dict:
    if path.suffix.lower() != ".png":
        raise ValueError(
            "portrait must be a PNG file. "
            "input_still_container_NOT_DYNAMIC.mp4 is a silent still-image "
            "container, not a dynamic driving video, and is not accepted here."
        )
    info = ffprobe_json(path)
    streams = info.get("streams") or []
    if not streams:
        raise ValueError("portrait has no ffprobe stream")
    stream = streams[0]
    codec = stream.get("codec_name")
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    pix_fmt = stream.get("pix_fmt") or ""
    if codec != PORTRAIT_CODEC:
        raise ValueError(f"portrait codec is {codec}, expected {PORTRAIT_CODEC}")
    if width != PORTRAIT_WIDTH or height != PORTRAIT_HEIGHT:
        raise ValueError(
            f"portrait size is {width}x{height}, expected {PORTRAIT_WIDTH}x{PORTRAIT_HEIGHT}"
        )
    if "rgb" not in pix_fmt.lower():
        raise ValueError(f"portrait pix_fmt is {pix_fmt}, expected an RGB format")
    return {"codec": codec, "width": width, "height": height, "pix_fmt": pix_fmt}


def validate_original_wav(path: Path) -> dict:
    if path.suffix.lower() != ".wav":
        raise ValueError("original audio must be a .wav path")
    info = ffprobe_json(path)
    audio = None
    for stream in info.get("streams") or []:
        if stream.get("codec_type") == "audio":
            audio = stream
            break
    if audio is None:
        raise ValueError("original audio has no audio stream")
    codec = audio.get("codec_name")
    rate = int(audio.get("sample_rate") or 0)
    channels = int(audio.get("channels") or 0)
    duration = float((info.get("format") or {}).get("duration") or audio.get("duration") or 0)
    if codec != AUDIO_CODEC:
        raise ValueError(f"audio codec is {codec}, expected {AUDIO_CODEC}")
    if rate != AUDIO_SAMPLE_RATE_HZ or channels != AUDIO_CHANNELS:
        raise ValueError(
            f"audio is {rate} Hz / {channels} ch, expected "
            f"{AUDIO_SAMPLE_RATE_HZ} Hz / {AUDIO_CHANNELS} ch"
        )
    if abs(duration - AUDIO_DURATION_SEC) > AUDIO_DURATION_TOLERANCE_SEC:
        raise ValueError(
            f"audio duration {duration} s is outside "
            f"{AUDIO_DURATION_SEC} ± {AUDIO_DURATION_TOLERANCE_SEC}"
        )
    return {
        "codec": codec,
        "sample_rate_hz": rate,
        "channels": channels,
        "duration_sec": duration,
    }


def full_decode_command(path: Path) -> list[str]:
    return ["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"]


def ffprobe_command(path: Path) -> list[str]:
    return [
        "ffprobe",
        "-v",
        "error",
        "-show_format",
        "-show_streams",
        str(path),
    ]
