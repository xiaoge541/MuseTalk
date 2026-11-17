"""Reproducible ffmpeg commands. They write a new file and refuse to overwrite the source."""

from __future__ import annotations

from pathlib import Path


def _reject_same_path(src: Path, dst: Path) -> None:
    if src.resolve() == dst.resolve():
        raise ValueError(f"refusing to overwrite source {src.name}")


def audio_to_16k_mono(src: Path, dst: Path) -> list[str]:
    """48 kHz stereo (or any input) -> 16 kHz mono PCM16 WAV, new file."""
    _reject_same_path(src, dst)
    return [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(dst),
    ]


def remux_original_audio(video_mp4: Path, original_wav: Path, dst: Path) -> list[str]:
    """Copy the generated picture and attach the untouched original WAV as AAC."""
    _reject_same_path(original_wav, dst)
    _reject_same_path(video_mp4, dst)
    return [
        "ffmpeg",
        "-y",
        "-i",
        str(video_mp4),
        "-i",
        str(original_wav),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(dst),
    ]


def still_container_from_png(portrait_png: Path, dst: Path) -> list[str]:
    """Document-only rebuild of a silent 25 fps still container.

    The Modal job does not run this. scripts.inference would treat the MP4 as
    a video and extract every repeated frame. Pass the PNG instead.
    """
    _reject_same_path(portrait_png, dst)
    return [
        "ffmpeg",
        "-y",
        "-loop",
        "1",
        "-i",
        str(portrait_png),
        "-t",
        "3.4",
        "-r",
        "25",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(dst),
    ]
