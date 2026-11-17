"""One local job directory: preprocess audio, call scripts.inference, remux, verify.

The default runners are the real ffmpeg binary and `python -m scripts.inference`.
Tests inject runners so this module can execute without weights or a GPU.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from modal_musetalk.media_commands import audio_to_16k_mono, remux_original_audio
from modal_musetalk.input_spec import ffprobe_json, full_decode_command
from modal_musetalk.usage import empty_usage
from modal_musetalk.weights import prepare_weights

_JOB_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_SAFE_PATH = re.compile(r"^[A-Za-z0-9_./-]+$")


def validate_job_id(job_id: str) -> str:
    if not _JOB_ID.fullmatch(job_id):
        raise ValueError("job id must match ^[a-z0-9][a-z0-9-]{0,63}$")
    return job_id


def inference_argv(repo_root: Path, config_path: Path, result_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "scripts.inference",
        "--inference_config",
        str(config_path),
        "--result_dir",
        str(result_dir),
        "--unet_model_path",
        str(repo_root / "models" / "musetalkV15" / "unet.pth"),
        "--unet_config",
        str(repo_root / "models" / "musetalkV15" / "musetalk.json"),
        "--whisper_dir",
        str(repo_root / "models" / "whisper"),
        "--vae_type",
        "sd-vae",
        "--version",
        "v15",
        "--use_float16",
        "--fps",
        "25",
        "--batch_size",
        "8",
        "--extra_margin",
        "10",
        "--parsing_mode",
        "jaw",
        "--left_cheek_width",
        "90",
        "--right_cheek_width",
        "90",
        "--audio_padding_length_left",
        "2",
        "--audio_padding_length_right",
        "2",
        "--ffmpeg_path",
        "ffmpeg",
        "--gpu_id",
        "0",
        "--output_vid_name",
        "result.mp4",
    ]


def _require_safe(path: Path) -> None:
    if not _SAFE_PATH.fullmatch(str(path)):
        raise ValueError(f"path is not shell-safe for scripts.inference: {path.name}")


def run_job(
    *,
    repo_root: Path,
    job_dir: Path,
    portrait_png: bytes,
    audio_wav: bytes,
    job_id: str,
    runner=None,
    weight_preparer=prepare_weights,
    measured_on_gpu: bool = False,
) -> dict:
    """Run one prepared job.

    Billing fields in the usage dict stay null unless `measured_on_gpu` is
    set by the Modal function after a real GPU process. A CPU stand-in does
    not copy its wall clock into those fields.
    """
    if runner is None:
        runner = default_runner
    validate_job_id(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    portrait_path = job_dir / "portrait.png"
    audio_path = job_dir / "audio_original.wav"
    audio_16k = job_dir / "audio_16k_mono.wav"
    portrait_path.write_bytes(portrait_png)
    audio_path.write_bytes(audio_wav)
    for path in (portrait_path, audio_path, audio_16k, job_dir):
        _require_safe(path)

    usage = empty_usage()
    local_wall: dict[str, float] = {}
    started = time.perf_counter()

    t0 = time.perf_counter()
    runner(audio_to_16k_mono(audio_path, audio_16k))
    local_wall["ffmpeg_audio_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    weight_preparer(repo_root / "models")
    local_wall["weight_download_s"] = time.perf_counter() - t0

    result_dir = job_dir / "inference_results"
    config_path = job_dir / "job.yaml"
    _require_safe(result_dir)
    _require_safe(config_path)
    config_path.write_text(
        "task_0:\n"
        f'  video_path: "{portrait_path}"\n'
        f'  audio_path: "{audio_16k}"\n'
        '  result_name: "result.mp4"\n',
        encoding="utf-8",
    )
    argv = inference_argv(repo_root, config_path, result_dir)
    if "-m" not in argv or "scripts.inference" not in argv:
        raise RuntimeError("inference argv is not scripts.inference")
    if "--version" not in argv or "v15" not in argv or "--use_float16" not in argv:
        raise RuntimeError("inference argv is not MuseTalk 1.5 fp16")
    t0 = time.perf_counter()
    if _accepts_cwd(runner):
        runner(argv, cwd=str(repo_root))
    else:
        runner(argv)
    local_wall["inference_subprocess_s"] = time.perf_counter() - t0

    produced = result_dir / "v15" / "result.mp4"
    if not produced.is_file() or produced.stat().st_size <= 0:
        raise RuntimeError(
            "scripts.inference did not leave result.mp4. "
            "The client must not submit this job id again."
        )

    final_path = job_dir / "result.mp4"
    _require_safe(final_path)
    t0 = time.perf_counter()
    runner(remux_original_audio(produced, audio_path, final_path))
    local_wall["ffmpeg_remux_s"] = time.perf_counter() - t0
    decode = full_decode_command(final_path)
    runner(decode)
    probe = ffprobe_json(final_path)
    local_wall["function_wall_s"] = time.perf_counter() - started
    # Timing numbers from a CPU stand-in stay out of the billing fields.
    usage["measured_on_gpu"] = bool(measured_on_gpu)
    if measured_on_gpu:
        for key, value in local_wall.items():
            usage[key] = value
    mp4 = final_path.read_bytes()
    return {
        "mp4": mp4,
        "mp4_sha256": hashlib.sha256(mp4).hexdigest(),
        "ffprobe": probe,
        "usage": usage,
        "output_name": "result.mp4",
        "job_id": job_id,
        "code_base_sha": os.environ.get("MUSETALK_CODE_BASE_SHA", ""),
    }


def _accepts_cwd(runner) -> bool:
    return getattr(runner, "accepts_cwd", False)


def default_runner(cmd: list[str], cwd: str | None = None) -> None:
    if "app.py" in cmd:
        raise RuntimeError("refusing to launch app.py; the image entry is scripts.inference")
    proc = subprocess.run(cmd, check=False, cwd=cwd)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {Path(cmd[0]).name}")


default_runner.accepts_cwd = True  # type: ignore[attr-defined]
