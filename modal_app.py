"""Modal app object for one MuseTalk 1.5 task.

Importing this file is refused while the spend gate is closed, so
`modal run` and `modal deploy` cannot build an image from this checkout
until a person has written an external attestation. This module does not
create a Volume, a Secret, or a web endpoint.
"""

from __future__ import annotations

import os

from modal_musetalk.gate import import_requires_gate, require_open_gate_or_exit

if import_requires_gate(os.environ):
    require_open_gate_or_exit()

import modal  # noqa: E402

from modal_musetalk.job import run_job  # noqa: E402
from modal_musetalk.spec import (  # noqa: E402
    APP_NAME,
    FUNCTION_TIMEOUT_S,
    GPU,
    MAX_CONTAINERS,
    REMOTE_REPO,
    RETRIES,
    SCALEDOWN_WINDOW_S,
    STARTUP_TIMEOUT_S,
    TORCH_INDEX_URL,
    modal_image_ignore,
)

app = modal.App(APP_NAME)

# Package pins live in requirements-modal-inference.txt. The image build that
# would execute these lines has not been run.
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "git", "libgl1", "libglib2.0-0", "wget", "ca-certificates")
    .pip_install(
        "torch==2.0.1",
        "torchvision==0.15.2",
        "torchaudio==2.0.2",
        index_url=TORCH_INDEX_URL,
    )
    .pip_install_from_requirements("requirements-modal-inference.txt")
    .run_commands(
        "pip install --no-cache-dir openmim==0.3.9",
        'mim install "mmcv==2.0.1"',
        "pip install --no-cache-dir 'mmdet==3.1.0' 'mmpose==1.1.0' 'mmengine==0.10.5'",
    )
    .workdir(REMOTE_REPO)
    .add_local_dir(".", remote_path=REMOTE_REPO, copy=False, ignore=modal_image_ignore)
)


@app.function(
    image=image,
    gpu=GPU,
    timeout=FUNCTION_TIMEOUT_S,
    startup_timeout=STARTUP_TIMEOUT_S,
    scaledown_window=SCALEDOWN_WINDOW_S,
    max_containers=MAX_CONTAINERS,
    retries=RETRIES,
)
def infer_once(portrait_png: bytes, audio_wav: bytes, job_id: str) -> dict:
    from pathlib import Path

    return run_job(
        repo_root=Path(REMOTE_REPO),
        job_dir=Path("/tmp") / "musetalk-job" / job_id,
        portrait_png=portrait_png,
        audio_wav=audio_wav,
        job_id=job_id,
        measured_on_gpu=True,
    )
