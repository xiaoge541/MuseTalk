"""Weight directories and pinned revisions for one MuseTalk 1.5 inference.

`plan()` is data. `prepare_weights()` performs downloads only when called.
This module does not create a Modal volume.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

# Byte sizes are the Hugging Face tree API `size` fields for these revisions,
# read 2026-11-17. They are file sizes, not measured download times.
HF_FILES = (
    {
        "repo_id": "TMElyralab/MuseTalk",
        "revision": "2bcb936e2fddb4d86db4c62fd45b387d0c061571",
        "local_subdir": ".",
        "allow_patterns": ["musetalkV15/musetalk.json", "musetalkV15/unet.pth"],
        "files": {
            "musetalkV15/musetalk.json": 748,
            "musetalkV15/unet.pth": 3400074924,
        },
        "hf_license_field": "mit",
    },
    {
        "repo_id": "stabilityai/sd-vae-ft-mse",
        "revision": "31f26fdeee1355a5c34592e401dd41e45d25a493",
        "local_subdir": "sd-vae",
        "allow_patterns": ["config.json", "diffusion_pytorch_model.bin"],
        "files": {
            "config.json": 547,
            "diffusion_pytorch_model.bin": 334707217,
        },
        "hf_license_field": "mit",
    },
    {
        "repo_id": "openai/whisper-tiny",
        "revision": "169d4a4341b33bc18d8881c4b69c2e104e1cc0af",
        "local_subdir": "whisper",
        "allow_patterns": ["config.json", "pytorch_model.bin", "preprocessor_config.json"],
        "files": {
            "config.json": 1983,
            "preprocessor_config.json": 184990,
            "pytorch_model.bin": 151095027,
        },
        "hf_license_field": "apache-2.0",
    },
    {
        "repo_id": "yzd-v/DWPose",
        "revision": "1a7144101628d69ee7a3768d1ee3a094070dc388",
        "local_subdir": "dwpose",
        "allow_patterns": ["dw-ll_ucoco_384.pth"],
        "files": {
            "dw-ll_ucoco_384.pth": 406878486,
        },
        "hf_license_field": "apache-2.0",
    },
)

UNPINNED_FILES = (
    {
        "kind": "gdown",
        "id": "154JgKpzCPW82qINcVieuPH3fZ2e0P812",
        "dest": "face-parse-bisent/79999_iter.pth",
        "revision": None,
        "sha256": None,
        "bytes": None,
    },
    {
        "kind": "http",
        "url": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
        "dest": "face-parse-bisent/resnet18-5c106cde.pth",
        "revision": "filename prefix 5c106cde only; full sha256 was not fetched",
        "sha256": None,
        "bytes": None,
    },
)

# Fetched by libraries during scripts.inference, not by prepare_weights().
RUNTIME_FETCHES_NOT_PREFETCHED = (
    {
        "by": "face-alignment==1.3.4",
        "url": "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth",
        "bytes": None,
    },
    {
        "by": "face-alignment==1.3.4",
        "url": "https://www.adrianbulat.com/downloads/python-fan/2DFAN4-cd938726ad.zip",
        "bytes": None,
    },
    {
        "by": "mmpose config init_cfg (whether init_model downloads it was not executed)",
        "url": (
            "https://download.openmmlab.com/mmpose/v1/projects/rtmpose/"
            "cspnext-l_udp-aic-coco_210e-256x192-273b7631_20230130.pth"
        ),
        "bytes": None,
    },
)

# Training-only. Not downloaded for this inference task.
NOT_DOWNLOADED = (
    {
        "repo_id": "ByteDance/LatentSync",
        "revision": "405eda8eab9f65c1a6e0c292a5dee5a08089e2ae",
        "file": "latentsync_syncnet.pt",
        "why": "loaded by train.py, not by scripts.inference",
        "hf_license_field": "openrail++",
    },
)


def known_hf_bytes() -> int:
    total = 0
    for repo in HF_FILES:
        total += sum(repo["files"].values())
    return total


def plan() -> dict:
    return {
        "hf": list(HF_FILES),
        "unpinned": list(UNPINNED_FILES),
        "runtime_fetches_not_prefetched": list(RUNTIME_FETCHES_NOT_PREFETCHED),
        "not_downloaded": list(NOT_DOWNLOADED),
        "known_hf_bytes": known_hf_bytes(),
        "volume": None,
    }


def _needed(models_dir: Path, relative: str) -> bool:
    return not (models_dir / relative).is_file()


def prepare_weights(models_dir: Path) -> list[str]:
    """Download missing inference weights. Not called by tests or import."""
    models_dir.mkdir(parents=True, exist_ok=True)
    done: list[str] = []
    from huggingface_hub import snapshot_download

    for repo in HF_FILES:
        relative_files = [Path(repo["local_subdir"]) / name for name in repo["files"]]
        if repo["local_subdir"] == ".":
            relative_files = [Path(name) for name in repo["files"]]
        if all((models_dir / rel).is_file() for rel in relative_files):
            continue
        snapshot_download(
            repo_id=repo["repo_id"],
            revision=repo["revision"],
            local_dir=str(models_dir / repo["local_subdir"]) if repo["local_subdir"] != "." else str(models_dir),
            allow_patterns=list(repo["allow_patterns"]),
        )
        done.append(repo["repo_id"])
    for item in UNPINNED_FILES:
        dest = models_dir / item["dest"]
        if dest.is_file():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        if item["kind"] == "http":
            urllib.request.urlretrieve(item["url"], dest)  # noqa: S310
        elif item["kind"] == "gdown":
            import gdown

            gdown.download(id=item["id"], output=str(dest), quiet=False)
        else:
            raise RuntimeError(f"unknown weight kind {item['kind']}")
        done.append(item["dest"])
    return done
