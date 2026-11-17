"""Items this checkout did not execute. Tests assert this list stays honest."""

UNTESTED = (
    "modal token, auth, and setup",
    "modal image build",
    "modal deploy",
    "modal volume, secret, and web endpoint creation",
    "GPU scheduling and GPU eligibility",
    "weight download from Hugging Face, gdown, pytorch.org, openmmlab, or adrianbulat.com",
    "MuseTalk 1.5 inference",
    "an MP4 produced by MuseTalk",
    "ffprobe or full decode of a MuseTalk MP4",
    "cold start seconds",
    "weight download seconds on a GPU",
    "inference seconds on a GPU",
    "idle shutdown seconds",
    "Modal dashboard billed seconds or USD",
    "whether T4 16GB is enough for a 1086x1448 fp16 batch-8 run",
    "installation of face-alignment 1.3.4, mmcv 2.0.1, mmdet 3.1.0, mmpose 1.1.0, mmengine 0.10.5",
    "sha256 of 79999_iter.pth, resnet18-5c106cde.pth, s3fd, and 2DFAN-4",
    "whether mmpose init_model downloads the CSPNeXt init checkpoint",
    "commercial clearance of weights or training data",
    "a saved $0 cash spend cap on the Modal account",
)
