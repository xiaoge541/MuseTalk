"""License and VRAM facts that were read from files or public API responses.

Unknowns stay unknown. A missing SPDX field is not a commercial grant.
"""

from __future__ import annotations

CODE_LICENSE = {
    "instrument": "LICENSE file in this repository",
    "name": "MIT",
    "commercial_use_of_this_code_text": (
        "The MIT text in LICENSE permits use, copy, modification, and sale "
        "of the Software, subject to the notice requirement."
    ),
}

README_MODEL_SENTENCE = (
    "README.md Disclaimer/License item 2 says: "
    "'model: The trained model are available for any purpose, even commercially.'"
)

VRAM_EVIDENCE = (
    {
        "where": "README.md Gradio section",
        "statement": (
            "Tested on a Windows NVIDIA GeForce RTX 3050 Ti Laptop GPU with 4GB VRAM. "
            "In fp16 mode, generating an 8-second video takes approximately 5 minutes."
        ),
        "applies_to_this_1086x1448_input": False,
        "measured_here": False,
    },
    {
        "where": "README.md overview and realtime notes",
        "statement": "30fps+ on an NVIDIA Tesla V100. The README does not say whether that V100 is 16GB or 32GB.",
        "applies_to_this_1086x1448_input": False,
        "measured_here": False,
    },
    {
        "where": "README.md training GPU Memory Requirements",
        "statement": (
            "Stage 1 about 32GB to 74GB per H20 GPU and stage 2 about 54GB to 85GB per H20 GPU, "
            "depending on batch size. Those figures are training, not this inference task."
        ),
        "applies_to_this_1086x1448_input": False,
        "measured_here": False,
    },
)

THIRD_PARTY = (
    {
        "name": "TMElyralab/MuseTalk weights",
        "hf_card_license_field": "mit",
        "hf_revision": "2bcb936e2fddb4d86db4c62fd45b387d0c061571",
        "note": (
            "The Hub card field is mit. The README also claims commercial use of the trained model. "
            "Training used HDTF and a private dataset; those dataset rights are not granted by the MIT code license. "
            "This record is not a clearance."
        ),
    },
    {
        "name": "stabilityai/sd-vae-ft-mse",
        "hf_card_license_field": "mit",
        "hf_revision": "31f26fdeee1355a5c34592e401dd41e45d25a493",
        "note": "Hub card field read from the models API. Not a clearance of downstream use.",
    },
    {
        "name": "openai/whisper-tiny",
        "hf_card_license_field": "apache-2.0",
        "code_license_cited_by_musetalk_license_file": "MIT (GitHub openai/whisper, as quoted in this repo's LICENSE)",
        "note": "The Hub card and the MuseTalk LICENSE citation do not use the same name. Both are recorded. Neither is a clearance.",
    },
    {
        "name": "yzd-v/DWPose",
        "hf_card_license_field": "apache-2.0",
        "note": "Matches the Apache-2.0 notice in this repo's LICENSE for DWpose.",
    },
    {
        "name": "ByteDance/LatentSync syncnet weight",
        "hf_card_license_field": "openrail++",
        "used_by_this_inference_job": False,
        "note": "openrail++ is not treated as a commercial grant. The file is not in the inference download plan. app.py's startup checker looks for it; scripts.inference does not load it.",
    },
    {
        "name": "face-parsing 79999_iter.pth",
        "hf_card_license_field": None,
        "note": "Distributed via the gdown id in download_weights.sh. The face-parsing.PyTorch code notice in LICENSE is MIT. The weight file's own license was not found in this repo. Unknown.",
    },
    {
        "name": "resnet18-5c106cde.pth",
        "hf_card_license_field": None,
        "note": "URL is the PyTorch model-zoo file named in face_parsing/resnet.py. Full license of that binary was not re-fetched. Unknown.",
    },
    {
        "name": "face-alignment 1.3.4 and its S3FD / 2DFAN downloads",
        "code_license": "BSD (package metadata)",
        "note": "Weight licenses for the adrianbulat.com files were not found in this repo. Unknown.",
    },
)
