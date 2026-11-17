"""Usage-log shape. Dashboard fields stay null until a person copies a real bill."""

from __future__ import annotations

from modal_musetalk.spec import (
    CPU_USD_PER_CORE_SEC,
    EGRESS_USD_PER_GIB,
    GPU,
    GPU_USD_PER_SEC,
    MEMORY_USD_PER_GIB_SEC,
    PRICE_FETCHED_ON,
    PRICE_SOURCE,
    SCALEDOWN_WINDOW_S,
    VOLUME_USD_PER_GIB_MONTH,
)


def empty_usage(gpu: str = GPU) -> dict:
    return {
        "measured_on_gpu": False,
        "gpu": gpu,
        "price_source": PRICE_SOURCE,
        "price_fetched_on": PRICE_FETCHED_ON,
        "list_gpu_usd_per_sec": GPU_USD_PER_SEC[gpu],
        "list_cpu_usd_per_core_sec": CPU_USD_PER_CORE_SEC,
        "list_memory_usd_per_gib_sec": MEMORY_USD_PER_GIB_SEC,
        "list_egress_usd_per_gib": EGRESS_USD_PER_GIB,
        "list_volume_usd_per_gib_month": VOLUME_USD_PER_GIB_MONTH,
        "volume_created": False,
        "cold_start_s": None,
        "weight_download_s": None,
        "inference_subprocess_s": None,
        "ffmpeg_audio_s": None,
        "ffmpeg_remux_s": None,
        "function_wall_s": None,
        "idle_tail_s": None,
        "idle_tail_bounds_s": [0, SCALEDOWN_WINDOW_S],
        "formula_gpu_usd": "gpu_seconds * list_gpu_usd_per_sec",
        "gpu_seconds_formula": (
            "cold_start_s + weight_download_s + inference_subprocess_s "
            "+ ffmpeg_audio_s + ffmpeg_remux_s + idle_tail_s"
        ),
        "formula_result_usd": None,
        "modal_dashboard_gpu_billed_s": None,
        "modal_dashboard_usd": None,
        "note": (
            "null fields were not measured. List rates are not a quote. "
            "function_wall_s excludes platform cold start before the function "
            "process starts and excludes idle time after it returns. "
            "Do not fill dashboard fields from this formula."
        ),
    }
