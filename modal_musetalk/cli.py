"""CLI. The gate is checked before Modal is imported."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from modal_musetalk.call_store import CallStore
from modal_musetalk.gate import evaluate_gate
from modal_musetalk.input_spec import (
    ffprobe_command,
    full_decode_command,
    validate_original_wav,
    validate_portrait_png,
)
from modal_musetalk.orchestrate import AlreadySubmitted, WaitTimeout, recover_once, submit_once
from modal_musetalk.spec import CLIENT_WAIT_S, CODE_BASE_SHA, GPU


def _store() -> CallStore:
    return CallStore(Path(".musetalk_modal_calls"))


def _refuse_gate() -> int:
    decision = evaluate_gate()
    if decision.opened:
        return 0
    print(decision.reason, file=sys.stderr)
    return 2


def submit_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Submit one MuseTalk 1.5 task. Does nothing while the spend gate is closed."
    )
    parser.add_argument("--portrait", required=True, help="PNG path. Example: PORTRAIT.png")
    parser.add_argument("--audio", required=True, help="WAV path. Example: AUDIO.wav")
    parser.add_argument("--output", required=True, help="Where to write the MP4. Example: ./out/result.mp4")
    parser.add_argument("--job-id", required=True, help="Example: job-001")
    parser.add_argument("--wait-s", type=float, default=CLIENT_WAIT_S)
    args = parser.parse_args(argv)
    closed = _refuse_gate()
    if closed:
        return closed
    portrait = Path(args.portrait)
    audio = Path(args.audio)
    try:
        validate_portrait_png(portrait)
        validate_original_wav(audio)
    except (OSError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    from modal_musetalk.modal_backend import build_backend

    try:
        result = submit_once(
            ops=build_backend(),
            store=_store(),
            job_id=args.job_id,
            spawn_args=(portrait.read_bytes(), audio.read_bytes(), args.job_id),
            wait_s=args.wait_s,
            gpu=GPU,
        )
    except AlreadySubmitted as exc:
        print(str(exc), file=sys.stderr)
        _print_recover(exc.call_id, args.output)
        return 3
    except WaitTimeout as exc:
        print(str(exc), file=sys.stderr)
        _print_recover(exc.call_id, args.output)
        return 3
    return _write_payload(result.payload, Path(args.output), result.call_id)


def recover_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download an existing call. Does not spawn.")
    parser.add_argument("--call-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--wait-s", type=float, default=CLIENT_WAIT_S)
    args = parser.parse_args(argv)
    closed = _refuse_gate()
    if closed:
        return closed
    from modal_musetalk.modal_backend import build_backend

    payload = recover_once(ops=build_backend(), call_id=args.call_id, wait_s=args.wait_s)
    return _write_payload(payload, Path(args.output), args.call_id)


def _print_recover(call_id: str, output: str) -> None:
    print(
        "recover (does not submit again):\n"
        f"  python -m modal_musetalk recover --call-id {call_id} --output {output}",
        file=sys.stderr,
    )


def _write_payload(payload: dict, output: Path, call_id: str) -> int:
    if not isinstance(payload, dict) or not isinstance(payload.get("mp4"), (bytes, bytearray)):
        print("result did not contain MP4 bytes", file=sys.stderr)
        return 1
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload["mp4"])
    usage = payload.get("usage") or {}
    usage["call_id"] = call_id
    usage["code_base_sha"] = CODE_BASE_SHA
    usage["mp4_sha256"] = hashlib.sha256(payload["mp4"]).hexdigest()
    log_path = output.with_suffix(".usage.json")
    log_path.write_text(json.dumps(usage, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    print(f"wrote {log_path}")
    print("verify:")
    print(" ", " ".join(ffprobe_command(output)))
    print(" ", " ".join(full_decode_command(output)))
    print("GPU was not claimed by this writer. Dashboard fields in the usage file stay null until a real run fills them.")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in {"submit", "recover"}:
        print(
            "usage:\n"
            "  python -m modal_musetalk submit --portrait PORTRAIT.png --audio AUDIO.wav "
            "--output ./out/result.mp4 --job-id job-001\n"
            "  python -m modal_musetalk recover --call-id CALL_ID --output ./out/result.mp4",
            file=sys.stderr,
        )
        return 2
    command, rest = argv[0], argv[1:]
    if command == "submit":
        return submit_main(rest)
    return recover_main(rest)
