"""Fail-closed spend gate.

Nothing in this module opens the gate by itself. A closed gate is the only
default. There is no skip flag, no credit-balance check that passes, and no
code path that writes an attestation.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "musetalk-modal-spend-gate/v1"
ATTESTATION_ENV = "MUSETALK_SPEND_ATTESTATION"
# The 2026-11-17 01:41 Asia/Shanghai window is expired and is not an authorization.
EXPIRED_WINDOW = datetime.fromisoformat("2026-11-17T01:41:00+08:00")
REQUIRED_STATEMENT = (
    "I personally verified the Modal billing page for this workspace: "
    "an effective saved cash spend cap of 0 USD is in force and is not the "
    "documentation default formula, I understand payment-method requirements "
    "for any remaining grant, and I authorize exactly one MuseTalk 1.5 task. "
    "The 2026-11-17 01:41 Asia/Shanghai window is not this authorization."
)


@dataclass(frozen=True)
class GateDecision:
    opened: bool
    reason: str


def evaluate_gate(env: dict[str, str] | None = None) -> GateDecision:
    env = os.environ if env is None else env
    raw_path = env.get(ATTESTATION_ENV, "").strip()
    if not raw_path:
        return GateDecision(
            False,
            "spend gate closed: MUSETALK_SPEND_ATTESTATION is unset. "
            "A usage limit, a marketing grant, or the docs-default $0 formula "
            "does not open this gate.",
        )
    path = Path(raw_path)
    if not path.is_file():
        return GateDecision(False, "spend gate closed: attestation file is missing")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return GateDecision(False, f"spend gate closed: attestation is not readable JSON ({exc})")
    if not isinstance(payload, dict):
        return GateDecision(False, "spend gate closed: attestation must be a JSON object")
    return _check_payload(payload)


def import_requires_gate(env: dict[str, str] | None = None) -> bool:
    """Local imports must hit the gate.

    Modal SDK 1.6.4 sets MODAL_IS_REMOTE=1 inside a container before user
    code imports the app module. That process is not a new authorization;
    the client already had to pass the gate to deploy or spawn. The CLI
    does not call this function.
    """
    env = os.environ if env is None else env
    return env.get("MODAL_IS_REMOTE") != "1"


def require_open_gate_or_exit() -> None:
    decision = evaluate_gate()
    if not decision.opened:
        raise SystemExit(decision.reason)


def _check_payload(payload: dict) -> GateDecision:
    if payload.get("schema") != SCHEMA:
        return GateDecision(False, "spend gate closed: schema mismatch")
    if payload.get("statement") != REQUIRED_STATEMENT:
        return GateDecision(False, "spend gate closed: statement does not match the required sentence")
    if payload.get("expired_window_is_not_authorization") is not True:
        return GateDecision(False, "spend gate closed: expired window was not rejected")
    if payload.get("single_task_only") is not True:
        return GateDecision(False, "spend gate closed: single_task_only is not true")
    if payload.get("gpu_eligibility_verified") is not True:
        return GateDecision(
            False,
            "spend gate closed: GPU eligibility has not been verified by a person",
        )
    if payload.get("cash_cap_is_saved_effective") is not True:
        return GateDecision(False, "spend gate closed: cash cap is not marked saved and effective")
    if payload.get("cash_cap_is_not_docs_default_formula") is not True:
        return GateDecision(
            False,
            "spend gate closed: a documentation-default $0 is not an explicit saved cap",
        )
    cap = payload.get("explicit_saved_cash_spend_cap_usd")
    if type(cap) is not int and type(cap) is not float:
        return GateDecision(False, "spend gate closed: cash cap must be a JSON number")
    if float(cap) != 0.0:
        return GateDecision(False, "spend gate closed: explicit cash spend cap is not 0")
    payment = payload.get("payment_requirement")
    if not isinstance(payment, str) or not payment.strip():
        return GateDecision(False, "spend gate closed: payment requirement text is empty")
    if "available_credit_usd" not in payload:
        return GateDecision(False, "spend gate closed: available credit was not recorded")
    # The recorded credit is not compared against a threshold. A positive
    # balance must not open the gate; the fields above are what open it.
    verified_at = _parse_time(payload.get("human_verified_at"))
    if verified_at is None:
        return GateDecision(False, "spend gate closed: human_verified_at is not a timestamp")
    if verified_at <= EXPIRED_WINDOW:
        return GateDecision(
            False,
            "spend gate closed: verification is not after the expired 2026-11-17 01:41 Asia/Shanghai window",
        )
    return GateDecision(True, "spend gate open by external attestation")


def _parse_time(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)
