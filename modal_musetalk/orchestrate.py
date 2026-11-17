"""Single submit: spawn, persist the call id, then wait.

A timeout or a remote failure does not spawn again. Recovery uses the saved id.
"""

from __future__ import annotations

from dataclasses import dataclass

from modal_musetalk.call_store import CallStore


class AlreadySubmitted(RuntimeError):
    def __init__(self, call_id: str) -> None:
        super().__init__(
            f"job already has call id {call_id}. Recover that id. Do not submit again."
        )
        self.call_id = call_id


class WaitTimeout(RuntimeError):
    def __init__(self, call_id: str) -> None:
        super().__init__(
            f"wait timed out for call id {call_id}. "
            "Query or download that id. Do not submit again."
        )
        self.call_id = call_id


@dataclass
class SubmitResult:
    call_id: str
    payload: dict


def submit_once(
    *,
    ops,
    store: CallStore,
    job_id: str,
    spawn_args: tuple,
    wait_s: float,
    gpu: str,
) -> SubmitResult:
    existing = store.read(job_id)
    if existing and existing.get("call_id"):
        raise AlreadySubmitted(str(existing["call_id"]))
    function = ops.ensure_function()
    call = ops.spawn(function, spawn_args)
    call_id = getattr(call, "object_id", "") or ""
    if not isinstance(call_id, str) or not call_id.strip():
        raise RuntimeError(
            "spawn returned no call id. Do not spawn again. "
            "Check the Modal dashboard for an unrecorded call."
        )
    store.persist(
        {
            "job_id": job_id,
            "call_id": call_id,
            "status": "spawned_awaiting_get",
            "gpu": gpu,
            "resubmit_forbidden": True,
        }
    )
    try:
        payload = call.get(timeout=wait_s)
    except TimeoutError as exc:
        store.update_status(job_id, "waiting_recovery")
        raise WaitTimeout(call_id) from exc
    except Exception:
        store.update_status(job_id, "remote_failed")
        raise
    store.update_status(job_id, "finished")
    return SubmitResult(call_id=call_id, payload=payload)


def recover_once(*, ops, call_id: str, wait_s: float):
    """Fetch an existing call. This function has no spawn method."""
    if not call_id.strip():
        raise ValueError("call id is empty")
    call = ops.from_id(call_id)
    return call.get(timeout=wait_s)
