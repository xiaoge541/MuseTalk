"""Persist a Modal call id before any wait.

The record is fsynced and renamed into place. A second submit for the same
job id is refused.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


class CallStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def path_for(self, job_id: str) -> Path:
        return self.root / f"{job_id}.json"

    def read(self, job_id: str) -> dict | None:
        path = self.path_for(job_id)
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def persist(self, record: dict) -> Path:
        job_id = record["job_id"]
        call_id = record["call_id"]
        if not isinstance(call_id, str) or not call_id.strip():
            raise ValueError("refusing to persist an empty call id")
        path = self.path_for(job_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        data = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8")
        with open(tmp, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return path

    def update_status(self, job_id: str, status: str) -> None:
        record = self.read(job_id)
        if record is None:
            raise FileNotFoundError(job_id)
        record["status"] = status
        self.persist(record)
