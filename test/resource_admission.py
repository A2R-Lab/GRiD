"""One host-memory admission authority for pool builds and an entire GPU shard.

Reservations include inline compilation. Observed resident memory has already
left MemAvailable, so only the unspent part of each reservation is subtracted
from live headroom. Predictions are not hard RSS limits; a pressure observation
latches a drain-only stop, never kills a peer or invents successful test rows.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path


class MemoryAdmission:
    def __init__(self, *, floor_kb, available, rss, log_path=None):
        self.floor_kb = floor_kb
        self.available = available
        self.rss = rss
        self.log_path = Path(log_path) if log_path else None
        self._lock = threading.RLock()
        self._jobs = {}
        self.blocked = None
        self.refusal = None
        self.min_available_kb = None
        self.max_tree_kb = 0

    @property
    def active(self):
        with self._lock:
            return bool(self._jobs)

    def _event(self, kind, **data):
        if self.log_path:
            with self.log_path.open("a") as f:
                f.write(json.dumps(dict(event=kind, monotonic=time.monotonic(), **data)) + "\n")

    def _sample(self):
        available = self.available()
        self.min_available_kb = (available if self.min_available_kb is None
                                 else min(available, self.min_available_kb))
        observed = 0
        outstanding = 0
        for job in self._jobs.values():
            current = self.rss(job["pid"]) if job["pid"] is not None else 0
            job["peak_kb"] = max(job["peak_kb"], current)
            observed += current
            outstanding += max(0, job["predicted_kb"] - current)
        self.max_tree_kb = max(self.max_tree_kb, observed)
        if available < self.floor_kb and self.blocked is None:
            self.blocked = f"live headroom {available} KiB below reserve {self.floor_kb} KiB"
            self._event("pressure", reason=self.blocked)
        return available, outstanding

    def pressure(self):
        with self._lock:
            self._sample()
            return self.blocked is not None

    def halt(self, reason):
        with self._lock:
            if self.blocked is None:
                self.blocked = reason
                self._event("halt", reason=reason)

    def reserve(self, token, predicted_kb):
        if predicted_kb <= 0:
            raise ValueError("A job must have a positive memory allowance")
        with self._lock:
            if token in self._jobs:
                raise ValueError(f"Duplicate memory reservation: {token}")
            available, outstanding = self._sample()
            if self.blocked or available - outstanding - predicted_kb < self.floor_kb:
                self.refusal = self.blocked or (
                    f"{token}: available {available} KiB, unspent reservations {outstanding} KiB, "
                    f"new allowance {predicted_kb} KiB, reserve {self.floor_kb} KiB")
                return False
            self._jobs[token] = dict(predicted_kb=predicted_kb, pid=None, peak_kb=0)
            self._event("admit", token=token, available_kb=available,
                        outstanding_kb=outstanding, predicted_kb=predicted_kb)
            return True

    def attach(self, token, pid):
        with self._lock:
            self._jobs[token]["pid"] = pid

    def release(self, token):
        with self._lock:
            self._sample()
            job = self._jobs.pop(token)
            self._event("release", token=token, **job,
                        min_available_kb=self.min_available_kb,
                        max_tree_kb=self.max_tree_kb)
            return job["peak_kb"]
