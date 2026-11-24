"""One host-memory admission authority for pool builds and an entire GPU shard.

Reservations include inline compilation. Observed resident memory has already
left MemAvailable, so only the unspent part of each reservation is subtracted
from live headroom. Predictions are not hard RSS limits; a pressure observation
latches a drain-only stop, never kills a peer or invents successful test rows.

With latch=False (the adaptive profile) pressure does not latch: a refusal is
a wait, admissions resume when headroom returns, and the caller gives up only
after patience_s with nothing of its own running.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path


class MemoryAdmission:
    def __init__(self, *, floor_kb, available, rss, log_path=None, cap_kb=None,
                 latch=True, patience_s=1800.0, clock=time.monotonic):
        self.floor_kb = floor_kb
        self.latch = latch
        self.patience_s = patience_s
        self.clock = clock
        self._idle_refused_since = None
        self.waits = 0
        # Optional operator ceiling on the sum of allowances held at once. It
        # is an extra refusal only: the floor and live headroom still apply.
        self.cap_kb = cap_kb
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
        if self.latch and available < self.floor_kb and self.blocked is None:
            self.blocked = f"live headroom {available} KiB below reserve {self.floor_kb} KiB"
            self._event("pressure", reason=self.blocked)
        return available, outstanding, observed + outstanding

    def pressure(self):
        with self._lock:
            self._sample()
            return self.blocked is not None

    def give_up(self):
        """After a refusal with nothing held: stop now (latched profiles), or
        only once headroom has not returned for patience_s (adaptive)."""
        with self._lock:
            if self.latch or self.blocked:
                return True
            if self._jobs:
                self._idle_refused_since = None
                return False
            now = self.clock()
            if self._idle_refused_since is None:
                self._idle_refused_since = now
                self.waits += 1
                self._event("wait", reason=self.refusal)
            return now - self._idle_refused_since >= self.patience_s

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
            available, outstanding, held = self._sample()
            if self.blocked or available - outstanding - predicted_kb < self.floor_kb:
                self.refusal = self.blocked or (
                    f"{token}: available {available} KiB, unspent reservations {outstanding} KiB, "
                    f"new allowance {predicted_kb} KiB, reserve {self.floor_kb} KiB")
                return False
            if self.cap_kb is not None and held + predicted_kb > self.cap_kb:
                self.refusal = (
                    f"{token}: held allowances {held} KiB, new allowance {predicted_kb} KiB, "
                    f"memory cap {self.cap_kb} KiB")
                return False
            self._idle_refused_since = None
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
