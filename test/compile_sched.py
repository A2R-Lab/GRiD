"""RAM-aware parallel compile scheduler (2026-08-18, user-directed).

Runs a queue of compile jobs (each an argv subprocess) in parallel, bounded by
PREDICTED peak RSS so the box never OOMs: a job is admitted only while the sum
of running jobs' predicted peaks stays under the memory budget. Predictions
come from a rolling max-over-history ledger (test/.split_suite/compile_rss.json)
measured via `/usr/bin/time -v` (its ru_maxrss folds in nvcc's cicc/ptxas
children); unmeasured jobs use a conservative default.

Design constraints honored:
- CONSERVATIVE by default (box crashes are costly): budget = MemAvailable at
  scheduler start minus a floor, margin multiplier on every prediction, hard
  max-jobs cap, plus a live MemAvailable backstop re-checked at each admission.
- Jobs sharing a `group` run SERIALLY relative to each other (e.g. a mimic
  robot's flagship cells share one header-cache key — unlocked concurrent
  writes would race), still parallel vs everything else.
- Workers run in their own sessions (setsid) so an interrupt can kill whole
  compiler trees; kill_all() is the SIGTERM path.
- The ledger only records peaks for jobs the caller marks as real builds
  (record_peak in the result is consumed by the caller) — cache-HIT invocations
  finish in ~1s with tiny RSS and must not poison future predictions.

Used by test/run_split_suite.py (parallel Phase A + cuda flagship pre-warm,
optionally overlapped with GPU shard execution). CPU-only; no GPU use.
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

_TIME_BIN = "/usr/bin/time"
_MAXRSS_RE = re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)")

DEFAULT_PEAK_KB = 6 * 1024 * 1024      # 6 GB — unmeasured compile assumption
DEFAULT_FLOOR_KB = 10 * 1024 * 1024    # never plan below 10 GB MemAvailable
DEFAULT_MARGIN = 1.3
DEFAULT_MAX_JOBS = 5


def mem_available_kb() -> int:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1])
    return 0


@dataclass
class Job:
    name: str                     # unique id; also the done-event key
    argv: list[str]               # command to run (wrapped with /usr/bin/time -v)
    ledger_key: str               # peak-RSS ledger key (stable across runs)
    env: dict | None = None       # extra env on top of os.environ
    group: str | None = None      # jobs sharing a group run serially vs each other
    log_path: str | None = None   # worker stdout+stderr destination


@dataclass
class JobResult:
    name: str
    rc: int
    secs: float
    peak_kb: int                  # 0 if /usr/bin/time output was unparsable
    stdout_tail: str = ""


class Ledger:
    """Max-over-history peak-RSS map, atomically persisted."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self.data: dict[str, int] = {}
        if self.path.exists():
            try:
                self.data = {k: int(v) for k, v in json.loads(self.path.read_text()).items()}
            except (ValueError, OSError):
                self.data = {}

    def predict_kb(self, key: str, default_kb: int, margin: float) -> int:
        base = self.data.get(key, default_kb)
        return int(base * margin)

    def record(self, key: str, peak_kb: int) -> None:
        if peak_kb <= 0:
            return
        with self._lock:
            if peak_kb > self.data.get(key, 0):
                self.data[key] = peak_kb
                tmp = self.path.with_suffix(f".{os.getpid()}.tmp")
                tmp.write_text(json.dumps(self.data, indent=0, sort_keys=True))
                tmp.replace(self.path)


class RamScheduler:
    """Admission-controlled parallel runner. Thread-based; each job is a
    subprocess in its own session. `run()` blocks; `run_async()` returns
    immediately with per-job done-events for pipeline overlap."""

    def __init__(
        self,
        ledger: Ledger,
        *,
        max_jobs: int = DEFAULT_MAX_JOBS,
        floor_kb: int = DEFAULT_FLOOR_KB,
        margin: float = DEFAULT_MARGIN,
        default_peak_kb: int = DEFAULT_PEAK_KB,
        label: str = "compile",
    ):
        self.ledger = ledger
        self.max_jobs = max(1, int(max_jobs))
        self.floor_kb = floor_kb
        self.margin = margin
        self.default_peak_kb = default_peak_kb
        self.label = label
        # Static pool sized at construction; the live MemAvailable backstop at
        # each admission covers external memory pressure appearing later.
        self.budget_kb = max(mem_available_kb() - floor_kb, default_peak_kb)
        self.results: dict[str, JobResult] = {}
        self.done_events: dict[str, threading.Event] = {}
        self._procs: dict[str, subprocess.Popen] = {}
        self._state_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def raise_floor(self, floor_kb: int) -> None:
        """Tighten the floor mid-run (e.g. once GPU test shards start and need
        host RAM); only ever raises."""
        if floor_kb > self.floor_kb:
            self.floor_kb = floor_kb

    def stop(self) -> None:
        """Stop admitting new jobs; running jobs finish."""
        self._stop.set()

    def kill_all(self) -> None:
        """SIGTERM every running worker's process group and stop admissions."""
        self._stop.set()
        with self._state_lock:
            procs = list(self._procs.values())
        for p in procs:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass

    def run_async(self, jobs: list[Job]) -> threading.Thread:
        for j in jobs:
            self.done_events[j.name] = threading.Event()
        self._thread = threading.Thread(
            target=self._run_pool, args=(list(jobs),), daemon=True
        )
        self._thread.start()
        return self._thread

    def run(self, jobs: list[Job]) -> dict[str, JobResult]:
        self.run_async(jobs).join()
        return self.results

    def wait(self, names: list[str]) -> None:
        for n in names:
            ev = self.done_events.get(n)
            if ev is not None:
                ev.wait()

    # -- internals ---------------------------------------------------------

    def _run_pool(self, pending: list[Job]) -> None:
        running: dict[str, tuple[Job, subprocess.Popen, int, float, object]] = {}
        busy_groups: set[str] = set()
        predicted_sum = 0

        def reap() -> None:
            nonlocal predicted_sum
            for name in list(running):
                job, proc, pred, t0, log_f = running[name]
                if proc.poll() is None:
                    continue
                del running[name]
                predicted_sum -= pred
                if job.group:
                    busy_groups.discard(job.group)
                if log_f is not None:
                    log_f.close()
                with self._state_lock:
                    self._procs.pop(name, None)
                self.results[name] = self._collect(job, proc, time.monotonic() - t0)
                self.done_events[name].set()

        while pending or running:
            reap()
            launched = False
            if not self._stop.is_set():
                for job in list(pending):
                    if len(running) >= self.max_jobs:
                        break
                    if job.group and job.group in busy_groups:
                        continue
                    pred = self.ledger.predict_kb(
                        job.ledger_key, self.default_peak_kb, self.margin
                    )
                    if running and predicted_sum + pred > self.budget_kb:
                        continue
                    # Live backstop: real MemAvailable must still clear the floor
                    # after this job's predicted peak (guards external pressure).
                    if running and mem_available_kb() - pred < self.floor_kb:
                        continue
                    pending.remove(job)
                    self._launch(job, pred, running, busy_groups)
                    predicted_sum += pred
                    launched = True
            elif not running:
                break  # stopped and drained
            if not launched:
                time.sleep(1.0)

        # Anything never launched (stop() mid-queue) resolves as rc=-1 so
        # waiters unblock and callers can distinguish "not attempted".
        for job in pending:
            self.results[job.name] = JobResult(job.name, -1, 0.0, 0, "not attempted (stopped)")
            self.done_events[job.name].set()

    def _launch(self, job: Job, pred_kb: int, running: dict, busy_groups: set) -> None:
        env = dict(os.environ)
        if job.env:
            env.update(job.env)
        time_out = Path(job.log_path).with_suffix(".time") if job.log_path else None
        argv = ([_TIME_BIN, "-v", "-o", str(time_out)] + job.argv
                if time_out is not None else [_TIME_BIN, "-v"] + job.argv)
        log_f = open(job.log_path, "w") if job.log_path else subprocess.DEVNULL
        proc = subprocess.Popen(
            argv,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        running[job.name] = (job, proc, pred_kb, time.monotonic(), log_f if job.log_path else None)
        if job.group:
            busy_groups.add(job.group)
        with self._state_lock:
            self._procs[job.name] = proc
        print(
            f"[{self.label}] + {job.name} (pred {pred_kb // 1024} MiB, "
            f"{len(running)} running, {mem_available_kb() // (1024 * 1024)} GiB avail)",
            flush=True,
        )

    def _collect(self, job: Job, proc: subprocess.Popen, secs: float) -> JobResult:
        peak_kb = 0
        tail = ""
        try:
            if job.log_path:
                time_path = Path(job.log_path).with_suffix(".time")
                text = time_path.read_text() if time_path.exists() else ""
                m = _MAXRSS_RE.search(text)
                if m:
                    peak_kb = int(m.group(1))
                log_text = Path(job.log_path).read_text(errors="replace")
                tail = log_text[-800:]
        except OSError:
            pass
        rc = proc.returncode
        # /usr/bin/time exits 127/126 if the command couldn't start; otherwise
        # it propagates the command's exit status.
        print(
            f"[{self.label}] - {job.name}: rc={rc} peak={peak_kb // 1024}MiB {secs:.0f}s",
            flush=True,
        )
        return JobResult(job.name, rc, secs, peak_kb, tail)
