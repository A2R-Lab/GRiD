"""RAM-aware parallel compile scheduler (2026-08-18, user-directed).

Runs a queue of compile jobs (each an argv subprocess) in parallel, bounded by
PREDICTED peak RSS to reduce OOM risk. Predictions are not enforced RSS limits;
other processes can also change pressure after admission. The legacy profile
allows a first job above budget for forward progress. With strict_memory=True,
that job is refused and all unattempted jobs are reported as such. Predictions
come from a rolling max-over-history ledger (test/.split_suite/compile_rss.json,
schema 2) measured by SAMPLING THE JOB'S WHOLE PROCESS TREE (every process in the
worker's session, summed RSS, 0.5 s cadence) — `/usr/bin/time -v`'s ru_maxrss
reports only the LARGEST single descendant, so it under-predicted a multi-unit
bindings build (4–8 parallel nvcc units) ~4x (2026-10-10: predicted 4.6 GB,
measured 15 GB). The ledger also keeps each job's wall time so the pool can
launch heavy jobs first. Unmeasured jobs use a conservative default.

Design constraints honored:
- CONSERVATIVE by default (box crashes are costly): budget = MemAvailable at
  scheduler start minus a floor, margin multiplier on every prediction, hard
  max-jobs cap, plus a live MemAvailable backstop re-checked at each admission.
- Jobs sharing a `group` run SERIALLY relative to each other (e.g. a mimic
  robot's flagship cells share one header-cache key; the harness caches are
  per-key flock'd since 2026-10-01, so the group is now belt-and-braces that
  also avoids two workers paying the same miss), still parallel vs everything else.
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
from dataclasses import dataclass
from pathlib import Path

_TIME_BIN = "/usr/bin/time"
# GNU time is optional: it records the per-job .time summary. Without it (a
# Jetson image ships none) the job still runs and the tree sampler still
# measures its peak; the first Orin receipt (2026-10-16) launched nothing for
# hours because the missing binary raised inside the pool thread, silently.
_HAVE_TIME_BIN = os.path.exists(_TIME_BIN)
_MAXRSS_RE = re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)")

DEFAULT_PEAK_KB = 8 * 1024 * 1024      # 8 GB — unmeasured compile assumption (tree peak)
DEFAULT_SECS = 600.0                   # unmeasured wall-time assumption (ordering only)
LEDGER_SCHEMA = 2                      # 1 = flat {key: ru_maxrss_kb} (dropped: under-predicts)
_PAGE_KB = os.sysconf("SC_PAGE_SIZE") // 1024


def session_tree_rss_kb(sid: int) -> int:
    """Session members plus descendants, including children that call setsid.

    Workers remain alive while waiting for their children. Session membership
    also retains orphaned ordinary compiler children until they are reaped.
    """
    rows = {}
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/stat") as f:
                stat = f.read()
        except OSError:
            continue  # exited between listdir and open
        rest = stat[stat.rfind(")") + 2:].split()   # comm may contain spaces/parens
        # fields after comm: [0]=state [1]=ppid [2]=pgrp [3]=session ... [21]=rss (pages)
        if len(rest) > 21:
            rows[int(name)] = (int(rest[1]), int(rest[3]), int(rest[21]) * _PAGE_KB)
    owned = {pid for pid, (_, session, _) in rows.items() if session == sid or pid == sid}
    while True:
        children = {pid for pid, (parent, _, _) in rows.items() if parent in owned}
        if children <= owned:
            break
        owned.update(children)
    return sum(rows[pid][2] for pid in owned)
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
    peak_kb: int                  # max(sampled process-tree peak, ru_maxrss); 0 if neither
    stdout_tail: str = ""


class Ledger:
    """Max-over-history {key: {peak_kb, secs}} map, atomically persisted.

    Schema 2 (2026-10-10): peaks are process-TREE peaks. A schema-1 file (flat
    ru_maxrss map) is dropped on load — its entries under-predict parallel
    builds ~4x and would poison admission; `dropped_legacy` says how many."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self.data: dict[str, dict] = {}
        self.dropped_legacy = 0
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text())
            except (ValueError, OSError):
                raw = {}
            if isinstance(raw, dict) and raw.get("schema") == LEDGER_SCHEMA:
                for k, v in (raw.get("jobs") or {}).items():
                    try:
                        self.data[k] = {"peak_kb": int(v["peak_kb"]),
                                        "secs": float(v.get("secs", 0.0))}
                    except (KeyError, TypeError, ValueError):
                        continue
            elif isinstance(raw, dict):
                self.dropped_legacy = len(raw)

    def predict_kb(self, key: str, default_kb: int, margin: float) -> int:
        # A warm variant may have measured only a cheap relink. Other variants
        # of the same robot provide a lower bound for a later cold object build.
        # Keep the default for unseen variants; do not group prewarm job keys.
        with self._lock:
            base = self.data.get(key, {}).get("peak_kb", default_kb)
            if key.startswith("warm:"):
                robot_key = key.split(",", 1)[0]
                base = max(base, *(v["peak_kb"] for k, v in self.data.items()
                                  if k == robot_key or k.startswith(robot_key + ",")), 0)
        return int(base * margin)

    def predict_secs(self, key: str, default_secs: float = DEFAULT_SECS) -> float:
        return float(self.data.get(key, {}).get("secs") or default_secs)

    def record(self, key: str, peak_kb: int, secs: float = 0.0) -> None:
        if peak_kb <= 0:
            return
        with self._lock:
            cur = self.data.get(key, {"peak_kb": 0, "secs": 0.0})
            new = {"peak_kb": max(cur["peak_kb"], int(peak_kb)),
                   "secs": max(cur["secs"], float(secs))}
            if new != cur:
                self.data[key] = new
                self.path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.path.with_suffix(f".{os.getpid()}.tmp")
                tmp.write_text(json.dumps({"schema": LEDGER_SCHEMA, "jobs": self.data},
                                          indent=0, sort_keys=True))
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
        strict_memory: bool = False,
        admission=None,
    ):
        self.ledger = ledger
        self.max_jobs = max(1, int(max_jobs))
        self.floor_kb = floor_kb
        self.margin = margin
        self.default_peak_kb = default_peak_kb
        self.label = label
        self.strict_memory = strict_memory
        self.admission = admission
        self.resource_blocked: str | None = None
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

    @property
    def stopped(self) -> bool:
        """True once stop()/kill_all() ran: later non-zero results are drained
        jobs (killed in flight or never attempted), not failures."""
        return self._stop.is_set()

    def running_pids(self) -> list[int]:
        """Snapshot worker process-group leaders for dependency-aware stall checks."""
        with self._state_lock:
            return [p.pid for p in self._procs.values() if p.poll() is None]

    def kill_all(self) -> None:
        """Stop admissions and reap owned worker groups before releasing the box."""
        with self._state_lock:
            self._stop.set()
            procs = list(self._procs.values())
        for p in procs:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=5)
        # A worker may have exited while an ignoring grandchild still lives in
        # its group. Kill only groups whose leaders we created with setsid.
        for p in procs:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=5)

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
        try:
            self._run_pool_inner(pending)
        except BaseException as exc:   # the thread must never die silently
            if self.admission is not None:
                self.stop()
                self.admission.halt(f"compile scheduler failure: {exc!r}")
                # A launch failure must not strand the reservations of healthy
                # earlier jobs. Drain our children before resolving their events.
                with self._state_lock:
                    owned = list(self._procs.items())
                for name, proc in owned:
                    proc.wait()
                    self.admission.release("compile:" + name)
                with self._state_lock:
                    self._procs.clear()
            # Every job without a result, including the one whose launch raised
            # (already taken off `pending`), resolves so waiters unblock.
            unresolved = [n for n in self.done_events if n not in self.results]
            print(f"[{self.label}] FATAL: scheduler stopped: {exc!r}; "
                  f"{len(unresolved)} job(s) unresolved", flush=True)
            for name in unresolved:
                self.results[name] = JobResult(name, -1, 0.0, 0, f"scheduler stopped: {exc!r}")
                self.done_events[name].set()
            raise

    def _run_pool_inner(self, pending: list[Job]) -> None:
        running: dict[str, tuple[Job, subprocess.Popen, int, float, object, list]] = {}
        busy_groups: set[str] = set()
        predicted_sum = 0

        def reap() -> None:
            nonlocal predicted_sum
            for name in list(running):
                job, proc, pred, t0, log_f, peak_holder = running[name]
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
                try:
                    self.results[name] = self._collect(job, proc, time.monotonic() - t0,
                                                       peak_holder[0])
                finally:
                    if self.admission is not None:
                        self.admission.release("compile:" + name)
                self.done_events[name].set()

        while pending or running:
            reap()
            if self.admission is not None and self.admission.pressure():
                self.resource_blocked = self.admission.blocked
                self.stop()
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
                    if self.strict_memory:
                        # Shared admission may use measured cold-build peaks,
                        # still floored at the conservative 8 GiB pool allowance.
                        # Unseen jobs retain default_peak_kb (16 GiB in the
                        # release profiles); ledger values never lower that.
                        floor = (min(DEFAULT_PEAK_KB, self.default_peak_kb)
                                 if self.admission is not None else self.default_peak_kb)
                        pred = max(pred, int(floor * self.margin))
                    available = mem_available_kb()
                    fits = (predicted_sum + pred <= self.budget_kb
                            and available - pred >= self.floor_kb)
                    token = "compile:" + job.name
                    if self.admission is not None:
                        fits = self.admission.reserve(token, pred)
                        if not fits:
                            if self.admission.active:
                                continue  # a shard may still own the headroom
                            self.resource_blocked = self.admission.refusal
                            self.stop()
                            break
                    if self.admission is None and self.strict_memory and not running and not fits:
                        self.resource_blocked = (
                            f"{job.name}: predicted {pred} KiB, available {available} KiB, "
                            f"reserve {self.floor_kb} KiB, budget {self.budget_kb} KiB")
                        print(f"[{self.label}] RESOURCE BLOCKED: {self.resource_blocked}", flush=True)
                        self.stop()
                        break
                    # Live backstop: real MemAvailable must still clear the floor
                    # after this job's predicted peak (guards external pressure).
                    # The legacy first-job progress exception is opt-out; the
                    # receipt's strict profile refuses instead of oversubscribing.
                    if (running or self.strict_memory) and not fits:
                        continue
                    pending.remove(job)
                    try:
                        if not self._launch(job, pred, running, busy_groups):
                            if self.admission is not None:
                                self.admission.release(token)
                            pending.append(job)
                            break
                    except BaseException:
                        if self.admission is not None:
                            self.admission.release(token)
                        raise
                    if self.admission is not None:
                        self.admission.attach(token, running[job.name][1].pid)
                    predicted_sum += pred
                    launched = True
            elif not running:
                break  # stopped and drained
            if not launched:
                time.sleep(1.0)

        # Anything never launched (stop() mid-queue) resolves as rc=-1 so
        # waiters unblock and callers can distinguish "not attempted".
        for job in pending:
            reason = (f"not attempted (resource blocked: {self.resource_blocked})"
                      if self.resource_blocked else "not attempted (stopped)")
            self.results[job.name] = JobResult(job.name, -1, 0.0, 0, reason)
            self.done_events[job.name].set()

    def _launch(self, job: Job, pred_kb: int, running: dict, busy_groups: set) -> bool:
        env = dict(os.environ)
        if job.env:
            env.update(job.env)
        time_out = Path(job.log_path).with_suffix(".time") if job.log_path else None
        if _HAVE_TIME_BIN:
            argv = ([_TIME_BIN, "-v", "-o", str(time_out)] + job.argv
                    if time_out is not None else [_TIME_BIN, "-v"] + job.argv)
        else:
            argv = list(job.argv)
        log_f = open(job.log_path, "w") if job.log_path else subprocess.DEVNULL
        with self._state_lock:
            if self._stop.is_set():
                if job.log_path:
                    log_f.close()
                return False
            proc = subprocess.Popen(
                argv, stdout=log_f, stderr=subprocess.STDOUT, env=env,
                start_new_session=True,
            )
            self._procs[job.name] = proc
        # Tree-peak sampler (schema 2): the worker is a session leader (setsid), so
        # every compiler it spawns shares its session id; sum their RSS until exit.
        peak_holder = [0]
        threading.Thread(target=self._sample_tree, args=(proc, peak_holder),
                         daemon=True).start()
        running[job.name] = (job, proc, pred_kb, time.monotonic(),
                             log_f if job.log_path else None, peak_holder)
        if job.group:
            busy_groups.add(job.group)
        print(
            f"[{self.label}] + {job.name} (pred {pred_kb // 1024} MiB, "
            f"{len(running)} running, {mem_available_kb() // (1024 * 1024)} GiB avail)",
            flush=True,
        )
        return True

    @staticmethod
    def _sample_tree(proc: subprocess.Popen, holder: list, period: float = 0.5) -> None:
        while proc.poll() is None:
            try:
                kb = session_tree_rss_kb(proc.pid)
            except OSError:
                kb = 0
            if kb > holder[0]:
                holder[0] = kb
            time.sleep(period)

    def _collect(self, job: Job, proc: subprocess.Popen, secs: float,
                 tree_peak_kb: int = 0) -> JobResult:
        # ru_maxrss (largest single descendant) is the floor for jobs too short
        # for the sampler to catch; the sampled tree sum is the real peak.
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
        peak_kb = max(peak_kb, int(tree_peak_kb))
        rc = proc.returncode
        # /usr/bin/time exits 127/126 if the command couldn't start; otherwise
        # it propagates the command's exit status.
        print(
            f"[{self.label}] - {job.name}: rc={rc} peak={peak_kb // 1024}MiB "
            f"(tree) {secs:.0f}s",
            flush=True,
        )
        return JobResult(job.name, rc, secs, peak_kb, tail)
