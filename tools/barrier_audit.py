#!/usr/bin/env python3
"""Static barrier audit of a generated grid.cuh (guide §7.z39).

For every `__syncthreads();` in every `__device__` function, find the STAGE
immediately before it and the stage immediately after it (a stage = one
block-parallel `for(int v = threadIdx.x + ...)` loop or one thread-0 serial
block, with only comments / blank lines / scalar or pointer declarations in
between) and classify the barrier by the shared-memory traffic it orders:

  independent  the stage after reads nothing the stage before wrote (and the
               stage before reads nothing the stage after writes) -> the
               barrier orders nothing; delete it.
  same-idx     every read in the later stage of an array the earlier stage
               wrote uses the SAME index expression the earlier stage wrote
               (after substituting the loop-local `int x = ...;` decodes and
               renaming the loop variable) -> fuse the two loops, keep the
               producer's values in registers (bit-identical by construction).
  cross-thread some read is at another index (or through a pointer / range) ->
               the barrier is real. Reported so a recompute-instead-of-share
               variant can be weighed by hand.
  opaque       the barrier is not bracketed by two stages (sequential loop
               boundary, function call, if/else, ...). The lines around it are
               reported for a manual look.

A serial thread-0 block counts as a stage whose "loop index" is the whole
block: a read of what it wrote is same-idx only from another thread-0 block,
so a thread-0 -> parallel pair is cross-thread unless independent.

Usage: .venv/bin/python tools/barrier_audit.py <grid.cuh> [--func REGEX] [--all]
Prints one line per barrier (default: only non-cross-thread findings; --all
prints everything) and a per-function summary.
"""
import argparse
import re
import sys
from collections import Counter
from pathlib import Path

PAR_HDR = re.compile(r"^\s*for\s*\(\s*int\s+(\w+)\s*=\s*threadIdx\.x\s*\+\s*threadIdx\.y\s*\*\s*blockDim\.x\s*;\s*\1\s*<\s*([^;]+);")
SER_HDR = re.compile(r"^\s*if\s*\(\s*threadIdx\.x\s*==\s*0\s*&&\s*threadIdx\.y\s*==\s*0\s*\)\s*\{")
SYNC = re.compile(r"^\s*__syncthreads\(\)\s*;")
FUNC_HDR = re.compile(r"^\s*(?:__device__|__host__|template|void|T\b|static|inline).*\b(\w+)\s*\([^;]*$")
NOISE = re.compile(r"^\s*(//.*|/\*.*\*/|T\s*\*\s*\w+\s*=.*;|const\s+T\s*\*\s*\w+\s*=.*;|int\s+\w+\s*=.*;|constexpr\s+.*;|(?:T|int)\s+\w+(?:\[\d*\])?\s*;)?\s*$")
INT_DEF = re.compile(r"\bint\s+(\w+)\s*=\s*([^;]+);")
ACCESS = re.compile(r"(&?)\b([A-Za-z_]\w*)\s*\[([^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*)\]")
ASSIGN = re.compile(r"(\b[A-Za-z_]\w*)\s*\[([^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*)\]\s*(\+=|-=|\*=|/=|=)(?!=)")
PTR_ALIAS = re.compile(r"\bT\s*\*\s*(\w+)\s*=\s*&?\s*(\w+)\b")
CALL = re.compile(r"\b(\w+)\s*(?:<[^()]*>)?\s*\(")
IGNORE_CALLS = {"for", "if", "while", "static_cast", "dot_prod", "crm", "crf", "icrf", "crm_mul", "icrf_mul",
                "sqrt", "fabs", "min", "max", "fmaxf", "fminf", "__syncthreads", "constexpr", "sizeof", "switch"}
REG_WORDS = {"threadIdx", "blockDim", "blockIdx", "gridDim"}


def norm(expr):
    return re.sub(r"\s+", "", expr)


def brace_delta(line):
    code = re.sub(r"//.*", "", line)
    code = re.sub(r'"[^"]*"', "", code)
    return code.count("{") - code.count("}")


class Stage:
    def __init__(self, kind, var, bound, start, lines, pointer_names=frozenset(), faliases=None):
        self.kind, self.var, self.bound, self.start, self.lines = kind, var, bound, start, lines
        self.faliases = faliases or {}
        body = "\n".join(lines[1:])
        self.defs = {}
        for m in INT_DEF.finditer(body):
            self.defs[m.group(1)] = m.group(2).strip()
        self.writes, self.reads, self.ptr_reads, self.calls = {}, {}, set(), set()
        for raw in lines[1:]:
            line = re.sub(r"//.*", "", raw)
            lhs_spans = []
            for m in ASSIGN.finditer(line):
                lhs_spans.append(m.span())
                self.writes.setdefault(m.group(1), set()).add(self.expand(m.group(2)))
                if m.group(3) != "=":
                    self.reads.setdefault(m.group(1), set()).add(self.expand(m.group(2)))
            for m in ACCESS.finditer(line):
                if any(a <= m.start() < b for a, b in lhs_spans):
                    continue
                if m.group(1) == "&":
                    self.ptr_reads.add(m.group(2))
                else:
                    self.reads.setdefault(m.group(2), set()).add(self.expand(m.group(3)))
            for m in CALL.finditer(line):
                if m.group(1) not in IGNORE_CALLS:
                    self.calls.add(m.group(1))
            # a shared pointer passed bare (`icrf<T>(idx, fs_IC_v)`) is a range read
            for m in re.finditer(r"\b([A-Za-z_]\w*)\b(?!\s*[\[(])", line):
                if m.group(1) in pointer_names and not re.search(r"\bT\s*\*\s*%s\b" % m.group(1), line):
                    self.ptr_reads.add(m.group(1))
        # local register arrays (T x[6];) are not shared traffic
        locals_ = set(re.findall(r"\bT\s+(\w+)\s*\[", body)) | set(re.findall(r"\bint\s+(\w+)\s*\[", body))
        for name in locals_:
            self.writes.pop(name, None); self.reads.pop(name, None); self.ptr_reads.discard(name)
        # in-stage pointer aliases (`T *Xdn = &s_oXi[jid*36];`, `T *S_u2 = S_u1 + 6;`) are RANGE
        # accesses on the base array: a write through one is non-injective, a read is a ptr read
        self.aliases = {}
        for m in re.finditer(r"\b(?:const\s+)?T\s*\*\s*(\w+)\s*=\s*&?\s*(\w+)", body):
            base = m.group(2)
            while base in self.aliases:
                base = self.aliases[base]
            if base != m.group(1):
                self.aliases[m.group(1)] = base
        self.range_writes = set()
        # function-level aliases: fold the alias name onto its base (same storage, index
        # space unknown -> range)
        for alias, base in self.faliases.items():
            if alias in self.writes:
                self.writes.pop(alias); self.writes.setdefault(base, set()).add("<range>"); self.range_writes.add(base)
            if alias in self.reads:
                self.reads.pop(alias); self.ptr_reads.add(base)
            if alias in self.ptr_reads:
                self.ptr_reads.discard(alias); self.ptr_reads.add(base)
        for alias, base in self.aliases.items():
            if alias in self.writes:
                self.writes.pop(alias); self.writes.setdefault(base, set()).add("<range>"); self.range_writes.add(base)
            if alias in self.reads:
                self.reads.pop(alias); self.ptr_reads.add(base)
            if alias in self.ptr_reads:
                self.ptr_reads.discard(alias); self.ptr_reads.add(base)

    def expand(self, expr, depth=0):
        e = norm(expr)
        if depth > 6:
            return e
        changed = False
        for name, val in self.defs.items():
            if re.search(r"\b%s\b" % re.escape(name), e):
                e = re.sub(r"\b%s\b" % re.escape(name), "(" + norm(val) + ")", e); changed = True
        return self.expand(e, depth + 1) if changed else e

    def rename(self, expr, new_var):
        if self.var is None or new_var is None:
            return expr
        return re.sub(r"\b%s\b" % re.escape(self.var), new_var, expr)

    def label(self):
        if self.kind == "serial":
            return "thread0"
        return f"par[{self.var}<{norm(self.bound)}]"


def parse_functions(text):
    """Yield (name, start_line, lines) for every brace-balanced function body."""
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        m = FUNC_HDR.match(lines[i])
        if m and "{" in lines[i] and not lines[i].strip().startswith("//") and not lines[i].strip().startswith("*"):
            name = m.group(1)
            depth = brace_delta(lines[i])
            j = i + 1
            while j < len(lines) and depth > 0:
                depth += brace_delta(lines[j]); j += 1
            yield name, i + 1, lines[i:j]
            i = j
        else:
            i += 1


def pointer_names_of(lines):
    text = "\n".join(lines)
    names = set(re.findall(r"\b(?:const\s+)?(?:T|int|float|double)\s*\*\s*(\w+)", text))
    names |= set(re.findall(r"\b([A-Za-z_]\w*)\s*\[", text))
    return frozenset(n for n in names if n not in REG_WORDS)


def function_aliases(lines):
    """Function-level pointer aliases (`T *p1 = t;`, `T *IC_v = aJ;`, `T *S_p = &S_vel[..]`)
    resolved to a canonical base name: two names on the same storage are one array."""
    aliases = {}
    for line in lines:
        if PAR_HDR.match(line) or SER_HDR.match(line):
            continue
        m = re.match(r"\s*(?:const\s+)?T\s*\*\s*(\w+)\s*=\s*&?\s*(\w+)\b([^;]*);", line)
        if not m or m.group(1) == m.group(2):
            continue
        base = m.group(2)
        # `T *X = s_temp + OFF;` / `T *X = Y + N*NUM_BODIES;` carve DISJOINT slabs of one pool;
        # only treat as the same array when the rhs is the bare name (or &name[...]).
        if m.group(3).strip() and not m.group(3).strip().startswith("["):
            continue
        while base in aliases:
            base = aliases[base]
        aliases[m.group(1)] = base
    return aliases


def stages_and_syncs(lines, base_line, pointer_names=None, faliases=None):
    """Walk a function body; return ordered list of ('stage', Stage) / ('sync', line_no) / ('other', line_no, text)."""
    if pointer_names is None:
        pointer_names = pointer_names_of(lines)
    if faliases is None:
        faliases = function_aliases(lines)
    items = []
    i = 1
    while i < len(lines):
        line = lines[i]
        pm, sm = PAR_HDR.match(line), SER_HDR.match(line)
        if (pm or sm) and "{" in line:
            depth, j = brace_delta(line), i + 1
            while j < len(lines) and depth > 0:
                depth += brace_delta(lines[j]); j += 1
            block = lines[i:j]
            # a parallel loop that itself contains a barrier or a nested stage is not a simple stage
            inner_hdr = any(PAR_HDR.match(l) or SER_HDR.match(l) for l in block[1:])
            if any(SYNC.match(l) for l in block[1:]) or inner_hdr:
                items.append(("other", base_line + i, line.strip()[:80] + "  [compound stage]"))
                # descend: treat its contents as a scope of their own
                items.extend(stages_and_syncs(block, base_line + i, pointer_names, faliases))
                items.append(("other", base_line + j - 1, "}  [end compound]"))
            elif pm:
                items.append(("stage", Stage("par", pm.group(1), pm.group(2), base_line + i, block, pointer_names, faliases)))
            else:
                items.append(("stage", Stage("serial", None, None, base_line + i, block, pointer_names, faliases)))
            i = j
            continue
        if SYNC.match(line):
            items.append(("sync", base_line + i))
        elif not NOISE.match(line):
            items.append(("other", base_line + i, line.strip()[:100]))
        i += 1
    return items


def injective(expr, var):
    """True when `expr` reads as an injective function of `var`: mentions it exactly once
    and never under `%`, `/` or inside a conditional/ternary."""
    if var is None:
        return False
    n = len(re.findall(r"\b%s\b" % re.escape(var), expr))
    if n != 1:
        return False
    return not re.search(r"(\b%s\b\s*[%%/])|([%%/]\s*\(?\s*\b%s\b)|\?" % (re.escape(var), re.escape(var)), expr)


def classify(a, b):
    """Classify the barrier between stage a (before) and stage b (after)."""
    wa, wb = a.writes, b.writes
    reasons = []
    # RAW: b reads what a wrote
    raw = {}
    for arr, idxs in b.reads.items():
        if arr in wa:
            raw[arr] = idxs
    raw_ptr = {arr for arr in b.ptr_reads if arr in wa}
    # WAR: a reads what b writes (b's writes would land before a's reads in other threads)
    war = {arr for arr in (set(a.reads) | a.ptr_reads) if arr in wb}
    waw = {arr for arr in wa if arr in wb}
    if a.calls or b.calls:
        reasons.append("calls:" + ",".join(sorted(a.calls | b.calls)))
    if not raw and not raw_ptr and not war and not waw:
        return "independent", reasons
    if raw_ptr:
        reasons.append("ptr-reads:" + ",".join(sorted(raw_ptr)))
    if a.kind == "serial" or b.kind == "serial":
        if raw or raw_ptr:
            reasons.append("serial-stage")
            return "cross-thread", reasons
    cross = []
    for arr, idxs in raw.items():
        written = {a.rename(w, b.var) for w in wa[arr]}
        # same-idx needs the producer's write index to be an injective function of its loop
        # variable (thread == iteration): `idx`, `jid*36+idx`, ... but NOT `h_idx%6` (several
        # iterations, hence several threads, write the same location).
        if any(not injective(w, a.var) for w in wa[arr]) or arr in a.range_writes:
            cross.append(f"{arr}: non-injective write index {{{','.join(sorted(wa[arr]))}}}")
            continue
        for idx in idxs:
            if idx not in written:
                cross.append(f"{arr}[{idx}] vs wrote {{{','.join(sorted(wa[arr]))}}}")
    for arr in war:
        # same-idx WAR is fine (same thread); any other read index is a hazard
        if arr in b.range_writes:
            cross.append(f"WAR range-write {arr}"); continue
        written = {b.rename(w, a.var) for w in wb[arr]}
        for idx in a.reads.get(arr, set()):
            if idx not in written:
                cross.append(f"WAR {arr}[{idx}]")
        if arr in a.ptr_reads:
            cross.append(f"WAR ptr {arr}")
    for arr in waw:
        wa_r = {a.rename(w, b.var) for w in wa[arr]}
        if wa_r != wb[arr] or arr in a.range_writes or arr in b.range_writes:
            cross.append(f"WAW {arr}")
    if raw_ptr:
        cross.append("ptr")
    if cross:
        reasons.extend(cross[:4])
        return "cross-thread", reasons
    reasons.append("same-idx:" + ",".join(sorted(raw)) + ("" if not war else " WAR-same:" + ",".join(sorted(war))) + ("" if not waw else " WAW-same:" + ",".join(sorted(waw))))
    return "same-idx", reasons


def audit(path, func_re, show_all):
    """Greedy forward pass per function: `pending` = the stages since the last KEPT barrier.
    A barrier is deletable iff every stage of the next group (up to the following barrier)
    is independent / same-idx w.r.t. EVERY pending stage, nothing opaque sits on either
    side, and no stage calls a helper that could hide pointer traffic (-> 'calls')."""
    text = Path(path).read_text()
    summary = Counter()
    per_func = {}
    for name, start, lines in parse_functions(text):
        if func_re and not re.search(func_re, name):
            continue
        if not any(SYNC.match(l) for l in lines):
            continue
        items = stages_and_syncs(lines, start)
        # split into groups separated by syncs
        groups, cur, sync_lines = [], [], []
        for it in items:
            if it[0] == "sync":
                groups.append(cur); sync_lines.append(it[1]); cur = []
            else:
                cur.append(it)
        groups.append(cur)
        counts, rows = Counter(), []
        pending = list(groups[0])
        for k, ln in enumerate(sync_lines):
            nxt = groups[k + 1]
            opaque = [g for g in pending + nxt if g[0] != "stage"]
            if opaque or not pending or not nxt:
                ctx = " | ".join((g[1].label() if g[0] == "stage" else g[2][:50]) for g in (pending[-1:] + nxt[:1])) or "<edge>"
                verdict, why = "opaque", ""
            else:
                verdicts, whys = [], []
                for pst in pending:
                    for nst in nxt:
                        v, r = classify(pst[1], nst[1])
                        verdicts.append(v); whys.extend(r)
                calls = any(c.startswith("calls:") for c in whys)
                if "cross-thread" in verdicts:
                    verdict = "cross-thread"
                elif calls:
                    verdict = "calls"
                elif "same-idx" in verdicts:
                    verdict = "same-idx"
                else:
                    verdict = "independent"
                why = "; ".join(dict.fromkeys(whys))[:160]
                ctx = f"{'+'.join(s[1].label() for s in pending)} -> {'+'.join(s[1].label() for s in nxt)}"
            counts[verdict] += 1
            rows.append((ln, verdict, ctx, why))
            if verdict in ("independent", "same-idx"):
                pending = pending + nxt          # barrier deleted: writes stay unpublished
            else:
                pending = list(nxt)              # barrier kept: everything before is published
        per_func[name] = (counts, rows)
        summary.update(counts)
    for name, (counts, rows) in per_func.items():
        if not show_all and not any(v in ("independent", "same-idx", "calls") for _, v, _, _ in rows):
            continue
        print(f"\n== {name}: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
        for ln, verdict, ctx, why in rows:
            if show_all or verdict in ("independent", "same-idx", "calls"):
                print(f"  L{ln:<7} {verdict:12s} {ctx[:70]:70s} {why}")
    print("\nTOTAL: " + ", ".join(f"{k}={v}" for k, v in sorted(summary.items())))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("header")
    ap.add_argument("--func", default=None)
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    audit(a.header, a.func, a.all)
