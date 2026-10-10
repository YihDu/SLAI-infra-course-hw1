"""Critical-path / stall attribution on the frozen timing model (dev analysis only).

Wraps codesign.challenge.pipeline_events.GlobalEventPipeline._issue with a
read-only observer that, before every issue, recomputes the same readiness terms
the frozen `_preview` uses (in-order slot, RAW, WAR/WAW, DMA slot, barrier
target) with the frozen helper functions, and records which term was binding
and which earlier instruction produced it.  Scheduling state is never mutated
by the observer, so the reported cycle count must equal the frozen estimate;
`--expect-cycles` asserts that.

Race validation inside GlobalEventPipeline.__init__ is skipped in this process
only (it does not affect timing and dominates memory).  Nothing here is a
score; official numbers still come from `challenge.py grade`.
"""
import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import codesign.challenge.hbm_race as hbm_race
from codesign.challenge import pipeline as _pl
from codesign.challenge import pipeline_events as pe

ENERGY_LOG = []  # (resource_key, start, finish, dynamic_pj)

_orig_energy_add = _pl._EnergyRuns.add


def _tagged_energy_add(self, key, start, finish, energy):
    if energy:
        ENERGY_LOG.append((key[0], start, finish, energy))
    return _orig_energy_add(self, key, start, finish, energy)


_pl._EnergyRuns.add = _tagged_energy_add
from codesign.challenge.hardware import Hardware
from codesign.challenge.isa import iter_parse
from codesign.challenge.pipeline import _MemoryTimes, _max_destination, _max_write, _view
from codesign.challenge.runner import required_hbm_words, step_requirements
from codesign.challenge.abi import build_layout
from codesign.challenge.workload import MODELS

COMPUTE = {"MMA.ACC", "VEC", "REDUCE", "SFU"}
TRANSFER = {"LD", "ST"}
SYNC = {"WAIT", "BARRIER", "STEP.COMMIT"}


def label_of(event):
    if event is None:
        return "-"
    return re.split(r"_e\d+", event, maxsplit=1)[0]


def stage_of(label):
    # pb0_qkvs1 -> p_qkv ; generic fallback strips digits
    m = re.match(r"([ps])b\d+_([a-z0-9]+?)s\d+$", label)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    return re.sub(r"\d+", "", label)


def space_kind(ins):
    if ins.op in TRANSFER:
        s, d = ins.args["src"]["space"], ins.args["dst"]["space"]
        return f"{ins.op}:{s}->{d}"
    return ins.op


class Observer(pe.GlobalEventPipeline):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.rec = []  # per issued instruction
        self.tape_seq = defaultdict(list)
        self.writer_rf = {}
        self.writer_sh = {}
        self.writer_hbm = _MemoryTimes.sized(self.hbm.written.size, sparse=True)
        self.last_tape_idx = {}

    # writer-index arrays mirror frozen timed-memory views; 0 = none, idx+1 stored
    def _writer_group(self, group):
        name = group.name
        if name not in self.writer_rf:
            self.writer_rf[name] = _MemoryTimes.sized(group.rf.written.size)
            self.writer_sh[name] = _MemoryTimes.sized(max(1, group.shared.written.size))
        return type(group)(
            group.name, group.sm, group.words_per_lane, 0, 0,
            self.writer_rf[name], self.writer_sh[name],
        )

    def _producer(self, group, operand):
        times = _view(group, self.hbm, operand)[0]
        if times.size == 0:
            return 0, -1
        flat = np.asarray(times).ravel()
        pos = int(flat.argmax())
        wg = self._writer_group(group)
        owners = np.asarray(_view(wg, self.writer_hbm, operand)[0]).ravel()
        return int(flat[pos]), int(owners[pos]) - 1

    def _issue(self, preview):
        begin, pc, tape, selected = preview
        ins = tape.next_command()[1]
        op = ins.op
        entry = {"tape": tape.name, "op": op, "kind": space_kind(ins), "begin": begin,
                 "event": ins.args.get("event"), "cause": "none", "wait": 0, "pred": -1}
        idx = len(self.rec)
        if op in COMPUTE or op in TRANSFER:
            group = self.group_states[tape]
            nxt = group.next_issue
            reads, writes = self._operands(ins)
            raw_t, raw_p = 0, -1
            for operand in reads:
                if "imm" in operand:
                    continue
                t, p = self._producer(group, operand)
                if t > raw_t:
                    raw_t, raw_p = t, p
            war_t = max((_max_destination(group, self.hbm, o) for o in writes), default=0)
            slot_t = 0
            if op in TRANSFER:
                ready = max(nxt, raw_t, war_t)
                slot = self._available_slot(tape.sm, ready)
                slot_t = slot[0] if slot else 0
            terms = {"issue_order": nxt, "raw": raw_t, "war": war_t, "dma_slot": slot_t}
            cause, bound = max(terms.items(), key=lambda kv: (kv[1], kv[0] == "raw"))
            if begin > bound:
                cause = "blocked_pending"
            entry["wait"] = max(0, begin - nxt)
            entry["cause"] = cause if entry["wait"] else "none"
            entry["pred"] = raw_p if cause == "raw" else self.last_tape_idx.get(tape.name, -1)
            entry["raw_producer"] = raw_p
        elif op in SYNC:
            peers = [p for p in selected if not p.control]
            arrive = {p.name: self.group_states[p].next_issue for p in peers}
            drain = {p.name: self.group_states[p].last_finish for p in peers}
            last_arrive = max(arrive.values(), default=0)
            entry["arrive"] = arrive
            entry["drain_target"] = max(drain.values(), default=0)
            entry["peers"] = [p.name for p in peers]
            entry["cause"] = "sync"
        entry["idx"] = idx
        try:
            super()._issue(preview)
        except AssertionError:
            recent = [r for r in self.rec[-400:] if r["tape"] == tape.name][-6:]
            peers = [r for r in self.rec[-60:] if r["begin"] >= begin - 16][-12:]
            print(json.dumps({"failed": entry, "sm": tape.sm, "current_cycle": self.current_cycle,
                              "same_tape_recent": recent, "global_recent": peers},
                             default=str, indent=1), file=sys.stderr)
            raise
        if op in COMPUTE or op in TRANSFER:
            group = self.group_states[tape]
            wg = self._writer_group(group)
            for operand in self._operands(ins)[1]:
                _view(wg, self.writer_hbm, operand)[0][...] = idx + 1
        self.rec.append(entry)
        for p in selected:
            self.tape_seq[p.name].append(idx)
            self.last_tape_idx[p.name] = idx


def run(hardware, program_text, case):
    model = MODELS[case.split("_")[0]] if case else None
    layout = build_layout(model, case.split("_")[1]) if case else None
    words = required_hbm_words(program_text, layout) if case else 1 << 22
    reqs = step_requirements(case, layout) if case else None
    trace = {}
    saved = hbm_race.validate_hbm_races
    hbm_race.validate_hbm_races = lambda *a, **k: None
    try:
        t0 = time.time()
        obs = Observer(hardware, iter_parse(program_text), words, reqs, trace)
        result = obs.run()
        elapsed = time.time() - t0
    finally:
        hbm_race.validate_hbm_races = saved
    for e in obs.rec:
        t = trace.get(e["event"]) if e["event"] else None
        e["finish"] = t["finish"] if t else e["begin"] + 1
    return obs, result, elapsed


def critical_path(rec, tape_seq):
    """Walk back from the latest finish, following the binding dependency."""
    pos_in_tape = {}
    for name, seq in tape_seq.items():
        for i, idx in enumerate(seq):
            pos_in_tape.setdefault(idx, {})[name] = i
    cur = max(range(len(rec)), key=lambda i: rec[i]["finish"])
    path = []
    end = rec[cur]["finish"]
    while cur >= 0:
        e = rec[cur]
        service = e["finish"] - e["begin"]
        path.append(cur)
        if e["op"] in SYNC:
            # binding is either the latest-draining participant instruction or late arrival
            best, best_t = -1, -1
            for name in e.get("peers", []):
                i = pos_in_tape[cur][name]
                for j in reversed(tape_seq[name][:i]):
                    if rec[j]["finish"] > best_t:
                        best, best_t = j, rec[j]["finish"]
                    if rec[j]["op"] in SYNC:
                        break
            cur = best
            continue
        if e["cause"] == "raw" and e["pred"] >= 0:
            cur = e["pred"]
        else:
            seq = tape_seq[e["tape"]]
            i = pos_in_tape[cur][e["tape"]]
            cur = seq[i - 1] if i > 0 else -1
    path.reverse()
    return path, end


def summarize(obs, result, elapsed, top=25):
    rec = obs.rec
    out = {"cycles": result.cycles, "instructions": len(rec), "sim_seconds": round(elapsed, 1)}
    for key in ("peak_window_power_w", "peak_power_upper_bound_w", "average_power_w",
                "total_energy_pj", "hbm_read_bytes", "hbm_write_bytes", "area_mm2"):
        out[key] = getattr(result, key, None)
    # 1. per-tape stall by cause (RAW split by producer kind)
    per_tape = defaultdict(Counter)
    per_stage = defaultdict(Counter)
    for e in rec:
        if e["op"] in SYNC:
            for name, arr in e["arrive"].items():
                last = max(e["arrive"].values())
                per_tape[name][f"sync:{e['op']}:imbalance"] += last - arr
                per_tape[name][f"sync:{e['op']}:drain"] += max(0, e["begin"] - last)
            continue
        if not e["wait"]:
            continue
        key = e["cause"]
        if key == "raw" and e["pred"] >= 0:
            key = f"raw<-{rec[e['pred']]['kind']}"
        per_tape[e["tape"]][key] += e["wait"]
        per_stage[stage_of(label_of(e["event"]))][key] += e["wait"]
    out["stall_by_tape"] = {k: dict(v.most_common()) for k, v in sorted(per_tape.items())}
    by_kind = Counter()
    for e in rec:
        if e["op"] not in SYNC and e["wait"] and e["tape"] == "g0":
            stage = stage_of(label_of(e["event"]))
            by_kind[f"{stage}|{e['kind']}|{e['cause']}"] += e["wait"]
    out["stall_g0_by_kind_cause"] = dict(by_kind.most_common(20))
    out["stall_by_stage"] = {
        k: dict(v.most_common(6)) for k, v in
        sorted(per_stage.items(), key=lambda kv: -sum(kv[1].values()))[:top]
    }
    # 2. service time by op kind (issue->finish), per tape
    service = defaultdict(Counter)
    for e in rec:
        service[e["tape"]][e["kind"]] += e["finish"] - e["begin"]
    out["service_by_tape"] = {k: dict(v.most_common(8)) for k, v in sorted(service.items())}
    # 3. critical path decomposition
    path, end = critical_path(rec, obs.tape_seq)
    cp_kind, cp_stage, cp_gap = Counter(), Counter(), Counter()
    prev_finish = 0
    covered = 0
    for i in path:
        e = rec[i]
        start = max(prev_finish, 0)
        gap = max(0, e["begin"] - start)
        svc = max(0, e["finish"] - max(e["begin"], start))
        cp_gap["gap:" + (e["cause"] if e["op"] not in SYNC else "sync")] += gap
        cp_kind[e["kind"]] += svc
        cp_stage[stage_of(label_of(e["event"]))] += svc + gap
        prev_finish = max(prev_finish, e["finish"])
        covered += gap + svc
    out["critical_path"] = {
        "length_instructions": len(path),
        "end_cycle": end,
        "covered_cycles": covered,
        "service_by_kind": dict(cp_kind.most_common()),
        "gaps": dict(cp_gap.most_common()),
        "by_stage": dict(cp_stage.most_common(top)),
    }
    # 4. energy attribution: totals by resource + composition of the peak window
    if ENERGY_LOG:
        cycles = result.cycles
        diff = np.zeros(cycles + 2)
        per_res = defaultdict(list)
        for res, start, finish, pj in ENERGY_LOG:
            per_res[res].append((start, finish, pj))
            rate = pj / (finish - start)
            diff[start] += rate
            diff[finish] -= rate
        rate_per_cycle = np.cumsum(diff[:cycles])
        window = getattr(result, "power_window_cycles", None) or 1000
        cumsum = np.concatenate(([0.0], np.cumsum(rate_per_cycle)))
        window_pj = cumsum[window:] - cumsum[:-window]
        w0 = int(np.argmax(window_pj))
        w1 = w0 + window
        res_totals = {r: sum(p for _, _, p in v) for r, v in per_res.items()}
        in_window = {
            r: sum(p * max(0, min(f, w1) - max(s, w0)) / (f - s) for s, f, p in v)
            for r, v in per_res.items()
        }
        stages = Counter()
        for e in rec:
            if e["begin"] < w1 and e["finish"] > w0 and e["op"] not in SYNC:
                stages[stage_of(label_of(e["event"]))] += min(e["finish"], w1) - max(e["begin"], w0)
        out["energy"] = {
            "total_pj_by_resource": dict(Counter(res_totals).most_common(15)),
            "peak_window": {
                "start_cycle": w0,
                "fraction": round(w0 / max(1, cycles), 3),
                "window_pj": round(float(window_pj[w0]), 1),
                "by_resource_pj": {k: round(v, 1) for k, v in Counter(in_window).most_common(12)},
                "stages_cycles": dict(stages.most_common(10)),
            },
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hardware", type=Path, required=True)
    ap.add_argument("--program", type=Path, required=True)
    ap.add_argument("--case", choices=("M1_P1", "M2_D1"))
    ap.add_argument("--expect-cycles", type=int)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    hardware = Hardware.from_dict(json.loads(args.hardware.read_text()))
    obs, result, elapsed = run(hardware, args.program.read_text(), args.case)
    out = summarize(obs, result, elapsed)
    out["inputs"] = {"hardware": str(args.hardware), "program": str(args.program), "case": args.case}
    if args.expect_cycles is not None:
        out["matches_frozen_cycles"] = result.cycles == args.expect_cycles
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("cycles", "instructions", "sim_seconds")}
                     | {"matches": out.get("matches_frozen_cycles")}))
    if args.expect_cycles is not None and result.cycles != args.expect_cycles:
        sys.exit("cycle mismatch: observer altered scheduling")


if __name__ == "__main__":
    main()
