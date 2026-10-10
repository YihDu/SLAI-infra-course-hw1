"""Attribute D1 issue intervals to Transformer stages using the frozen trace hook."""
import argparse
import importlib.util
import inspect
import json
import re
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from codesign.challenge.hardware import Hardware
from codesign.challenge.isa import iter_parse
from codesign.challenge.pipeline import estimate_pipeline
from codesign.challenge.runner import required_hbm_words, step_requirements


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    candidate = root / "project/experiments" / args.candidate
    spec = importlib.util.spec_from_file_location(
        "profile_d1_compiler", candidate / "inputs/compiler.py"
    )
    compiler = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = compiler
    spec.loader.exec_module(compiler)

    event_stage = {}
    original_emit = compiler.Builder.emit

    def emit(self, op, **kw):
        original_emit(self, op, **kw)
        if "event" in kw:
            event = json.loads(self.lines[-1].split(" ", 1)[1])["event"]
            event_stage[event.split("_{")[0]] = getattr(
                self, "_profile_stage", "output"
            )

    compiler.Builder.emit = emit
    methods = (
        "layernorm", "gemm", "attention_decode", "add_rows",
        "add_bias", "bias_gelu", "gelu",
    )
    for method in methods:
        original = getattr(compiler.Builder, method)

        def wrap(self, *a, _method=method, _original=original, **kw):
            bound = inspect.signature(_original).bind(self, *a, **kw)
            name = bound.arguments["name"]
            stage = {
                "attention_decode": "attention",
                "bias_gelu": "gelu",
                "add_bias": "bias",
                "gelu": "gelu",
            }.get(_method)
            if _method == "gemm":
                stage = next(
                    key for key in ("qkv", "wo", "w1", "w2")
                    if re.search(key + r"(?:g\d+)?$", name)
                )
            if _method == "layernorm":
                stage = "ln1" if name.endswith("ln1") else "ln2"
            if _method == "add_rows":
                stage = "residual1" if name.endswith("res") else "residual2"
            old = getattr(self, "_profile_stage", "output")
            self._profile_stage = stage
            try:
                return _original(self, *a, **kw)
            finally:
                self._profile_stage = old

        setattr(compiler.Builder, method, wrap)

    recipe = json.loads((candidate / "inputs/recipe.json").read_text())
    generated, layout, _ = compiler.generate_m1_d1(recipe=recipe)
    program = (candidate / "programs/M2_D1.asm").read_text()
    assert generated == program, "Instrumentation changed emitted program"
    hardware = Hardware.from_dict(
        json.loads((candidate / "inputs/hardware.json").read_text())
    )
    trace = {}
    timing = estimate_pipeline(
        hardware, iter_parse(program), required_hbm_words(program, layout),
        step_requirements("M2_D1", layout), trace=trace,
    )
    stages = defaultdict(lambda: {
        "instructions": 0, "issue_to_next_cycles": 0, "event_span_sum": 0,
    })
    ordered = []
    for event, event_timing in trace.items():
        base = re.match(r".*?e\d+", event).group()
        ordered.append((
            event_timing["issue"], event_timing["finish"],
            event_stage[base], event,
        ))
    ordered.sort()
    for index, (issue, finish, stage, _event) in enumerate(ordered):
        block = stages[stage]
        block["instructions"] += 1
        block["event_span_sum"] += finish - issue
        next_issue = ordered[index + 1][0] if index + 1 < len(ordered) else timing.cycles
        block["issue_to_next_cycles"] += next_issue - issue
    for block in stages.values():
        block["issue_to_next_fraction"] = block["issue_to_next_cycles"] / timing.cycles
    output = {
        "candidate": args.candidate,
        "cycles": timing.cycles,
        "first_issue": ordered[0][0],
        "stages": stages,
        "timing": asdict(timing),
        "method": (
            "Global issue-to-next attribution across four WG streams; intervals sum "
            "to makespan but overlapping work means this is a bottleneck locator, "
            "not isolated kernel latency."
        ),
        "trace_events": len(trace),
    }
    (candidate / "reports/d1-profile.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )
    print(json.dumps({k: v for k, v in output.items() if k != "timing"}, indent=2))


if __name__ == "__main__":
    main()
