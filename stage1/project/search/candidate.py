"""Candidate record: one point in the (hardware, program) design space.

A candidate owns an immutable inputs/ snapshot (compiler.py + hardware.json),
the programs generated from them, evaluator reports, a parsed metrics.json,
and a verdict. Everything under project/experiments/<id>/ is written once and
then treated as read-only evidence.
"""

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXPERIMENTS = REPO / "project" / "experiments"

# Files that define a candidate's design point. Everything else (generated asm,
# reports, metrics) is derived and reproducible.
INPUT_FILES = ("compiler.py", "hardware.json")
# Optional inputs: staged when present, removed from the worktree when absent.
OPTIONAL_INPUT_FILES = ("recipe.json",)
WORKTREE_PATHS = {
    "compiler.py": REPO / "project" / "compiler.py",
    "hardware.json": REPO / "hardware.json",
    "recipe.json": REPO / "project" / "recipe.json",
}
PROGRAM_FILES = ("M1_P1.asm", "M2_D1.asm")
STATUSES = ("created", "evaluating", "incorrect", "evaluated", "kept",
            "reverted", "pruned", "dominated", "not_verified", "ineligible")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class Candidate:
    id: str
    parent: str | None
    hypothesis: str
    summary: str
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    status: str = "created"
    verdict_note: str | None = None
    # CandidateNode fields for autonomous search.
    ops: list | None = None
    family: str | None = None
    bias: str | None = None
    kind: str | None = None          # exploit | expand | novelty | manual
    round: int | None = None
    struct_sig: str | None = None
    fingerprint: str | None = None
    failure_class: str | None = None
    provenance_note: str | None = None

    @property
    def dir(self) -> Path:
        return EXPERIMENTS / self.id

    @property
    def inputs(self) -> Path:
        return self.dir / "inputs"

    @property
    def reports(self) -> Path:
        return self.dir / "reports"

    @property
    def programs(self) -> Path:
        return self.dir / "programs"

    def path(self, name: str) -> Path:
        return self.dir / name

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        for sub in (self.inputs, self.reports, self.programs):
            sub.mkdir(exist_ok=True)
        (self.dir / "candidate.json").write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"
        )

    @classmethod
    def load(cls, cid: str) -> "Candidate":
        data = json.loads((EXPERIMENTS / cid / "candidate.json").read_text())
        return cls(**data)

    def input_hashes(self) -> dict:
        names = [n for n in INPUT_FILES + OPTIONAL_INPUT_FILES
                 if (self.inputs / n).exists()]
        return {name: sha256(self.inputs / name) for name in names}

    def stage_to_worktree(self) -> None:
        """Copy candidate inputs into the live repo files."""
        for name in INPUT_FILES:
            src = self.inputs / name
            if not src.exists():
                raise FileNotFoundError(f"Candidate {self.id} missing input {name}")
            WORKTREE_PATHS[name].write_bytes(src.read_bytes())
        for name in OPTIONAL_INPUT_FILES:
            src = self.inputs / name
            dst = WORKTREE_PATHS[name]
            if src.exists():
                dst.write_bytes(src.read_bytes())
            elif dst.exists():
                dst.unlink()

    def snapshot_from_worktree(self) -> None:
        """Capture the live repo files as this candidate's inputs."""
        for name in INPUT_FILES:
            src = WORKTREE_PATHS[name]
            if not src.exists():
                raise FileNotFoundError(f"Worktree missing {src}")
            (self.inputs / name).write_bytes(src.read_bytes())
        for name in OPTIONAL_INPUT_FILES:
            src = WORKTREE_PATHS[name]
            if src.exists():
                (self.inputs / name).write_bytes(src.read_bytes())
