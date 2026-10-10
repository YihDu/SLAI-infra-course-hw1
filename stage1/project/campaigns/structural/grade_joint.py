"""Run a frozen joint grade independently of the busy registry controller."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

directory=Path(sys.argv[1]).resolve()
root=Path(__file__).resolve().parents[3]
report=directory/"grade.json"
if report.exists():
    raise FileExistsError(report)
command=[sys.executable,str(root/"challenge.py"),"grade",
         "--hardware",str(directory/"hardware.json"),
         "--program-p1",str(directory/"programs/M1_P1.asm"),
         "--program-d1",str(directory/"programs/M2_D1.asm"),
         "--baseline",str(root/"baseline_manifest.json"),
         "--seed","7","--report",str(report)]
meta={"command":command,"status":"running","started_utc":datetime.now(timezone.utc).isoformat()}
(directory/"grade-run.json").write_text(json.dumps(meta,indent=2))
start=time.monotonic()
try:
    with (directory/"grade.stdout").open("w") as out, (directory/"grade.stderr").open("w") as err:
        result=subprocess.run(command,cwd=root,stdout=out,stderr=err,timeout=10800)
    meta.update(status="completed" if result.returncode==0 else "failed",returncode=result.returncode)
    if report.exists():
        raw=json.loads(report.read_text())
        meta.update(score=raw.get("experimental_score"),eligible=raw.get("eligible"))
        meta["cycles"]={case:block.get("timing",{}).get("cycles")
                        for case,block in raw.get("cases",{}).items()}
    print(json.dumps(meta),flush=True)
finally:
    meta["elapsed_seconds"]=round(time.monotonic()-start,2)
    if meta["status"]=="running": meta["status"]="interrupted_or_error"
    (directory/"grade-run.json").write_text(json.dumps(meta,indent=2))
    with (root/"project/iteration-log.md").open("a") as log:
        log.write(f"\n## {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} — Joint grade {directory.parent.name}\n"
                  f"- Change: measured P1/D1 combined with one shared hardware; snapshots `{directory.relative_to(root)}`.\n"
                  f"- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/structural/grade_joint.py {directory.relative_to(root)}` -> `{report.relative_to(root)}`.\n"
                  f"- Result: status={meta['status']}; score={meta.get('score')}; eligible={meta.get('eligible')}; cycles={meta.get('cycles')}.\n"
                  "- Decision: preserve full grade; registry promotion pending controlled import after writer lock is free.\n")
