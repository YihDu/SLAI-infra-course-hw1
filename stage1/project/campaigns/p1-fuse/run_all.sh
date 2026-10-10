#!/bin/bash
# dev loop: exact frozen timing (critpath observer, race check skipped) + dev functional
cd /home/jovyan/projects/SLAI-infra-course-hw1
HW=${HW:-project/experiments/cand-146/inputs/hardware.json}
for v in "$@"; do
  D=project/campaigns/p1-fuse/$v
  PYTHONPATH=. pixi run python project/campaigns/bound-model/critpath.py --hardware $HW --program $D/M1_P1.asm --case M1_P1 --out $D/critpath.json > $D/critpath.log 2>&1 &
  PYTHONPATH=. pixi run python -m project.search.dev_functional --case M1_P1 --hardware $HW --program $D/M1_P1.asm --seed 7 > $D/devfunc.json 2> $D/devfunc.err &
done
wait
