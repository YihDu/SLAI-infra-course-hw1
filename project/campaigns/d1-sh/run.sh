#!/bin/bash
# D1 dev loop: frozen check_case-equivalent functional (race skipped) + critpath timing
cd /home/jovyan/projects/SLAI-infra-course-hw1
HW=$1; shift
for v in "$@"; do
  D=project/campaigns/d1-sh/$v
  PYTHONPATH=. pixi run python project/campaigns/bound-model/critpath.py --hardware $HW --program $D/M2_D1.asm --case M2_D1 --out $D/critpath.json > $D/critpath.log 2>&1 &
  PYTHONPATH=. pixi run python -m project.search.dev_functional --case M2_D1 --hardware $HW --program $D/M2_D1.asm --seed 7 > $D/devfunc.json 2> $D/devfunc.err &
done
wait
