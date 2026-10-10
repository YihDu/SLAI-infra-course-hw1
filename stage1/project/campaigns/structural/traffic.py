"""Instruction-derived traffic/service audit, not a timing prediction."""
from collections import Counter, defaultdict
import argparse
import json
from pathlib import Path
import sys

from codesign.challenge.hardware import Hardware
from codesign.challenge.isa import iter_parse
from codesign.challenge.service import mma_service

def audit(directory):
    hp=directory/"hardware.json"
    pp=directory/"M1_P1.asm"
    if not hp.exists(): hp=directory/"inputs/hardware.json"
    if not pp.exists(): pp=directory/"programs/M1_P1.asm"
    hw=Hardware.from_dict(json.loads(hp.read_text()))
    total=Counter()
    per=defaultdict(Counter)
    sm={}
    for ins in iter_parse(pp.read_text()):
        a=ins.args
        if ins.op=="WG.BEGIN": sm[a["wg"]]=a["sm"]
        if ins.op in ("LD","ST"):
            src,dst=a["src"],a["dst"]
            key=f'{src["space"]}->{dst["space"]}_bytes'
            n=src["count"]*4
            total[key]+=n
            wg=src["wg"] or dst["wg"]
            per[sm[wg]][key]+=n
        elif ins.op=="MMA.ACC":
            service=mma_service(hw,a["m"],a["n"],a["k"])
            for k,n in (("mma_rf_read_bytes",service.rf_read_bytes),
                        ("mma_tc_service_cycles",service.compute_cycles),
                        ("logical_MAC",a["m"]*a["n"]*a["k"])):
                total[k]+=n
                per[sm[a["acc"]["wg"]]][k]+=n
        total["instructions"]+=1
    return {"totals":dict(total),"max_per_sm":{
        k:max(p[k] for p in per.values()) for k in total if k!="instructions"},
        "note":"Requested payload/service sums. Cache, overlap and stalls require frozen timing."}

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("directories",nargs="+",type=Path)
parser.add_argument("--output",type=Path,default=Path("project/campaigns/structural/traffic.json"))
args=parser.parse_args()
out={str(p):audit(p) for p in args.directories}
args.output.write_text(json.dumps(out,indent=2)+"\n")
for name,r in out.items():
    print(name,json.dumps(r["totals"]))
