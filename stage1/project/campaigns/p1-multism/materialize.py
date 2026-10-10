"""Materialize one explicit P1 structural proposal through the existing DAG."""
import argparse
import json
from pathlib import Path
from project.search import state, dsl, engine

parser=argparse.ArgumentParser()
parser.add_argument('--serial-control', action='store_true')
args=parser.parse_args()
root=Path(__file__).resolve().parents[3]
reg=state.load()
recipe=json.loads((root/'project/experiments/cand-098/inputs/recipe.json').read_text())
recipe['p1_schedule']={'kind':'column_head','shards':4}
if args.serial_control:
    recipe['p1_schedule']['placement']='serial'
hw=json.loads((root/'project/experiments/cand-098/inputs/hardware.json').read_text())
compiler=root/'project/compiler.py'
cid=state.next_id(reg)
prop=dict(parent='cand-098',hypothesis=('Serialized numerical control' if args.serial_control else
          'P1 end-to-end column GEMMs, head attention and row vector stages on four SMs')+'; D1/hardware unchanged',
          ops=[dict(op='p1_schedule',**recipe['p1_schedule'])],family='p1_multism',bias='p1',kind='manual',
          struct_sig='p1-column-head-4'+('-serial' if args.serial_control else ''),
          fingerprint=dsl.fingerprint(recipe,hw,compiler.read_text()),
          recipe=recipe,hardware=hw,compiler_path=str(compiler))
engine._materialize(reg,prop,reg.get('search',{}).get('round',0),cid)
state.save(reg)
print(cid)
