
import numpy as np
import json
from codesign.challenge.hardware import Hardware
from codesign.challenge.pipeline import estimate_pipeline
import codesign.challenge.isa as isa

with open('hardware.json') as f:
    hw = Hardware.from_dict(json.load(f))

d = 128
f = 512

x_off = 0
w1_off = x_off + d
b1_off = w1_off + d * f
w2_off = b1_off + f
b2_off = w2_off + f * d
res_off = b2_off + d
out_off = res_off + d

ff1_off = out_off + d
act_off = ff1_off + f
ff2_off = act_off + f
hbm_words = ff2_off + d

ev = 0
def get_ev():
    global ev
    ev += 1
    return f'e{ev}'

lines_base = ['WG.BEGIN {"wg": "g", "sm": 0, "shared_bytes": 0}']
for n in range(0, f, 64):
    lines_base.append(f'VEC {{"kind": "add", "src": [{{"imm": 0.0}}, {{"imm": 0.0}}], "dst": {{"space": "RF", "wg": "g", "lane": 2, "offset": 0, "count": 64}}, "event": "{get_ev()}"}}')
    for k in range(0, d, 16):
        lines_base.append(f'LD {{"src": {{"space": "HBM", "wg": null, "lane": 0, "offset": {x_off + k}, "count": 16}}, "dst": {{"space": "RF", "wg": "g", "lane": 0, "offset": 0, "count": 16}}, "event": "{get_ev()}"}}')
        b_off = w1_off + k * f + n
        lines_base.append(f'LD {{"src": {{"space": "HBM", "wg": null, "lane": 0, "offset": {b_off}, "count": 1024, "shape": [16, 64], "strides": [{f}, 1]}}, "dst": {{"space": "RF", "wg": "g", "lane": 1, "offset": 0, "count": 1024}}, "event": "{get_ev()}"}}')
        lines_base.append(f'MMA.ACC {{"a": {{"space": "RF", "wg": "g", "lane": 0, "offset": 0, "count": 16}}, "b": {{"space": "RF", "wg": "g", "lane": 1, "offset": 0, "count": 1024}}, "acc": {{"space": "RF", "wg": "g", "lane": 2, "offset": 0, "count": 64}}, "m": 1, "n": 64, "k": 16, "event": "{get_ev()}"}}')
    lines_base.append(f'ST {{"src": {{"space": "RF", "wg": "g", "lane": 2, "offset": 0, "count": 64}}, "dst": {{"space": "HBM", "wg": null, "lane": 0, "offset": {ff1_off + n}, "count": 64}}, "event": "{get_ev()}"}}')

for c, cnt in [(0, 384), (384, 128)]:
    lines_base.append(f'LD {{"src": {{"space": "HBM", "wg": null, "lane": 0, "offset": {ff1_off + c}, "count": {cnt}}}, "dst": {{"space": "RF", "wg": "g", "lane": 0, "offset": 0, "count": {cnt}}}, "event": "{get_ev()}"}}')
    lines_base.append(f'LD {{"src": {{"space": "HBM", "wg": null, "lane": 0, "offset": {b1_off + c}, "count": {cnt}}}, "dst": {{"space": "RF", "wg": "g", "lane": 2, "offset": 0, "count": {cnt}}}, "event": "{get_ev()}"}}')
    lines_base.append(f'VEC {{"kind": "add", "src": [{{"space": "RF", "wg": "g", "lane": 0, "offset": 0, "count": {cnt}}}, {{"space": "RF", "wg": "g", "lane": 2, "offset": 0, "count": {cnt}}}], "dst": {{"space": "RF", "wg": "g", "lane": 0, "offset": 0, "count": {cnt}}}, "event": "{get_ev()}"}}')
    lines_base.append(f'VEC {{"kind": "mul", "src": [{{"space": "RF", "wg": "g", "lane": 0, "offset": 0, "count": {cnt}}}, {{"space": "RF", "wg": "g", "lane": 0, "offset": 0, "count": {cnt}}}], "dst": {{"space": "RF", "wg": "g", "lane": 1, "offset": 0, "count": {cnt}}}, "event": "{get_ev()}"}}')
    lines_base.append(f'VEC {{"kind": "mul", "src": [{{"space": "RF", "wg": "g", "lane": 1, "offset": 0, "count": {cnt}}}, {{"space": "RF", "wg": "g", "lane": 0, "offset": 0, "count": {cnt}}}], "dst": {{"space": "RF", "wg": "g", "lane": 1, "offset": 0, "count": {cnt}}}, "event": "{get_ev()}"}}')
    lines_base.append(f'VEC {{"kind": "fma", "src": [{{"space": "RF", "wg": "g", "lane": 1, "offset": 0, "count": {cnt}}}, {{"imm": 0.044715}}, {{"space": "RF", "wg": "g", "lane": 0, "offset": 0, "count": {cnt}}}], "dst": {{"space": "RF", "wg": "g", "lane": 1, "offset": 0, "count": {cnt}}}, "event": "{get_ev()}"}}')
    lines_base.append(f'VEC {{"kind": "mul", "src": [{{"space": "RF", "wg": "g", "lane": 1, "offset": 0, "count": {cnt}}}, {{"imm": 0.7978845818}}], "dst": {{"space": "RF", "wg": "g", "lane": 1, "offset": 0, "count": {cnt}}}, "event": "{get_ev()}"}}')
    lines_base.append(f'SFU {{"kind": "tanh", "src": {{"space": "RF", "wg": "g", "lane": 1, "offset": 0, "count": {cnt}}}, "dst": {{"space": "RF", "wg": "g", "lane": 1, "offset": 0, "count": {cnt}}}, "event": "{get_ev()}"}}')
    lines_base.append(f'VEC {{"kind": "add", "src": [{{"space": "RF", "wg": "g", "lane": 1, "offset": 0, "count": {cnt}}}, {{"imm": 1.0}}], "dst": {{"space": "RF", "wg": "g", "lane": 1, "offset": 0, "count": {cnt}}}, "event": "{get_ev()}"}}')
    lines_base.append(f'VEC {{"kind": "mul", "src": [{{"space": "RF", "wg": "g", "lane": 0, "offset": 0, "count": {cnt}}}, {{"imm": 0.5}}], "dst": {{"space": "RF", "wg": "g", "lane": 0, "offset": 0, "count": {cnt}}}, "event": "{get_ev()}"}}')
    lines_base.append(f'VEC {{"kind": "mul", "src": [{{"space": "RF", "wg": "g", "lane": 0, "offset": 0, "count": {cnt}}}, {{"space": "RF", "wg": "g", "lane": 1, "offset": 0, "count": {cnt}}}], "dst": {{"space": "RF", "wg": "g", "lane": 0, "offset": 0, "count": {cnt}}}, "event": "{get_ev()}"}}')
    lines_base.append(f'ST {{"src": {{"space": "RF", "wg": "g", "lane": 0, "offset": 0, "count": {cnt}}}, "dst": {{"space": "HBM", "wg": null, "lane": 0, "offset": {act_off + c}, "count": {cnt}}}, "event": "{get_ev()}"}}')

for n in range(0, d, 64):
    lines_base.append(f'VEC {{"kind": "add", "src": [{{"imm": 0.0}}, {{"imm": 0.0}}], "dst": {{"space": "RF", "wg": "g", "lane": 2, "offset": 0, "count": 64}}, "event": "{get_ev()}"}}')
    for k in range(0, f, 16):
        lines_base.append(f'LD {{"src": {{"space": "HBM", "wg": null, "lane": 0, "offset": {act_off + k}, "count": 16}}, "dst": {{"space": "RF", "wg": "g", "lane": 0, "offset": 0, "count": 16}}, "event": "{get_ev()}"}}')
        b_off = w2_off + k * d + n
        lines_base.append(f'LD {{"src": {{"space": "HBM", "wg": null, "lane": 0, "offset": {b_off}, "count": 1024, "shape": [16, 64], "strides": [{d}, 1]}}, "dst": {{"space": "RF", "wg": "g", "lane": 1, "offset": 0, "count": 1024}}, "event": "{get_ev()}"}}')
        lines_base.append(f'MMA.ACC {{"a": {{"space": "RF", "wg": "g", "lane": 0, "offset": 0, "count": 16}}, "b": {{"space": "RF", "wg": "g", "lane": 1, "offset": 0, "count": 1024}}, "acc": {{"space": "RF", "wg": "g", "lane": 2, "offset": 0, "count": 64}}, "m": 1, "n": 64, "k": 16, "event": "{get_ev()}"}}')
    lines_base.append(f'ST {{"src": {{"space": "RF", "wg": "g", "lane": 2, "offset": 0, "count": 64}}, "dst": {{"space": "HBM", "wg": null, "lane": 0, "offset": {ff2_off + n}, "count": 64}}, "event": "{get_ev()}"}}')

for n in range(0, d, 64):
    lines_base.append(f'LD {{"src": {{"space": "HBM", "wg": null, "lane": 0, "offset": {res_off + n}, "count": 64}}, "dst": {{"space": "RF", "wg": "g", "lane": 0, "offset": 0, "count": 64}}, "event": "{get_ev()}"}}')
    lines_base.append(f'LD {{"src": {{"space": "HBM", "wg": null, "lane": 0, "offset": {ff2_off + n}, "count": 64}}, "dst": {{"space": "RF", "wg": "g", "lane": 1, "offset": 0, "count": 64}}, "event": "{get_ev()}"}}')
    lines_base.append(f'VEC {{"kind": "add", "src": [{{"space": "RF", "wg": "g", "lane": 0, "offset": 0, "count": 64}}, {{"space": "RF", "wg": "g", "lane": 1, "offset": 0, "count": 64}}], "dst": {{"space": "RF", "wg": "g", "lane": 2, "offset": 0, "count": 64}}, "event": "{get_ev()}"}}')
    lines_base.append(f'LD {{"src": {{"space": "HBM", "wg": null, "lane": 0, "offset": {b2_off + n}, "count": 64}}, "dst": {{"space": "RF", "wg": "g", "lane": 3, "offset": 0, "count": 64}}, "event": "{get_ev()}"}}')
    lines_base.append(f'VEC {{"kind": "add", "src": [{{"space": "RF", "wg": "g", "lane": 2, "offset": 0, "count": 64}}, {{"space": "RF", "wg": "g", "lane": 3, "offset": 0, "count": 64}}], "dst": {{"space": "RF", "wg": "g", "lane": 2, "offset": 0, "count": 64}}, "event": "{get_ev()}"}}')
    lines_base.append(f'ST {{"src": {{"space": "RF", "wg": "g", "lane": 2, "offset": 0, "count": 64}}, "dst": {{"space": "HBM", "wg": null, "lane": 0, "offset": {out_off + n}, "count": 64}}, "event": "{get_ev()}"}}')

lines_base.append('WG.END {"wg": "g"}')

p_base = tuple(isa.parse('\n'.join(lines_base) + '\n'))
res_base = estimate_pipeline(hw, p_base, hbm_words)
print(f'BASELINE:    cycles = {res_base.cycles:6d}, energy = {res_base.total_energy_pj / 1e6:8.2f} uJ, read_bytes = {res_base.hbm_read_bytes:7d}, write_bytes = {res_base.hbm_write_bytes:6d}')

lines_a = [
    'WG.BEGIN {"wg": "g", "sm": 0, "shared_bytes": 0}',
    f'LD {{"src": {{"space": "HBM", "wg": null, "lane": 0, "offset": {x_off}, "count": {d}}}, "dst": {{"space": "RF", "wg": "g", "lane": 5, "offset": 0, "count": {d}}}, "event": "{get_ev()}"}}',
]
for n in range(0, f, 64):
    lines_a.append(f'VEC {{"kind": "add", "src": [{{"imm": 0.0}}, {{"imm": 0.0}}], "dst": {{"space": "RF", "wg": "g", "lane": 2, "offset": 0, "count": 64}}, "event": "{get_ev()}"}}')
    for k in range(0, d, 16):
        b_off = w1_off + k * f + n
        lines_a.append(f'LD {{"src": {{"space": "HBM", "wg": null, "lane": 0, "offset": {b_off}, "count": 1024, "shape": [16, 64], "strides": [{f}, 1]}}, "dst": {{"space": "RF", "wg": "g", "lane": 1, "offset": 0, "count": 1024}}, "event": "{get_ev()}"}}')
        lines_a.append(f'MMA.ACC {{"a": {{"space": "RF", "wg": "g", "lane": 5, "offset": {k}, "count": 16}}, "b": {{"space": "RF", "wg": "g", "lane": 1, "offset": 0, "count": 1024}}, "acc": {{"space": "RF", "wg": "g", "lane": 2, "offset": 0, "count": 64}}, "m": 1, "n": 64, "k": 16, "event": "{get_ev()}"}}')
    lines_a.append(f'LD {{"src": {{"space": "HBM", "wg": null, "lane": 0, "offset": {b1_off + n}, "count": 64}}, "dst": {{"space": "RF", "wg": "g", "lane": 0, "offset": 0, "count": 64}}, "event": "{get_ev()}"}}')
    lines_a.append(f'VEC {{"kind": "add", "src": [{{"space": "RF", "wg": "g", "lane": 2, "offset": 0, "count": 64}}, {{"space": "RF", "wg": "g", "lane": 0, "offset": 0, "count": 64}}], "dst": {{"space": "RF", "wg": "g", "lane": 0, "offset": 0, "count": 64}}, "event": "{get_ev()}"}}')
    lines_a.append(f'VEC {{"kind": "mul", "src": [{{"space": "RF", "wg": "g", "lane": 0, "offset": 0, "count": 64}}, {{"space": "RF", "wg": "g", "lane": 0, "offset": 0, "count": 64}}], "dst": {{"space": "RF", "wg": "g", "lane": 1, "offset": 0, "count": 64}}, "event": "{get_ev()}"}}')
    lines_a.append(f'VEC {{"kind": "mul", "src": [{{"space": "RF", "wg": "g", "lane": 1, "offset": 0, "count": 64}}, {{"space": "RF", "wg": "g", "lane": 0, "offset": 0, "count": 64}}], "dst": {{"space": "RF", "wg": "g", "lane": 1, "offset": 0, "count": 64}}, "event": "{get_ev()}"}}')
    lines_a.append(f'VEC {{"kind": "fma", "src": [{{"space": "RF", "wg": "g", "lane": 1, "offset": 0, "count": 64}}, {{"imm": 0.044715}}, {{"space": "RF", "wg": "g", "lane": 0, "offset": 0, "count": 64}}], "dst": {{"space": "RF", "wg": "g", "lane": 1, "offset": 0, "count": 64}}, "event": "{get_ev()}"}}')
    lines_a.append(f'VEC {{"kind": "mul", "src": [{{"space": "RF", "wg": "g", "lane": 1, "offset": 0, "count": 64}}, {{"imm": 0.7978845818}}], "dst": {{"space": "RF", "wg": "g", "lane": 1, "offset": 0, "count": 64}}, "event": "{get_ev()}"}}')
    lines_a.append(f'SFU {{"kind": "tanh", "src": {{"space": "RF", "wg": "g", "lane": 1, "offset": 0, "count": 64}}, "dst": {{"space": "RF", "wg": "g", "lane": 1, "offset": 0, "count": 64}}, "event": "{get_ev()}"}}')
    lines_a.append(f'VEC {{"kind": "add", "src": [{{"space": "RF", "wg": "g", "lane": 1, "offset": 0, "count": 64}}, {{"imm": 1.0}}], "dst": {{"space": "RF", "wg": "g", "lane": 1, "offset": 0, "count": 64}}, "event": "{get_ev()}"}}')
    lines_a.append(f'VEC {{"kind": "mul", "src": [{{"space": "RF", "wg": "g", "lane": 0, "offset": 0, "count": 64}}, {{"imm": 0.5}}], "dst": {{"space": "RF", "wg": "g", "lane": 0, "offset": 0, "count": 64}}, "event": "{get_ev()}"}}')
    lines_a.append(f'VEC {{"kind": "mul", "src": [{{"space": "RF", "wg": "g", "lane": 0, "offset": 0, "count": 64}}, {{"space": "RF", "wg": "g", "lane": 1, "offset": 0, "count": 64}}], "dst": {{"space": "RF", "wg": "g", "lane": 4, "offset": {n}, "count": 64}}, "event": "{get_ev()}"}}')

for n2 in range(0, d, 64):
    lines_a.append(f'VEC {{"kind": "add", "src": [{{"imm": 0.0}}, {{"imm": 0.0}}], "dst": {{"space": "RF", "wg": "g", "lane": 2, "offset": 0, "count": 64}}, "event": "{get_ev()}"}}')
    for k2 in range(0, f, 16):
        b_off = w2_off + k2 * d + n2
        lines_a.append(f'LD {{"src": {{"space": "HBM", "wg": null, "lane": 0, "offset": {b_off}, "count": 1024, "shape": [16, 64], "strides": [{d}, 1]}}, "dst": {{"space": "RF", "wg": "g", "lane": 1, "offset": 0, "count": 1024}}, "event": "{get_ev()}"}}')
        lines_a.append(f'MMA.ACC {{"a": {{"space": "RF", "wg": "g", "lane": 4, "offset": {k2}, "count": 16}}, "b": {{"space": "RF", "wg": "g", "lane": 1, "offset": 0, "count": 1024}}, "acc": {{"space": "RF", "wg": "g", "lane": 2, "offset": 0, "count": 64}}, "m": 1, "n": 64, "k": 16, "event": "{get_ev()}"}}')
    lines_a.append(f'LD {{"src": {{"space": "HBM", "wg": null, "lane": 0, "offset": {res_off + n2}, "count": 64}}, "dst": {{"space": "RF", "wg": "g", "lane": 0, "offset": 0, "count": 64}}, "event": "{get_ev()}"}}')
    lines_a.append(f'VEC {{"kind": "add", "src": [{{"space": "RF", "wg": "g", "lane": 2, "offset": 0, "count": 64}}, {{"space": "RF", "wg": "g", "lane": 0, "offset": 0, "count": 64}}], "dst": {{"space": "RF", "wg": "g", "lane": 2, "offset": 0, "count": 64}}, "event": "{get_ev()}"}}')
    lines_a.append(f'LD {{"src": {{"space": "HBM", "wg": null, "lane": 0, "offset": {b2_off + n2}, "count": 64}}, "dst": {{"space": "RF", "wg": "g", "lane": 0, "offset": 0, "count": 64}}, "event": "{get_ev()}"}}')
    lines_a.append(f'VEC {{"kind": "add", "src": [{{"space": "RF", "wg": "g", "lane": 2, "offset": 0, "count": 64}}, {{"space": "RF", "wg": "g", "lane": 0, "offset": 0, "count": 64}}], "dst": {{"space": "RF", "wg": "g", "lane": 2, "offset": 0, "count": 64}}, "event": "{get_ev()}"}}')
    lines_a.append(f'ST {{"src": {{"space": "RF", "wg": "g", "lane": 2, "offset": 0, "count": 64}}, "dst": {{"space": "HBM", "wg": null, "lane": 0, "offset": {out_off + n2}, "count": 64}}, "event": "{get_ev()}"}}')

lines_a.append('WG.END {"wg": "g"}')
p_a = tuple(isa.parse('\n'.join(lines_a) + '\n'))
res_a = estimate_pipeline(hw, p_a, hbm_words)
print(f'APPROACH A:  cycles = {res_a.cycles:6d}, energy = {res_a.total_energy_pj / 1e6:8.2f} uJ, read_bytes = {res_a.hbm_read_bytes:7d}, write_bytes = {res_a.hbm_write_bytes:6d} (speedup: {res_base.cycles / res_a.cycles:.2f}x)')

lines_b = [
    'WG.BEGIN {"wg": "g", "sm": 0, "shared_bytes": 0}',
    f'LD {{"src": {{"space": "HBM", "wg": null, "lane": 0, "offset": {x_off}, "count": {d}}}, "dst": {{"space": "RF", "wg": "g", "lane": 5, "offset": 0, "count": {d}}}, "event": "{get_ev()}"}}',
    f'VEC {{"kind": "add", "src": [{{"imm": 0.0}}, {{"imm": 0.0}}], "dst": {{"space": "RF", "wg": "g", "lane": 4, "offset": 0, "count": 64}}, "event": "{get_ev()}"}}',
    f'VEC {{"kind": "add", "src": [{{"imm": 0.0}}, {{"imm": 0.0}}], "dst": {{"space": "RF", "wg": "g", "lane": 6, "offset": 0, "count": 64}}, "event": "{get_ev()}"}}',
]
for c in range(0, f, 64):
    lines_b.append(f'VEC {{"kind": "add", "src": [{{"imm": 0.0}}, {{"imm": 0.0}}], "dst": {{"space": "RF", "wg": "g", "lane": 2, "offset": 0, "count": 64}}, "event": "{get_ev()}"}}')
    for k in range(0, d, 16):
        b_off = w1_off + k * f + c
        lines_b.append(f'LD {{"src": {{"space": "HBM", "wg": null, "lane": 0, "offset": {b_off}, "count": 1024, "shape": [16, 64], "strides": [{f}, 1]}}, "dst": {{"space": "RF", "wg": "g", "lane": 1, "offset": 0, "count": 1024}}, "event": "{get_ev()}"}}')
        lines_b.append(f'MMA.ACC {{"a": {{"space": "RF", "wg": "g", "lane": 5, "offset": {k}, "count": 16}}, "b": {{"space": "RF", "wg": "g", "lane": 1, "offset": 0, "count": 1024}}, "acc": {{"space": "RF", "wg": "g", "lane": 2, "offset": 0, "count": 64}}, "m": 1, "n": 64, "k": 16, "event": "{get_ev()}"}}')
    lines_b.append(f'LD {{"src": {{"space": "HBM", "wg": null, "lane": 0, "offset": {b1_off + c}, "count": 64}}, "dst": {{"space": "RF", "wg": "g", "lane": 0, "offset": 0, "count": 64}}, "event": "{get_ev()}"}}')
    lines_b.append(f'VEC {{"kind": "add", "src": [{{"space": "RF", "wg": "g", "lane": 2, "offset": 0, "count": 64}}, {{"space": "RF", "wg": "g", "lane": 0, "offset": 0, "count": 64}}], "dst": {{"space": "RF", "wg": "g", "lane": 0, "offset": 0, "count": 64}}, "event": "{get_ev()}"}}')
    lines_b.append(f'VEC {{"kind": "mul", "src": [{{"space": "RF", "wg": "g", "lane": 0, "offset": 0, "count": 64}}, {{"space": "RF", "wg": "g", "lane": 0, "offset": 0, "count": 64}}], "dst": {{"space": "RF", "wg": "g", "lane": 1, "offset": 0, "count": 64}}, "event": "{get_ev()}"}}')
    lines_b.append(f'VEC {{"kind": "mul", "src": [{{"space": "RF", "wg": "g", "lane": 1, "offset": 0, "count": 64}}, {{"space": "RF", "wg": "g", "lane": 0, "offset": 0, "count": 64}}], "dst": {{"space": "RF", "wg": "g", "lane": 1, "offset": 0, "count": 64}}, "event": "{get_ev()}"}}')
    lines_b.append(f'VEC {{"kind": "fma", "src": [{{"space": "RF", "wg": "g", "lane": 1, "offset": 0, "count": 64}}, {{"imm": 0.044715}}, {{"space": "RF", "wg": "g", "lane": 0, "offset": 0, "count": 64}}], "dst": {{"space": "RF", "wg": "g", "lane": 1, "offset": 0, "count": 64}}, "event": "{get_ev()}"}}')
    lines_b.append(f'VEC {{"kind": "mul", "src": [{{"space": "RF", "wg": "g", "lane": 1, "offset": 0, "count": 64}}, {{"imm": 0.7978845818}}], "dst": {{"space": "RF", "wg": "g", "lane": 1, "offset": 0, "count": 64}}, "event": "{get_ev()}"}}')
    lines_b.append(f'SFU {{"kind": "tanh", "src": {{"space": "RF", "wg": "g", "lane": 1, "offset": 0, "count": 64}}, "dst": {{"space": "RF", "wg": "g", "lane": 1, "offset": 0, "count": 64}}, "event": "{get_ev()}"}}')
    lines_b.append(f'VEC {{"kind": "add", "src": [{{"space": "RF", "wg": "g", "lane": 1, "offset": 0, "count": 64}}, {{"imm": 1.0}}], "dst": {{"space": "RF", "wg": "g", "lane": 1, "offset": 0, "count": 64}}, "event": "{get_ev()}"}}')
    lines_b.append(f'VEC {{"kind": "mul", "src": [{{"space": "RF", "wg": "g", "lane": 0, "offset": 0, "count": 64}}, {{"imm": 0.5}}], "dst": {{"space": "RF", "wg": "g", "lane": 0, "offset": 0, "count": 64}}, "event": "{get_ev()}"}}')
    lines_b.append(f'VEC {{"kind": "mul", "src": [{{"space": "RF", "wg": "g", "lane": 0, "offset": 0, "count": 64}}, {{"space": "RF", "wg": "g", "lane": 1, "offset": 0, "count": 64}}], "dst": {{"space": "RF", "wg": "g", "lane": 3, "offset": 0, "count": 64}}, "event": "{get_ev()}"}}')

    for k_sub in range(0, 64, 16):
        b2_off_0 = w2_off + (c + k_sub) * d + 0
        lines_b.append(f'LD {{"src": {{"space": "HBM", "wg": null, "lane": 0, "offset": {b2_off_0}, "count": 1024, "shape": [16, 64], "strides": [{d}, 1]}}, "dst": {{"space": "RF", "wg": "g", "lane": 1, "offset": 0, "count": 1024}}, "event": "{get_ev()}"}}')
        lines_b.append(f'MMA.ACC {{"a": {{"space": "RF", "wg": "g", "lane": 3, "offset": {k_sub}, "count": 16}}, "b": {{"space": "RF", "wg": "g", "lane": 1, "offset": 0, "count": 1024}}, "acc": {{"space": "RF", "wg": "g", "lane": 4, "offset": 0, "count": 64}}, "m": 1, "n": 64, "k": 16, "event": "{get_ev()}"}}')
        b2_off_1 = w2_off + (c + k_sub) * d + 64
        lines_b.append(f'LD {{"src": {{"space": "HBM", "wg": null, "lane": 0, "offset": {b2_off_1}, "count": 1024, "shape": [16, 64], "strides": [{d}, 1]}}, "dst": {{"space": "RF", "wg": "g", "lane": 1, "offset": 0, "count": 1024}}, "event": "{get_ev()}"}}')
        lines_b.append(f'MMA.ACC {{"a": {{"space": "RF", "wg": "g", "lane": 3, "offset": {k_sub}, "count": 16}}, "b": {{"space": "RF", "wg": "g", "lane": 1, "offset": 0, "count": 1024}}, "acc": {{"space": "RF", "wg": "g", "lane": 6, "offset": 0, "count": 64}}, "m": 1, "n": 64, "k": 16, "event": "{get_ev()}"}}')

for tile_idx, (lane_acc, n_start) in enumerate([(4, 0), (6, 64)]):
    lines_b.append(f'LD {{"src": {{"space": "HBM", "wg": null, "lane": 0, "offset": {res_off + n_start}, "count": 64}}, "dst": {{"space": "RF", "wg": "g", "lane": 0, "offset": 0, "count": 64}}, "event": "{get_ev()}"}}')
    lines_b.append(f'VEC {{"kind": "add", "src": [{{"space": "RF", "wg": "g", "lane": {lane_acc}, "offset": 0, "count": 64}}, {{"space": "RF", "wg": "g", "lane": 0, "offset": 0, "count": 64}}], "dst": {{"space": "RF", "wg": "g", "lane": {lane_acc}, "offset": 0, "count": 64}}, "event": "{get_ev()}"}}')
    lines_b.append(f'LD {{"src": {{"space": "HBM", "wg": null, "lane": 0, "offset": {b2_off + n_start}, "count": 64}}, "dst": {{"space": "RF", "wg": "g", "lane": 0, "offset": 0, "count": 64}}, "event": "{get_ev()}"}}')
    lines_b.append(f'VEC {{"kind": "add", "src": [{{"space": "RF", "wg": "g", "lane": {lane_acc}, "offset": 0, "count": 64}}, {{"space": "RF", "wg": "g", "lane": 0, "offset": 0, "count": 64}}], "dst": {{"space": "RF", "wg": "g", "lane": {lane_acc}, "offset": 0, "count": 64}}, "event": "{get_ev()}"}}')
    lines_b.append(f'ST {{"src": {{"space": "RF", "wg": "g", "lane": {lane_acc}, "offset": 0, "count": 64}}, "dst": {{"space": "HBM", "wg": null, "lane": 0, "offset": {out_off + n_start}, "count": 64}}, "event": "{get_ev()}"}}')

lines_b.append('WG.END {"wg": "g"}')
p_b = tuple(isa.parse('\n'.join(lines_b) + '\n'))
res_b = estimate_pipeline(hw, p_b, hbm_words)
print(f'APPROACH B:  cycles = {res_b.cycles:6d}, energy = {res_b.total_energy_pj / 1e6:8.2f} uJ, read_bytes = {res_b.hbm_read_bytes:7d}, write_bytes = {res_b.hbm_write_bytes:6d} (speedup: {res_base.cycles / res_b.cycles:.2f}x)')
