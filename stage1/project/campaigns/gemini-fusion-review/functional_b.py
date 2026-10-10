
import numpy as np
import json
from codesign.challenge.hardware import Hardware
from codesign.challenge.micro import MicroMachine
import codesign.challenge.isa as isa

with open('hardware.json') as f:
    hw = Hardware.from_dict(json.load(f))

d = 128
f = 512

rng = np.random.default_rng(1234)
x_val = rng.standard_normal((1, d)).astype(np.float32)
w1_val = rng.standard_normal((d, f)).astype(np.float32)
b1_val = rng.standard_normal(f).astype(np.float32)
w2_val = rng.standard_normal((f, d)).astype(np.float32)
b2_val = rng.standard_normal(d).astype(np.float32)
res_val = rng.standard_normal((1, d)).astype(np.float32)

x_off = 0
w1_off = x_off + d
b1_off = w1_off + d * f
w2_off = b1_off + f
b2_off = w2_off + f * d
res_off = b2_off + d
out_off = res_off + d
hbm_size = out_off + d

hbm = np.zeros(hbm_size, dtype=np.float32)
hbm[x_off : x_off + d] = x_val.ravel()
hbm[w1_off : w1_off + d * f] = w1_val.ravel()
hbm[b1_off : b1_off + f] = b1_val.ravel()
hbm[w2_off : w2_off + f * d] = w2_val.ravel()
hbm[b2_off : b2_off + d] = b2_val.ravel()
hbm[res_off : res_off + d] = res_val.ravel()

def gelu_ref(x):
    c = np.float32(np.sqrt(2.0 / np.pi))
    return np.float32(0.5) * x * (np.float32(1.0) + np.tanh(c * (x + np.float32(0.044715) * x**3)))

h1_ref = np.matmul(x_val, w1_val) + b1_val
a_ref = gelu_ref(h1_ref)
y_ref = np.matmul(a_ref, w2_val)
out_ref = res_val + y_ref + b2_val

ev = 0
def get_ev():
    global ev
    ev += 1
    return f'e{ev}'

lines = [
    'WG.BEGIN {"wg": "g", "sm": 0, "shared_bytes": 0}',
    # Load x (128 words) into rf(5, 128)
    f'LD {{"src": {{"space": "HBM", "wg": null, "lane": 0, "offset": {x_off}, "count": {d}}}, "dst": {{"space": "RF", "wg": "g", "lane": 5, "offset": 0, "count": {d}}}, "event": "{get_ev()}"}}',
    # Initialize W2 accumulators: y0 in rf(4, 64), y1 in rf(6, 64)
    f'VEC {{"kind": "add", "src": [{{"imm": 0.0}}, {{"imm": 0.0}}], "dst": {{"space": "RF", "wg": "g", "lane": 4, "offset": 0, "count": 64}}, "event": "{get_ev()}"}}',
    f'VEC {{"kind": "add", "src": [{{"imm": 0.0}}, {{"imm": 0.0}}], "dst": {{"space": "RF", "wg": "g", "lane": 6, "offset": 0, "count": 64}}, "event": "{get_ev()}"}}',
]

# Outer loop over FFN dimension in chunks of 64
for c in range(0, f, 64):
    # Step 1: Compute W1 tile: (1, 128) x W1[:, c:c+64] -> rf(2, 64)
    lines.append(f'VEC {{"kind": "add", "src": [{{"imm": 0.0}}, {{"imm": 0.0}}], "dst": {{"space": "RF", "wg": "g", "lane": 2, "offset": 0, "count": 64}}, "event": "{get_ev()}"}}')
    for k in range(0, d, 16):
        b_off = w1_off + k * f + c
        lines.append(f'LD {{"src": {{"space": "HBM", "wg": null, "lane": 0, "offset": {b_off}, "count": 1024, "shape": [16, 64], "strides": [{f}, 1]}}, "dst": {{"space": "RF", "wg": "g", "lane": 1, "offset": 0, "count": 1024}}, "event": "{get_ev()}"}}')
        lines.append(f'MMA.ACC {{"a": {{"space": "RF", "wg": "g", "lane": 5, "offset": {k}, "count": 16}}, "b": {{"space": "RF", "wg": "g", "lane": 1, "offset": 0, "count": 1024}}, "acc": {{"space": "RF", "wg": "g", "lane": 2, "offset": 0, "count": 64}}, "m": 1, "n": 64, "k": 16, "event": "{get_ev()}"}}')

    # Step 2: Bias + GELU on rf(2, 64) -> rf(3, 64)
    lines.append(f'LD {{"src": {{"space": "HBM", "wg": null, "lane": 0, "offset": {b1_off + c}, "count": 64}}, "dst": {{"space": "RF", "wg": "g", "lane": 0, "offset": 0, "count": 64}}, "event": "{get_ev()}"}}')
    lines.append(f'VEC {{"kind": "add", "src": [{{"space": "RF", "wg": "g", "lane": 2, "offset": 0, "count": 64}}, {{"space": "RF", "wg": "g", "lane": 0, "offset": 0, "count": 64}}], "dst": {{"space": "RF", "wg": "g", "lane": 0, "offset": 0, "count": 64}}, "event": "{get_ev()}"}}')
    lines.append(f'VEC {{"kind": "mul", "src": [{{"space": "RF", "wg": "g", "lane": 0, "offset": 0, "count": 64}}, {{"space": "RF", "wg": "g", "lane": 0, "offset": 0, "count": 64}}], "dst": {{"space": "RF", "wg": "g", "lane": 1, "offset": 0, "count": 64}}, "event": "{get_ev()}"}}')
    lines.append(f'VEC {{"kind": "mul", "src": [{{"space": "RF", "wg": "g", "lane": 1, "offset": 0, "count": 64}}, {{"space": "RF", "wg": "g", "lane": 0, "offset": 0, "count": 64}}], "dst": {{"space": "RF", "wg": "g", "lane": 1, "offset": 0, "count": 64}}, "event": "{get_ev()}"}}')
    lines.append(f'VEC {{"kind": "fma", "src": [{{"space": "RF", "wg": "g", "lane": 1, "offset": 0, "count": 64}}, {{"imm": 0.044715}}, {{"space": "RF", "wg": "g", "lane": 0, "offset": 0, "count": 64}}], "dst": {{"space": "RF", "wg": "g", "lane": 1, "offset": 0, "count": 64}}, "event": "{get_ev()}"}}')
    c_val = float(np.sqrt(2.0 / np.pi))
    lines.append(f'VEC {{"kind": "mul", "src": [{{"space": "RF", "wg": "g", "lane": 1, "offset": 0, "count": 64}}, {{"imm": {c_val}}}], "dst": {{"space": "RF", "wg": "g", "lane": 1, "offset": 0, "count": 64}}, "event": "{get_ev()}"}}')
    lines.append(f'SFU {{"kind": "tanh", "src": {{"space": "RF", "wg": "g", "lane": 1, "offset": 0, "count": 64}}, "dst": {{"space": "RF", "wg": "g", "lane": 1, "offset": 0, "count": 64}}, "event": "{get_ev()}"}}')
    lines.append(f'VEC {{"kind": "add", "src": [{{"space": "RF", "wg": "g", "lane": 1, "offset": 0, "count": 64}}, {{"imm": 1.0}}], "dst": {{"space": "RF", "wg": "g", "lane": 1, "offset": 0, "count": 64}}, "event": "{get_ev()}"}}')
    lines.append(f'VEC {{"kind": "mul", "src": [{{"space": "RF", "wg": "g", "lane": 0, "offset": 0, "count": 64}}, {{"imm": 0.5}}], "dst": {{"space": "RF", "wg": "g", "lane": 0, "offset": 0, "count": 64}}, "event": "{get_ev()}"}}')
    # Destination into rf(3, 64)
    lines.append(f'VEC {{"kind": "mul", "src": [{{"space": "RF", "wg": "g", "lane": 0, "offset": 0, "count": 64}}, {{"space": "RF", "wg": "g", "lane": 1, "offset": 0, "count": 64}}], "dst": {{"space": "RF", "wg": "g", "lane": 3, "offset": 0, "count": 64}}, "event": "{get_ev()}"}}')

    # Step 3: Accumulate into W2: a_c in rf(3, 64). We need to multiply (1, 64) x W2[c:c+64, 0:128]
    # W2 has 4 sub-chunks of k (size 16):
    for k_sub in range(0, 64, 16):
        # Tile 0 (columns 0..64) -> accumulate into rf(4, 64)
        b2_off_0 = w2_off + (c + k_sub) * d + 0
        lines.append(f'LD {{"src": {{"space": "HBM", "wg": null, "lane": 0, "offset": {b2_off_0}, "count": 1024, "shape": [16, 64], "strides": [{d}, 1]}}, "dst": {{"space": "RF", "wg": "g", "lane": 1, "offset": 0, "count": 1024}}, "event": "{get_ev()}"}}')
        lines.append(f'MMA.ACC {{"a": {{"space": "RF", "wg": "g", "lane": 3, "offset": {k_sub}, "count": 16}}, "b": {{"space": "RF", "wg": "g", "lane": 1, "offset": 0, "count": 1024}}, "acc": {{"space": "RF", "wg": "g", "lane": 4, "offset": 0, "count": 64}}, "m": 1, "n": 64, "k": 16, "event": "{get_ev()}"}}')
        # Tile 1 (columns 64..128) -> accumulate into rf(6, 64)
        b2_off_1 = w2_off + (c + k_sub) * d + 64
        lines.append(f'LD {{"src": {{"space": "HBM", "wg": null, "lane": 0, "offset": {b2_off_1}, "count": 1024, "shape": [16, 64], "strides": [{d}, 1]}}, "dst": {{"space": "RF", "wg": "g", "lane": 1, "offset": 0, "count": 1024}}, "event": "{get_ev()}"}}')
        lines.append(f'MMA.ACC {{"a": {{"space": "RF", "wg": "g", "lane": 3, "offset": {k_sub}, "count": 16}}, "b": {{"space": "RF", "wg": "g", "lane": 1, "offset": 0, "count": 1024}}, "acc": {{"space": "RF", "wg": "g", "lane": 6, "offset": 0, "count": 64}}, "m": 1, "n": 64, "k": 16, "event": "{get_ev()}"}}')

# Final epilogue: add res and b2 to rf(4, 64) and rf(6, 64), store to HBM
for tile_idx, (lane_acc, n_start) in enumerate([(4, 0), (6, 64)]):
    lines.append(f'LD {{"src": {{"space": "HBM", "wg": null, "lane": 0, "offset": {res_off + n_start}, "count": 64}}, "dst": {{"space": "RF", "wg": "g", "lane": 0, "offset": 0, "count": 64}}, "event": "{get_ev()}"}}')
    lines.append(f'VEC {{"kind": "add", "src": [{{"space": "RF", "wg": "g", "lane": {lane_acc}, "offset": 0, "count": 64}}, {{"space": "RF", "wg": "g", "lane": 0, "offset": 0, "count": 64}}], "dst": {{"space": "RF", "wg": "g", "lane": {lane_acc}, "offset": 0, "count": 64}}, "event": "{get_ev()}"}}')
    lines.append(f'LD {{"src": {{"space": "HBM", "wg": null, "lane": 0, "offset": {b2_off + n_start}, "count": 64}}, "dst": {{"space": "RF", "wg": "g", "lane": 0, "offset": 0, "count": 64}}, "event": "{get_ev()}"}}')
    lines.append(f'VEC {{"kind": "add", "src": [{{"space": "RF", "wg": "g", "lane": {lane_acc}, "offset": 0, "count": 64}}, {{"space": "RF", "wg": "g", "lane": 0, "offset": 0, "count": 64}}], "dst": {{"space": "RF", "wg": "g", "lane": {lane_acc}, "offset": 0, "count": 64}}, "event": "{get_ev()}"}}')
    lines.append(f'ST {{"src": {{"space": "RF", "wg": "g", "lane": {lane_acc}, "offset": 0, "count": 64}}, "dst": {{"space": "HBM", "wg": null, "lane": 0, "offset": {out_off + n_start}, "count": 64}}, "event": "{get_ev()}"}}')

lines.append('WG.END {"wg": "g"}')
prog_text = '\n'.join(lines) + '\n'

p = tuple(isa.parse(prog_text))
m = MicroMachine(hw, hbm)
res_hbm = m.execute(p, hbm_races_validated=True)

sim_out = res_hbm[out_off : out_off + d].reshape(1, d)
max_err = np.max(np.abs(sim_out - out_ref))
rel_err = np.max(np.abs(sim_out - out_ref) / (np.abs(out_ref) + 1e-7))
print('Approach B executed successfully!')
print(f'Max absolute error: {max_err:.6e}')
print(f'Max relative error: {rel_err:.6e}')
tol_check = np.all(np.abs(sim_out - out_ref) <= 1e-3 + 1e-3 * np.abs(out_ref))
print(f'Within tolerance: {tol_check}')
