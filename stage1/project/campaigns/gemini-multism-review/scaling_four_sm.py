
import json
from codesign.challenge.hardware import Hardware
from codesign.challenge.isa import iter_parse
from codesign.challenge.pipeline import estimate_pipeline
from codesign.challenge.hbm_race import validate_hbm_races

with open('hardware.json') as f:
    hw = Hardware(**json.load(f))

M, K, N = 64, 256, 256
m_tile = 8
k_chunk = 48
tile_n = 16

def build_gemm_program(num_sms):
    cols_per_sm = N // num_sms
    lines = []
    for sm in range(num_sms):
        lines.append(f'WG.BEGIN {{"wg": "wg{sm}", "sm": {sm}, "shared_bytes": 0}}')
    for sm in range(num_sms):
        wg_name = f'wg{sm}'
        col_start = sm * cols_per_sm
        col_end = col_start + cols_per_sm
        for col in range(col_start, col_end, tile_n):
            for row in range(0, M, m_tile):
                lines.append(f'VEC {{"kind": "add", "src": [{{"imm": 0.0}}, {{"imm": 0.0}}], "dst": {{"space": "RF", "offset": {row * tile_n}, "count": {m_tile * tile_n}, "wg": "{wg_name}", "lane": 2}}, "event": "e_z_{sm}_{col}_{row}"}}')
            for inner in range(0, K, k_chunk):
                depth = min(k_chunk, K - inner)
                b_off = 100000 + inner * N + col
                lines.append(f'LD {{"src": {{"space": "HBM", "offset": {b_off}, "count": {depth * tile_n}, "shape": [{depth}, {tile_n}], "strides": [{N}, 1], "wg": null, "lane": 0}}, "dst": {{"space": "RF", "offset": 0, "count": {depth * tile_n}, "wg": "{wg_name}", "lane": 1}}, "event": "e_b_{sm}_{col}_{inner}"}}')
                for row in range(0, M, m_tile):
                    a_off = row * K + inner
                    lines.append(f'LD {{"src": {{"space": "HBM", "offset": {a_off}, "count": {m_tile * depth}, "shape": [{m_tile}, {depth}], "strides": [{K}, 1], "wg": null, "lane": 0}}, "dst": {{"space": "RF", "offset": 0, "count": {m_tile * depth}, "wg": "{wg_name}", "lane": 0}}, "event": "e_a_{sm}_{col}_{inner}_{row}"}}')
                    lines.append(f'MMA.ACC {{"a": {{"space": "RF", "offset": 0, "count": {m_tile * depth}, "wg": "{wg_name}", "lane": 0}}, "b": {{"space": "RF", "offset": 0, "count": {depth * tile_n}, "wg": "{wg_name}", "lane": 1}}, "acc": {{"space": "RF", "offset": {row * tile_n}, "count": {m_tile * tile_n}, "wg": "{wg_name}", "lane": 2}}, "m": {m_tile}, "n": {tile_n}, "k": {depth}, "event": "e_mma_{sm}_{col}_{inner}_{row}"}}')
            for row in range(0, M, m_tile):
                c_off = 500000 + row * N + col
                lines.append(f'ST {{"src": {{"space": "RF", "offset": {row * tile_n}, "count": {m_tile * tile_n}, "wg": "{wg_name}", "lane": 2}}, "dst": {{"space": "HBM", "offset": {c_off}, "count": {m_tile * tile_n}, "shape": [{m_tile}, {tile_n}], "strides": [{N}, 1], "wg": null, "lane": 0}}, "event": "e_st_{sm}_{col}_{row}"}}')
    for sm in range(num_sms):
        lines.append(f'WG.END {{"wg": "wg{sm}"}}')
    return '\n'.join(lines) + '\n'

base_cycles = None
for num_sms in [1, 2, 4]:
    prog = build_gemm_program(num_sms)
    validate_hbm_races(prog)
    res = estimate_pipeline(hw, iter_parse(prog), 1000000)
    if base_cycles is None:
        base_cycles = res.cycles
    speedup = base_cycles / res.cycles
    efficiency = speedup / num_sms * 100
    print(f'SMs: {num_sms:1d} | Cycles: {res.cycles:8d} | Speedup: {speedup:.2f}x | Efficiency: {efficiency:.1f}% | Avg Power: {res.average_power_w:.2f}W | Peak Power: {res.peak_window_power_w:.2f}W')
