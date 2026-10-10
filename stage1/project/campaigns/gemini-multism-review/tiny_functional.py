
import numpy as np
import json
from codesign.challenge.hardware import Hardware
from codesign.challenge.isa import Instruction, iter_parse
from codesign.challenge.micro import MicroMachine, SparseHBM
from codesign.challenge.pipeline import estimate_pipeline
from codesign.challenge.hbm_race import validate_hbm_races

# Load default hardware
with open('hardware.json') as f:
    hw_data = json.load(f)
hw = Hardware(**hw_data)

# Test matrix dimensions: M=8, K=48, N=32
# WG0 computes columns 0..16, WG1 computes columns 16..32
# Memory layout:
# A: offset 0, shape [8, 48] = 384 words
# B: offset 384, shape [48, 32] = 1536 words
# C: offset 1920, shape [8, 32] = 256 words
M, K, N = 8, 48, 32
N_split = 16

lines = []
# WG0 on SM 0, WG1 on SM 1
lines.append('WG.BEGIN ' + json.dumps({'wg': 'wg0', 'sm': 0, 'shared_bytes': 0}))
lines.append('WG.BEGIN ' + json.dumps({'wg': 'wg1', 'sm': 1, 'shared_bytes': 0}))

# Zero accumulators for WG0 and WG1
lines.append('VEC ' + json.dumps({'kind': 'add', 'src': [{'imm': 0.0}, {'imm': 0.0}], 'dst': {'space': 'RF', 'offset': 0, 'count': M * N_split, 'wg': 'wg0', 'lane': 2}, 'event': 'e_zero0'}))
lines.append('VEC ' + json.dumps({'kind': 'add', 'src': [{'imm': 0.0}, {'imm': 0.0}], 'dst': {'space': 'RF', 'offset': 0, 'count': M * N_split, 'wg': 'wg1', 'lane': 2}, 'event': 'e_zero1'}))

# WG0: load A, load B[:, 0:16], MMA, ST C[:, 0:16]
# A is at offset 0
lines.append('LD ' + json.dumps({'src': {'space': 'HBM', 'offset': 0, 'count': M * K, 'shape': [M, K], 'strides': [K, 1], 'wg': None, 'lane': 0}, 'dst': {'space': 'RF', 'offset': 0, 'count': M * K, 'wg': 'wg0', 'lane': 0}, 'event': 'e_lda0'}))
# B[:, 0:16] is at offset 384, shape [48, 16], strides [32, 1]
lines.append('LD ' + json.dumps({'src': {'space': 'HBM', 'offset': 384, 'count': K * N_split, 'shape': [K, N_split], 'strides': [N, 1], 'wg': None, 'lane': 0}, 'dst': {'space': 'RF', 'offset': 0, 'count': K * N_split, 'wg': 'wg0', 'lane': 1}, 'event': 'e_ldb0'}))
lines.append('MMA.ACC ' + json.dumps({'a': {'space': 'RF', 'offset': 0, 'count': M * K, 'wg': 'wg0', 'lane': 0}, 'b': {'space': 'RF', 'offset': 0, 'count': K * N_split, 'wg': 'wg0', 'lane': 1}, 'acc': {'space': 'RF', 'offset': 0, 'count': M * N_split, 'wg': 'wg0', 'lane': 2}, 'm': M, 'n': N_split, 'k': K, 'event': 'e_mma0'}))
# ST C[:, 0:16] to offset 1920, shape [8, 16], strides [32, 1]
lines.append('ST ' + json.dumps({'src': {'space': 'RF', 'offset': 0, 'count': M * N_split, 'wg': 'wg0', 'lane': 2}, 'dst': {'space': 'HBM', 'offset': 1920, 'count': M * N_split, 'shape': [M, N_split], 'strides': [N, 1], 'wg': None, 'lane': 0}, 'event': 'e_st0'}))

# WG1: load A, load B[:, 16:32], MMA, ST C[:, 16:32]
lines.append('LD ' + json.dumps({'src': {'space': 'HBM', 'offset': 0, 'count': M * K, 'shape': [M, K], 'strides': [K, 1], 'wg': None, 'lane': 0}, 'dst': {'space': 'RF', 'offset': 0, 'count': M * K, 'wg': 'wg1', 'lane': 0}, 'event': 'e_lda1'}))
# B[:, 16:32] is at offset 384 + 16 = 400, shape [48, 16], strides [32, 1]
lines.append('LD ' + json.dumps({'src': {'space': 'HBM', 'offset': 400, 'count': K * N_split, 'shape': [K, N_split], 'strides': [N, 1], 'wg': None, 'lane': 0}, 'dst': {'space': 'RF', 'offset': 0, 'count': K * N_split, 'wg': 'wg1', 'lane': 1}, 'event': 'e_ldb1'}))
lines.append('MMA.ACC ' + json.dumps({'a': {'space': 'RF', 'offset': 0, 'count': M * K, 'wg': 'wg1', 'lane': 0}, 'b': {'space': 'RF', 'offset': 0, 'count': K * N_split, 'wg': 'wg1', 'lane': 1}, 'acc': {'space': 'RF', 'offset': 0, 'count': M * N_split, 'wg': 'wg1', 'lane': 2}, 'm': M, 'n': N_split, 'k': K, 'event': 'e_mma1'}))
# ST C[:, 16:32] to offset 1920 + 16 = 1936, shape [8, 16], strides [32, 1]
lines.append('ST ' + json.dumps({'src': {'space': 'RF', 'offset': 0, 'count': M * N_split, 'wg': 'wg1', 'lane': 2}, 'dst': {'space': 'HBM', 'offset': 1936, 'count': M * N_split, 'shape': [M, N_split], 'strides': [N, 1], 'wg': None, 'lane': 0}, 'event': 'e_st1'}))

lines.append('WG.END ' + json.dumps({'wg': 'wg0'}))
lines.append('WG.END ' + json.dumps({'wg': 'wg1'}))

prog_text = '\n'.join(lines) + '\n'

print('Validating races...')
try:
    validate_hbm_races(prog_text)
    print('Race validation PASSED!')
except Exception as e:
    print('Race validation FAILED:', e)

# Test functional execution
hbm = SparseHBM(4096)
np.random.seed(42)
mat_a = np.random.randn(M, K).astype(np.float32)
mat_b = np.random.randn(K, N).astype(np.float32)
expected_c = mat_a @ mat_b

hbm._data[0:M*K] = mat_a.ravel()
hbm._valid[0:M*K] = 1
hbm._data[384:384+K*N] = mat_b.ravel()
hbm._valid[384:384+K*N] = 1

mm = MicroMachine(hw, hbm)
mm.execute(tuple(iter_parse(prog_text)), hbm_races_validated=True)

# Read output C
out_c = np.zeros((M, N), dtype=np.float32)
for r in range(M):
    for c in range(N):
        out_c[r, c] = hbm._data[1920 + r * N + c]

diff = np.max(np.abs(out_c - expected_c))
print(f'Max absolute error vs numpy matmul: {diff:.6e}')

# Timing simulation
timing = estimate_pipeline(hw, iter_parse(prog_text), 4096)
print(f'2-WG cycles: {timing.cycles}')
print('SM 0 busy issue cycles:', timing.resource_stats['sms']['0']['busy_cycles']['issue'])
print('SM 1 busy issue cycles:', timing.resource_stats['sms']['1']['busy_cycles']['issue'])
