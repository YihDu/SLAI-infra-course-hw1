"""Verify an existing submission with the supplied official starter, in isolation."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
import zipfile

ROOT = Path(__file__).resolve().parents[1]
STAMP = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
DEST = ROOT / 'project' / ('release-verification-' + STAMP)
DEST.mkdir(exist_ok=False)
STARTER = ROOT / 'transformer-codesign-starter (1).zip'
SUBMISSION = ROOT / 'submission.zip'
OUTPUT = ROOT / ('submission-v073-' + STAMP + '.zip')

def digest(data):
    return hashlib.sha256(data).hexdigest()

with zipfile.ZipFile(STARTER) as archive:
    for entry in archive.infolist():
        target = (DEST / entry.filename).resolve()
        if not target.is_relative_to(DEST.resolve()):
            raise ValueError('Unsafe starter archive member')
    archive.extractall(DEST)

with zipfile.ZipFile(SUBMISSION) as archive:
    inputs = {name: archive.read(name) for name in
              ('hardware.json', 'programs/M1_P1.asm', 'programs/M2_D1.asm')}
for name, data in inputs.items():
    (DEST / name).write_bytes(data)

cmd = [sys.executable, 'challenge.py', 'grade', '--hardware', 'hardware.json',
       '--program-p1', 'programs/M1_P1.asm', '--program-d1', 'programs/M2_D1.asm',
       '--baseline', 'baseline_manifest.json', '--seed', '7',
       '--report', 'local-grade.json']
print('Verification directory:', DEST, flush=True)
print('Command:', ' '.join(cmd), flush=True)
with (DEST / 'grade.stdout').open('w') as out, (DEST / 'grade.stderr').open('w') as err:
    result = subprocess.run(cmd, cwd=DEST, stdout=out, stderr=err)
report_path = DEST / 'local-grade.json'
if not report_path.exists():
    raise RuntimeError(f'Grade failed ({result.returncode}); inspect {DEST}/grade.stderr')
report = json.loads(report_path.read_text())
summary = {
    'starter_sha256': digest(STARTER.read_bytes()),
    'submission_sha256': digest(SUBMISSION.read_bytes()),
    'input_sha256': {name: digest(data) for name, data in inputs.items()},
    'returncode': result.returncode,
    'eligible': report.get('eligible'),
    'experimental_score': report.get('experimental_score'),
    'cases': {name: {'functional_passed': case.get('functional_passed'),
                     'cycles': case.get('timing', {}).get('cycles'),
                     'peak_window_power_w': case.get('timing', {}).get('peak_window_power_w')}
              for name, case in report['cases'].items()},
}
with (ROOT / 'project/iteration-log.md').open('a') as log:
    log.write(f'\n## {STAMP} UTC — Official starter v0.7.3 revalidation\n'
              '- Change: same submitted hardware/programs; isolated updated official scorer and frozen baseline.\n'
              f'- Verify: public grade, seed 7 -> {report_path.relative_to(ROOT)}; exit={result.returncode}.\n'
              f'- Result: {json.dumps(summary["cases"])}; eligible={summary["eligible"]}; score={summary["experimental_score"]}.\n'
              '- Decision: preserve original submission; package updated report only if eligible.\n')
if result.returncode == 0 and report.get('eligible'):
    replacements = {'local-grade.json': report_path.read_bytes(),
                    'baseline_manifest.json': (DEST / 'baseline_manifest.json').read_bytes(),
                    'project/iteration-log.md': (ROOT / 'project/iteration-log.md').read_bytes(),
                    'project/verify_current_starter.py': Path(__file__).read_bytes()}
    # Keep existing project/history; omit stale release documents and old scorer copies.
    skip = {'MANIFEST.json', 'README.md', 'ASSEMBLY_GUIDE.md', 'local-grade.json',
            'baseline_manifest.json'}
    with zipfile.ZipFile(SUBMISSION) as old, zipfile.ZipFile(OUTPUT, 'x', zipfile.ZIP_DEFLATED) as new:
        for entry in old.infolist():
            if entry.filename not in skip and entry.filename not in replacements and not entry.filename.startswith('codesign/'):
                new.writestr(entry, old.read(entry.filename))
        for name, data in replacements.items():
            new.writestr(name, data)
    with zipfile.ZipFile(OUTPUT) as archive:
        assert archive.testzip() is None
        for name, data in inputs.items():
            assert archive.read(name) == data
        assert archive.read('local-grade.json') == report_path.read_bytes()
    assert OUTPUT.stat().st_size <= 25 * 1024 * 1024
    summary['output_zip'] = str(OUTPUT)
(DEST / 'verification-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2), flush=True)
if result.returncode:
    sys.exit(result.returncode)
