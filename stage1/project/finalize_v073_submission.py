"""Include the completed verification log without changing graded inputs."""
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parents[1]
source = root / 'submission-v073-20261001-144837.zip'
output = root / 'submission-v073-20261001-144837-ready.zip'
replacements = {
    'project/iteration-log.md': (root / 'project/iteration-log.md').read_bytes(),
    'project/verify_current_starter.py': (root / 'project/verify_current_starter.py').read_bytes(),
    'project/release-verification-summary.json': (root / 'project/release-verification-20261001-144837/verification-summary.json').read_bytes(),
}
with zipfile.ZipFile(source) as old, zipfile.ZipFile(output, 'x', zipfile.ZIP_DEFLATED) as new:
    for entry in old.infolist():
        if entry.filename not in replacements:
            new.writestr(entry, old.read(entry.filename))
    for name, content in replacements.items():
        new.writestr(name, content)
with zipfile.ZipFile(source) as old, zipfile.ZipFile(output) as new:
    assert new.testzip() is None
    for name in ('hardware.json', 'programs/M1_P1.asm', 'programs/M2_D1.asm', 'local-grade.json', 'baseline_manifest.json'):
        assert old.read(name) == new.read(name)
    assert b'Official starter v0.7.3 revalidation' in new.read('project/iteration-log.md')
assert output.stat().st_size < 25 * 1024 * 1024
print(output)
print('All graded inputs/report unchanged; log included; archive verified.')
