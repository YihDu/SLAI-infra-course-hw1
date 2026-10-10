"""Extend the verified v0.7.3 submission with project evidence and native traces."""
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'submission-v073-20261001-144837-ready.zip'
OUTPUT = ROOT / 'submission-v073-20261007-final-100mib.zip'
FIXED = ('hardware.json', 'programs/M1_P1.asm', 'programs/M2_D1.asm',
         'local-grade.json', 'baseline_manifest.json')


def main():
    tracked = subprocess.check_output(
        ['git', 'ls-files', 'project', 'agent-trace'], cwd=ROOT, text=True
    ).splitlines()
    assert tracked and 'project/iteration-log.md' in tracked
    with zipfile.ZipFile(SOURCE) as old:
        entries = {n: old.read(n) for n in old.namelist()
                   if not n.startswith('agent-trace/')}
        tracked = [n for n in tracked if n in entries or n.startswith('agent-trace/')
                   or n.endswith(('.py', '.sh', '.md'))
                   or '/inputs/' in n or n.endswith(('candidate.json', 'proposal.json'))
                   or n in ('project/submission-audit-20261007.json',
                            'project/trace-commit-coverage-20261007.json')]
        assert all(n in tracked for n in subprocess.check_output(
            ['git', 'ls-files', 'project'], cwd=ROOT, text=True
        ).splitlines() if n.endswith('metrics.json'))
        for name in tracked:
            entries[name] = (ROOT / name).read_bytes()
        source_index = {}
        source_by_hash = {}
        for name in sorted(list(entries)):
            if name.startswith('project/experiments/') and '/inputs/' in name and name.endswith('compiler.py'):
                digest = hashlib.sha256(entries[name]).hexdigest()
                canonical = source_by_hash.setdefault(digest, name)
                source_index[name] = {'path': canonical, 'sha256': digest}
                if canonical != name:
                    del entries[name]
        entries['project/compiler-source-index.json'] = json.dumps(
            {'description': 'Original candidate compiler path -> byte-identical retained source; '
             'resolve missing compiler.py files through this index.', 'sources': source_index},
            ensure_ascii=False, separators=(',', ':')).encode()
        for name in list(entries):
            if name.startswith('project/') and name.endswith('.json'):
                entries[name] = json.dumps(json.loads(entries[name]), ensure_ascii=False,
                                          separators=(',', ':')).encode()
        assert sum(map(len, entries.values())) <= 100 * 1024**2
        for name in FIXED[:3]:
            assert entries[name] == (ROOT / name).read_bytes(), name
        assert all(len(entries[n]) <= 8 * 1024**2 for n in FIXED[1:3])
        report = json.loads(entries['local-grade.json'])
        provenance = report['provenance']
        hardware = json.dumps(json.loads(entries['hardware.json']),
                              sort_keys=True, separators=(',', ':')).encode()
        assert hashlib.sha256(hardware).hexdigest() == provenance['hardware_sha256']
        for case, name in zip(('M1_P1', 'M2_D1'), FIXED[1:3]):
            assert hashlib.sha256(entries[name]).hexdigest() == provenance['program_sha256'][case]
        assert report['eligible']
        assert hashlib.sha256(entries['baseline_manifest.json']).hexdigest() == report['baseline_manifest_sha256']
        assert all(not n.startswith('/') and '..' not in Path(n).parts for n in entries)
        with zipfile.ZipFile(OUTPUT, 'x', zipfile.ZIP_DEFLATED, compresslevel=9) as new:
            for name, content in sorted(entries.items()):
                new.writestr(name, content)
        with zipfile.ZipFile(OUTPUT) as new:
            assert new.testzip() is None
            assert len(new.namelist()) == len(set(new.namelist()))
            for name in FIXED:
                assert old.read(name) == new.read(name), name
            for name in tracked:
                if name in source_index:
                    assert new.read(source_index[name]['path']) == (ROOT / name).read_bytes(), name
                elif name.startswith('project/') and name.endswith('.json'):
                    assert json.loads(new.read(name)) == json.loads((ROOT / name).read_bytes()), name
                else:
                    assert new.read(name) == (ROOT / name).read_bytes(), name
            assert sum(e.file_size for e in new.infolist()) <= 100 * 1024**2
    assert OUTPUT.stat().st_size <= 25 * 1024**2
    print(json.dumps({'zip': str(OUTPUT), 'bytes': OUTPUT.stat().st_size,
                      'mib': OUTPUT.stat().st_size / 1024**2,
                      'expanded_mib': sum(map(len, entries.values())) / 1024**2,
                      'files': len(entries), 'git_commit': subprocess.check_output(
                          ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                      'sha256': hashlib.sha256(OUTPUT.read_bytes()).hexdigest()}, indent=2))


if __name__ == '__main__':
    main()
