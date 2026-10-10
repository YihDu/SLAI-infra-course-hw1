"""Bounded subprocesses with process-group cleanup and physical-memory accounting."""
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

_active = set()
_lock = threading.Lock()

def footprint(pid):
    if sys.platform == 'darwin':
        # rusage_info_v2: 16 byte uuid, 6 uint64 counters, then resident_size and phys_footprint.
        buf = ctypes.create_string_buffer(1024)
        lib = ctypes.CDLL('/usr/lib/libproc.dylib')
        if lib.proc_pid_rusage(pid, 2, ctypes.byref(buf)) == 0:
            return ctypes.c_uint64.from_buffer(buf, 72).value
        return 0
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return 0

def stop(proc):
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    except ProcessLookupError:
        proc.wait()

def cancel_all():
    with _lock:
        active = list(_active)
    for proc in active:
        stop(proc)

def run(cmd, cwd, prefix, timeout_seconds=600, memory_mib=3072, env=None):
    prefix = Path(prefix)
    start = time.monotonic()
    peak = 0
    failure = None
    with Path(str(prefix)+'.stdout').open('x') as out, Path(str(prefix)+'.stderr').open('x') as err:
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=out, stderr=err, env=env, start_new_session=True)
        with _lock:
            _active.add(proc)
        heartbeat = Path(str(prefix) + '.heartbeat.json')
        def progress(status):
            payload = {'pid': proc.pid, 'status': status, 'elapsed': time.monotonic()-start,
                       'peak_memory_mib': peak / 1024**2, 'updated_at': time.time(), 'command': cmd}
            temporary = heartbeat.with_suffix('.tmp')
            temporary.write_text(json.dumps(payload))
            temporary.replace(heartbeat)
        last_progress = 0
        try:
            while proc.poll() is None:
                peak = max(peak, footprint(proc.pid))
                if time.monotonic() - last_progress >= 1:
                    progress('running'); last_progress = time.monotonic()
                if peak > memory_mib * 1024**2:
                    failure = 'memory_limit'; break
                if time.monotonic() - start > timeout_seconds:
                    failure = 'timeout'; break
                time.sleep(.25)
        finally:
            stop(proc)
            progress(failure or ("completed" if proc.returncode == 0 else "failed"))
            with _lock:
                _active.discard(proc)
    return {'command': cmd, 'exit_code': proc.returncode, 'elapsed': time.monotonic()-start,
            'peak_memory_mib': peak / 1024**2, 'failure_class': failure}


def install_signal_handlers():
    def terminate(signum, frame):
        cancel_all()
        raise KeyboardInterrupt(f"received signal {signum}")
    signal.signal(signal.SIGTERM, terminate)


def cleanup_stale(candidate):
    """Reap a leftover child only when its current command names this candidate."""
    for heartbeat in candidate.reports.glob('*.heartbeat.json'):
        try:
            saved = json.loads(heartbeat.read_text())
            if saved.get('status') != 'running':
                continue
            pid = int(saved['pid'])
            command = subprocess.check_output(['ps', '-p', str(pid), '-o', 'command='], text=True)
            if str(candidate.dir) not in command:
                continue
            if os.getpgid(pid) == pid:
                os.killpg(pid, signal.SIGKILL)
        except (OSError, ValueError, KeyError, subprocess.CalledProcessError):
            continue
