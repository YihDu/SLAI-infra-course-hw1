"""Single writer, bounded evaluator scheduling and exact artifact reuse."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
import copy
import fcntl
import json
import threading
import time

from . import state
from .candidate import EXPERIMENTS, Candidate


@dataclass
class Budget:
    maximum: int = 12
    wall_seconds: float = 7200
    used: int = 0
    audit_path: Path | None = None
    started: float = field(default_factory=time.monotonic)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def claim(self, mode, candidate=None):
        with self.lock:
            if self.used >= self.maximum or time.monotonic() - self.started >= self.wall_seconds:
                raise RuntimeError('evaluation_budget_exhausted')
            self.used += 1
            if self.audit_path:
                self.audit_path.parent.mkdir(parents=True, exist_ok=True)
                with self.audit_path.open("a") as stream:
                    stream.write(json.dumps({"launch": self.used, "mode": mode,
                                             "candidate": candidate, "time": time.time()}) + "\n")

    def available(self):
        return self.used < self.maximum and time.monotonic() - self.started < self.wall_seconds

    def summary(self):
        return {'used': self.used, 'maximum': self.maximum,
                'wall_seconds': round(time.monotonic() - self.started, 3)}


@contextmanager
def writer_lock():
    """Prevent two independent controller processes from writing the registry."""
    EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    with (EXPERIMENTS / '.controller.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('another search controller is active') from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def recover(reg):
    """Called under writer lock; no previous coordinator can still own work."""
    from .records import record_result
    for cid, meta in list(reg['candidates'].items()):
        cand = Candidate.load(cid)
        if meta.get('status') != 'evaluating' and cand.status != 'evaluating':
            continue
        path = cand.dir / 'metrics.json'
        metrics = json.loads(path.read_text()) if path.exists() else {
            'candidate': cid, 'reports': {}, 'verdict': 'not_verified',
            'failure_class': 'interrupted', 'failure_note': 'Recovered incomplete evaluation',
            'evaluation_attempts': 0,
        }
        path.write_text(json.dumps(metrics, indent=2) + '\n')
        record_result(reg, cand, metrics)
    state.save(reg)


def evaluate_many(reg, candidates, seeds, budget, *, jobs=2, validation='grade',
                  timeout_seconds=600, memory_mib=3072, force=False):
    """Only evaluation subprocesses run in parallel; state/cache writes stay here."""
    from . import cache, evaluator, runtime
    from .records import record_result
    jobs = max(1, min(int(jobs), 2))
    results = []
    candidates = list(candidates)
    if not getattr(cache, "_history_indexed", False):
        cache.index_history(seeds, validation)
        cache._history_indexed = True

    def finish(cand, metrics, key=None):
        metrics['candidate'] = cand.id
        (cand.dir / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
        record_result(reg, cand, metrics)
        if key and not metrics.get('cache_hit'):
            cache.store(key, cand, metrics)
        state.save(reg)
        results.append(metrics)
        score = metrics.get('reports', {}).get('grade', {}).get('score')
        print(f'[{cand.id}] {metrics.get("verdict")} score={score} budget={budget.used}/{budget.maximum}', flush=True)

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for start in range(0, len(candidates), jobs):
            wave = []
            for cand in candidates[start:start + jobs]:
                if (cand.dir / 'metrics.json').exists() or list(cand.reports.glob('*.json')):
                    raise FileExistsError(f'{cand.id} has results; create a new candidate/attempt')
                cand.status = 'evaluating'; cand.save()
                reg['candidates'][cand.id]['status'] = 'evaluating'; state.save(reg)
                try:
                    prepared = evaluator.prepare(cand, timeout_seconds=timeout_seconds, memory_mib=memory_mib)
                    if prepared.get('failure_class') or prepared.get('verdict') in ('not_verified', 'pruned'):
                        finish(cand, prepared); continue
                    key = cache.execution_key(cand, seeds, validation)
                    hit = None if force else cache.lookup(key)
                    if hit:
                        hit = copy.deepcopy(hit)
                        source = hit.get('candidate')
                        hit.update(cache_hit=True, cache_source=source, evaluation_attempts=0)
                        finish(cand, hit); continue
                    duplicate = next((item for item in wave if item[2] == key), None)
                    if duplicate:
                        wave.append((cand, duplicate[1], key, True)); continue
                    if not budget.available():
                        finish(cand, {'reports': {}, 'verdict': 'not_verified',
                                      'failure_class': 'budget_exhausted', 'evaluation_attempts': 0}); continue
                    fut = pool.submit(evaluator.evaluate, cand, seeds, validation=validation,
                                      timeout_seconds=min(timeout_seconds, max(0.1, budget.wall_seconds - (time.monotonic() - budget.started))), memory_mib=memory_mib,
                                      prepared=True, before_launch=lambda mode, cid=cand.id: budget.claim(mode, cid))
                    wave.append((cand, fut, key, False))
                except Exception as exc:
                    finish(cand, {'reports': {}, 'verdict': 'not_verified',
                                  'failure_class': 'prepare_error', 'failure_note': str(exc),
                                  'evaluation_attempts': 0})
            # Integrate in proposal order, independent of worker completion order.
            for cand, fut, key, duplicate in wave:
                try:
                    metrics = copy.deepcopy(fut.result())
                except KeyboardInterrupt:
                    runtime.cancel_all()
                    raise
                except Exception as exc:
                    metrics = {'reports': {}, 'verdict': 'not_verified',
                               'failure_class': 'evaluator_error', 'failure_note': str(exc)}
                if duplicate:
                    source = metrics.get('candidate')
                    metrics.update(cache_hit=True, cache_source=source, evaluation_attempts=0)
                    if metrics.get('verdict') != 'evaluated':
                        metrics.update(verdict='not_verified', failure_class='duplicate_unverified')
                finish(cand, metrics, None if duplicate else key)
    return results
