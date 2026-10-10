"""Single writer, bounded evaluator scheduling and exact artifact reuse."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
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
        from .runtime import cleanup_stale
        cleanup_stale(cand)
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

    # Running slots are replenished on completion. Publication stays in proposal
    # order, so slower earlier candidates cannot change promotion/delta ordering.
    entries = []
    next_publish = 0
    by_key = {}
    active = set()

    def publish_ready():
        nonlocal next_publish
        while next_publish < len(entries):
            cand, future, key, duplicate, immediate = entries[next_publish]
            if future is not None and not future.done():
                break
            try:
                metrics = copy.deepcopy(future.result() if future is not None else immediate)
            except Exception as exc:
                metrics = {'reports': {}, 'verdict': 'not_verified',
                           'failure_class': 'evaluator_error', 'failure_note': str(exc)}
            if duplicate:
                source = metrics.get('candidate')
                metrics.update(cache_hit=True, cache_source=source, evaluation_attempts=0)
                if metrics.get('verdict') != 'evaluated':
                    metrics.update(verdict='not_verified', failure_class='duplicate_unverified')
            finish(cand, metrics, None if duplicate else key)
            next_publish += 1

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        next_candidate = 0
        try:
            while next_candidate < len(candidates) or active:
                while next_candidate < len(candidates) and len(active) < jobs:
                    cand = candidates[next_candidate]
                    next_candidate += 1
                    if (cand.dir / 'metrics.json').exists() or list(cand.reports.glob('*.json')):
                        raise FileExistsError(f'{cand.id} has results; create a new candidate/attempt')
                    cand.status = 'evaluating'; cand.save()
                    reg['candidates'][cand.id]['status'] = 'evaluating'; state.save(reg)
                    future, key, duplicate, immediate = None, None, False, None
                    try:
                        candidate_started = time.monotonic()
                        limit = min(timeout_seconds, max(.01, budget.wall_seconds - (candidate_started-budget.started)))
                        prepared = evaluator.prepare(cand, timeout_seconds=limit, memory_mib=memory_mib)
                        if prepared.get('failure_class') or prepared.get('verdict') in ('not_verified', 'pruned'):
                            immediate = prepared
                        else:
                            key = cache.execution_key(cand, seeds, validation)
                            hit = None if force else cache.lookup(key)
                            if hit:
                                immediate = copy.deepcopy(hit)
                                immediate.update(cache_hit=True, cache_source=hit.get('candidate'), evaluation_attempts=0)
                            elif key in by_key:
                                future, duplicate = by_key[key], True
                            elif not budget.available():
                                immediate = {'reports': {}, 'verdict': 'not_verified',
                                             'failure_class': 'budget_exhausted', 'evaluation_attempts': 0}
                                key = None
                            else:
                                future = pool.submit(evaluator.evaluate, cand, seeds, validation=validation,
                                    timeout_seconds=max(.01, limit - (time.monotonic() - candidate_started)),
                                    memory_mib=memory_mib, prepared=True,
                                    before_launch=lambda mode, cid=cand.id: budget.claim(mode, cid))
                                by_key[key] = future
                                active.add(future)
                    except Exception as exc:
                        immediate = {'reports': {}, 'verdict': 'not_verified',
                                     'failure_class': 'prepare_error', 'failure_note': str(exc),
                                     'evaluation_attempts': 0}
                        key = None
                    entries.append((cand, future, key, duplicate, immediate))
                    publish_ready()
                if active:
                    completed, _ = wait(active, return_when=FIRST_COMPLETED)
                    active.difference_update(completed)
                publish_ready()
        except KeyboardInterrupt:
            runtime.cancel_all()
            raise
    return results
