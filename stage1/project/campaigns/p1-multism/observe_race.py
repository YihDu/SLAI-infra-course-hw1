"""Observe the unmodified frozen race checker on the complete candidate program.

Python tracing reads progress locals only. It never alters instructions, locals,
knowledge, or validation results. Run under the existing runtime memory bound.
"""
import json
import resource
import sys
from pathlib import Path
from codesign.challenge.hbm_race import validate_hbm_races

candidate=Path(sys.argv[1])
progress=Path(sys.argv[2])
last=-1

def trace(frame,event,arg):
    global last
    if frame.f_code.co_name != 'validate_hbm_races':
        return None
    if event=='line' and 'visited' in frame.f_locals:
        visited=frame.f_locals['visited']
        bucket=visited//1000
        if bucket>last:
            last=bucket
            knowledge=frame.f_locals.get('event_knowledge',{})
            sets=[v.events for v in knowledge.values()]
            row=dict(visited=visited,total=len(frame.f_locals.get('program',[])),
                     saved_events=len(sets),max_dependency_events=max(map(len,sets),default=0),
                     event_set_bytes=sum(map(sys.getsizeof,sets)),
                     peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                     op=frame.f_locals.get('op'),pc=frame.f_locals.get('pc'))
            with progress.open('a') as out:out.write(json.dumps(row)+'\n')
    return trace

sys.settrace(trace)
validate_hbm_races((candidate/'programs/M1_P1.asm').read_text())
sys.settrace(None)
print('Full race validation passed')
