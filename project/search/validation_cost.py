"""Read-only ISA cost model, NOT a race validator or eligibility check.

Counts the real expanded synchronization graph. A longest dependency path using
RF/SH writes, barriers, WG predecessors and commits gives a lower bound on each
frozen checker event-set cardinality. HBM dependencies and disjoint-path unions
are deliberately omitted, so they can only increase the real cost. No simulated
score/correctness is produced. CPython logical heap != macOS physical footprint.
"""
from collections import Counter, defaultdict, deque
import json
import re
from pathlib import Path
from codesign.challenge.isa import iter_parse
from codesign.challenge.hbm_race import (_graph, _operands, _local_key, _bounds,
    _candidates, _overlap, _erase, _key, _insert, _Node, _LocalWrite, _priority)


def line_stages(source):
    stack=[];out={}
    for number,line in enumerate(source.splitlines(),1):
        if not line.strip() or line.startswith('#'):continue
        op,args=line.split(' ',1);args=json.loads(args)
        if op=='FOR':stack.append(args['var'])
        names=[args.get('event',''),*stack]
        phase='unknown';stage='control'
        for name in names:
            m=re.match(r'([ps])b\d+(?:l\d+)?(?:g\d+)?_?(.*)',name)
            if not m:continue
            phase='prompt' if m[1]=='p' else 'step'
            rest=m[2]
            for prefix,result in [('attention','attention'),('biasgelu','gelu'),('residual1','residual1'),('residual2','residual2'),('ln1','ln1'),('ln2','ln2'),('qkv','qkv'),('wo','wo'),('w1','w1'),('w2','w2'),('gelu','gelu'),('bias1','gelu'),('res','residual1'),('output','output'),('out','residual2'),('a','attention')]:
                if rest.startswith(prefix):stage=result;break
            if stage!='control':break
        out[number]=(phase,stage)
        if op=='END.FOR':stack.pop()
    return out


def analyze(source, budget_mib=3072):
    locations=line_stages(source)
    program=tuple(iter_parse(source)); counts=Counter(i.op for i in program)
    stages=defaultdict(Counter)
    for ins in program:
        phase,stage=locations[ins.line]
        block=stages[phase+'/'+stage]
        block['instructions']+=1
        block[ins.op]+=1
        if 'event' in ins.args:block['events']+=1
    queues,tape_ids,owners,edges,indegree,commit_before=_graph(program,None)
    ready=deque(i for i,n in enumerate(indegree) if n==0)
    depths=[0]*len(queues.groups);event_depth={};commits={};local={}
    event_pc={ins.args['event']:pc for pc,ins in enumerate(program) if 'event' in ins.args}
    sum_depth=0;max_depth=0;checkpoints=[]
    named_cross=0;barrier_pairs=0;barrier_incidence=0
    for pc,ins in enumerate(program):
        if ins.op in ('WAIT','BARRIER'):
            named_cross+=sum(1 for ev in ins.args['events'] for peer in owners[pc]
                             if peer not in owners[event_pc[ev]])
            if ins.op=='BARRIER':
                n=len(owners[pc]);barrier_pairs+=n*(n-1);barrier_incidence+=2*n
    visited=0
    while ready:
        pc=ready.popleft();ins=program[pc];op=ins.op;a=ins.args;participants=owners[pc]
        if commit_before[pc] is not None:
            for t in participants:depths[t]=max(depths[t],commits[commit_before[pc]])
        if op=='WG.BEGIN':
            t=participants[0];pred=queues.groups[t].predecessor
            if pred is not None:depths[t]=max(depths[t],depths[tape_ids[pred]])
        elif op in ('BARRIER','WAIT'):
            dep=max((event_depth[event_pc[e]] for e in a['events']),default=0)
            if op=='BARRIER':dep=max(dep,*(depths[t] for t in participants))
            for t in participants:depths[t]=max(depths[t],dep)
        elif op=='STEP.COMMIT':
            dep=max(depths);commits[pc]=dep
            for t in participants:depths[t]=dep
        elif op!='WG.END':
            t=participants[0];reads,writes=_operands(ins)
            for operand in (*reads,*writes):
                key=_local_key(t,operand)
                if key is None:continue
                lo,hi=_bounds(operand)
                for old in _candidates(local.get(key),lo,hi):
                    if _overlap(old.descriptor,operand):depths[t]=max(depths[t],event_depth[old.pc])
            depth=depths[t]+1;event_depth[pc]=depth
            sum_depth+=depth;max_depth=max(max_depth,depth)
            phase,stage=locations[ins.line]
            stages[phase+'/'+stage]['sum_dependency_depth_lower_bound']+=depth
            for operand in writes:
                key=_local_key(t,operand)
                if key is None:continue
                lo,hi=_bounds(operand)
                for old in list(_candidates(local.get(key),lo,hi)):
                    if old.descriptor==operand:local[key]=_erase(local[key],_key(old))
                local[key]=_insert(local.get(key),_Node(_LocalWrite(pc,operand,a['event'],lo,hi),_priority(pc)))
        visited+=1
        if visited%10000==0:checkpoints.append(dict(visited=visited,max_depth=max_depth,sum_depth=sum_depth))
        for nxt in edges[pc]:
            indegree[nxt]-=1
            if indegree[nxt]==0:ready.append(nxt)
    assert visited==len(program)
    events=len(event_depth);bypass=counts['WG.BEGIN']<=1
    logical_lower=0 if bypass else 152*events+16*sum_depth
    return dict(expanded_instructions=len(program),source_lines=len(source.splitlines()),
        static_loops=sum(line.startswith('FOR ') for line in source.splitlines()),
        workgroup_lifetimes=counts['WG.BEGIN'],workgroup_names=len({i.args['wg'] for i in program if i.op=='WG.BEGIN'}),
        opcode_counts=counts,events=events,graph_edges=sum(map(len,edges)),
        named_cross_wg_sync_edges=named_cross,barrier_directed_pair_relations=barrier_pairs,
        barrier_incidence_edges=barrier_incidence,
        global_commit_incidence_edges=sum(1 for u,ns in enumerate(edges) for v in ns if program[u].op=='STEP.COMMIT' or program[v].op=='STEP.COMMIT'),
        stages=stages,race_checker_single_wg_fast_path=bypass,
        max_dependency_depth_lower_bound=max_depth,sum_dependency_depth_lower_bound=sum_depth,
        retained_event_sets_logical_bytes_lower_bound=logical_lower,
        decision='prune_validation_cost' if logical_lower>budget_mib*1024**2 else 'within_lower_bound_only',
        budget_mib=budget_mib,checkpoints=checkpoints,
        caveat='Lower bound on uncompressed CPython event-set allocation, not peak RSS prediction. Passing cost screen never establishes race correctness. Multi-path/HBM unions, tuples, prefixes and graph storage add memory.')


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('program');p.add_argument('--report',required=True);p.add_argument('--memory-mib',type=int,default=3072)
    a=p.parse_args();r=analyze(Path(a.program).read_text(),a.memory_mib)
    Path(a.report).write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({k:v for k,v in r.items() if k not in ('stages','checkpoints')},indent=2))

if __name__=='__main__':main()
