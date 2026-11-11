#!/usr/bin/env python3
"""Aggregate explicitly selected episode JSONs. Never select a best retry automatically."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
from statistics import fmean


def episode(path):
    data = json.loads(path.read_text())
    if data.get('rollout_error'): raise ValueError(f'Infrastructure error: {path.name}')
    m = data['benchmark_metrics']
    collisions = m['collision_count']
    hazards = sum(data[k] for k in ('fall_count','oil_count','water_count'))
    violations = data['red_light_violations_count'] + data['illegal_crossing_violations_count']
    if collisions != m['active_collision_count'] + m['passive_collision_count']:
        raise ValueError(f'Inconsistent collision accounting: {path.name}')
    if m['unsafe_event_count'] != collisions + hazards + violations:
        raise ValueError(f'Inconsistent event accounting: {path.name}')
    moves = {'1m':0, '2m':0, '4m':0, 'turn':0, 'wait':0}
    for step in data['decision_trace']:
        action = step.get('executed_action')
        if not action or step.get('execution_override'): continue
        kind, param = action.get('type'), str(action.get('param'))
        # Parse-valid, executed choices without overrides; raw decisions may include errors.
        raw = step.get('decision') or {}
        if isinstance(raw,dict) and (raw.get('action') or raw.get('type')):
            if (raw.get('action') or raw.get('type')) != kind or str(raw.get('param')) != param: continue
        if kind == 'move_to' and param in {'1','2','3','4','5','6','7'}:
            moves['1m' if param=='1' else '2m' if param in {'2','3','4'} else '4m'] += 1
        elif kind == 'wait' and param in {'1','2','3'}: moves['wait'] += 1
        elif kind == 'turn_around' and param in {'L30','L60','L90','R30','R60','R90'}: moves['turn'] += 1
    return dict(runtime_protocol=data.get('runtime_protocol', 'unversioned'), model=data['model'], mode='realtime' if data['realtime_thinking'] else 'static',
                difficulty='hard' if data['difficulty']=='default' else data['difficulty'],
                effort=data.get('reasoning_effort') or 'default', seed=data['seed'], task_id=data['task_id'],
                success=m['success_indicator'],safe_success=m['safe_success_indicator'],spl=m['spl'],
                collisions=collisions,active=m['active_collision_count'],passive=m['passive_collision_count'],
                safety_score=-(collisions+hazards+10*violations),hazards=hazards,violations=violations,
                distance_m=m['traveled_path_length_cm']/100,sim_seconds=m['sim_time_seconds'],
                decisions=data['decision_count'],mean_response_seconds=m['latency']['mean_response_seconds'],
                **moves,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def summarize(rows):
    groups=defaultdict(list)
    if len({r.get('runtime_protocol', 'unversioned') for r in rows}) > 1:
        raise ValueError('Select records from one runtime protocol.')
    for r in rows: groups[tuple(r[k] for k in ('model','difficulty','mode','effort'))].append(r)
    output=[]
    for key, group in sorted(groups.items()):
        total=lambda k:sum(r[k] for r in group)
        avg=lambda k:fmean(r[k] for r in group)
        n_actions=sum(total(k) for k in ('1m','2m','4m','turn','wait'))
        output.append(dict(zip(('model','difficulty','mode','effort'),key),runtime_protocol=group[0].get('runtime_protocol', 'unversioned'),episodes=len(group),
            sr_percent=100*avg('success'),safe_sr_percent=100*avg('safe_success'),spl=avg('spl'),
            collisions=avg('collisions'),active=avg('active'),passive=avg('passive'),
            safety_score=avg('safety_score'),mean_decisions=avg('decisions'),
            mean_episode_response_seconds=avg('mean_response_seconds'),
            collisions_per_100m_traveled=100*total('collisions')/total('distance_m') if total('distance_m') else None,
            collisions_per_simulated_minute=60*total('collisions')/total('sim_seconds') if total('sim_seconds') else None,
            move_1m_percent=100*total('1m')/n_actions if n_actions else None,
            move_4m_percent=100*total('4m')/n_actions if n_actions else None))
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--episodes',type=Path,nargs='+')
    g.add_argument('--manifest',type=Path,help='JSON list with file paths relative to the manifest; explicitly selects final episodes.')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    paths=args.episodes
    if args.manifest:
        paths=[args.manifest.parent / x['file'] for x in json.loads(args.manifest.read_text())]
    rows=[episode(path) for path in paths]
    keys=[tuple(r[k] for k in ('model','mode','difficulty','effort','seed','task_id')) for r in rows]
    if len(keys)!=len(set(keys)):p.error('Duplicate model/condition/seed/task: explicitly choose a single finalized episode per cell.')
    args.output.mkdir(parents=True,exist_ok=True)
    for name,items in [('episodes.csv',rows),('summary.csv',summarize(rows))]:
        with (args.output/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(items[0]));w.writeheader();w.writerows(items)
    print('| Model | Difficulty | Mode | Effort | N | SR (%) | SafeSR (%) | SPL | Collisions | Safety score |')
    print('| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |')
    for r in summarize(rows):
        print(f"| {r['model']} | {r['difficulty']} | {r['mode']} | {r['effort']} | {r['episodes']} | {r['sr_percent']:.1f} | {r['safe_sr_percent']:.1f} | {r['spl']:.3f} | {r['collisions']:.2f} | {r['safety_score']:.2f} |")

if __name__=='__main__':main()
