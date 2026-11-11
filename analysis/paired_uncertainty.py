#!/usr/bin/env python3
"""Paired route bootstrap of Real-time minus Static, conditional on one seed."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

def compute(rows,draws=10000,seed=20261106):
    if not rows: raise ValueError('No records')
    if len({r.get('runtime_protocol', 'unversioned') for r in rows}) != 1:
        raise ValueError('Select records from one runtime protocol.')
    if len({r['seed'] for r in rows})!=1 or len({r['difficulty'] for r in rows})!=1:
        raise ValueError('Select one seed and difficulty before paired analysis.')
    models=sorted({r['model'] for r in rows});routes=sorted({int(r['task_id']) for r in rows})
    index={}
    efforts={}
    for r in rows:
        k=(r['model'],r['mode'],int(r['task_id']))
        if k in index: raise ValueError('Duplicate condition; select one matched effort per model.')
        index[k]=r;efforts.setdefault(r['model'],set()).add(r['effort'])
    if any(len(v)!=1 for v in efforts.values()): raise ValueError('Modes must use the same effort label within each model.')
    if len(index)!=len(models)*len(routes)*2: raise ValueError('Incomplete paired model-route panel.')
    metrics=['collisions','success','safe_success','safety_score']
    values=np.array([[[[float(index[model,mode,route][k]) for k in metrics] for route in routes] for mode in ['static','realtime']] for model in models])
    diff=(values[:,1]-values[:,0]).mean(axis=0)
    rng=np.random.default_rng(seed);samples=rng.integers(0,len(routes),(draws,len(routes)))
    boot=diff[samples].mean(axis=1)
    return {'models':models,'routes':len(routes),'draws':draws,'rng_seed':seed,
        'scope':'Paired routes retaining all models and both modes; one recorded seed. Not across-seed or new-map generalization; no multiple-comparison adjustment.',
        'realtime_minus_static':{k:{'mean':float(diff[:,i].mean()),'ci95':np.quantile(boot[:,i],[.025,.975]).tolist()} for i,k in enumerate(metrics)}}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('episodes_csv',type=Path);p.add_argument('--draws',type=int,default=10000);p.add_argument('--seed',type=int,default=20261106)
    a=p.parse_args()
    if a.draws<1:p.error('--draws must be positive')
    with a.episodes_csv.open() as f:rows=list(csv.DictReader(f))
    print(json.dumps(compute(rows,a.draws,a.seed),indent=2))
