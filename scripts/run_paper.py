#!/usr/bin/env python3
"""Plan paper task/model settings with the RT-SAFE core runtime."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    models = json.loads((ROOT / 'configs/paper_models.json').read_text())
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=models, default='Gemini')
    p.add_argument('--study', choices=['rq1', 'rq2', 'rq3'], default='rq2')
    p.add_argument('--task-ids', nargs='+', type=int)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--ue-port', type=int, default=9000)
    p.add_argument('--ue-launcher', default=os.environ.get('SIMWORLD_UE_LAUNCHER'))
    p.add_argument('--output', type=Path, default=ROOT / 'results')
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    model = models[args.model]
    if not model['transport_in_core_release']:
        p.error('Claude Code transport is not included. Recorded model settings are in configs/paper_models.json.')
    if args.execute and (not args.ue_launcher or not Path(args.ue_launcher).is_file()):
        p.error('--execute requires --ue-launcher or SIMWORLD_UE_LAUNCHER pointing to a compatible runtime.')
    if args.task_ids:
        ids = {t['task_id'] for t in json.loads((ROOT / 'configs/paper_tasks.json').read_text())}
        if not set(args.task_ids) <= ids: p.error('Unknown task ID')
    env = os.environ.copy()
    env.update({
        'SIMWORLD_CAMPAIGN_MODEL': model['model'],
        'SIMWORLD_CAMPAIGN_PROVIDER': model['provider'],
        'SIMWORLD_CAMPAIGN_API_MODE': model['api_mode'],
        'SIMWORLD_CAMPAIGN_DIFFICULTIES': 'easy,medium,hard' if args.study == 'rq1' else 'hard',
        'SIMWORLD_CAMPAIGN_ENV_MODES': 'static,realtime' if args.study == 'rq2' else 'realtime',
        'SIMWORLD_CAMPAIGN_EFFORTS': ','.join(model['effort_ladder'].values()) if args.study == 'rq3' else model['recorded_default_request'],
    })
    command = [sys.executable, str(ROOT / 'evaluation/run_openai_frontier_matrix.py'),
               '--mode', 'coordinator',
               '--suite-dir', str(args.output.resolve() / args.study / args.model),
               '--workers', '1', '--worker-count', '1', '--gpus', str(args.gpu),
               '--ports', str(args.ue_port), '--fast-simulation', '--ue-no-rhi-thread',
               '--record-png-compress-level', '3']
    if args.ue_launcher: command += ['--ue-launcher', args.ue_launcher]
    if args.task_ids: command += ['--task-ids', *map(str,args.task_ids)]
    if not args.execute:
        tasks = json.loads((ROOT / 'configs/paper_tasks.json').read_text())
        selected = [t for t in tasks if not args.task_ids or t['task_id'] in args.task_ids]
        conditions = (3 if args.study in ('rq1', 'rq3') else 2)
        print(json.dumps({'starts_rollouts': False, 'runtime_protocol': 'rt_safe_core_v1', 'model': args.model,
            'study': args.study, 'tasks': len(selected), 'conditions': conditions,
            'episodes': len(selected) * conditions,
            'environment': {k: v for k,v in env.items() if k in (
                'SIMWORLD_CAMPAIGN_MODEL','SIMWORLD_CAMPAIGN_PROVIDER','SIMWORLD_CAMPAIGN_API_MODE',
                'SIMWORLD_CAMPAIGN_DIFFICULTIES','SIMWORLD_CAMPAIGN_ENV_MODES','SIMWORLD_CAMPAIGN_EFFORTS')},
            'command': command}, indent=2))
        return 0
    return subprocess.run(command, cwd=ROOT, env=env, check=False).returncode

if __name__ == '__main__': raise SystemExit(main())
