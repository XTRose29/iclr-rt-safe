#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
STUDIES = ('neutral_prompt', 'fixed_delay', 'stationary')


def build_commands(args):
    config = json.loads((ROOT / 'configs' / f'{args.study}.json').read_text())
    tasks = json.loads((ROOT / config['task_manifest']).read_text())
    if args.task_ids and not set(args.task_ids) <= {t['task_id'] for t in tasks}:
        raise ValueError('Unknown task ID')
    conditions = config['conditions']
    if args.condition and not set(args.condition) <= {c['name'] for c in conditions}:
        raise ValueError('Unknown condition')
    commands = []
    for condition in conditions:
        if args.condition and condition['name'] not in args.condition:
            continue
        for task in tasks:
            if args.task_ids and task['task_id'] not in args.task_ids:
                continue
            command = [sys.executable, str(ROOT / 'evaluation/run_qwen3vl8b_all_maps.py')]
            values = {
                'maps': task['map'], 'task-indices': task['task_index'],
                'difficulties': 'default', 'rounds': config['rounds'], 'seed': config['seed'],
                'prompt-style': config['prompt_style'], 'prompt-timing-mode': config['prompt_timing_mode'],
                'traffic-policy': config['traffic_policy'], 'env-modes': condition['env_mode'],
                'control-mode': condition['control_mode'],
                'max-steps': condition.get('max_steps', task['max_steps']),
                'ue-port': args.ue_port, 'ue-gpu': args.gpu,
                'output-root': str(args.output.resolve()),
                'suite-name': f"{condition['name']}__task{task['task_id']:03d}",
                'rollout-profile': 'full', 'ue-ready-timeout': 600,
                'ue-max-attempts': 1, 'cell-max-attempts': 1,
                'observation-width': config['observation']['width'],
                'observation-height': config['observation']['height'],
                'observation-fov-deg': config['observation']['fov_degrees'],
                'observation-camera-pitch-deg': config['observation']['pitch_degrees'],
            }
            for key in ('fixed_thinking_seconds', 'reasoning_effort', 'stationary_horizon_seconds'):
                if condition.get(key) is not None:
                    values[key.replace('_', '-')] = condition[key]
            if args.ue_launcher:
                values['ue-launcher'] = args.ue_launcher
            model = config.get('model')
            if model:
                if (model['temperature'], model['top_p'], model['seed']) != (0.7, 1.0, config['seed']):
                    raise ValueError('Sampling settings must match the evaluation runner')
                values.update({'model': model['id'], 'qwen-model-path': model['id'],
                    'qwen-max-tokens': model['max_tokens'], 'qwen-mode': 'external',
                    'qwen-url': args.qwen_url, 'model-request-timeout': 1800})
                command.append('--enable-thinking' if model['enable_thinking'] else '--no-enable-thinking')
            else:
                values['qwen-mode'] = 'external'
            for key, value in values.items():
                command.extend([f'--{key}', str(value)])
            command.extend(['--uniform-waypoint-movement', '--record-per-step', '--fast-simulation',
                            '--no-ue-no-rhi-thread', '--use-action-frames' if config['use_action_frames'] else '--no-use-action-frames'])
            command.append('--concurrent-realtime-inference' if condition.get('fixed_thinking_seconds') is None
                           else '--no-concurrent-realtime-inference')
            commands.append(command)
    return commands


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', choices=STUDIES, required=True)
    parser.add_argument('--condition', nargs='+')
    parser.add_argument('--task-ids', nargs='+', type=int)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--ue-port', type=int, default=9000)
    parser.add_argument('--ue-launcher', default=os.environ.get('SIMWORLD_UE_LAUNCHER'))
    parser.add_argument('--qwen-url', default=os.environ.get('QWEN_BASE_URL', 'http://127.0.0.1:30001/v1'))
    parser.add_argument('--output', type=Path, default=ROOT / 'results' / 'controls')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    if args.execute and (not args.ue_launcher or not Path(args.ue_launcher).is_file()):
        parser.error('--execute requires --ue-launcher or SIMWORLD_UE_LAUNCHER')
    try:
        commands = build_commands(args)
    except ValueError as exc:
        parser.error(str(exc))
    if not args.execute:
        print(json.dumps({'starts_rollouts': False, 'study': args.study,
                          'episodes': len(commands), 'commands': commands}, indent=2))
        return 0
    env = os.environ.copy()
    env.setdefault('SIMWORLD_TIME_ADVANCE_MODE', 'resume_pause')
    for command in commands:
        result = subprocess.run(command, cwd=ROOT, env=env, check=False)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
