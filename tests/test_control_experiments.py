import contextlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import test_inference_accounting
from test_native_runtime import RTAgent, Vector
from evaluation import run_qwen3vl8b_all_maps as runner
from scripts.run_controls import build_commands, ROOT


class ControlChecks(unittest.TestCase):
    def test_plans_are_executable_and_cover_conditions(self):
        for study, count in [('neutral_prompt', 72), ('fixed_delay', 324), ('stationary', 36)]:
            args = SimpleNamespace(study=study, task_ids=None, condition=None, gpu=0, ue_port=19000,
                ue_launcher='/runtime/SimWorld.sh', output=Path('/tmp/control-test'),
                qwen_url='http://127.0.0.1:30001/v1')
            commands = build_commands(args)
            self.assertEqual(len(commands), count)
            tasks = json.loads((ROOT/'configs/paper_tasks.json').read_text())
            by_route = {(t['map'], t['task_index']): t for t in tasks}
            for command in commands:
                parsed = runner.parse_args(command[2:])
                self.assertEqual(parsed.seed, 0)
                self.assertEqual(parsed.prompt_timing_mode, 'neutral')
                task = by_route[(parsed.maps[0], parsed.task_indices[0])]
                self.assertEqual(parsed.max_steps, 120 if study == 'stationary' else task['max_steps'])
                if study == 'fixed_delay':
                    self.assertIn(parsed.fixed_thinking_seconds, (0, 3, 5))
                    self.assertFalse(parsed.concurrent_realtime_inference)
                if study != 'stationary':
                    self.assertEqual(parsed.qwen_max_tokens, 8192)
                    self.assertEqual(parsed.model, 'Qwen/Qwen3.8-27B-FP8')
            args.task_ids = [999]
            with self.assertRaises(ValueError):
                build_commands(args)

    def test_fixed_delay_is_applied_once_even_on_parse_failure(self):
        helper = test_inference_accounting.InferenceAccountingChecks()
        for delay in (0., 3., 5.):
            for parse_error in (False, True):
                for realtime in (False, True):
                    with self.subTest(delay=delay, parse_error=parse_error, realtime=realtime):
                        world, agent, _ = helper.setup_agent(parse_error=parse_error, realtime=realtime)
                        agent.concurrent_realtime_inference = True
                        agent.fixed_thinking_seconds = delay
                        world.resume_simulation = Mock(side_effect=AssertionError('Inference must stay paused'))
                        advances = []
                        agent._advance_simulation_time = lambda dt: advances.append(dt)
                        agent.take_action = lambda *args: (True, 'test action')
                        helper.step(agent)
                        self.assertEqual(advances, [delay] if delay and realtime else [])
                        self.assertEqual(agent.decision_trace[-1]['modeled_thinking_latency_seconds'], delay if realtime else 0)
                        world.resume_simulation.assert_not_called()

    def test_stationary_horizon_and_no_model_service(self):
        agent = SimpleNamespace(step_num=0, sim_time_elapsed=0, collision_count=0)
        def step(mode):
            self.assertEqual(mode, 'stationary')
            agent.step_num += 1
            agent.sim_time_elapsed += 1
        agent.step = step
        manager = SimpleNamespace(agent=agent, max_steps=120, control_mode='stationary')
        with patch.dict(os.environ, {'SIMWORLD_STATIONARY_HORIZON_SECONDS': '60'}):
            runner.run_agent_until_terminal(manager)
        self.assertEqual(agent.step_num, 60)
        self.assertTrue(agent.stationary_horizon_reached)
        service = Mock()
        runner.ensure_qwen_ready_and_sync_agent_config(service, Path('/unused'), SimpleNamespace(control_mode='stationary'))
        service.ensure_ready.assert_not_called()

    def test_invalid_delays_rejected(self):
        for value in ('-1', 'nan', 'inf'):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                runner.parse_args(['--fixed-thinking-seconds', value])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            runner.parse_args(['--fixed-thinking-seconds', '3', '--token-based'])

    def test_neutral_prompts_match_across_timing_modes(self):
        prompts = [prompt_agent(rt).plan({'ego_view': object(), 'waypoints': []}, 'llm')[-1] for rt in (False, True)]
        for key in ('system_prompt', 'user_prompt'):
            self.assertEqual(prompts[0][key], prompts[1][key])
        self.assertNotIn('simulator is paused', prompts[0]['system_prompt'])
        self.assertNotIn('continues to evolve during both your reasoning', prompts[0]['system_prompt'])

def prompt_agent(realtime):
    a=object.__new__(RTAgent)
    a.realtime_thinking=realtime;a.prompt_timing_mode='neutral';a.fixed_thinking_seconds=3.
    a.decision_count=a.step_num=0;a.position=Vector(0,0);a.current_destination=Vector(100,0)
    a.disable_internal_planning_diagnostics=True
    a.format_traffic_light_context=lambda *a,**k:'';a._format_active_route_context=lambda *a:''
    a.speed=200;a.direction=0.;a.required_time=100;a.sim_time_elapsed=0
    a.text_history=[];a.use_action_frames=True;a.action_frames=[];a.prompt_style='instructional'
    a.traffic_policy='visual_only';a.logger=SimpleNamespace(debug=lambda *a:None)
    a.total_char_count=a.total_token_count=a.total_completion_tokens=a.total_reasoning_tokens=0
    a.reasoning_token_usage_reported_decisions=a.total_api_cost_usd=a.total_cached_prompt_tokens=a.total_cache_write_tokens=0
    a.parse_error_count=0
    a.llm=SimpleNamespace(generate_response_openai=lambda **k:(None,12.5,0,'',0,0,0))
    return a

