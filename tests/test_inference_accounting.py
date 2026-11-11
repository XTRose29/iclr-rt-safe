"""Inference exposure survives step-zero and invalid model responses."""
import itertools
import unittest
from unittest.mock import patch
from test_native_runtime import SweepWorld, make_agent, Vector
from base.rt_action_space import RTActionSpace


class InferenceAccountingChecks(unittest.TestCase):
    def setup_agent(self, *, step=0, concurrent=False, parse_error=False,
                    realtime=True, vehicle=False, token_based=False):
        world = SweepWorld()
        agent = make_agent(world)
        agent.step_num = step
        agent.realtime_thinking = realtime
        agent.concurrent_realtime_inference = concurrent
        agent.token_based = token_based
        agent.time_alpha = .2
        agent.time_beta = 0
        agent.disable_step_evaluation = True
        agent.use_action_frames = True
        world.set_camera_resolution = lambda *args: None
        world.set_camera_fov = lambda *args: None
        world.resume_simulation = lambda: None
        world.pause_simulation = lambda: None
        agent.sync_ue = lambda: None
        agent._remember_collision_free_state = lambda reason: None
        agent._accepted_stationary_pose = lambda **kw: None
        agent.get_observation = lambda: {'ego_view': None, 'waypoints': [Vector(100, 0)]}
        agent._observation_geometry = lambda *args, **kw: {}
        agent._should_record_step_data = lambda mode, prompt: parse_error
        saved = []
        agent._record_step_data = lambda *args, **kw: saved.append((args[2], kw))
        def contact():
            world.human = 1
            if vehicle:
                world.object = 1
                agent.queue_swept_vehicle_collision()
        action = None if parse_error else RTActionSpace(action_type='move_to', action_param='1')
        def plan(observation, mode):
            if concurrent and realtime and not token_based:
                contact()
            return action, 1, 20, 5, ({'full_response': 'invalid output'} if parse_error else None)
        agent.plan = plan
        advance = agent._advance_simulation_time
        def tick(dt):
            advance(dt)
            if dt == 1:
                contact()
        agent._advance_simulation_time = tick
        def take(action, waypoints):
            agent.move_to(Vector(100, 0))
            return True, 'Move 1 m'
        agent.take_action = take
        return world, agent, saved

    def step(self, agent):
        clock = itertools.count(0, .1)
        with patch('base.rt_agent.time.perf_counter', side_effect=lambda: next(clock)):
            agent.step()

    def test_first_and_later_inference_contacts_are_counted_before_move_reset(self):
        for step in (0, 1):
            for concurrent in (False, True):
                with self.subTest(step=step, concurrent=concurrent):
                    _, agent, _ = self.setup_agent(step=step, concurrent=concurrent)
                    self.step(agent)
                    self.assertEqual(agent.collision_count, 1)
                    self.assertEqual(agent.passive_collision_count, 1)
                    self.assertEqual(agent.decision_trace[-1]['passive_collision_details']['human'], 1)
                    self.assertEqual(agent.decision_trace[-1]['collision_details']['human'], 0)

    def test_parse_failure_retains_exposure_contacts_and_saved_metadata(self):
        for step in (0, 1):
            for concurrent in (False, True):
                with self.subTest(step=step, concurrent=concurrent):
                    _, agent, saved = self.setup_agent(step=step, concurrent=concurrent, parse_error=True)
                    agent.take_action = lambda *args: self.fail('Parse failure must not execute an action')
                    self.step(agent)
                    trace = agent.decision_trace[-1]
                    self.assertEqual(agent.collision_count, 1)
                    self.assertEqual(agent.passive_collision_count, 1)
                    self.assertGreater(trace['modeled_thinking_latency_seconds'], 0)
                    self.assertEqual(trace['sim_time_seconds'], trace['modeled_thinking_latency_seconds'])
                    self.assertIsNone(trace['executed_action'])
                    self.assertEqual(trace['evaluation_safety_events']['during_inference']['human'], 1)
                    self.assertEqual(saved[0][0]['passive_collision_details']['human'], 1)
                    self.assertIn('collided', saved[0][1]['feedback'])
                    # The same native count is not charged again on another failed response.
                    self.step(agent)
                    self.assertEqual(agent.collision_count, 1)
                    self.assertEqual(agent.decision_trace[-1]['passive_collision_details']['human'], 0)

    def test_first_parse_failure_can_terminate_on_vehicle_impact(self):
        for concurrent in (False, True):
            with self.subTest(concurrent=concurrent):
                _, agent, saved = self.setup_agent(concurrent=concurrent, parse_error=True, vehicle=True)
                agent.take_action = lambda *args: self.fail('Terminal inference must not execute an action')
                self.step(agent)
                self.assertEqual(agent.failure_reason, 'vehicle_collision')
                self.assertEqual(agent.collision_count, 3)
                self.assertEqual(agent.passive_collision_count, 3)
                self.assertEqual(agent.decision_trace[-1]['passive_collision_details'],
                                 {'human': 1, 'object': 1, 'building': 0, 'vehicle': 1})
                self.assertEqual(saved[0][0]['passive_collision_details']['human'], 1)

    def test_static_parse_failure_does_not_advance_environment(self):
        _, agent, _ = self.setup_agent(parse_error=True, realtime=False)
        self.step(agent)
        self.assertEqual(agent.sim_time_elapsed, 0)
        self.assertEqual(agent.collision_count, 0)
        self.assertEqual(agent.decision_trace[-1]['modeled_thinking_latency_seconds'], 0)

    def test_token_based_parse_failure_retains_assigned_delay(self):
        _, agent, _ = self.setup_agent(parse_error=True, token_based=True)
        self.step(agent)
        self.assertEqual(agent.sim_time_elapsed, 1)
        self.assertEqual(agent.passive_collision_count, 1)


if __name__ == '__main__':
    unittest.main()
