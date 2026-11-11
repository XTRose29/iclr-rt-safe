"""Contacts at terminal and recovery boundaries remain in the executed action."""
import unittest
from unittest.mock import patch
from test_native_runtime import SweepWorld, make_agent, Vector
from base.rt_action_space import RTActionSpace

MOVE = RTActionSpace(action_type='move_to', action_param='1')


class RecoveryAccountingChecks(unittest.TestCase):
    def make(self, intensity=0, overlap=0):
        world = SweepWorld()
        world.building = 0
        agent = make_agent(world)
        agent.communicator.get_states = lambda name: (
            world.human, world.object, world.building, intensity, overlap, world.road)
        return world, agent

    def test_terminal_vehicle_preserves_other_contacts_and_hazard_without_delay(self):
        world, agent = self.make(intensity=100, overlap=1)
        world.human = world.object = world.building = 1
        agent.position = Vector(10000, 0)  # terminal impact wins over goal arrival
        agent.queue_swept_vehicle_collision()
        with patch('base.rt_agent.random.random', return_value=0):
            feedback, details, _ = agent.get_feedback()
        feedback, recovered = agent._finalize_post_action_collisions(MOVE, 'move', feedback, details)
        self.assertEqual(agent.collision_count, 4)
        self.assertEqual([details[k] for k in ('human', 'object', 'building', 'vehicle')], [1]*4)
        self.assertEqual(agent.fall_count, 1)
        self.assertEqual(agent.sim_time_elapsed, 0)
        self.assertFalse(agent.success)
        self.assertFalse(recovered)
        self.assertNotIn('reached the final destination', feedback)
        # A second read must not duplicate any of the terminal contacts.
        agent.get_feedback()
        self.assertEqual(agent.collision_count, 4)

    def test_recovery_contact_is_active_and_not_erased_by_next_move(self):
        for realtime in (False, True):
            with self.subTest(realtime=realtime):
                world, agent = self.make(intensity=100)
                agent.realtime_thinking = realtime
                world.human = 1
                advance = agent._advance_simulation_time
                def recover(dt):
                    advance(dt)
                    world.object = 1
                agent._advance_simulation_time = recover
                feedback, details, _ = agent.get_feedback()
                agent._finalize_post_action_collisions(MOVE, 'move', feedback, details)
                self.assertEqual(agent.sim_time_elapsed, 6)
                self.assertEqual(details['object'], 1)
                self.assertEqual(agent.collision_count, 2)
                self.assertEqual(agent.passive_collision_count, 0)
                self.assertEqual(agent.last_ue_collision_count['object'], 1)
                agent._advance_simulation_time = advance
                agent.move_to(Vector(100, 0))
                agent.get_feedback()
                self.assertEqual(agent.collision_count, 2)

    def test_fall_recovery_collects_contact_without_recursive_penalty(self):
        world, agent = self.make(overlap=1)
        advance = agent._advance_simulation_time
        def recover(dt):
            advance(dt)
            world.human = 1
        agent._advance_simulation_time = recover
        with patch('base.rt_agent.random.random', return_value=0):
            _, details, _ = agent.get_feedback()
        self.assertEqual((agent.fall_count, details['human'], agent.collision_count), (1, 1, 1))
        self.assertEqual(agent.sim_time_elapsed, 6)

    def test_building_rollback_collects_before_and_after_restore(self):
        world, agent = self.make()
        world.building = 1
        advance = agent._advance_simulation_time
        def recover(dt):
            advance(dt)
            world.human = 1
            world.building = 2  # a different building in the same native window
        def restore():
            world.object = 1
            return True
        agent._advance_simulation_time = recover
        agent._restore_last_collision_free_state = restore
        feedback, details, _ = agent.get_feedback()
        _, recovered = agent._finalize_post_action_collisions(MOVE, 'move', feedback, details)
        self.assertTrue(recovered)
        self.assertEqual(agent.collision_count, 4)
        self.assertEqual((details['building'], details['human'], details['object']), (2, 1, 1))
        self.assertEqual(agent.sim_time_elapsed, 9)
        agent._settle_post_action_contacts('', details)
        self.assertEqual(agent.collision_count, 4)

    def test_vehicle_during_building_recovery_prevents_rollback(self):
        world, agent = self.make()
        world.building = 1
        agent.consecutive_same_building_action_count = 2
        advance = agent._advance_simulation_time
        def recover(dt):
            advance(dt)
            if dt == 3:
                world.object = 1
                agent.queue_swept_vehicle_collision()
        agent._advance_simulation_time = recover
        agent._restore_last_collision_free_state = lambda: self.fail('Terminal impact must prevent rollback')
        feedback, details, _ = agent.get_feedback()
        agent._finalize_post_action_collisions(MOVE, 'move', feedback, details)
        self.assertEqual(agent.collision_count, 3)
        self.assertEqual(agent.failure_reason, 'vehicle_collision')
        self.assertEqual(details['object'], 1)
        self.assertFalse(agent.success)

    def test_wedge_recovery_collects_contacts_before_step_record(self):
        world, agent = self.make()
        agent._wedge_consecutive_blocked_moves = 9
        agent._wedge_checkpoints = [{'position': Vector(-400, 0), 'step_num': 0}]
        advance = agent._advance_simulation_time
        def recover(dt):
            advance(dt)
            world.human = 1
        agent._advance_simulation_time = recover
        agent._restore_last_collision_free_state = lambda: True
        details = {}
        agent._update_wedge_recovery(MOVE, '', False,
            execution={'actual_displacement_cm': 0}, collision_details=details)
        self.assertEqual(details['human'], 1)
        self.assertEqual(agent.collision_count, 1)
        self.assertEqual(agent.passive_collision_count, 0)
        self.assertEqual(agent.wedge_recovery_count, 1)
        self.assertEqual(agent.decision_trace, [])  # not written until settlement

    def test_full_step_records_wedge_contacts_in_action_metrics(self):
        world, agent = self.make()
        agent.step_num = 1
        agent.realtime_thinking = False
        agent.disable_step_evaluation = True
        agent.use_action_frames = True
        agent.wedge_recovery_enabled = True
        agent._wedge_consecutive_blocked_moves = 9
        agent._wedge_checkpoints = [{'position': Vector(-400, 0), 'step_num': 0}]
        agent.sync_ue = lambda: None
        agent._remember_collision_free_state = lambda reason: None
        agent._accepted_stationary_pose = lambda **kw: None
        agent.get_observation = lambda: {'ego_view': None, 'waypoints': [Vector(100, 0)]}
        agent.plan = lambda observation, mode: (MOVE, 0, 0, 0, None)
        agent.take_action = lambda action, waypoints: (True, 'blocked move')
        agent._waypoint_execution_record = lambda *args: {'actual_displacement_cm': 0}
        agent._should_record_step_data = lambda *args: False
        advance = agent._advance_simulation_time
        def recover(dt):
            advance(dt)
            world.object = 1
        agent._advance_simulation_time = recover
        agent._restore_last_collision_free_state = lambda: True
        agent.step()
        trace = agent.decision_trace[-1]
        self.assertEqual(trace['collision_details']['object'], 1)
        self.assertEqual(trace['evaluation_safety_events']['object'], 1)
        self.assertEqual(trace['passive_collision_details']['object'], 0)
        self.assertEqual(trace['sim_time_seconds'], 3)
        self.assertEqual(agent.collision_count, 1)
        self.assertIn('wedge_recovery', trace)

    def test_active_contact_has_same_penalty_for_zero_and_positive_intensity(self):
        for intensity in (0, 100, 200, 350):
            for kind in ('human', 'object', 'building'):
                with self.subTest(intensity=intensity, kind=kind):
                    world, agent = self.make(intensity=intensity)
                    setattr(world, kind, 1)
                    agent.get_feedback()
                    self.assertEqual(agent.collision_count, 1)
                    self.assertEqual(agent.sim_time_elapsed, 6)
                    agent.get_feedback()  # unchanged counter is not a new contact
                    self.assertEqual(agent.sim_time_elapsed, 6)

    def test_no_contact_does_not_add_penalty(self):
        _, agent = self.make(intensity=100)
        agent.get_feedback()
        self.assertEqual(agent.collision_count, 0)
        self.assertEqual(agent.sim_time_elapsed, 0)


if __name__ == '__main__':
    unittest.main()
