"""Offline behavioral checks using a deterministic Unreal transport double."""
import json
import math
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Reuse the runner's setup to avoid unrelated city-generation dependencies.
import evaluation.run_openai_benchmark
import numpy as np
from base.rt_agent import RTAgent
from base.rt_unrealcv import RTUnrealCV
from simworld.utils.vector import Vector


class SweepWorld:
    def __init__(self, wall=None):
        self.location = [0., 0., 110.]
        self.orientation = [0., 0., 0.]
        self.wall = wall
        self.human = 0
        self.object = 0
        self.road = 0
        self.vehicle = 0
        self.camera = None
        self.captures = []
        self.samples = []

    def get_location(self, name): return list(self.location)
    def get_orientation(self, name): return list(self.orientation)
    def set_orientation(self, value, name): self.orientation[:] = value
    def humanoid_stop(self, name): pass
    def initialize_collision_window(self, name):
        self.human = 0
        self.object = 0
    def set_location(self, *args): raise AssertionError('Movement must not teleport')
    def set_camera_location(self, camera_id, value): self.camera = tuple(value)
    def set_camera_rotation(self, *args): pass
    def move_actor_with_sweep(self, name, value):
        self.location[:] = value
        if self.wall is not None and value[0] > self.wall:
            self.location[0] = self.wall
            self.human += 1
        self.samples.append(tuple(self.location))
        return tuple(self.location)


def make_agent(world, frames=False, edges=True):
    comm = SimpleNamespace(
        unrealcv=world,
        get_states=lambda name: (world.human, world.object, 0, 0., 0, world.road),
        get_vehicle_collision_number=lambda name: world.vehicle,
    )
    agent = RTAgent(Vector(*world.location[:2]), Vector(1, 0), Vector(10000, 0),
                    [Vector(10000, 0)], 100, comm, None,
                    task_edges=[{'node1': [0., 0.], 'node2': [10000., 0.], 'type': 'sidewalk'}] if edges else None,
                    use_action_frames=frames)
    agent.name = 'RT_AGENT'
    agent.camera_id = 2
    agent.first_person_camera_initialized = True
    agent.record_action_frames = False
    agent._get_traffic_light_snapshot = lambda: {}
    agent._advance_simulation_time = lambda dt: setattr(agent, 'sim_time_elapsed', round(agent.sim_time_elapsed+dt, 6))
    def image(*args):
        world.captures.append(world.camera)
        return np.zeros((2, 2, 3), dtype=np.uint8)
    comm.get_camera_observation = image
    return agent


class NativeRuntimeChecks(unittest.TestCase):
    def test_clear_rays_end_at_selected_waypoint_with_matching_camera(self):
        for edges in (True, False):
            for angle in (-45, 0, 45):
                with self.subTest(edges=edges, angle=angle):
                    world = SweepWorld()
                    agent = make_agent(world, frames=True, edges=edges)
                    target = Vector(400*math.cos(math.radians(angle)), 400*math.sin(math.radians(angle)))
                    agent.last_selected_image_waypoint = target
                    agent.move_to(target)
                    self.assertLess(agent.position.distance(target), .001)
                    self.assertLess(agent.last_commanded_waypoint.distance(target), .001)
                    self.assertAlmostEqual(world.camera[0], target.x)
                    self.assertAlmostEqual(world.camera[1], target.y)
                    self.assertAlmostEqual(agent.traveled_path_length_cm, 400, delta=0.001)
                    self.assertEqual(len(world.samples), 20)
                    self.assertEqual(len(world.captures), 3)
                    for fraction, pose in zip((.25, .5, .75), world.captures):
                        self.assertAlmostEqual(pose[0], target.x*fraction, delta=0.001)
                        self.assertAlmostEqual(pose[1], target.y*fraction, delta=0.001)

    def test_blocked_ray_keeps_actual_pose_and_counts_native_contact_once(self):
        world = SweepWorld(wall=155)
        agent = make_agent(world, frames=True)
        agent.last_action_type = 'MOVE'
        agent.last_action_start_position = Vector(0, 0)
        agent.move_to(Vector(400, 0))
        self.assertEqual(agent.position, Vector(155, 0))
        self.assertEqual(world.camera[:2], (155., 0.))
        self.assertEqual(agent.last_commanded_waypoint, Vector(400, 0))
        self.assertAlmostEqual(agent.traveled_path_length_cm, 155)
        self.assertEqual(world.human, 1)
        self.assertEqual(agent.sim_time_elapsed, 2)
        _, first, _ = agent.get_feedback()
        _, second, _ = agent.get_feedback()
        self.assertEqual((first['human'], second['human']), (1, 0))
        self.assertEqual(agent.collision_count, 1)

    def test_proximity_and_endpoint_miss_do_not_create_human_or_object_collision(self):
        for kind in ('human', 'object'):
            with self.subTest(kind=kind):
                agent = make_agent(SweepWorld())
                agent._physical_off_route_move_endpoint_missed = lambda: True
                agent._ue_physical_block_contact_kind = lambda: kind
                _, details, _ = agent.get_feedback()
                self.assertEqual(details[kind], 0)
                self.assertEqual(agent.collision_count, 0)

    def test_native_object_contact_remains_counted_without_intensity(self):
        world = SweepWorld();world.object = 2
        agent = make_agent(world)
        _, details, _ = agent.get_feedback()
        self.assertEqual(details['object'], 2)
        self.assertEqual(agent.collision_count, 2)

    def test_road_contact_uses_ue_and_retains_legal_crossing_exclusion(self):
        for ue, geometry, crossing, sidewalk, unsafe in (
            (0, True, False, False, 0),
            (1, False, False, False, 1),
            (1, True, True, False, 0),
            (1, True, False, True, 0),
        ):
            with self.subTest(ue=ue, geometry=geometry, crossing=crossing, sidewalk=sidewalk):
                world = SweepWorld();world.road = ue
                agent = make_agent(world)
                agent._is_geometrically_on_road = lambda _: geometry
                agent._is_within_marked_crosswalk = lambda _: crossing
                agent._is_within_authored_sidewalk = lambda _: sidewalk
                agent._is_within_route_crosswalk = lambda _: crossing
                observed=[]
                agent._check_illegal_crossing_trigger = lambda current, previous, detection_source=None: observed.append((current,detection_source))
                _, details, _ = agent.get_feedback()
                self.assertEqual(details['touched_road'], ue)
                self.assertEqual(details['unsafe_touched_road'], unsafe)
                self.assertEqual(observed, [(unsafe, 'ue_touched_road' if unsafe else None)])

    def test_vehicle_compensation_is_preserved_and_deduplicated(self):
        for ue, swept in ((0, 1), (1, 0), (1, 1)):
            with self.subTest(ue=ue, swept=swept):
                world=SweepWorld();world.vehicle=ue
                agent=make_agent(world);agent.red_light_conflict_active=True
                if swept: agent.queue_swept_vehicle_collision()
                self.assertEqual(agent._check_vehicle_collision_failure(), 1)
                self.assertEqual(agent._check_vehicle_collision_failure(), 0)
                self.assertEqual(agent.collision_count, 1)
                self.assertEqual(agent.failure_reason, 'vehicle_collision')

    def test_pending_vehicle_impact_stops_further_translation(self):
        world=SweepWorld();agent=make_agent(world)
        advance=agent._advance_simulation_time
        def tick(dt):
            advance(dt)
            if agent.sim_time_elapsed == .2:
                agent.queue_swept_vehicle_collision()
        agent._advance_simulation_time=tick
        agent.move_to(Vector(400,0))
        self.assertEqual(agent.position,Vector(20,0))
        self.assertEqual(len(world.samples),1)
        self.assertEqual(agent.sim_time_elapsed,2)

    def test_sweep_preserves_current_physics_height(self):
        world=SweepWorld();agent=make_agent(world)
        advance=agent._advance_simulation_time;sweep=world.move_actor_with_sweep
        def tick(dt):
            advance(dt);world.location[2]+=.25
        def checked(name, target):
            self.assertAlmostEqual(target[2],world.location[2])
            return sweep(name,target)
        agent._advance_simulation_time=tick;world.move_actor_with_sweep=checked
        agent.move_to(Vector(400,0))
        self.assertAlmostEqual(world.location[2],115)

    def test_new_native_window_counts_equal_counter_value_as_new_contact(self):
        world=SweepWorld(wall=155);agent=make_agent(world)
        agent.move_to(Vector(400,0));agent.get_feedback()
        self.assertEqual(agent.collision_count,1)
        agent.move_to(Vector(0,0));agent.get_feedback()
        agent.move_to(Vector(400,0));agent.get_feedback()
        self.assertEqual(agent.collision_count,2)
        self.assertEqual(world.human,1)

    def test_execution_record_keeps_physical_endpoint_before_recovery(self):
        from base.rt_action_space import RTActionSpace
        agent=make_agent(SweepWorld())
        agent.last_selected_image_waypoint=Vector(400,0)
        agent.last_commanded_waypoint=Vector(400,0)
        agent.last_move_command_duration_seconds=2
        agent.last_move_execution_mode='ue_swept_segment_v1'
        agent.position=Vector(0,0)  # checkpoint after recovery
        record=agent._waypoint_execution_record(
            RTActionSpace(action_type='move_to',action_param='5'),Vector(0,0),Vector(155,0))
        self.assertEqual(record['actual_displacement_cm'],155)
        self.assertEqual(record['post_action_position_cm'],{'x':155.,'y':0.})
        self.assertEqual(record['post_feedback_position_cm'],{'x':0.,'y':0.})
        self.assertFalse(record['reached_selected_waypoint'])

    def test_native_collision_initialization_checks_ack_and_reset(self):
        ue=RTUnrealCV.__new__(RTUnrealCV);ue.lock=threading.Lock()
        ue.client=SimpleNamespace(request=lambda _: '{}')
        ue.get_collision_num=lambda _: json.dumps({'HumanCollision':0,'ObjectCollision':0,'BuildingCollision':0})
        ue.initialize_collision_window('RT_AGENT')
        ue.get_collision_num=lambda _: json.dumps({'HumanCollision':1,'ObjectCollision':0,'BuildingCollision':0})
        with self.assertRaises(RuntimeError):ue.initialize_collision_window('RT_AGENT')
        ue.client=SimpleNamespace(request=lambda _: '{"error":"missing function"}')
        with self.assertRaises(RuntimeError):ue.initialize_collision_window('RT_AGENT')

    def test_native_api_rejects_unacknowledged_command_and_nonfinite_target(self):
        ue=RTUnrealCV.__new__(RTUnrealCV)
        ue.lock=threading.Lock()
        ue.client=SimpleNamespace(request=lambda _: 'error unsupported')
        with self.assertRaises(RuntimeError):ue.move_actor_with_sweep('RT_AGENT',(1,2,3))
        with self.assertRaises(ValueError):ue.move_actor_with_sweep('RT_AGENT',(float('nan'),2,3))
        commands=[]
        ue.client=SimpleNamespace(request=lambda cmd: (commands.append(cmd),json.dumps({'ReturnValue':False}))[1])
        ue.get_location=lambda _: (0,0,3)
        self.assertEqual(ue.move_actor_with_sweep('RT_AGENT',(1,2,3)),(0.,0.,3.))
        self.assertIn('true false',commands[0])

if __name__=='__main__':unittest.main()
