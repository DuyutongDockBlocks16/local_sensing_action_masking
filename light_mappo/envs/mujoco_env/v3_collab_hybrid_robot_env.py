import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import numpy as np
import mujoco
import mujoco.viewer
import concurrent.futures
# from envs.mujoco_env.util_threads.object_placer import place_object_on_table_random
# from envs.mujoco_env.util_threads.object_remover_step_counter import remove_object_on_plane_with_step_counter_with_flag
import threading
import random
import time
from enum import Enum
# from light_mappo.envs.mujoco_env.utils.mujoco_object_color_randomiser import randomize_materials_at_runtime
from light_mappo.envs.env_core import Action

class AgentRobot(Enum):
    ROBOT2 = 0
    ROBOT3 = 1
    ROBOT4 = 2
    ROBOT5 = 3
    ROBOT6 = 4
    ROBOT7 = 5

class V3CollabHybridMuJoCoEnv(gym.Env):
    
    def _get_data_and_model(self):
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        xml_file = ""
        if self.map_type == "a":
            xml_file = "collab_mirobot_7_3.xml"
        elif self.map_type == "b":
            xml_file = "collab_mirobot_9_9.xml"
        elif self.map_type == "c" :
            xml_file = "collab_mirobot_7_7_2agents.xml"
        elif self.map_type == "d" :
            xml_file = "collab_mirobot_7_7_4agents.xml"

        xml_path = os.path.join(current_dir, "xml", xml_file)
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        time_step = 0.005
        model.opt.timestep = time_step
        return model, data
    
    def _initialize_robot_state(self, robot_index):
        robot_rover_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"robot{robot_index}:rover")
        robot_bodies = [
            f"robot{robot_index}:rover",         # chassis
            f"robot{robot_index}:r-l-wheel",     # rear left wheel
            f"robot{robot_index}:r-r-wheel",     # rear right wheel
            f"robot{robot_index}:f-l-wheel",     # front left wheel
            f"robot{robot_index}:f-l-wheel-hub", f"robot{robot_index}:f-l-wheel-1", f"robot{robot_index}:f-l-wheel-2",  # front left wheel hub and spokes
            f"robot{robot_index}:f-r-wheel-hub", f"robot{robot_index}:f-r-wheel-1", f"robot{robot_index}:f-r-wheel-2",  # front right wheel hub and spokes
            f"robot{robot_index}:f-r-wheel",     # front right wheel
            # f"robot{robot_index}:base",          # arm base
            # f"robot{robot_index}:base_link",     # arm base link
            # "robot2:link1",         # arm joint 1
            # "robot2:link2",         # arm joint 2
            # "robot2:link3",         # arm joint 3
            # "robot2:link4",         # arm joint 4
            # "robot2:link5",         # arm joint 5
            # "robot2:link6",          # arm end effector
            # "robot2:vacuum_sphere"
        ]

        rover_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"robot{robot_index}:centroid")

        robot_body_ids = []
        for body_name in robot_bodies:
            try:
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                robot_body_ids.append(body_id)
            except:
                continue
            
        agent_robot = AgentRobot(robot_index - 2) 
            
        self.robot_infos[agent_robot] = {
            "robot_rover_id": robot_rover_id,
            "rover_joint_id": rover_joint_id,
            "robot_body_ids": robot_body_ids
        }

    def __init__(self, action_repeat=1, n_agents = 2, map_type = "a"):
        
        self.map_type = map_type
        
        self.direction_to_quat = {
                0: [0.707, 0, 0, 0.707], # UP
                2: [0, 0, 0, 1],        # LEFT
                1: [0.707, 0, 0, -0.707],  # DOWN
                3: [1, 0, 0, 0],        # RIGHT
            }
        
        self.colors = {
            'yellow': [1.0, 1.0, 0.2, 1.0],  # yellow
            'green': [0.2, 1.0, 0.2, 1.0],     # green
        }
        
        self.n_agents = n_agents
        self.agents = []
        for i in range(self.n_agents):
            self.agents.append(AgentRobot(i))
        
        self.sensor_range = 2
        
        super().__init__()
        self.action_repeat = action_repeat
        
        self.model, self.data = self._get_data_and_model()
        
        self.initial_qpos = np.copy(self.data.qpos)
        self.initial_qvel = np.copy(self.data.qvel)
        self.initial_ctrl = np.copy(self.data.ctrl)
        
        mujoco.mj_forward(self.model, self.data)
        
        self.left_object_position = [-1, -1.9, 0.28]
        self.right_object_position = [1, -1.9, 0.28]
        
        initial_index_of_robot = 2
        
        self.robot_infos = {}
        
        for i in range(initial_index_of_robot, initial_index_of_robot + len(self.agents)):
            self._initialize_robot_state(i)
        
        
        self.forbidden_geoms = [
            "wall_front", 
            "wall_back", "wall_left", "wall_right",
            "placingplace2:low_plane",
            "placingplace1:low_plane" 
        ]
        
        self.active_joint_id = None
        
        self.placingplace1_low_plane_pos = [1.9, 1, 0.23]
        self.placingplace2_low_plane_pos = [1.9, -0.5, 0.23]
        
        self.placingplace1_pos = [1.9, 1]
        self.placingplace2_pos = [1.9, -0.5]

        self.picking_positions = [
            [1, -1.9],
            [-1, -1.9],
        ]

        self.placing_positions = [
            [1.9, 1],
            [1.9, -0.5]
        ]
        # General setup end
        
        self.object_geoms = [
            "object0_geom", "object1_geom", "object2_geom", "object3_geom",
            "object4_geom", "object5_geom", "object6_geom", "object7_geom",
            "object8_geom", "object9_geom"
        ]
        
        self.object_body_names = [
            f"object{i}" for i in range(len(self.object_geoms))
        ]
        
        self.object_body_ids = []
        for body_name in self.object_body_names:
            try:
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                self.object_body_ids.append(body_id)
            except:
                continue
        
        self.object_joints = []
        for i in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if joint_name and joint_name.startswith("object") and joint_name.endswith(":joint"):
                try:
                    object_id = int(joint_name.split("object")[1].split(":")[0])
                    self.object_joints.append((object_id, i, joint_name))
                except (ValueError, IndexError):
                    continue

        self.object_joints.sort(key=lambda x: x[0])
        
        self.floor_body_name = ["floor"]   
        self.floor_body_ids = []
        for body_name in self.floor_body_name:
            try:
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                self.floor_body_ids.append(body_id)
            except:
                continue 
            
        object_ids = self._get_object_ids(self.model)
        
        self.object_joint_ids = []
        
        for i in object_ids:
            joint_name = f"object{i}:joint"
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            self.object_joint_ids.append((i, joint_id))
            
        self.shared_state = {"current_object_index": 0, "current_object_position": None, "stop": False, "stopped": True}
            
        # self.start_object_placer_thread(self.model, self.data, self.object_joint_ids, self.left_object_position, self.right_object_position, self.shared_state)
        
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self.remover_futures = []
        self.remover_shared_state = {"should_stop": False}
        # self._start_object_remover_threads(self.model, self.data, self.object_joint_ids, self.remover_shared_state)

        # obs = self._get_obs()
        
    def reset(self, info, seed=None, options=None):
        
        self.data.qpos[:] = self.initial_qpos
        self.data.qvel[:] = self.initial_qvel
        self.data.ctrl[:] = self.initial_ctrl
        
        self.next_object_index = 0
        
        self._place_agents(info)
        
        if self.shared_state["stop"] is False:
            self.shared_state["stop"] = True
            self.shared_state = {"current_object_index": 0, "current_object_position": None, "stop": False, "stopped": False}     
        
        self.remover_shared_state["should_stop"] = True
        self.remover_shared_state = {"should_stop": False}
        
        info = {}
        
        mujoco.mj_forward(self.model, self.data)
    
    def step(self, action):

        total_reward = 0
        terminated = False
        truncated = False
        final_obs = None
        final_info = {}
        
        for _ in range(self.action_repeat):
            obs, reward, terminated, truncated, info = self._original_step(action)
            total_reward += reward
            final_obs = obs
            final_info.update(info)
            
            if terminated or truncated:
                break
        
        return final_obs, total_reward, terminated, truncated, final_info
        
    def _original_step(self, actions):

        collision_with_env_occurred, collision_with_robot_occurred, previous_location_before_actions = self._process_actions(actions)

        mujoco.mj_step(self.model, self.data)
        if hasattr(self, 'viewer') and self.viewer.is_running():
            self.viewer.sync()

        if self._check_robot_robot_collision():
            collision_with_robot_occurred = True

        if self._check_robot_forbidden_collision():
            # print("Robot collision with forbidden area detected! Terminating episode.")
            collision_with_env_occurred = True
            
        return collision_with_env_occurred, collision_with_robot_occurred, previous_location_before_actions

    def _process_actions(self, actions):
        
        collision_with_env_occurred = False
        collision_with_robot_occurred = False
        
        previous_location_before_actions = []
        for agent_robot, info in self.robot_infos.items():
            rover_joint_id = info["rover_joint_id"]
            qpos_adr = self.model.jnt_qposadr[rover_joint_id]
            previous_location_before_actions.append(self.data.qpos[qpos_adr:qpos_adr+3].copy())
        
        parsed_actions = []
        for i in range(len(actions)):
            action = Action(np.argmax(actions[i]))
            agent_robot = self.agents[i]
            parsed_actions.append((agent_robot, action))
            
        max_forward_steps = 5
        for step in range(max_forward_steps):
            for agent_robot, action in parsed_actions:
                if action == Action.FORWARD:
                    self.forward_robot(0.1, agent_robot)
                elif action == Action.LEFT and step == 0: 
                    self.turn_robot("left", agent_robot)
                elif action == Action.RIGHT and step == 0:
                    self.turn_robot("right", agent_robot)

            mujoco.mj_step(self.model, self.data)
            if hasattr(self, 'viewer') and self.viewer.is_running():
                self.viewer.sync()
            
            if self._check_robot_robot_collision():
                collision_with_robot_occurred = True
                return collision_with_env_occurred, collision_with_robot_occurred, previous_location_before_actions
            
            if self._check_robot_forbidden_collision():
                collision_with_env_occurred = True
                return collision_with_env_occurred, collision_with_robot_occurred, previous_location_before_actions    

        return collision_with_env_occurred, collision_with_robot_occurred, previous_location_before_actions


    def forward_robot(self, offset_value, agent_robot):
            
        rover_joint_id = self.robot_infos[agent_robot]["rover_joint_id"]
        qpos_adr = self.model.jnt_qposadr[rover_joint_id]

        current_pos = self.data.qpos[qpos_adr:qpos_adr+3].copy()
        current_dir = self.data.qpos[qpos_adr+3:qpos_adr+7].copy()
        
        atol = 1e-3
        
        if np.allclose(current_dir, self.direction_to_quat[0], atol=atol):  # UP 
            current_pos[1] += offset_value  # y
        elif np.allclose(current_dir, self.direction_to_quat[1], atol=atol): # DOWN
            current_pos[1] -= offset_value  # y
        elif np.allclose(current_dir, self.direction_to_quat[2], atol=atol): # LEFT
            current_pos[0] -= offset_value  # x
        elif np.allclose(current_dir, self.direction_to_quat[3], atol=atol): # RIGHT
            current_pos[0] += offset_value  # x

        self.data.qpos[qpos_adr:qpos_adr+3] = current_pos
            
    def turn_robot(self, direction, agent_robot):
        rover_joint_id = self.robot_infos[agent_robot]["rover_joint_id"]
        qpos_adr = self.model.jnt_qposadr[rover_joint_id]
        current_quat = self.data.qpos[qpos_adr+3:qpos_adr+7].copy()
        new_quat = self.calculate_new_quaternion(current_quat, direction)
        target_quat = self.direction_to_quat[new_quat]
        self.data.qpos[qpos_adr+3:qpos_adr+7] = target_quat

        qvel_adr = self.model.jnt_dofadr[rover_joint_id]
        self.data.qvel[qvel_adr:qvel_adr+6] = 0.0
            
    def calculate_new_quaternion(self, current_quat, direction):
        
        tolerance = 1e-3
        
        if direction == "left":
            if np.allclose(current_quat, self.direction_to_quat[0], atol=tolerance):
                return 2
            elif np.allclose(current_quat, self.direction_to_quat[1], atol=tolerance):
                return 3
            elif np.allclose(current_quat, self.direction_to_quat[3], atol=tolerance):
                return 0
            elif np.allclose(current_quat, self.direction_to_quat[2], atol=tolerance):
                return 1
        elif direction == "right":
            if np.allclose(current_quat, self.direction_to_quat[0], atol=tolerance):
                return 3
            elif np.allclose(current_quat, self.direction_to_quat[1], atol=tolerance):
                return 2
            elif np.allclose(current_quat, self.direction_to_quat[3], atol=tolerance):
                return 1
            elif np.allclose(current_quat, self.direction_to_quat[2], atol=tolerance):
                return 0
            
        print(f"Warning: Current quaternion {current_quat} did not match any known direction within tolerance. No change in direction applied.")
        return 0

    def render(self):
        if not hasattr(self, "viewer") or self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        if self.viewer.is_running():
            self.viewer.sync()
    
    def _check_robot_robot_collision(self):
        
        robot_body_to_agent = {} 
        for agent_robot, info in self.robot_infos.items():
            for body_id in info["robot_body_ids"]:
                robot_body_to_agent[body_id] = agent_robot
        
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1_id = contact.geom1
            geom2_id = contact.geom2
            
            body1_id = self.model.geom_bodyid[geom1_id]
            body2_id = self.model.geom_bodyid[geom2_id]
            
            robot1 = robot_body_to_agent.get(body1_id)
            robot2 = robot_body_to_agent.get(body2_id)
            
            if robot1 is not None and robot2 is not None and robot1 != robot2:
                geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom1_id)
                geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom2_id)
                
                geom1_name = geom1_name if geom1_name else f"geom_{geom1_id}"
                geom2_name = geom2_name if geom2_name else f"geom_{geom2_id}"
                
                robot1_rover_id = self.robot_infos[robot1]["robot_rover_id"]
                robot2_rover_id = self.robot_infos[robot2]["robot_rover_id"]
                
                robot1_pos = self.data.xpos[robot1_rover_id]
                robot2_pos = self.data.xpos[robot2_rover_id]

                print(f"🚨 {robot1.name}-{robot2.name} COLLISION: {geom1_name} <-> {geom2_name}")
                print(f"    {robot1.name} position: x={robot1_pos[0]:.3f}, y={robot1_pos[1]:.3f}, z={robot1_pos[2]:.3f}")
                print(f"    {robot2.name} position: x={robot2_pos[0]:.3f}, y={robot2_pos[1]:.3f}, z={robot2_pos[2]:.3f}")
                print(f"    Contact position: {contact.pos}")
                
                return True
        
        return False
    
    def _check_robot_forbidden_collision(self, agent_robot = None):
        if agent_robot is not None:
            robots_to_check = [agent_robot]
        else:
            robots_to_check = list(self.robot_infos.keys())
        
        body_to_robot = {}
        for robot in robots_to_check:
            for body_id in self.robot_infos[robot]["robot_body_ids"]:
                body_to_robot[body_id] = robot
        
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1_id = contact.geom1
            geom2_id = contact.geom2
            
            body1_id = self.model.geom_bodyid[geom1_id]
            body2_id = self.model.geom_bodyid[geom2_id]
            
            geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom1_id)
            geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom2_id)
            
            if body1_id in body_to_robot and geom2_name in self.forbidden_geoms:
                robot = body_to_robot[body1_id]
                rover_id = self.robot_infos[robot]["robot_rover_id"]
                robot_pos = self.data.xpos[rover_id]
                
                print(f"🚨 {robot.name}-ENV COLLISION: {geom1_name} <-> {geom2_name}")
                print(f"    {robot.name} position: x={robot_pos[0]:.3f}, y={robot_pos[1]:.3f}, z={robot_pos[2]:.3f}")
                print(f"    Contact position: {contact.pos}")
                return True
            
            if body2_id in body_to_robot and geom1_name in self.forbidden_geoms:
                robot = body_to_robot[body2_id]
                rover_id = self.robot_infos[robot]["robot_rover_id"]
                robot_pos = self.data.xpos[rover_id]
                
                print(f"🚨 {robot.name}-ENV COLLISION: {geom1_name} <-> {geom2_name}")
                print(f"    {robot.name} position: x={robot_pos[0]:.3f}, y={robot_pos[1]:.3f}, z={robot_pos[2]:.3f}")
                print(f"    Contact position: {contact.pos}")
                return True
        
        return False

    def _get_object_ids(self, model):
        object_ids = []
        for i in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name and name.startswith("object") and name.endswith(":joint"):
                try:
                    num = int(name.split(":")[0][6:])  
                    object_ids.append(num)
                except Exception:
                    continue
        return sorted(object_ids)

    def _place_agents(self, info):
        # Initial info: {'shelf': 2, 'color': 0, 
        # 'agent_positions': [(5, 5), (0, 4)], 
        # 'agent_directions': [<Direction.LEFT: 2>, <Direction.DOWN: 1>]}

        agent_positions = info.get("agent_positions")
        agent_directions = info.get("agent_directions")
        for i, agent in enumerate(self.agents):
            agent_position = agent_positions[i]
            agent_direction = agent_directions[i]
            
            if self.map_type == "a":
                x_position_in_mujoco = -2.0 + (agent_position[0] + 1) * 0.5
                y_position_in_mujoco = 1.0 - (agent_position[1] + 1) * 0.5
            
            elif self.map_type == "b":
                x_position_in_mujoco = -2.5 + (agent_position[0] + 1) * 0.5
                y_position_in_mujoco = 2.5 - (agent_position[1] + 1) * 0.5
            
            elif self.map_type == "c" or self.map_type == "d":
                 x_position_in_mujoco = -2.0 + (agent_position[0] + 1) * 0.5
                 y_position_in_mujoco = 2.0 - (agent_position[1] + 1) * 0.5

            quat = self.direction_to_quat.get(agent_direction.value, [1, 0, 0, 0]) 
            
            
            rover_joint_id = self.robot_infos[agent]["rover_joint_id"]
            qpos_adr = self.model.jnt_qposadr[rover_joint_id]
            
            current_pos = self.data.qpos[qpos_adr:qpos_adr+3].copy()
            current_pos[0] = x_position_in_mujoco  # x
            current_pos[1] = y_position_in_mujoco  # y
            self.data.qpos[qpos_adr:qpos_adr+3] = current_pos
            self.data.qpos[qpos_adr+3:qpos_adr+7] = quat            