import torch
import numpy as np
import time

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from light_mappo.envs.env_discrete import DiscreteActionEnv
from light_mappo.envs.mujoco_env.v3_collab_hybrid_robot_env import V3CollabHybridMuJoCoEnv
import argparse
import os
from light_mappo.envs.env_core import Action
from maps import layout_7_3, layout_7_7, layout_9_9


class ModelTester:
    def __init__(self, map, model_path, render=True, action_masking=False, n_agents=2, map_type="a"):

        self.render = render

        self.action_masking = action_masking

        self.env = DiscreteActionEnv(action_masking=self.action_masking, n_agents=n_agents, map=map, map_type=map_type)
        self.mujoco_env = V3CollabHybridMuJoCoEnv(n_agents=n_agents, map_type=map_type)

        self.n_agents = self.env.n_agents
        print(f"Environment created with {self.n_agents} agents")
        
        self.load_model(model_path)
        
    def load_model(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        print(f"Loading model from: {model_path}")
        
        try:
            from light_mappo.algorithms.algorithm.r_actor_critic import R_Actor
            
            class Args:
                def __init__(self):
                    self.hidden_size = 64 
                    self.activation_id = 1 
                    self.use_orthogonal = True  
                    self.use_policy_active_masks = True 
                    self.use_naive_recurrent_policy = False
                    self.use_recurrent_policy = False
                    self.recurrent_N = 1
                    self.use_influence_policy = False
                    self.influence_layer_N = 1
                    self.use_policy_vhead = False
                    self.gain = 0.01
                    
                    self.stacked_frames = 1
                    self.layer_N = 1
                    
                    self.use_ReLU = True
                    
                    self.use_feature_normalization = True 
                    self.use_popart = False
                    self.use_valuenorm = False
                    self.use_single_network = False
                    
            args = Args()
            
            self.actor = R_Actor(
                args=args,
                obs_space=self.env.observation_space[0],
                action_space=self.env.action_space[0], 
                device='cpu'
            )
            
            state_dict = torch.load(model_path, map_location='cpu')
            self.actor.load_state_dict(state_dict)
            self.actor.eval()
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")

            import traceback

            traceback.print_exc()
            raise

    def get_action(self, obs, action_masks=None, deterministic=True):

        obs_tensor = torch.FloatTensor(obs)
        
        with torch.no_grad():
            batch_size = obs_tensor.shape[0] 
            
            rnn_states = torch.zeros(batch_size, self.actor._recurrent_N, self.actor.hidden_size)
            
            masks = torch.ones(batch_size, 1)
            
            available_actions = None
            if action_masks is not None:
                available_actions = torch.FloatTensor(action_masks)
            
            actions, action_log_probs, new_rnn_states = self.actor(
                obs_tensor, 
                rnn_states, 
                masks, 
                available_actions=available_actions,
                deterministic=deterministic
            )
            
            actions_np = actions.cpu().numpy().flatten()
            
            if action_masks is not None:
                for i, (action, mask) in enumerate(zip(actions_np, action_masks)):
                    action = int(action)
                    if mask[action] == 0 or mask[action] == False:
                        print(f"  ❌ Agent {i}: select illeage action {action}, mask={mask}")
                    else:
                        pass
            
            return actions.cpu().numpy().astype(int)

    def test_episode(self, render_delay, eval_pattern, map_type, max_steps=500, deterministic=False):
        need_mujoco_env = False
        if eval_pattern == "base":
            need_mujoco_env = False
        elif eval_pattern == "cont" or eval_pattern == "lsam":
            need_mujoco_env = True
        
        obs, info = self.env.reset()
        # if len(info) > 0:
        #     print(f"Initial info: {info}")
        if need_mujoco_env:
            self.mujoco_env.reset(info)
        total_rewards = np.zeros(self.n_agents)
        step_count = 0
        
        print(f"\n=== Starting Episode ===")
        
        self.env.render()
        if need_mujoco_env:
            self.mujoco_env.render()
        # time.sleep(1000)

        collision_result = "No collision"
        position_mismatch_occurred = False
        
        sum_reward = 0
        
        action_masks = info.get("action_masks", None)
        
        for step in range(max_steps):
            if self.render:
                self.env.render()
                if need_mujoco_env:
                    self.mujoco_env.render()
                if render_delay > 0:
                    time.sleep(render_delay)

            # print(obs)
            
            actions = self.get_action(obs, action_masks=action_masks, deterministic=deterministic)
            
            if hasattr(self.env, 'discrete_action_input') and not self.env.discrete_action_input:
                actions_onehot = []
                for agent_id in range(self.n_agents):
                    action_onehot = np.zeros(self.env.signal_action_dim)
                    action_onehot[actions[agent_id]] = 1.0
                    actions_onehot.append(action_onehot)
                actions = np.array(actions_onehot)
            
            next_obs, rewards, dones, truncated, info = self.env.step(actions)
            previous_action_masks = action_masks
            action_masks = info.get("action_masks", None)
            agent_positions_before_action_from_rware = info.get("agent_positions_before_action", None)
            
            collision_with_env_occurred = False
            collision_with_robot_occurred = False
            
            if need_mujoco_env:

                collision_with_env_occurred, collision_with_robot_occurred, previous_location_before_actions_from_mujoco = self.mujoco_env._original_step(actions)

                if agent_positions_before_action_from_rware is not None and previous_location_before_actions_from_mujoco is not None:
                    for i in range(self.n_agents):
                        pos_from_rware = agent_positions_before_action_from_rware[i]
                        pos_from_mujoco = previous_location_before_actions_from_mujoco[i].tolist()[0:2] # only x,y
                            
                        if map_type == "a":
                            x_position_in_mujoco_calculation = -2.0 + (pos_from_rware[0] + 1) * 0.5
                            y_position_in_mujoco_calculation = 1.0 - (pos_from_rware[1] + 1) * 0.5
                        
                        elif map_type == "b":
                            x_position_in_mujoco_calculation = -2.5 + (pos_from_rware[0] + 1) * 0.5
                            y_position_in_mujoco_calculation = 2.5 - (pos_from_rware[1] + 1) * 0.5
                        
                        elif map_type == "c" or map_type == "d":
                            x_position_in_mujoco_calculation = -2.0 + (pos_from_rware[0] + 1) * 0.5
                            y_position_in_mujoco_calculation = 2.0 - (pos_from_rware[1] + 1) * 0.5

                        x_position_in_mujoco_real = round(pos_from_mujoco[0], 3)
                        y_position_in_mujoco_real = round(pos_from_mujoco[1], 3)

                        if abs(x_position_in_mujoco_calculation - x_position_in_mujoco_real) > 0.01 or abs(y_position_in_mujoco_calculation - y_position_in_mujoco_real) > 0.01:
                            print(f"❌ Agent {i+2} position mismatch!")
                            print(f"  From RWARE: {pos_from_rware}")
                            print(f"  Calculated Mujoco: ({x_position_in_mujoco_calculation:.3f}, {y_position_in_mujoco_calculation:.3f})")
                            print(f"  Actual Mujoco: ({x_position_in_mujoco_real}, {y_position_in_mujoco_real})")
                            position_mismatch_occurred = True
                            break
                
                if collision_with_env_occurred or collision_with_robot_occurred:
                    print("previous_action_masks:", previous_action_masks)
                    print(f"\n[Step {step}] action: {actions}")
                    # print(f"- obs: {obs}")
                    print(f"- agent_positions_before_action_from_rware: {agent_positions_before_action_from_rware}")
                    print(f"- previous_location_before_actions_from_mujoco: {previous_location_before_actions_from_mujoco}")
                    break

                # info: {'shelf': 1, 'color': 1} shelf 1: left 2: right, color 0: yellow 1: green. 0: yellow, 1: green

            total_rewards += rewards.squeeze()
            step_count += 1
            
            # print(f"Step {step}: Rewards = {rewards.squeeze()}, Done = {dones}")
            
            obs = next_obs
            
            if np.any(dones):
                print(f"Episode finished at step {step}")
                break
            
            sum_reward = np.sum(total_rewards)
            
            if sum_reward > 19: # max 20 item to test
                # print(f"Episode finished at step {step} with max reward")
                break
        
        print(f"=== Episode Summary ===")
        print(f"Total Steps: {step_count}")
        print(f"Total Rewards per Agent: {total_rewards}")
        print(f"Sum Reward: {sum_reward:.3f}")
        if collision_with_env_occurred:
            collision_result = "Collision with environment"
        elif collision_with_robot_occurred:
            collision_result = "Collision with robot"
        print(f"Collision Result: {collision_result}")
        
        # if collision_with_env_occurred or collision_with_robot_occurred
        
        return sum_reward, step_count, collision_result, position_mismatch_occurred

    def test_multiple_episodes(self, render_delay, eval_pattern, map_type, num_episodes=10, max_steps=400):
        all_rewards = []
        all_steps = []
        collisions_with_robot = 0
        collisions_with_env = 0
            
        for episode in range(num_episodes):
            print(f"\n{'='*50}")
            print(f"Testing Episode {episode + 1}/{num_episodes}")
            print(f"{'='*50}")

            rewards, steps, collisions_result, position_mismatch_occurred = self.test_episode(render_delay, eval_pattern, map_type, max_steps)
            if position_mismatch_occurred:
                print("Position mismatch occurred in this episode. Skipping result.")
                continue
            all_rewards.append(rewards)
            print(f"Length of all_rewards: {len(all_rewards)}")
            all_steps.append(steps)
            if "Collision with robot" in collisions_result:
                collisions_with_robot += 1
            if "Collision with environment" in collisions_result:
                collisions_with_env += 1

            if len(all_steps) == 100:
                print("100 episodes without position_mismatch_occurred, stopping early to avoid long runtime.")
                break

            # time.sleep(0.1)

        all_rewards = np.array(all_rewards)
        all_steps = np.array(all_steps)

        print(f"\n{'='*60}")
        print(f"FINAL RESULTS ({len(all_steps)} episodes)")
        print(f"{'='*60}")
        print(f"Average Steps per Episode: {np.mean(all_steps):.2f} ± {np.std(all_steps):.2f}")
        print(f"Total Average Reward: {np.mean(all_rewards):.3f} ± {np.std(all_rewards):.3f}")
        print(f"Collision Rate (TOTAL): {(collisions_with_env + collisions_with_robot) / len(all_steps) * 100:.1f}%")
        print(f"Collision Rate (ENV): {collisions_with_env / len(all_steps) * 100:.1f}%")
        print(f"Collision Rate (Robot): {collisions_with_robot / len(all_steps) * 100:.1f}%")

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model_path", type=str, required=True,
    #                    help="Path to the trained model file")
    parser.add_argument("--episodes", type=int, default=10,
                       help="Number of episodes to test")
    parser.add_argument("--max_steps", type=int, default=500,
                       help="Maximum steps per episode")
    parser.add_argument("--no_render", action="store_true",
                       help="Disable rendering")
    # parser.add_argument("--stochastic", action="store_true",
    #                    help="Use stochastic policy instead of deterministic")
    parser.add_argument("--map", type=str, default="a")
    parser.add_argument("--eval_pattern", type=str, default="base")
    parser.add_argument("--render_speed", type=str, default="slow")
    
    args = parser.parse_args()

    if args.map == "a":
        map = layout_7_3
        agents_num = 3
    elif args.map == "b":
        map = layout_9_9
        agents_num = 6
    elif args.map == "c":
        map = layout_7_7
        agents_num = 2
    elif args.map == "d":
        map = layout_7_7
        agents_num = 4
        
    if args.eval_pattern == "base":
        action_masking = False
    elif args.eval_pattern == "cont":
        action_masking = False
    elif args.eval_pattern == "lsam":
        action_masking = True
    eval_pattern = args.eval_pattern
    
    if args.render_speed == "slow":
        render_delay = 0.5
    elif args.render_speed == "fast":
        render_delay = 0.0
    # print(f"Using render_delay: {render_delay} seconds")

    if args.map == "a": 
        if args.eval_pattern == "base" or args.eval_pattern == "cont":
            model_path = "../light_mappo/results/MyEnv/MyEnv/mappo/check/run31/models/actor.pt"
        elif args.eval_pattern == "lsam":
            model_path = "../light_mappo/results/MyEnv/MyEnv/mappo/check/run32/models/actor.pt"
    elif args.map == "b":
        if args.eval_pattern == "base" or args.eval_pattern == "cont":
            model_path = "../light_mappo/results/MyEnv/MyEnv/mappo/check/run38/models/actor.pt"
        elif args.eval_pattern == "lsam":
            model_path = "../light_mappo/results/MyEnv/MyEnv/mappo/check/run39/models/actor.pt"
    elif args.map == "c":
        if args.eval_pattern == "base" or args.eval_pattern == "cont":
            model_path = "../light_mappo/results/MyEnv/MyEnv/mappo/check/run1/models/actor.pt"
        elif args.eval_pattern == "lsam":
            model_path = "../light_mappo/results/MyEnv/MyEnv/mappo/check/run19/models/actor.pt"
    elif args.map == "d":
        if args.eval_pattern == "base" or args.eval_pattern == "cont":
            model_path = "../light_mappo/results/MyEnv/MyEnv/mappo/check/run27/models/actor.pt"
        elif args.eval_pattern == "lsam":
            model_path = "../light_mappo/results/MyEnv/MyEnv/mappo/check/run26/models/actor.pt"

    tester = ModelTester(
        map=map,
        map_type=args.map,
        model_path=model_path,
        render=not args.no_render,
        action_masking=action_masking,
        n_agents=agents_num,
    )

    tester.test_multiple_episodes(
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        render_delay = render_delay,
        eval_pattern = eval_pattern,
        map_type = args.map
    )

# python eval_model.py --map "a" --eval_pattern "base" --episodes 100 --render_speed "slow"
if __name__ == "__main__":
    main()