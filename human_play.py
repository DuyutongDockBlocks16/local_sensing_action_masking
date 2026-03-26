from argparse import ArgumentParser
import warnings

import numpy as np
import gymnasium as gym

import sys
import os
from maps import layout_7_3, layout_7_7, layout_9_9

current_dir = os.path.dirname(os.path.abspath(__file__))
rware_path = os.path.join(current_dir, 'robotic-warehouse')

if rware_path not in sys.path:
    sys.path.insert(0, rware_path)

# from rware.warehouse import Action

import inspect
from light_mappo.envs.env_core import EnvCore, Action, RewardType

import importlib
import sys


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--env",
        type=str,
        default="rware-tiny-2ag-v2",
        help="Environment to use",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=500,
        help="Maximum number of steps per episode",
    )
    parser.add_argument(
        "--display_info",
        action="store_true",
        help="Display agent info per step",
    )
    parser.add_argument(
        "--map",
        type=str,
        default="a",
        help="choice of map (a, b, c, or d)",
    )
    return parser.parse_args()


class InteractiveRWAREEnv:
    """Use this script to interactively play RWARE"""

    def __init__(
        self,
        env: str,
        max_steps,
        display_info: bool = True,
        layout: str = None,
        map_type: str = None,
    ):
        self.env = EnvCore(
            # env, 
            render_mode="human", 
            max_steps=max_steps, 
            map=layout, 
            map_type=map_type,
            action_masking=True,
            request_queue_size=1,
            n_agents=4,
            msg_bits=0,
            sensor_range=2,
            max_inactivity_steps=None,
            reward_type=RewardType.INDIVIDUAL,
        )
        self.n_agents = self.env.unwrapped.n_agents
        self.running = True
        self.current_agent_index = 0
        self.current_action = None

        self.t = 0
        self.ep_returns = np.zeros(self.n_agents)
        self.reset = False

        self.display_info = display_info

        obss, _ = self.env.reset()
        self.env.render()
        self.env.unwrapped.renderer.window.on_key_press = self._key_press

        if self.display_info:
            self._display_info(obss, [0] * self.n_agents, False)

        self._cycle()

    def _help(self):
        print("Use the up arrow key to move the current agent forward")
        print("Use the left/ right arrow keys to rotate the current agent left/ right")
        print("Press P or L to pickup/ drop shelf")
        print("Use the SPACE key to do nothing")
        print("Press TAB to change the current agent")
        print("Press R to reset the environment")
        print("Press H to show help")
        print("Press D to display agent info")
        print("Press ESC to exit")
        print()

    def _format_pos(self, pos):
        return f"row {pos[0] + 1}, col {pos[1] + 1}"

    def _get_current_agent_info(self):
        agent_carrying = self.env.unwrapped.agents[self.current_agent_index].carrying_item
        agent_x = self.env.unwrapped.agents[self.current_agent_index].x
        agent_y = self.env.unwrapped.agents[self.current_agent_index].y
        agent_str = f"Agent {self.current_agent_index + 1} (at row {agent_y + 1}, col {agent_x + 1}"
        if agent_carrying:
            agent_str += ", carrying shelf)"
        else:
            agent_str += ")"
        return agent_str
        

    def _display_info(self, obss, rews, done):
        print(f"Step {self.t}:")
        # print(f"\tSelected: {self._get_current_agent_info()}")
        print(f"\tObs: {obss[self.current_agent_index]}")
        current_obs = obss[self.current_agent_index]
        if hasattr(current_obs, 'shape'):
            print(f"\tObs shape: {current_obs.shape}")
            print(f"\tObs: {current_obs}")
        elif hasattr(current_obs, '__len__'):
            print(f"\tObs length: {len(current_obs)}")
            print(f"\tObs: {current_obs}")
        else:
            print(f"\tObs: {current_obs}")
        print(f"\tRew: {round(rews[self.current_agent_index], 3)}")
        print(f"\tDone: {done}")
        print()

    def _increment_current_agent_index(self, index: int):
        index += 1
        if index == self.n_agents:
            index = 0
        return index

    def _key_press(self, k, mod):
        from pyglet.window import key

        if k == key.LEFT:
            self.current_action = Action.LEFT
        elif k == key.RIGHT:
            self.current_action = Action.RIGHT
        elif k == key.UP:
            self.current_action = Action.FORWARD
        elif k == key.P or k == key.L:
            self.current_action = Action.TOGGLE_LOAD
            # print("Toggled load")
        elif k == key.SPACE:
            self.current_action = Action.NOOP
        elif k == key.TAB:
            self.current_action = None
            self.current_agent_index = self._increment_current_agent_index(
                self.current_agent_index
            )
            if self.display_info:
                print(f"Now selected: {self._get_current_agent_info()}")
        # elif k == key.R:
        #     self.current_action = None
        #     self.reset = True
        elif k == key.H:
            self.current_action = None
            self._help()
        elif k == key.D:
            self.current_action = None
            self.display_info = not self.display_info
        elif k == key.ESCAPE:
            self.running = False
        else:
            self.current_action = None
            warnings.warn(f"Key {k} not recognized")

    def _cycle(self):
        while self.running:
            if self.reset:
                # print("Resetting environment...")
                if self.display_info:
                    print(f"Finished episode with episodic returns: {[round(ret, 3) for ret in self.ep_returns]}")
                    print()
                obss, _ = self.env.reset()
                self.reset = False
                self.ep_returns = np.zeros(self.n_agents)
                self.t = 0

                if self.display_info:
                    self._display_info(obss, [0] * self.n_agents, False)

            if self.current_action is not None:
                actions = [Action.NOOP] * self.n_agents
                actions[self.current_agent_index] = self.current_action
                actions_onehot = []
                for act in actions:
                    onehot = [0] * 5 
                    onehot[act.value] = 1
                    actions_onehot.append(onehot)

                obss, rews, done, trunc, info = self.env.step(actions_onehot)
                action_masks = info["action_masks"]
                # print(f"Action masks: {action_masks}")
                self.ep_returns += np.array(rews)
                self.t += 1

                if self.display_info:
                    self._display_info(obss, rews, done or trunc)

                if done[0] or trunc[0]:
                    self.reset = True

                self.current_action = None
            
            self.env.render()
        self.env.close()

# python human_play.py --map "a"
if __name__ == "__main__":
    args = parse_args()
    
    if args.map == "a":
        layout = layout_7_3
    elif args.map == "b":
        layout = layout_9_9
    elif args.map == "c" or args.map == "d":
        layout = layout_7_7

    InteractiveRWAREEnv(layout=layout, map_type=args.map, env=args.env, max_steps=args.max_steps, display_info=False)
