from collections import OrderedDict
from enum import Enum
from typing import List, Tuple, Optional, Dict

import gymnasium as gym
from gymnasium.utils import seeding
import networkx as nx
import numpy as np


_COLLISION_LAYERS = 3
_LAYER_AGENTS = 0
_LAYER_SHELFS = 1
_LAYER_ITEMS = 2


class _VectorWriter:
    def __init__(self, size: int):
        self.vector = np.zeros(size, dtype=np.float32)
        self.idx = 0

    def write(self, data):
        data_size = len(data)
        self.vector[self.idx : self.idx + data_size] = data
        self.idx += data_size

    def skip(self, bits):
        self.idx += bits


class Action(Enum):
    NOOP = 0
    FORWARD = 1
    LEFT = 2
    RIGHT = 3
    TOGGLE_LOAD = 4


class Direction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class RewardType(Enum):
    GLOBAL = 0
    INDIVIDUAL = 1
    TWO_STAGE = 2


class ObservationType(Enum):
    DICT = 0
    FLATTENED = 1
    IMAGE = 2
    IMAGE_DICT = 3
    
class ItemType(Enum):
    TYPE_1 = 0
    TYPE_2 = 1


class ImageLayer(Enum):
    """
    Input layers of image-style observations
    """

    SHELVES = 0  # binary layer indicating shelves (also indicates carried shelves)
    REQUESTS = 1  # binary layer indicating requested shelves
    AGENTS = 2  # binary layer indicating agents in the environment (no way to distinguish agents)
    AGENT_DIRECTION = 3  # layer indicating agent directions as int (see Direction enum + 1 for values)
    AGENT_LOAD = 4  # binary layer indicating agents with load
    GOALS = 5  # binary layer indicating goal/ delivery locations
    ACCESSIBLE = 6  # binary layer indicating accessible cells (all but occupied cells/ out of map)


class Entity:
    def __init__(self, id_: int, x: int, y: int):
        self.id = id_
        self.prev_x = None
        self.prev_y = None
        self.x = x
        self.y = y


class Agent(Entity):
    counter = 0

    def __init__(self, x: int, y: int, dir_: Direction, msg_bits: int):
        Agent.counter += 1
        super().__init__(Agent.counter, x, y)
        self.dir = dir_
        self.message = np.zeros(msg_bits)
        self.req_action: Optional[Action] = None
        self.carrying_item: Optional[Item] = None
        self.canceled_action = None
        self.has_delivered = False

    # @property
    # def collision_layers(self):
    #     if self.loaded:
    #         return (_LAYER_AGENTS, _LAYER_SHELFS)
    #     else:
    #         return (_LAYER_AGENTS,)
    
    @property
    def collision_layers(self):
        return (_LAYER_AGENTS,)

    def req_location(self, grid_size) -> Tuple[int, int]:
        if self.req_action != Action.FORWARD:
            return self.x, self.y
        elif self.dir == Direction.UP:
            return self.x, max(0, self.y - 1)
        elif self.dir == Direction.DOWN:
            return self.x, min(grid_size[0] - 1, self.y + 1)
        elif self.dir == Direction.LEFT:
            return max(0, self.x - 1), self.y
        elif self.dir == Direction.RIGHT:
            return min(grid_size[1] - 1, self.x + 1), self.y

        raise ValueError(
            f"Direction is {self.dir}. Should be one of {[v for v in Direction]}"
        )

    def req_direction(self) -> Direction:
        wraplist = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
        if self.req_action == Action.RIGHT:
            return wraplist[(wraplist.index(self.dir) + 1) % len(wraplist)]
        elif self.req_action == Action.LEFT:
            return wraplist[(wraplist.index(self.dir) - 1) % len(wraplist)]
        else:
            return self.dir
        
    @property
    def loaded(self):
        return self.carrying_item is not None
        
class Item(Entity):
    counter = 0
    item_id = 1

    def __init__(self, x, y, shelf_id, item_type):
        Item.counter += 1
        super().__init__(Item.counter, x, y)
        self.shelf_id = shelf_id
        self.item_type = item_type

    @property
    def collision_layers(self):
        return () 


class Shelf(Entity):
    counter = 0

    def __init__(self, x, y, occupied_by_item: Optional[Item] = None):
        Shelf.counter += 1
        super().__init__(Shelf.counter, x, y)
        self.occupied_by_item = occupied_by_item

    @property
    def collision_layers(self):
        return (_LAYER_SHELFS,)
    
class Goal():
    def __init__(self, location: Tuple[int, int], accepted_item_type: ItemType):
        self.location = location
        self.accepted_item_type = accepted_item_type

class EnvCore(gym.Env):
    # metadata = {
    #     "render_modes": ["human", "rgb_array"],
    #     "render_fps": 10,
    # }
    

    def __init__(
        self,
        n_agents: int,
        msg_bits: int,
        sensor_range: int,
        request_queue_size: int,
        max_inactivity_steps: Optional[int],
        max_steps: Optional[int],
        reward_type: RewardType,
        layout: Optional[str] = None,
        observation_type: ObservationType = ObservationType.FLATTENED,
        normalised_coordinates: bool = False,
        render_mode: Optional[str] = None,     
        action_masking: bool = False,
    ):
        self.action_masking = action_masking
        if self.action_masking:
            self.obs_dim = 210 + n_agents
        else:
            self.obs_dim = 210
        self.action_dim = 5 
        
        self.goals: List[Tuple[int, int]] = []
        
        self._make_layout_from_str(layout)
        self.n_agents = n_agents 
        self.msg_bits = msg_bits
        self.sensor_range = sensor_range
        self.max_inactivity_steps: Optional[int] = max_inactivity_steps
        self.reward_type = reward_type
        self.reward_range = (0, 1)

        self._cur_inactive_steps = None
        self._cur_steps = 0
        self.max_steps = max_steps

        self.normalised_coordinates = normalised_coordinates

        sa_action_space = [len(Action), *msg_bits * (2,)]
        if len(sa_action_space) == 1:
            sa_action_space = gym.spaces.Discrete(sa_action_space[0])
        else:
            sa_action_space = gym.spaces.MultiDiscrete(sa_action_space)
        self.action_space = gym.spaces.Tuple(tuple(n_agents * [sa_action_space]))

        self.request_queue_size = request_queue_size
        self.request_queue = []

        self.agents: List[Agent] = []
        
        self.fast_obs = True
        self.image_obs = None
        self.image_dict_obs = None
        
        self.observation_space = self._use_fast_obs()

        self.renderer = None
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        """
        # self.n_agents设定为2个智能体时，返回值为一个list，每个list里面为一个shape = (self.obs_dim, )的观测数据
        # When self.n_agents is set to 2 agents, the return value is a list, each list contains a shape = (self.obs_dim, ) observation data
        """
        # sub_agent_obs = []
        # for i in range(self.n_agents):
        #     sub_obs = np.random.random(size=(14,))
        #     sub_agent_obs.append(sub_obs)
        # return sub_agent_obs
    
        if seed is not None:
            # setting seed
            super().reset(seed=seed, options=options)

        Shelf.counter = 0
        Agent.counter = 0
        Item.counter = 0
        self._cur_inactive_steps = 0
        self._cur_steps = 0

        # n_xshelf = (self.grid_size[1] - 1) // 3
        # n_yshelf = (self.grid_size[0] - 2) // 9

        # make the shelfs
        self.shelfs = [
            Shelf(x, y)
            for y, x in zip(
                np.indices(self.grid_size)[0].reshape(-1),
                np.indices(self.grid_size)[1].reshape(-1),
            )
            if not self._is_highway(x, y)
        ]
        
        if self.shelfs:
            selected_shelf = self.np_random.choice(self.shelfs)
            selected_type = self.np_random.choice(list(ItemType))
            new_item = Item(selected_shelf.x, selected_shelf.y, selected_shelf.id, selected_type)
            self.items = [new_item]
            selected_shelf.occupied_by_item = new_item
            
            info = {
                "shelf": selected_shelf.id,
                "color": new_item.item_type.value,
            }
        else:
            self.items = []

        # spawn agents at random locations
        agent_locs = self.np_random.choice(
            np.arange(self.grid_size[0] * self.grid_size[1]),
            size=self.n_agents,
            replace=False,
        )
        agent_locs = np.unravel_index(agent_locs, self.grid_size)
        # and direction
        agent_dirs = self.np_random.choice([d for d in Direction], size=self.n_agents)
        self.agents = [
            Agent(x, y, dir_, self.msg_bits)
            for y, x, dir_ in zip(*agent_locs, agent_dirs)
        ]
            
        # append to info
        info["agent_positions"] = [(agent.x, agent.y) for agent in self.agents]
        info["agent_directions"] = [agent.dir for agent in self.agents]

        self._recalc_grid()

        self.request_queue = self.items.copy()
        
        sub_agent_obs = []
        sub_agent_action_masks = []
        for agent in self.agents:
            sub_obs = self._make_obs(agent)
            sub_agent_obs.append(sub_obs)
            sub_agent_action_mask = self._get_action_mask(sub_obs)
            sub_agent_action_masks.append(sub_agent_action_mask)

        if self.action_masking:
            info["action_masks"] = sub_agent_action_masks

        return sub_agent_obs, info
        
        # return tuple([self._make_obs(agent) for agent in self.agents]), self._get_info()

    def step(self, actions):
        """
        # self.n_agents设定为2个智能体时，actions的输入为一个2纬的list，每个list里面为一个shape = (self.action_dim, )的动作数据
        # 默认参数情况下，输入为一个list，里面含有两个元素，因为动作维度为5，所里每个元素shape = (5, )
        # When self.n_agents is set to 2 agents, the input of actions is a 2-dimensional list, each list contains a shape = (self.action_dim, ) action data
        # The default parameter situation is to input a list with two elements, because the action dimension is 5, so each element shape = (5, )
        """
        self._resolve_movement_conflicts(actions)

        rewards = np.zeros(self.n_agents)
        
        info = self._get_info()
        
        item_delivered = False
        for agent in self.agents:
            agent.prev_x, agent.prev_y = agent.x, agent.y

            if agent.req_action == Action.FORWARD:
                # rewards[agent_id - 1] += 1
                agent.x, agent.y = agent.req_location(self.grid_size)
                if agent.carrying_item:
                    agent.carrying_item.x, agent.carrying_item.y = agent.x, agent.y
            elif agent.req_action in [Action.LEFT, Action.RIGHT]:
                agent.dir = agent.req_direction()
            elif agent.req_action == Action.TOGGLE_LOAD and not agent.carrying_item and agent.dir == Direction.DOWN:
                item_id = self.grid[_LAYER_ITEMS, agent.y, agent.x]
                # print(item_id)
                if item_id:
                    for item in self.items:
                        if item.id == item_id:
                            # print(f"Item with ID {item_id} found in items list!")
                            item_to_pickup = item
                            break
                    
                    if item_to_pickup:
                        # print(f"Agent {agent.id} picked up item {item_id} from shelf {item_to_pickup.shelf_id}")
                        # print(len(self.items))
                        agent.carrying_item = item_to_pickup

                        current_shelf = self.shelfs[item_to_pickup.shelf_id - 1]
                        current_shelf.occupied_by_item = None

                        # generate new request
                        new_shelf = self.np_random.choice(self.shelfs)
                        new_type = self.np_random.choice(list(ItemType))
                        new_item = Item(new_shelf.x, new_shelf.y, new_shelf.id, new_type)
                        self.items.append(new_item)
                        new_shelf.occupied_by_item = new_item
                        self.request_queue.append(new_item)
                        # add information of the new request to info
                        info = {
                            "shelf": new_shelf.id,
                            "color": new_item.item_type.value,
                        }

                    else:
                        print(f"Warning: Item with ID {item_id} not found in items list!")
            elif agent.req_action == Action.TOGGLE_LOAD and agent.carrying_item and agent.dir == Direction.RIGHT:
                
                item_id = self.grid[_LAYER_ITEMS, agent.y, agent.x]
                for item in self.items:
                    if item.id == item_id:
                        break
                    
                if item.item_type == ItemType.TYPE_1 and (agent.x, agent.y) in [goal.location for goal in self.goals if goal.accepted_item_type == ItemType.TYPE_1]:
                    item_delivered = True
                    self.request_queue.remove(item)
                    self.items.remove(item)
                    
                    agent.carrying_item = None
                    
                    # print(f"Agent {agent.id} delivered item {item.id} to goal at location {(agent.x, agent.y)}")
                    
                    # also reward the agents
                    if self.reward_type.value == RewardType.GLOBAL.value:
                        rewards += 1
                    elif self.reward_type.value == RewardType.INDIVIDUAL.value:
                        rewards[agent.id-1] += 1
                    elif self.reward_type.value == RewardType.TWO_STAGE.value:
                        rewards[agent.id-1] += 1
                elif item.item_type == ItemType.TYPE_2 and (agent.x, agent.y) in [goal.location for goal in self.goals if goal.accepted_item_type == ItemType.TYPE_2]:
                    item_delivered = True
                    self.request_queue.remove(item)
                    self.items.remove(item)
                    
                    agent.carrying_item = None
                    
                    # print(f"Agent {agent.id} delivered item {item.id} to goal at location {(agent.x, agent.y)}")
                    
                    # also reward the agents
                    if self.reward_type.value == RewardType.GLOBAL.value:
                        rewards += 1
                    elif self.reward_type.value == RewardType.INDIVIDUAL.value:
                        rewards[agent.id-1] += 1
                    elif self.reward_type.value == RewardType.TWO_STAGE.value:
                        self.agents[agent.id-1].has_delivered = True
                        rewards[agent.id-1] += 0.5

        self._recalc_grid()

        if item_delivered:
            self._cur_inactive_steps = 0
        else:
            self._cur_inactive_steps += 1
        self._cur_steps += 1

        if (
            self.max_inactivity_steps
            and self._cur_inactive_steps >= self.max_inactivity_steps
        ) or (self.max_steps and self._cur_steps >= self.max_steps):
            done = True
        else:
            done = False
        truncated = False

        sub_agent_obs = []
        sub_agent_action_masks = []
        for agent in self.agents:
            sub_obs = self._make_obs(agent)
            sub_agent_obs.append(sub_obs)
            sub_agent_action_mask = self._get_action_mask(sub_obs)
            sub_agent_action_masks.append(sub_agent_action_mask)
        
        if self.action_masking:
            info["action_masks"] = sub_agent_action_masks

        # print(list(rewards))
        return [sub_agent_obs, list(rewards), [done, done], [truncated, truncated], info]

    def render(self):
        if not self.renderer:
            from .rendering import Viewer
            self.renderer = Viewer(self.grid_size)

        return self.renderer.render(self, return_rgb_array=self.render_mode == "rgb_array")
    
    def _resolve_movement_conflicts(self, actions: List[np.ndarray]) -> None:
        """
        Resolves movement conflicts between agents using a graph-based approach.

        This function takes the raw actions from the policy, builds a directed graph
        of movement intentions, detects cycles and paths to identify valid collective
        movements, and finally updates each agent's 'req_action' to either their
        intended action or Action.NOOP if a conflict forces them to wait.

        Args:
            actions: A list of action arrays, one for each agent.
        """
        commited_agents = set()
        G = nx.DiGraph()

        # 1. Build graph of movement intentions
        for i, agent in enumerate(self.agents):
            agent.req_action = Action(np.argmax(actions[i]))
            start = agent.x, agent.y
            target = agent.req_location(self.grid_size)

            # Special case: agent with shelf can't move to a cell with another shelf
            if (
                agent.carrying_item
                and start != target
                and self.grid[_LAYER_SHELFS, target[1], target[0]]
                and not (
                    self.grid[_LAYER_AGENTS, target[1], target[0]]
                    and self.agents[
                        self.grid[_LAYER_AGENTS, target[1], target[0]] - 1
                    ].carrying_item
                )
            ):
                agent.req_action = Action.NOOP
                G.add_edge(start, start)
            else:
                G.add_edge(start, target)

        # 2. Analyze graph components to find valid movements
        wcomps = [G.subgraph(c).copy() for c in nx.weakly_connected_components(G)]

        for comp in wcomps:
            try:
                # Commit agents in cycles (collective rotational movement)
                cycle = nx.algorithms.find_cycle(comp)
                if len(cycle) == 2:  # Impossible head-on swap
                    continue
                for edge in cycle:
                    start_node = edge[0]
                    agent_id = self.grid[_LAYER_AGENTS, start_node[1], start_node[0]]
                    if agent_id > 0:
                        commited_agents.add(agent_id)
            except nx.NetworkXNoCycle:
                # Commit agents on the longest path in DAGs (chain movement)
                longest_path = nx.algorithms.dag_longest_path(comp)
                for x, y in longest_path:
                    agent_id = self.grid[_LAYER_AGENTS, y, x]
                    if agent_id:
                        commited_agents.add(agent_id)

        # 3. Finalize committed and failed agents
        commited_agent_objects = {self.agents[id_ - 1] for id_ in commited_agents}
        failed_agents = set(self.agents) - commited_agent_objects

        # Force failed agents to do nothing to prevent collisions
        for agent in failed_agents:
            # Only agents wanting to move forward can "fail" in this way.
            # Rotations or load/unload actions don't cause this type of conflict.
            if agent.req_action == Action.FORWARD:
                agent.req_action = Action.NOOP

    def _make_layout_from_str(self, layout):
        layout = layout.strip()
        layout = layout.replace(" ", "")
        grid_height = layout.count("\n") + 1
        lines = layout.split("\n")
        grid_width = len(lines[0])
        for line in lines:
            assert len(line) == grid_width, "Layout must be rectangular"

        self.grid_size = (grid_height, grid_width)
        self.grid = np.zeros((_COLLISION_LAYERS, *self.grid_size), dtype=np.int32)
        self.highways = np.zeros(self.grid_size, dtype=np.uint8)

        for y, line in enumerate(lines):
            for x, char in enumerate(line):
                assert char.lower() in "x.12"
                if char.lower() == "1":
                    goal = Goal((x, y), accepted_item_type=ItemType.TYPE_1)
                    self.goals.append(goal)
                    self.highways[y, x] = 1
                if char.lower() == "2":
                    goal = Goal((x, y), accepted_item_type=ItemType.TYPE_2)
                    self.goals.append(goal)
                    self.highways[y, x] = 1
                elif char.lower() == ".":
                    self.highways[y, x] = 1

        assert len(self.goals) >= 1, "At least one goal is required"
        
    def _get_info(self):
        return {}
    
    def _is_highway(self, x: int, y: int) -> bool:
        return self.highways[y, x]
    
    def _recalc_grid(self):
        self.grid[:] = 0
        for s in self.shelfs:
            self.grid[_LAYER_SHELFS, s.y, s.x] = s.id
            
        for item in self.items:
            self.grid[_LAYER_ITEMS, item.y, item.x] = item.id

        for a in self.agents:
            self.grid[_LAYER_AGENTS, a.y, a.x] = a.id

    def _make_obs(self, agent):
        if self.action_masking:
            # obs based on sensor and masking
            return self._get_default_obs_for_masking(agent)
        
        # original obs
        return self._get_default_obs(agent)

    def _get_default_obs(self, agent):
        
        sensor_range = self.sensor_range

        min_x = agent.x - sensor_range
        max_x = agent.x + sensor_range + 1

        min_y = agent.y - sensor_range
        max_y = agent.y + sensor_range + 1

        # range 2 sensors
        if (
                (min_x < 0)
                or (min_y < 0)
                or (max_x > self.grid_size[1])
                or (max_y > self.grid_size[0])
            ):
                padded_agents = np.pad(
                    self.grid[_LAYER_AGENTS], sensor_range, mode="constant"
                )
                padded_shelfs = np.pad(
                    self.grid[_LAYER_SHELFS], sensor_range, mode="constant"
                )
                # + self.sensor_range due to padding
                min_x += sensor_range
                max_x += sensor_range
                min_y += sensor_range
                max_y += sensor_range

        else:
            padded_agents = self.grid[_LAYER_AGENTS]
            padded_shelfs = self.grid[_LAYER_SHELFS]

        agents = padded_agents[min_y:max_y, min_x:max_x].reshape(-1)
        shelfs = padded_shelfs[min_y:max_y, min_x:max_x].reshape(-1)
        
        # write flattened observations
        flatdim = gym.spaces.flatdim(self.observation_space[agent.id - 1])
        obs = _VectorWriter(flatdim)
        
        if self.normalised_coordinates:
            agent_x = agent.x / (self.grid_size[1] - 1)
            agent_y = agent.y / (self.grid_size[0] - 1)
        else:
            agent_x = agent.x
            agent_y = agent.y

        obs.write([agent_x, agent_y, int(agent.carrying_item is not None)])
        if agent.carrying_item is not None:
            if agent.carrying_item.item_type == ItemType.TYPE_1:
                obs.write([1.0, 0.0])  # carrying item type 1
            elif agent.carrying_item.item_type == ItemType.TYPE_2:
                obs.write([0.0, 1.0])  # carrying item type 2
        else:
            obs.write([0.0, 0.0])  # not carrying any item
        
        direction = np.zeros(4)
        direction[agent.dir.value] = 1.0
        obs.write(direction)
        obs.write([int(self._is_highway(agent.x, agent.y))])

        # 'has_agent': MultiBinary(1),
        # 'direction': Discrete(4),
        # 'local_message': MultiBinary(2)
        # 'has_shelf': MultiBinary(1),
        # 'shelf_requested': MultiBinary(1),
        
        for i, (id_agent, id_shelf) in enumerate(zip(agents, shelfs)):
            if id_agent == 0:
                # no agent, direction, or message
                obs.write([0.0])  # no agent present
                obs.write([1.0, 0.0, 0.0, 0.0])  # agent direction
                obs.skip(self.msg_bits)  # agent message
            else:
                obs.write([1.0])  # agent present
                direction = np.zeros(4)
                direction[self.agents[id_agent - 1].dir.value] = 1.0
                obs.write(direction)  # agent direction as onehot
                if self.msg_bits > 0:
                    obs.write(self.agents[id_agent - 1].message)  # agent message
            if id_shelf == 0:
                obs.write([0.0, 0.0, 0.0])  # no shelf
            elif id_shelf and not self.shelfs[id_shelf - 1].occupied_by_item:  # has an empty shelf (no item on it)
                obs.write([1.0, 0.0, 0.0])  # shelf present, no item
            elif id_shelf and self.shelfs[id_shelf - 1].occupied_by_item:  # has a shelf with an item on it
                shelf = self.shelfs[id_shelf - 1]
                item = shelf.occupied_by_item
                if item.item_type == ItemType.TYPE_1:
                    obs.write([1.0, 1.0, 0.0])  # shelf present, item type 1
                elif item.item_type == ItemType.TYPE_2:
                    obs.write([1.0, 0.0, 1.0])  # shelf present, item type 2

        return obs.vector

    def _get_default_obs_for_masking(self, agent):
        sensor_range = 1

        min_x = agent.x - sensor_range
        max_x = agent.x + sensor_range + 1

        min_y = agent.y - sensor_range
        max_y = agent.y + sensor_range + 1

        # range 1 sensors
        if (
                (min_x < 0)
                or (min_y < 0)
                or (max_x > self.grid_size[1])
                or (max_y > self.grid_size[0])
            ):
                padded_agents = np.pad(
                    self.grid[_LAYER_AGENTS], sensor_range, mode="constant"
                )
                padded_shelfs = np.pad(
                    self.grid[_LAYER_SHELFS], sensor_range, mode="constant"
                )
                padded_wall = np.pad(
                    np.ones_like(self.grid[_LAYER_AGENTS]), sensor_range, mode="constant", constant_values=1
                )
                
                # + self.sensor_range due to padding
                min_x += sensor_range
                max_x += sensor_range
                min_y += sensor_range
                max_y += sensor_range

        else:
            padded_agents = self.grid[_LAYER_AGENTS]
            padded_shelfs = self.grid[_LAYER_SHELFS]
            walls = np.ones_like(self.grid[_LAYER_AGENTS])

        agents = padded_agents[min_y:max_y, min_x:max_x].reshape(-1)
        shelfs = padded_shelfs[min_y:max_y, min_x:max_x].reshape(-1)
        walls = padded_wall[min_y:max_y, min_x:max_x].reshape(-1)
        
        # write flattened observations
        flatdim = gym.spaces.flatdim(self.observation_space[agent.id - 1])
        obs = _VectorWriter(flatdim)
            
        if self.action_masking:
            # agent id onehot
            agent_id_onehot = np.zeros(self.n_agents)
            agent_id_onehot[agent.id - 1] = 1.0 
            obs.write(agent_id_onehot)

        # write flattened observations
        flatdim = gym.spaces.flatdim(self.observation_space[agent.id - 1])
        obs = _VectorWriter(flatdim)

        for i, (id_agent, id_shelf, wall_flag) in enumerate(zip(agents, shelfs, walls)):
            if id_agent == 0:
                # no agent, direction, or message
                obs.write([0.0])  # no agent present
                obs.write([1.0, 0.0, 0.0, 0.0])  # agent direction
                obs.skip(self.msg_bits)  # agent message
            else:
                obs.write([1.0])  # agent present
                direction = np.zeros(4)
                direction[self.agents[id_agent - 1].dir.value] = 1.0
                obs.write(direction)  # agent direction as onehot
                if self.msg_bits > 0:
                    obs.write(self.agents[id_agent - 1].message)  # agent message
            if id_shelf == 0:
                obs.write([0.0, 0.0, 0.0])  # no shelf
            elif id_shelf and not self.shelfs[id_shelf - 1].occupied_by_item:  # has an empty shelf (no item on it)
                obs.write([1.0, 0.0, 0.0])  # shelf present, no item
            elif id_shelf and self.shelfs[id_shelf - 1].occupied_by_item:  # has a shelf with an item on it
                shelf = self.shelfs[id_shelf - 1]
                item = shelf.occupied_by_item
                if item.item_type == ItemType.TYPE_1:
                    obs.write([1.0, 1.0, 0.0])  # shelf present, item type 1
                elif item.item_type == ItemType.TYPE_2:
                    obs.write([1.0, 0.0, 1.0])  # shelf present, item type 2
            # if wall_flag == 1:

        # return obs.vector
    
    def _use_fast_obs(self):

        self.fast_obs = True
        
        flatdim = self.obs_dim
        
        ma_spaces = []
        for sa_obs in range(self.n_agents):
            ma_spaces.append(
                gym.spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(flatdim,),
                    dtype=np.float32,
                )
            )

        return gym.spaces.Tuple(tuple(ma_spaces))
    
    def _get_action_mask(self, sub_obs) -> List[bool]:
        """
        Generates a boolean mask for the legal actions of a given agent.
        [NOOP, FORWARD, LEFT, RIGHT, TOGGLE_LOAD]
        """
        
        mask = [True] * 5
        
        idx = 0
        
        idx += self.n_agents
        
        agent_x = sub_obs[idx]; idx += 1
        agent_y = sub_obs[idx]; idx += 1
        carrying_item = sub_obs[idx]; idx += 1
        carrying_type_1 = sub_obs[idx]; idx += 1
        carrying_type_2 = sub_obs[idx]; idx += 1
        
        direction_onehot = sub_obs[idx:idx+4]; idx += 4
        direction_names = ["UP", "DOWN", "LEFT", "RIGHT"]
        my_direction = direction_names[int(np.argmax(direction_onehot))]
        
        on_highway = sub_obs[idx]; idx += 1 
        
        sensor_info_per_loc = 1 + 4 + self.msg_bits + 3
        n_sensors = (1 + 2 * self.sensor_range) ** 2
        sensor_grid_size = int(np.sqrt(n_sensors))
        center = sensor_grid_size // 2
        
        if agent_x == 0 and my_direction == "LEFT":
            mask[Action.FORWARD.value] = False
        elif agent_x == self.grid_size[0] - 1 and my_direction == "RIGHT":
            mask[Action.FORWARD.value] = False
        elif agent_y == 0 and my_direction == "UP":
            mask[Action.FORWARD.value] = False
        elif agent_y == self.grid_size[1] - 1 and my_direction == "DOWN":
            mask[Action.FORWARD.value] = False
        
        # print(f"[传感器] 范围:{self.sensor_range} 网格:{sensor_grid_size}x{sensor_grid_size}")
        
        agents_found = []
        for i in range(n_sensors):
            has_agent = sub_obs[idx]; idx += 1
            agent_dir_onehot = sub_obs[idx:idx+4]; idx += 4
            idx += self.msg_bits
            idx += 3  
            
            if has_agent:
                row = i // sensor_grid_size
                col = i % sensor_grid_size
                rel_x = col - center
                rel_y = row - center
                agent_dir = direction_names[int(np.argmax(agent_dir_onehot))]
                agents_found.append((rel_x, rel_y, agent_dir))
        
        # if self is the only agent detected, then all actions are legal
        if len(agents_found) > 1:
            
            other_agents_position = []
            other_agent_dirs = []
            for rel_x, rel_y, agent_dir in agents_found:
                if rel_x == 0 and rel_y == 0:
                    pass
                else:
                    other_agents_position.append((rel_x, rel_y))
                    other_agent_dirs.append(agent_dir)

            if my_direction == "UP":
                # detect (+0,-1), if occupied by agent no matter its dir, the foward is illegal;
                if (0, -1) in other_agents_position:
                    mask[Action.FORWARD.value] = False
                if (0, -2) in other_agents_position:
                    # find the index of (0, -2)
                    index = other_agents_position.index((0, -2))
                    # get the diection of this agent
                    agent_dir = other_agent_dirs[index]
                    # if this agent is moving down, then the forward is illegal
                    if agent_dir == "DOWN":
                        mask[Action.FORWARD.value] = False
                if (-1, -1) in other_agents_position:
                    # find the index of (-1, -1)
                    index = other_agents_position.index((-1, -1))
                    # get the diection of this agent
                    agent_dir = other_agent_dirs[index]
                    # if this agent is moving right, then the forward is illegal
                    if agent_dir == "RIGHT":
                        mask[Action.FORWARD.value] = False
                if (1, -1) in other_agents_position:
                    # find the index of (1, -1)
                    index = other_agents_position.index((1, -1))
                    # get the diection of this agent
                    agent_dir = other_agent_dirs[index]
                    # if this agent is moving left, then the forward is illegal
                    if agent_dir == "LEFT":
                        mask[Action.FORWARD.value] = False
            elif my_direction == "DOWN":
                if (0, 1) in other_agents_position:
                    mask[Action.FORWARD.value] = False
                if (0, 2) in other_agents_position:
                    index = other_agents_position.index((0, 2))
                    agent_dir = other_agent_dirs[index]
                    if agent_dir == "UP":
                        mask[Action.FORWARD.value] = False
                if (-1, 1) in other_agents_position:
                    index = other_agents_position.index((-1, 1))
                    agent_dir = other_agent_dirs[index]
                    if agent_dir == "RIGHT":
                        mask[Action.FORWARD.value] = False
                if (1, 1) in other_agents_position:
                    index = other_agents_position.index((1, 1))
                    agent_dir = other_agent_dirs[index]
                    if agent_dir == "LEFT":
                        mask[Action.FORWARD.value] = False
                
            elif my_direction == "LEFT":
                if (-1, 0) in other_agents_position:
                    mask[Action.FORWARD.value] = False
                if (-2, 0) in other_agents_position:
                    index = other_agents_position.index((-2, 0))
                    agent_dir = other_agent_dirs[index]
                    if agent_dir == "RIGHT":
                        mask[Action.FORWARD.value] = False
                if (-1, -1) in other_agents_position:
                    index = other_agents_position.index((-1, -1))
                    agent_dir = other_agent_dirs[index]
                    if agent_dir == "DOWN":
                        mask[Action.FORWARD.value] = False
                if (-1, 1) in other_agents_position:
                    index = other_agents_position.index((-1, 1))
                    agent_dir = other_agent_dirs[index]
                    if agent_dir == "UP":
                        mask[Action.FORWARD.value] = False
            elif my_direction == "RIGHT":
                if (1, 0) in other_agents_position:
                    mask[Action.FORWARD.value] = False
                if (2, 0) in other_agents_position:
                    index = other_agents_position.index((2, 0))
                    agent_dir = other_agent_dirs[index]
                    if agent_dir == "LEFT":
                        mask[Action.FORWARD.value] = False
                if (1, -1) in other_agents_position:
                    index = other_agents_position.index((1, -1))
                    agent_dir = other_agent_dirs[index]
                    if agent_dir == "DOWN":
                        mask[Action.FORWARD.value] = False
                if (1, 1) in other_agents_position:
                    index = other_agents_position.index((1, 1))
                    agent_dir = other_agent_dirs[index]
                    if agent_dir == "UP":
                        mask[Action.FORWARD.value] = False

        return mask
    


if __name__ == "__main__":
    env = Warehouse(9, 8, 3, 10, 3, 1, 5, None, None, RewardType.INDIVIDUAL)
    env.reset()
    from tqdm import tqdm

    # env.render()

    for _ in tqdm(range(1000000)):
        # time.sleep(0.05)
        # env.render()
        actions = env.action_space.sample()
        env.step(actions)
