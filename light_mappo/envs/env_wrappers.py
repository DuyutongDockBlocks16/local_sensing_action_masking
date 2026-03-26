
import numpy as np

# single env
class DummyVecEnv():
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        self.num_envs = len(env_fns)
        self.observation_space = env.observation_space
        self.share_observation_space = env.share_observation_space
        self.action_space = env.action_space
        self.actions = None

    def step(self, actions):
        """
        Step the environments synchronously.
        This is available for backwards compatibility.
        """
        self.step_async(actions)
        return self.step_wait()

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        obs, rews, dones, truncated, infos = map(np.array, zip(*results))

        for (i, done) in enumerate(dones):
            if 'bool' in done.__class__.__name__:
                if done:
                    reset_result = self.envs[i].reset()
                    if isinstance(reset_result, tuple):
                        obs[i], infos[i] = reset_result
                    else:
                        obs[i] = reset_result
            else:
                if np.all(done):
                    reset_result = self.envs[i].reset()
                    if isinstance(reset_result, tuple):
                        obs[i], infos[i] = reset_result
                    else:
                        obs[i] = reset_result
                    
        if len(rews.shape) == 4 and rews.shape[1] == 1:
            rews = rews.squeeze(1)  # (5,1,2,1) -> (5,2,1)

        self.actions = None
        # print(f"Shape of rewards: {rews.shape}")
        # print(f"Rewards: {rews}")
        
        combined_info = {}
        if len(infos) > 0 and isinstance(infos[0], dict) and "action_masks" in infos[0]:
            combined_info["action_masks"] = np.array([info["action_masks"] for info in infos])

        return obs, rews, dones, truncated, combined_info

    def reset(self):
        results = [env.reset() for env in self.envs]
        
        obs_list = []
        infos_list = []
        
        for result in results:
            if isinstance(result, tuple) and len(result) == 2:
                obs, info = result
            else:
                obs = result
                info = {}
            obs_list.append(obs)
            infos_list.append(info)
        
        obs = np.array(obs_list)  # shape: (n_envs, n_agents, obs_dim)
        
        combined_info = {}
        if infos_list and isinstance(infos_list[0], dict) and "action_masks" in infos_list[0]:
            combined_info["action_masks"] = np.array([info["action_masks"] for info in infos_list])
        
        return obs, combined_info

    def close(self):
        for env in self.envs:
            env.close()

    def render(self, mode="human"):
        if mode == "rgb_array":
            return np.array([env.render(mode=mode) for env in self.envs])
        elif mode == "human":
            for env in self.envs:
                env.render(mode=mode)
        else:
            raise NotImplementedError