import gymnasium as gym

def make_env(slippery: bool = True, render_mode=None, seed: int | None = None):
    env = gym.make("FrozenLake-v1",
                   is_slippery=bool(slippery),
                   render_mode=render_mode)
    obs, info = env.reset(seed=seed)
    if seed is not None:
        env.action_space.seed(seed)
    return env
