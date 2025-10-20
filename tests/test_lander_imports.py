def test_imports():
    import gymnasium as gym
    env = gym.make("LunarLander-v3")
    assert env.observation_space.shape[0] == 8
