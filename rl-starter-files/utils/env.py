import gym
import sys
from os import path
root_path = path.dirname(path.dirname(path.dirname(__file__)))
# print("root_path", root_path)
gym_minigrid_path = path.join(root_path, 'gym-minigrid')
# print("gym_minigrid_path",gym_minigrid_path)
sys.path.append(gym_minigrid_path)
import gym_minigrid


def make_env(env_key, seed=None):
    print("utils/env.py make_env(), env_key=", env_key)
    env = gym.make(env_key)
    env.seed(seed)
    return env
