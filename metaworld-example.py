import metaworld
import random
from rlpy.Agents import Greedy_GQ
from rlpy.Policies import eGreedy
from rlpy.Representations import IncrementalTabular

print(metaworld.ML1.ENV_NAMES)  # Check out the available environments

ml1 = metaworld.ML1('pick-place-wall-v2') # Construct the benchmark, sampling tasks

env = ml1.train_classes['pick-place-wall-v2']()  # Create an environment with task `pick_place`
task = random.choice(ml1.train_tasks)
env.set_task(task)  # Set task

obs = env.reset()  # Reset environment
representation = IncrementalTabular()
agent = Greedy_GQ(discount_factor=0.95, initial_learn_rate=0.1)

a = env.action_space.sample()  # Sample an action
print('action', a)
obs, reward, done, info = env.step(a)  # Step the environoment with the sampled random action
agent.learn(obs, a, reward, ns, np_actions, na, terminal)
print('state', obs)
print('reward', reward) 

