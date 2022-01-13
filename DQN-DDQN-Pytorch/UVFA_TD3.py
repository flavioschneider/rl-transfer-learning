import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from tensorboardX import SummaryWriter

import gym
# import roboschool
import sys

# %%

import metaworld
import wandb
from tqdm import tqdm


class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(in_dim, hid_dim)
        self.l2 = nn.Linear(hid_dim, hid_dim)
        self.l3 = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        #         print(x)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class ReplayBuffer(object):
    """Buffer to store tuples of experience replay"""

    def __init__(self, max_size=1000000):
        """
        Args:
            max_size (int): total amount of tuples to store
        """

        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, data):
        """Add experience tuples to buffer

        Args:
            data (tuple): experience replay tuple
        """

        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        """Samples a random amount of experiences from buffer of batch size

        Args:
            batch_size (int): size of sample
        """

        ind = np.random.randint(0, len(self.storage), size=batch_size)
        states, actions, next_states, rewards, dones = [], [], [], [], []

        for i in ind:
            s, a, s_, r, d = self.storage[i]
            states.append(np.array(s, copy=False))
            actions.append(np.array(a, copy=False))
            next_states.append(np.array(s_, copy=False))
            rewards.append(np.array(r, copy=False))
            dones.append(np.array(d, copy=False))

        return np.array(states), np.array(actions), np.array(next_states), np.array(rewards).reshape(-1, 1), np.array(
            dones).reshape(-1, 1)

    def save(self):
        np_buffer = np.asarray(self.storage)
        with open('replaybuffer' + str(env._last_rand_vec[0]) + '_' + str(env._last_rand_vec[1]) + '.npy', 'wb') as f:
            print("Saving replay buffer in ", f)
            np.save(f, np_buffer)

    def load(self, filename):
        with open(filename, 'rb') as f:
            self.storage = np.load(f, allow_pickle=True)


class Actor(nn.Module):
    """Initialize parameters and build model.
        Args:
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            max_action (float): highest action to take
            seed (int): Random seed
            h1_units (int): Number of nodes in first hidden layer
            h2_units (int): Number of nodes in second hidden layer

        Return:
            action output of network with tanh activation
    """

    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 100)
        self.l2 = nn.Linear(100, 100)
        self.l3 = nn.Linear(100, action_dim)

        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x

    def load(self, filename="best_avg"):
        state_dict = torch.load(filename)
        for el in state_dict.keys():
            e = state_dict[el]
            print(e.shape)
        self.load_state_dict(state_dict)

def evaluate_policy(policy, env, eval_episodes=100, render=False):
    """run several episodes using the best agent policy

        Args:
            policy (agent): agent to evaluate
            env (env): gym environment
            eval_episodes (int): how many test episodes to run
            render (bool): show training

        Returns:
            avg_reward (float): average reward over the number of evaluations

    """

    avg_reward = 0.
    for i in tqdm(range(eval_episodes)):
        obs = env.reset()
        done = False
        while env.curr_path_length < env.max_path_length:
            if render:
                env.render()
            g = torch.Tensor([EVAL_GOAL_X, EVAL_GOAL_Y]).to(device)
            action = policy.select_action(np.array(obs), goal=g, noise=0)
            obs, reward, done, _ = env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("\n---------------------------------------")
    print("Evaluation over {:d} episodes: {:f}".format(eval_episodes, avg_reward))
    print("---------------------------------------")
    return avg_reward

class UVFA_actor_agent(object):
    def __init__(self, env):
        self.env = env

    def select_action(self, state, goal=None, noise=0.1):
        """Select an appropriate action from the agent policy

            Args:
                state (array): current state of environment
                noise (float): how much noise to add to acitons

            Returns:
                action (float): action clipped within action range

        """

        state = torch.FloatTensor(state.reshape(1, -1)).to(device)

        action = torch.mul(goal_MLP(goal), state_MLP(state)).cpu().data.numpy().flatten()
        if noise != 0:
            action = (action + np.random.normal(0, noise, size=self.env.action_space.shape[0]))

        return action.clip(-1, 1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state_dim = 39
target_actor1,target_actor2,target_actor3,target_actor4 = Actor(state_dim, 4, 1.0),Actor(state_dim, 4, 1.0),Actor(state_dim, 4, 1.0),Actor(state_dim, 4, 1.0)
target_actor1.load('models/best_avg-0.09_0.86_actor.pth')
target_actor2.load('models/best_avg-0.09_0.89_actor.pth')
target_actor3.load('models/best_avg0.07_0.86_actor.pth')
target_actor4.load('models/best_avg0.07_0.89_actor.pth')
target_actors = [target_actor1,target_actor2,target_actor3,target_actor4 ]
for a in target_actors:
    a.to(device)
    a.eval()

r1, r2, r3, r4 = ReplayBuffer(), ReplayBuffer(), ReplayBuffer(), ReplayBuffer()

r1.load('replaybuffers/replaybuffer-0.09_0.86.npy')
r2.load('replaybuffers/replaybuffer-0.09_0.89.npy')
r3.load('replaybuffers/replaybuffer0.07_0.86.npy')
r4.load('replaybuffers/replaybuffer0.07_0.89.npy')

replay_buffers = [r1,r2,r3,r4]

goal_MLP = MLP(2, 100, 4).to(device)
state_MLP = MLP(39, 100, 4).to(device)
goal_optimizer = torch.optim.Adam(goal_MLP.parameters(), lr=1e-3)
state_optimizer = torch.optim.Adam(state_MLP.parameters(), lr=1e-3)

import wandb
wandb.init(project="RL-transfer-learning", entity="frl", settings=wandb.Settings(start_method="fork"))
wandb.watch(goal_MLP, log_freq=10000)
wandb.watch(state_MLP, log_freq=10000)

iterations = 1000000
batch_size = 100
eval_freq = 50
best_avg = -200000
goals = [[-0.09, 0.86], [-0.09, 0.89], [0.07, 0.86], [0.07, 0.89]]
EVAL_GOAL_X, EVAL_GOAL_Y = 0.02, 0.88 # eval 0.00, 0.87

SEED = 0
mt1 = metaworld.MT1('button-press-v2')  # Construct the benchmark, sampling tasks
env = mt1.train_classes['button-press-v2']()  # Create an environment with task `pick_place`
task = mt1.train_tasks[1]
env.set_task(task)  # Set task
env._last_rand_vec[0] = EVAL_GOAL_X  # -0.09 or 0.07
env._last_rand_vec[1] = EVAL_GOAL_Y  # 0.86 or 0.89
env.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

eval_agent = UVFA_actor_agent(env)

# test code (comment for training)
state_MLP.load_state_dict(torch.load('saves/best_stateMLP0.0_0.87'))
goal_MLP.load_state_dict(torch.load('saves/best_goalMLP0.0_0.87'))
eval_reward = evaluate_policy(eval_agent, env)

# train code (comment for test)
# for it in range(iterations):
#     for goal_idx in range(4):
#         #         goal_idx = 3
#         goal = torch.Tensor(goals[goal_idx]).to(device)
#         s, a, next_s, r, d = replay_buffers[goal_idx].sample(batch_size)
#         #         print(s)
#         loss = 0
#         for state in s:
#             ss = torch.Tensor(state).to(device)
#             #             print(ss, ss.shape)
#             target_action = target_actors[goal_idx](ss)
#             #         print(target_action)
#             pred_action = torch.mul(goal_MLP(goal), state_MLP(ss))
#
#             loss += F.mse_loss(pred_action, target_action)
#         print("loss:", loss.item())
#         wandb.log({"loss": loss.item(), "step": it})
#
#         goal_MLP.zero_grad()
#         state_MLP.zero_grad()
#
#         loss.backward()
#
#         goal_optimizer.step()
#         state_optimizer.step()
#
#     if it % eval_freq == 0:
#         eval_reward = evaluate_policy(eval_agent, env)
#         if best_avg < eval_reward:
#             best_avg = eval_reward
#             print("saving best model....\n")
#             torch.save(state_MLP.state_dict(), "saves/best_stateMLP" + str(env._last_rand_vec[0]) + '_' + str(env._last_rand_vec[1]))
#             torch.save(goal_MLP.state_dict(), "saves/best_goalMLP" + str(env._last_rand_vec[0]) + '_' + str(env._last_rand_vec[1]))
#         wandb.log({"uvfa eval reward": eval_reward, "step": it})

