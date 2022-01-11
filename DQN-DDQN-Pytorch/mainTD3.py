import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from tensorboardX import SummaryWriter

import gym
# import roboschool
import sys

#%%

import metaworld
import wandb
from tqdm import tqdm

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

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

class Critic(nn.Module):
    """Initialize parameters and build model.
        Args:
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            max_action (float): highest action to take
            seed (int): Random seed
            h1_units (int): Number of nodes in first hidden layer
            h2_units (int): Number of nodes in second hidden layer

        Return:
            value output of network
    """

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 100)
        self.l2 = nn.Linear(100, 100)
        self.l3 = nn.Linear(100, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 100)
        self.l5 = nn.Linear(100, 100)
        self.l6 = nn.Linear(100, 1)

    def forward(self, x, u):
        xu = torch.cat([x, u], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)

        x2 = F.relu(self.l4(xu))
        x2 = F.relu(self.l5(x2))
        x2 = self.l6(x2)
        return x1, x2

    def Q1(self, x, u):
        xu = torch.cat([x, u], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)
        return x1

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


class TD3(object):
    """Agent class that handles the training of the networks and provides outputs as actions

        Args:
            state_dim (int): state size
            action_dim (int): action size
            max_action (float): highest action to take
            device (device): cuda or cpu to process tensors
            env (env): gym environment to use

    """

    def __init__(self, state_dim, action_dim, max_action, env, device):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        self.max_action = max_action
        self.env = env

        wandb.watch(self.actor, log_freq=10000)
        wandb.watch(self.actor_target, log_freq=10000)
        wandb.watch(self.critic, log_freq=10000)
        wandb.watch(self.critic_target, log_freq=10000)

    def select_action(self, state, noise=0.1):
        """Select an appropriate action from the agent policy

            Args:
                state (array): current state of environment
                noise (float): how much noise to add to acitons

            Returns:
                action (float): action clipped within action range

        """

        state = torch.FloatTensor(state.reshape(1, -1)).to(device)

        action = self.actor(state).cpu().data.numpy().flatten()
        if noise != 0:
            action = (action + np.random.normal(0, noise, size=self.env.action_space.shape[0]))

        return action.clip(self.env.action_space.low, self.env.action_space.high)

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2,
              noise_clip=0.5, policy_freq=2):
        """Train and update actor and critic networks

            Args:
                replay_buffer (ReplayBuffer): buffer for experience replay
                iterations (int): how many times to run training
                batch_size(int): batch size to sample from replay buffer
                discount (float): discount factor
                tau (float): soft update for main networks to target networks

            Return:
                actor_loss (float): loss from actor network
                critic_loss (float): loss from critic network

        """

        for it in range(iterations):

            # Sample replay buffer
            x, y, u, r, d = replay_buffer.sample(batch_size)
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(1 - d).to(device)
            reward = torch.FloatTensor(r).to(device)

            # Select action according to policy and add clipped noise
            noise = torch.FloatTensor(u).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (done * discount * target_Q).detach()

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Delayed policy updates
            if it % policy_freq == 0:

                # Compute actor loss
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

    def load(self, filename="best_avg", directory="./saves"):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))

class Runner():
    """Carries out the environment steps and adds experiences to memory"""

    def __init__(self, env, agent, replay_buffer):
        self.env = env
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.obs = env.reset()
        self.done = False

    def next_step(self, episode_timesteps, noise=0.1):
        action = self.agent.select_action(np.array(self.obs), noise=0.1)

        # Perform action
        new_obs, reward, done, _ = self.env.step(action)
        done_bool = 0

        # Store data in replay buffer
        replay_buffer.add((self.obs, new_obs, action, reward, done_bool))

        self.obs = new_obs

        if env.curr_path_length >= env.max_path_length:
            self.obs = self.env.reset()
            done = False

            return reward, True

        return reward, done

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
    for i in range(eval_episodes):
        obs = env.reset()
        done = False
        while env.curr_path_length < env.max_path_length:
            if render:
                env.render()
            action = policy.select_action(np.array(obs), noise=0)
            obs, reward, done, _ = env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("\n---------------------------------------")
    print("Evaluation over {:d} episodes: {:f}".format(eval_episodes, avg_reward))
    print("---------------------------------------")
    return avg_reward

def evaluate_policy_store(policy, env, eval_episodes=100, render=False):
    """run several episodes using the best agent policy

        Args:
            policy (agent): agent to evaluate
            env (env): gym environment
            eval_episodes (int): how many test episodes to run
            render (bool): show training

        Returns:
            avg_reward (float): average reward over the number of evaluations

    """
    state_dim = 12
    action_dim = 4
    state_store = np.ndarray((eval_episodes * 200, state_dim))
    action_store = np.ndarray((eval_episodes * 200, action_dim))
    avg_reward = 0.
    j = 0
    for i in range(eval_episodes):
        obs = env.reset()
        done = False
        c = 0
        while env.curr_path_length < env.max_path_length:
            state_store[j, :] = obs
            if render:
                env.render()
            action = policy.select_action(np.array(obs), noise=0)
            action_store[j, :] = action
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
            j += 1
            c += 1
    #             print(c)

    avg_reward /= eval_episodes

    print("\n---------------------------------------")
    print("Evaluation over {:d} episodes: {:f}".format(eval_episodes, avg_reward))
    print("---------------------------------------")
    return avg_reward, state_store, action_store

def observe(env, replay_buffer, observation_steps):
    """run episodes while taking random actions and filling replay_buffer

        Args:
            env (env): gym environment
            replay_buffer(ReplayBuffer): buffer to store experience replay
            observation_steps (int): how many steps to observe for

    """

    time_steps = 0
    obs = env.reset()
    done = False

    while time_steps < observation_steps:
        action = env.action_space.sample()
        new_obs, reward, done, _ = env.step(action)

        replay_buffer.add((obs, new_obs, action, reward, done))

        obs = new_obs
        time_steps += 1

        if env.curr_path_length >= env.max_path_length:
            obs = env.reset()
            done = False

        print("\rPopulating Buffer {}/{}.".format(time_steps, observation_steps), end="")
        sys.stdout.flush()

def train(agent, env, replay_buffer, runner):
    """Train the agent for exploration steps

        Args:
            agent (Agent): agent to use
            env (environment): gym environment
            writer (SummaryWriter): tensorboard writer
            exploration (int): how many training steps to run

    """
    test_env = env

    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    episode_reward = 0
    episode_timesteps = 0
    done = False
    obs = env.reset()
    evaluations = []
    rewards = []
    best_avg = -2000

    writer = SummaryWriter(comment="-TD3_Baseline_HalfCheetah")

    while total_timesteps < EXPLORATION:

        if done:

            if total_timesteps != 0:
                rewards.append(episode_reward)
                avg_reward = np.mean(rewards[-100:])

                writer.add_scalar("avg_reward", avg_reward, total_timesteps)
                writer.add_scalar("reward_step", reward, total_timesteps)
                writer.add_scalar("episode_reward", episode_reward, total_timesteps)

                if best_avg < avg_reward:
                    best_avg = avg_reward
                    print("saving best model....\n")
                    agent.save("best_avg"+str(env._last_rand_vec[0]) + '_' + str(env._last_rand_vec[1]), "saves")

                print("\rTotal T: {:d} Episode Num: {:d} Reward: {:f} Avg Reward: {:f}".format(
                    total_timesteps, episode_num, episode_reward, avg_reward), end="")
                sys.stdout.flush()

                # if avg_reward >= REWARD_THRESH:
                #     break

                agent.train(replay_buffer, episode_timesteps, BATCH_SIZE, GAMMA, TAU, NOISE, NOISE_CLIP,
                            POLICY_FREQUENCY)

                # Evaluate episode
                if timesteps_since_eval >= EVAL_FREQUENCY:
                    timesteps_since_eval %= EVAL_FREQUENCY
                    eval_reward = evaluate_policy(agent, test_env)
                    evaluations.append(avg_reward)
                    writer.add_scalar("eval_reward", eval_reward, total_timesteps)

                    if best_avg < eval_reward:
                        best_avg = eval_reward
                        print("saving best model....\n")
                        agent.save("best_avg"+str(env._last_rand_vec[0]) + '_' + str(env._last_rand_vec[1]), "saves")

                    wandb.log({"reward": eval_reward, "step": total_timesteps})
                    replay_buffer.save()

                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

        reward, done = runner.next_step(episode_timesteps, EXPLORE_NOISE)
        episode_reward += reward

        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1

ENV = "RoboschoolHalfCheetah-v1"  # "Pendulum-v0"
SEED = 0
OBSERVATION = 3000     # 10000
EXPLORATION = 1000000
BATCH_SIZE = 100
GAMMA = 0.99
TAU = 0.005
NOISE = 0.2
NOISE_CLIP = 0.5
EXPLORE_NOISE = 0.1
POLICY_FREQUENCY = 2
EVAL_FREQUENCY = 5000
REWARD_THRESH = 8000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wandb.init(project="RL-transfer-learning", entity="frl", settings=wandb.Settings(start_method="fork"))
env_with_dw = False
# env = gym.make(EnvName[EnvIdex])
mt1 = metaworld.MT1('button-press-v2')  # Construct the benchmark, sampling tasks
env = mt1.train_classes['button-press-v2']()  # Create an environment with task `pick_place`
task = mt1.train_tasks[1]
env.set_task(task)  # Set task
env._last_rand_vec[0] = -0.09   #-0.09 or 0.07
env._last_rand_vec[1] = 0.86    #0.86 or 0.89
wandb.log({"goalX": env._last_rand_vec[0], "goalY": env._last_rand_vec[1]})

# Set seeds
env.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

policy = TD3(state_dim, action_dim, max_action, env, device)

replay_buffer = ReplayBuffer()

runner = Runner(env, policy, replay_buffer)

total_timesteps = 0
timesteps_since_eval = 0
episode_num = 0
done = True

# %%

# Populate replay buffer
observe(env, replay_buffer, OBSERVATION)

# %%

# Train agent
train(policy, env, replay_buffer, runner)

# %%

policy.load()

for i in tqdm(range(100)):
    evaluate_policy(policy, env, render=False)

# %%

reward, state_store, action_store = evaluate_policy_store(policy, env, render=False)

# %%

replay_buffer.save()

env.close()

