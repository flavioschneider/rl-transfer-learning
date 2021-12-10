import metaworld
import random
import gym
import itertools
import matplotlib
import numpy as np
import pandas as pd
import sys
from collections import defaultdict


def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.

    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action. Float between 0 and 1.
        nA: Number of actions in the environment.

    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """

    def policy_fn(observation):
        nA = 5**4
        A = np.ones(nA, dtype=float) * epsilon / nA
        # best_action = np.argmax(Q[observation])
        best_action = np.array([1,1,1,1],dtype=float)
        A[-1] += (1.0 - epsilon)
        print("sum:",sum(A))
        return A

    return policy_fn


def action2box(action):
    box = np.zeros(4)
    # print("action:",action)
    for i in range(4):
        box[i] = (action % 5)/4
        action = action // 5
    # print("box:",box)
    return box


def box2state(next_state):
    pass


def q_learning(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy

    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance to sample a random action. Float between 0 and 1.

    Returns:
        A tuple (Q, episode_lengths).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # Keeps track of useful statistics
    # stats = plotting.EpisodeStats(
    #     episode_lengths=np.zeros(num_episodes),
    #     episode_rewards=np.zeros(num_episodes))

    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.shape[0])

    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()

        # Reset the environment and pick the first action
        state = env.reset()

        # One step in the environment
        # total_reward = 0.0
        for t in itertools.count():

            # Take a step
            # implement epsilon greedy policy
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            action = action2box(action)
            next_state, reward, done, _ = env.step(action)

            # use tile coding to 

            # calculate phi_bar_t
            phi_bar_t = np.sum()
            # calculate delta_t
            # update e_t
            # update theta_t
            # update w_t



            # # Update statistics
            # stats.episode_rewards[i_episode] += reward
            # stats.episode_lengths[i_episode] = t
            print("rew:",reward)
            print("next state",next_state)
            next_state = box2state(next_state)

            # TD Update
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta

            if done:
                break

            state = next_state

    return Q, stats


print(metaworld.ML1.ENV_NAMES)  # Check out the available environments

mt1 = metaworld.MT1('push-v1') # Construct the benchmark, sampling tasks

env = mt1.train_classes['push-v1']()  # Create an environment with task `pick_place`
task = mt1.train_tasks[0]
env.set_task(task)  # Set task

obs = env.reset()  # Reset environment

print("env actions:",env.action_space)
# for i in range(20):
#     print('action', env.action_space.sample())


Q, stats = q_learning(env, 50)

# a = env.action_space.sample()  # Sample an action
# print('action', a)
# obs, reward, done, info = env.step(a)  # Step the environoment with the sampled random action
# agent.learn(obs, a, reward, ns, np_actions, na, terminal)
# print('state', obs)
# print('reward', reward)

