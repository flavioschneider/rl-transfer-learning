import metaworld
import random
import gym
import itertools
import matplotlib
import numpy as np
import pandas as pd
import sys
from collections import defaultdict


def make_epsilon_greedy_policy(epsilon, nA):
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

    def policy_fn(Q):
        nA = 5**4
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q)
        # best_action = np.array([1,1,1,1],dtype=float)
        A[best_action] += (1.0 - epsilon)
        # print("sum:",sum(A))
        return A

    return policy_fn


def action2box(action):
    box = np.zeros(4)
    # print("action:",action)
    for i in range(4):
        box[i] = (action % 5)/2 - 1
        action = action // 5
    # print("box:",box)
    return box


def box2state(next_state):
    pass


def tile_coding(state, dim):
    return state


def q_learning(env, num_episodes, discount_factor=0.9, lambda_=0.4, alpha_theta=0.1, alpha_w=0.00001):
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
    # Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # Keeps track of useful statistics
    # stats = plotting.EpisodeStats(
    #     episode_lengths=np.zeros(num_episodes),
    #     episode_rewards=np.zeros(num_episodes))

    # The policy we're following
    # target policy: greedy(Q)
    pi_policy = make_epsilon_greedy_policy(0.0, env.action_space.shape[0])
    # behavior policy: random
    b_policy = make_epsilon_greedy_policy(0.99, env.action_space.shape[0])

    phi_dimension = 16  # 4 + state dimension
    w_t = theta_t = np.ones(phi_dimension)
    Q_t_state = np.zeros(625)
    tot_timesteps = 10

    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 10 == 0:
            state = env.reset()
            for t in range(tot_timesteps):
                phi_t_S = tile_coding(state, phi_dimension - 4)
                all_actions = np.zeros((625, 4))
                for a in range(625):
                    all_actions[a, :] = action2box(a)
                phi_t_S_allA = np.zeros((625, 16))
                phi_t_S_allA[:, 0:12] = np.tile(phi_t_S, 625).reshape(625, 12)
                phi_t_S_allA[:, 12:16] = all_actions
                # current approximation of Q(state) = phi_t_S_allA @ theta_t (625x16)*(16x1)
                Q_t_state = phi_t_S_allA @ theta_t
                # Take a step
                # implement epsilon greedy policy
                action_probs_pi = pi_policy(Q_t_state)
                num_actions = len(action_probs_pi)
                action_index_pi = np.random.choice(np.arange(num_actions), p=action_probs_pi)
                action_pi = action2box(action_index_pi)
                next_state, reward, done, _ = env.step(action_pi)
                print("action:",action_pi)
                print("evaluation reward:",reward)
                if done:
                    break

                state = next_state

            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()

        # Reset the environment and pick the first action
        state = env.reset()
        # print("state:",state, len(state))

        # One step in the environment
        # total_reward = 0.0
        e_t = np.zeros(phi_dimension)

        for t in range(tot_timesteps):
            # use tile coding to encode state into phi
            phi_t_S = tile_coding(state, phi_dimension-4)
            all_actions = np.zeros((625, 4))
            for a in range(625):
                all_actions[a, :] = action2box(a)
            phi_t_S_allA = np.zeros((625, 16))
            # print("phi_t_S:",phi_t_S)
            phi_t_S_allA[:, 0:12] = np.tile(phi_t_S, 625).reshape(625,12)
            phi_t_S_allA[:, 12:16] = all_actions
            # print("phi_t_S_allA:",phi_t_S_allA)
            # current approximation of Q(state) = phi_t_S_allA @ theta_t (625x16)*(16x1)
            Q_t_state = phi_t_S_allA @ theta_t
            # Take a step
            # implement greedy policy w.r.t. approximate Q as target policy
            action_probs_pi = pi_policy(Q_t_state)
            # implement behavior policy (random)
            action_probs_b = b_policy(Q_t_state)
            num_actions = len(action_probs_b)
            action_index_b = np.random.choice(np.arange(num_actions), p=action_probs_b)
            action_b = action2box(action_index_b)
            next_state, reward, done, _ = env.step(action_b)
            next_action_probs_pi = pi_policy(next_state)

            # print("action",action)

            # phi_t_Snext = phi_t(S_{t+1})
            phi_t_Snext = tile_coding(next_state, phi_dimension-1)
            # calculate phi_bar_t
            phi_bar_t = np.zeros(16)
            for a in range(num_actions):
                phi_bar_t += next_action_probs_pi[a] * np.append(phi_t_Snext, action2box(a))
            # print("phi_bar_t", phi_bar_t)
            # phi_t_S_A = phi_t(S_t, A_t)
            phi_t_S_A = np.append(phi_t_S, action_b)
            # calculate delta_t (scalar)
            delta_t = reward + discount_factor * np.dot(theta_t, phi_bar_t) - np.dot(theta_t, phi_t_S_A)
            # update e_t (vector)
            e_t = phi_t_S_A + discount_factor * lambda_ * action_probs_pi[action_index_b] / (1 / num_actions) * e_t
            # print("e_t", e_t)
            # update theta_t (vector)
            theta_t = theta_t + alpha_theta * (delta_t * e_t - discount_factor*(1-lambda_)*np.dot(w_t,e_t)*phi_bar_t)
            # print("theta_t",theta_t)
            # update w_t (vector)
            w_t = w_t + alpha_w * (delta_t * e_t - (np.dot(w_t,phi_t_S_A))*phi_t_S_A)


            # # Update statistics
            # stats.episode_rewards[i_episode] += reward
            # stats.episode_lengths[i_episode] = t
            if t==tot_timesteps-1:
                print("rew:",reward)
                # print("Q_state:",Q_t_state)
            # print("next state",next_state)
            # next_state = box2state(next_state)

            # # TD Update
            # best_next_action = np.argmax(Q[next_state])
            # td_target = reward + discount_factor * Q[next_state][best_next_action]
            # td_delta = td_target - Q[state][action]
            # Q[state][action] += alpha * td_delta

            if done:
                break

            state = next_state

    return Q_t_state


print(metaworld.ML1.ENV_NAMES)  # Check out the available environments

mt1 = metaworld.MT1('push-v1') # Construct the benchmark, sampling tasks

env = mt1.train_classes['push-v1']()  # Create an environment with task `pick_place`
task = mt1.train_tasks[0]
env.set_task(task)  # Set task

obs = env.reset()  # Reset environment

print("env actions:",env.action_space)
# for i in range(20):
#     print('action', env.action_space.sample())


Q = q_learning(env, 5000, discount_factor=0.9, alpha_theta=0.01)

# a = env.action_space.sample()  # Sample an action
# print('action', a)
# obs, reward, done, info = env.step(a)  # Step the environoment with the sampled random action
# agent.learn(obs, a, reward, ns, np_actions, na, terminal)
# print('state', obs)
# print('reward', reward)

