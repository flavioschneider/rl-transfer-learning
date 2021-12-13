import time
import numpy as np
import torch
import gym
from DQN import DQN_Agent,ReplayBuffer,device
from torch.utils.tensorboard import SummaryWriter
import os, shutil
from datetime import datetime
import argparse
from utils import evaluate_policy,str2bool
import metaworld
import wandb

'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--EnvIdex', type=int, default=0, help='CP-v1, LLd-v2')
parser.add_argument('--write', type=str2bool, default=True, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--pth', type=str, default='', help='which model to load')

parser.add_argument('--seed', type=int, default=532, help='random seed')
parser.add_argument('--Max_train_steps', type=int, default=1e6, help='Max training steps')
parser.add_argument('--save_interval', type=int, default=5e4, help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=1e3, help='Model evaluating interval, in steps.')
parser.add_argument('--random_steps', type=int, default=3e3, help='steps for random policy to explore')
parser.add_argument('--update_every', type=int, default=50, help='training frequency')

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--net_width', type=int, default=200, help='Hidden net width')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=512, help='lenth of sliced trajectory')
parser.add_argument('--exp_noise', type=float, default=0.2, help='explore noise')
parser.add_argument('--noise_decay', type=float, default=0.99, help='decay rate of explore noise')
parser.add_argument('--DDQN', type=str2bool, default=True, help='True:DDQN; False:DQN')
opt = parser.parse_args()
print(opt)

def action2box(action):
    box = np.zeros(4)
    # print("action2:",action)
    for i in range(4):
        box[i] = (action % 5)/2 - 1
        action = action // 5
    # print("2box:",box)
    return box


def box2action(box):
    # print("box2:",box)
    a = 0
    for i in range(4):
        a += (round(box[i])+1)*2 * (5**i)
    # print("2action:",a)
    # print("origbox:",box,"a2b:",action2box(a))
    return a


def transform_reward(r):
    # return max(0,np.log2(r+200))
    return r

def main():
    wandb.init(project="ddqn-metaworld", entity="fgossi")
    # EnvName = ['CartPole-v1','LunarLander-v2']
    # BriefEnvName = ['CPV1', 'LLdV2']
    # Env_With_DW = [True, True] #DW: Die or Win
    # EnvIdex = opt.EnvIdex
    # env_with_dw = Env_With_DW[EnvIdex]
    env_with_dw = False
    # env = gym.make(EnvName[EnvIdex])
    mt1 = metaworld.MT1('push-v1')  # Construct the benchmark, sampling tasks
    env = mt1.train_classes['push-v1']()  # Create an environment with task `pick_place`
    task = mt1.train_tasks[1]
    env.set_task(task)  # Set task
    env._last_rand_vec[3] = -0.09   #-0.09 or 0.09
    env._last_rand_vec[4] = 0.81    #0.81 or 0.89
    wandb.log({"goalX": env._last_rand_vec[3], "goalY":env._last_rand_vec[4]})
    # eval_env = gym.make(EnvName[EnvIdex])
    eval_env = env
    state_dim = 6
    action_dim = 625
    max_e_steps = 150


    #Use DDQN or DQN
    if opt.DDQN: algo_name = 'DDQN'
    else: algo_name = 'DQN'

    seed = opt.seed
    torch.manual_seed(seed)
    env.seed(seed)
    eval_env.seed(seed)
    np.random.seed(seed)

    print('Algorithm:',algo_name,'  Env:','metaworld','  state_dim:',state_dim,'  action_dim:',action_dim,'  Random Seed:',seed, '  max_e_steps:',max_e_steps)
    print('\n')

    if opt.write:
        timenow = str(datetime.now())[0:-10]
        timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
        writepath = 'runs/{}_{}'.format(algo_name,'metaworld') + timenow
        if os.path.exists(writepath): shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)

    kwargs = {
        "env_with_dw":env_with_dw,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "gamma": opt.gamma,
        "hid_shape": (opt.net_width,opt.net_width),
        "lr": opt.lr,
        "batch_size":opt.batch_size,
        "exp_noise":opt.exp_noise,
        "double_dqn": opt.DDQN
    }
    if not os.path.exists('model'): os.mkdir('model')
    model = DQN_Agent(**kwargs)
    if opt.Loadmodel: model.load(opt.pth)

    buffer = ReplayBuffer(state_dim, action_dim, max_size=int(1e6))

    minS = np.ones(state_dim)*10
    maxS = np.ones(state_dim)*(-10)

    wandb.watch(model.q_net, log_freq=10000)
    wandb.watch(model.q_target, log_freq=10000)

    print(str(kwargs['batch_size']))

    if opt.render:
        score, _ = evaluate_policy(eval_env, model, True, 20)
        print('EnvName:', 'metaworld', 'seed:', seed, 'score:', score)
    else:
        total_steps = 0
        while total_steps < opt.Max_train_steps:

            s, done, ep_r, steps = env.reset(), False, 0, 0
            ep_r = transform_reward(ep_r)
            s = s[:6]

            minS = np.minimum(s, minS)
            maxS = np.maximum(s, maxS)

            while not done:
                # print("step:", steps)
                steps += 1  #steps in current episode
                if buffer.size < opt.random_steps:
                    a = env.action_space.sample()
                else:
                    a = action2box(model.select_action(s, deterministic=False))
                s_prime, r, done, info = env.step(a)
                r = transform_reward(r)
                s_prime = s_prime[:6]

                minS = np.minimum(s_prime, minS)
                maxS = np.maximum(s_prime, maxS)

                '''Avoid impacts caused by reaching max episode steps'''
                if (done and steps != max_e_steps):
                    dw = True  # dw: dead and win
                else:
                    dw = False

                buffer.add(s, box2action(a), r, s_prime, dw)
                s = s_prime
                ep_r += r

                '''update if its time'''
                # train 50 times every 50 steps rather than 1 training per step. Better!
                if total_steps >= opt.random_steps and total_steps % opt.update_every == 0:
                    for j in range(opt.update_every):
                        model.train(buffer)

                '''record & log'''
                if (total_steps) % opt.eval_interval == 0:
                    model.exp_noise *= opt.noise_decay
                    score, positive_eps = evaluate_policy(eval_env, model, render=False, state_dim=state_dim)
                    if opt.write:
                        writer.add_scalar('ep_r', score, global_step=total_steps)
                        writer.add_scalar('noise', model.exp_noise, global_step=total_steps)
                    print('EnvName:','metaworld','seed:',seed,'steps: {}k'.format(int(total_steps/1000)),'score:', score)
                    wandb.log({"reward": score, "step": total_steps, "pos_eps":positive_eps})
                total_steps += 1

                '''save model'''
                if (total_steps) % opt.save_interval == 0:
                    model.save(algo_name,'metaworld'+'_bs'+str(kwargs['batch_size'])+'gamma'+str(opt.gamma)+'nDec'+str(opt.noise_decay),total_steps)

                '''print minS, maxS'''
                if (total_steps) % 10000 == 0:
                    print("minS", minS, "\nmaxS", maxS)
    env.close()

    print("minS",minS,"\nmaxS",maxS)

if __name__ == '__main__':
    main()








