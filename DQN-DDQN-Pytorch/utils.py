import numpy as np


def action2box(action):
    box = np.zeros(4)
    # print("action2:",action)
    for i in range(4):
        box[i] = (action % 5)/2 - 1
        action = action // 5
    # print("2box:",box)
    return box

def evaluate_policy(env, model, render, turns = 3,state_dim=None):
    scores = 0
    positive_eps = 0
    for j in range(turns):
        s, done, ep_r, steps = env.reset(), False, 0, 0
        s = s[:state_dim]
        while not done:
            # Take deterministic actions at test time
            a = model.select_action(s, deterministic=True)
            s_prime, r, done, info = env.step(action2box(a))
            s_prime = s_prime[:state_dim]
            ep_r += r
            steps += 1
            if steps == 500:
                done = True
            s = s_prime
            if render:
                env.render()
        scores += ep_r
        if ep_r > 0:
            positive_eps += 1
    return int(scores/turns), positive_eps


#You can just ignore this funciton. Is not related to the RL.
def str2bool(v):
    '''transfer str to bool for argparse'''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')