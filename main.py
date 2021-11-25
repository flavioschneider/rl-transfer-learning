import random

import numpy
import torch
import os, time
import numpy as np
import matplotlib.pyplot as plt

import scipy.stats as stats
from torch.distributions import Categorical, MultivariateNormal
from torch.utils.data.dataset import T_co

GAMMA = 0.9 # Discount rate
ALPHA = 0.1 # Update rate
EPSILON = 0.1 # Exploration factor (epsilon-greedy)
BETA = 10 # Softmax inverse temperature, set to e.g. 50 for max instead of softmax
MOVECOST = 0.01

import wandb

wandb.init(project="rl4rooms", entity="fgossi")

wandb.config = {
  "epochs": 30000,
  "learning_rate": 0.005,
  "hsize": 128,
  "zerofy": False,
  "do_clamp": False,
  "batch_size": 20,
}

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

class DQN_pl(pl.LightningModule):
    """ Who cares about Bellman equations when you have *deep* neural networks? """

    def __init__(self, isize, osize, lr, hsize=128, zerofy=False, do_clamp=False):
        """ Instantiates object, defines network architecture. """
        super().__init__()
        self.learning_rate = lr
        self.L1 = torch.nn.Linear(isize, hsize)
        self.L2 = torch.nn.Linear(hsize, hsize)
        self.L3 = torch.nn.Linear(hsize, osize)
        self.relu = torch.nn.ReLU()
        # self.relu = torch.nn.Sigmoid()
        self.do_clamp = do_clamp
        if do_clamp:
            self.sigmoid = torch.nn.Sigmoid()
        if zerofy:
            torch.nn.init.constant_(self.L1.weight, 0.)
            torch.nn.init.constant_(self.L1.bias, 0.)
            torch.nn.init.constant_(self.L2.weight, 0.)
            torch.nn.init.constant_(self.L2.bias, 0.)

    def configure_optimizers(self):
        return torch.optim.Adagrad(self.parameters(), lr=(self.lr or self.learning_rate))

    def forward(self, X):
        """ Computes the forward pass. """
        out = self.relu(self.L1(X))
        out = self.relu(self.L2(out))
        out = self.L3(out)
        if self.do_clamp:
            out = self.sigmoid(out)
        return out


class DQN(torch.nn.Module):
    """ Who cares about Bellman equations when you have *deep* neural networks? """

    def __init__(self, isize, osize, hsize=50, zerofy=True, do_clamp=False):
        """ Instantiates object, defines network architecture. """
        super(DQN, self).__init__()
        self.L1 = torch.nn.Linear(isize, hsize)
        self.L2 = torch.nn.Linear(hsize, hsize)
        self.L3 = torch.nn.Linear(hsize, osize)
        self.relu = torch.nn.ReLU()
        # self.relu = torch.nn.Sigmoid()
        self.do_clamp = do_clamp
        if do_clamp:
            self.sigmoid = torch.nn.Sigmoid()
        if zerofy:
            torch.nn.init.constant_(self.L1.weight, 0.)
            torch.nn.init.constant_(self.L1.bias, 0.)
            torch.nn.init.constant_(self.L2.weight, 0.)
            torch.nn.init.constant_(self.L2.bias, 0.)

    def forward(self, X):
        """ Computes the forward pass. """
        out = self.relu(self.L1(X))
        out = self.relu(self.L2(out))
        out = self.L3(out)
        if self.do_clamp:
            out = self.sigmoid(out)
        return out

class MultivariateRegression(torch.nn.Module):
    def __init__(self, isize, hsize, osize, zerofy, do_clamp):
        super(MultivariateRegression, self).__init__()
        self.Linear = torch.nn.Linear(isize, osize)

    def forward(self, X):
        out = self.Linear(X)
        return out


def top_row(x, y):
    return y == 0


def right_column(x, y):
    return x == 3


def bottom_row(x, y):
    return y == 3


def left_column(x, y):
    return x == 0


class FourRoomsDataset(Dataset):
    def __init__(self, MTRL):
        self.MTRL = MTRL

    def __getitem__(self, index) -> T_co:
        random_room = random.randint(0, 2)
        goal_cell = self.MTRL.change_goal_4rooms(random_room)
        return torch.tensor([goal_cell], dtype=torch.float)


def room_goal2xy(random_room, goal_cell):
    x = goal_cell % 4
    y = goal_cell // 4
    if random_room == 1 or random_room == 2:
        x += 5
    if random_room == 2 or random_room == 3:
        y += 5
    return x, y


def state2xy(STATE):
    state_room = STATE // 16
    state_cell = STATE % 16
    state_x = state_cell % 4
    state_y = state_cell // 4
    if state_room == 1 or state_room == 2:
        state_x += 5
    if state_room == 2 or state_room == 3:
        state_y += 5
    elif STATE == 64:
        state_x = 4
        state_y = 1
    elif STATE == 65:
        state_x = 7
        state_y = 4
    elif STATE == 66:
        state_x = 4
        state_y = 7
    elif STATE == 67:
        state_x = 1
        state_y = 4
    return state_x, state_y


def xy2polarencoding(x, y):
    x = torch.Tensor([x])
    y = torch.Tensor([y])
    x -= 4
    y -= 4
    r = torch.sqrt(x ** 2 + y ** 2)
    sin = y
    cos = x
    return r, sin, cos


def state2polarencoding(state):
    x,y = state2xy(state)
    r, sin, cos = xy2polarencoding(x,y)
    return r, sin, cos


class MTRL:
    """ Multi-task RL on a toy problem.
        Function Approximator: DQN.
        Algorithms: Value Iteration, Q-Learning (Tabular + DQN), UVFA, SF, USFA.
    """
    def __init__(self, Wtrain, Wtest, load_index=1):
        """ Instantiates object, load dataset. """
        getattr(self, f'load_dataset_{load_index}')()
        self.Wtrain = torch.tensor(Wtrain)
        self.Ntrain = len(self.Wtrain)
        self.Wtest = torch.tensor(Wtest)
        self.Ntest = len(self.Wtest)

    def load_dataset_4rooms(self):
        """ Load MDP = (S, A, T, R) for Dataset 1. """

        self.width = 9
        self.height = 9
        self.NS = 68
        self.NA = 4 # NA = |A|, number of actions
        self.S = torch.arange(self.width * self.height).reshape(self.width, self.height) # S, set of states
        self.A = torch.arange(self.NA) # A, set of actions
        # actions 0, 1, 2, 3 = up, right, down, left

        # T(s,a,s') -- state transition function. (NS, NA, NS) tensor.
        self.reset_transitions_4rooms()
        goal_cell = 0  # goal in cell 0
        # R(s,a) rewards. (NS,NA) tensor.
        self.R = torch.zeros(self.NS, self.NA)
        self.R[1, 3] = 1   # 1,left
        self.R[4, 0] = 1   # 4,up

        # (NS, ) boolean tensor specifying if state is terminal.
        self.is_terminal = torch.zeros(self.NS)
        self.is_terminal[goal_cell] = 1
        for a in range(4):
            for s in range(self.NS):
                self.T[goal_cell, a, s] = 0

    def change_goal_4rooms(self, room=0, goal_cell=-1):
        """Set a new goal in a random cell of the specified room"""
        self.reset_transitions_4rooms()
        if goal_cell == -1:
            goal_cell = random.randint(0, 15)
        # room = 2
        # goal_cell = 8
        x = goal_cell % 4
        y = goal_cell // 4
        # print("x,y:",x,y)
        self.R = torch.zeros(self.NS, self.NA)
        if room==0:
            if x==3 and y==1:
                self.R[64, 3] = 1
            elif x==1 and y==3:
                self.R[67, 0] = 1
        elif room==1:
            if x==0 and y==1:
                self.R[64, 1] = 1
            elif x==2 and y==3:
                self.R[65, 0] = 1
        elif room==2:
            if x==2 and y==0:
                self.R[65, 2] = 1
            elif x==0 and y==2:
                self.R[66, 1] = 1
        elif room==3:
            if x==1 and y== 0:
                self.R[67, 2] = 1
            elif x==3 and y==2:
                self.R[66, 3] = 1

        goal_cell += room * 16
        if not bottom_row(x,y):
            self.R[goal_cell+4, 0] = 1
        if not left_column(x,y):
            self.R[goal_cell-1, 1] = 1
        if not top_row(x,y):
            self.R[goal_cell-4, 2] = 1
        if not right_column(x,y):
            self.R[goal_cell+1, 3] = 1

        self.is_terminal = torch.zeros(self.NS)
        self.is_terminal[goal_cell] = 1
        for a in range(4):
            for s in range(self.NS):
                self.T[goal_cell, a, s] = 0

        return goal_cell


    def load_dataset_0(self):
        """ Load MDP = (S, A, T, R) for Dataset 1. """

        self.NS = 13 # NS = |S|, number of states
        self.NA = 3 # NA = |A|, number of actions
        self.NP = 3 # NP = |phi| = |w|, dimensionality of task-space and feature-space
        self.S = torch.arange(self.NS) # S, set of states
        self.A = torch.arange(self.NA) # A, set of actions

        # T(s,a,s') -- state transition function. (NS, NA, NS) tensor.
        self.T = torch.zeros(self.NS, self.NA, self.NS)
        self.T[0, 0, 1] = 1.
        self.T[0, 1, 2] = 1.
        self.T[0, 2, 3] = 1.
        self.T[1, 0, 4] = 1.
        self.T[1, 1, 5] = 1.
        self.T[1, 2, 6] = 1.
        self.T[2, 0, 7] = 1.
        self.T[2, 1, 8] = 1.
        self.T[2, 2, 9] = 1.
        self.T[3, 0, 10] = 1.
        self.T[3, 1, 11] = 1.
        self.T[3, 2, 12] = 1.

        # Φ(s) -- state features. (NS, 3) tensor.
        self.phi = []
        self.phi.append([0, 0, 0])
        self.phi.append([0, 0, 0])
        self.phi.append([0, 0, 0])
        self.phi.append([0, 0, 0])
        self.phi.append([0, 0, 0.9])
        self.phi.append([1, 0, 0.9])
        self.phi.append([0, 0, 0.9])
        self.phi.append([0.9, 0, 0.9])
        self.phi.append([0.8, 0.8, 0.9])
        self.phi.append([0, 0.9, 0.9])
        self.phi.append([0, 0, 0.9])
        self.phi.append([0, 1, 1.5])
        self.phi.append([0, 0, 0.9])
        self.phi = torch.tensor(self.phi)

        # (NS, ) boolean tensor specifying if state is terminal.
        self.is_terminal = torch.tensor([0,0,0,0,1,1,1,1,1,1,1,1,1])

    def load_dataset_1f(self):
        """ Load MDP = (S, A, T, R) for Dataset 1. """

        self.NS = 13 # NS = |S|, number of states
        self.NA = 3 # NA = |A|, number of actions
        self.NP = 3 # NP = |phi| = |w|, dimensionality of task-space and feature-space
        self.S = torch.arange(self.NS) # S, set of states
        self.A = torch.arange(self.NA) # A, set of actions

        # T(s,a,s') -- state transition function. (NS, NA, NS) tensor.
        self.T = torch.zeros(self.NS, self.NA, self.NS)
        self.T[0, 0, 1] = 1.
        self.T[0, 1, 2] = 1.
        self.T[0, 2, 3] = 1.
        self.T[1, 0, 4] = 1.
        self.T[1, 1, 5] = 1.
        self.T[1, 2, 6] = 1.
        self.T[2, 0, 7] = 1.
        self.T[2, 1, 8] = 1.
        self.T[2, 2, 9] = 1.
        self.T[3, 0, 10] = 1.
        self.T[3, 1, 11] = 1.
        self.T[3, 2, 12] = 1.

        # Φ(s) -- state features. (NS, 3) tensor.
        self.phi = []
        self.phi.append([0, 0, 0])
        self.phi.append([0, 0, 0])
        self.phi.append([0, 0, 0])
        self.phi.append([0, 0, 0])
        self.phi.append([0, 0.2, 0])
        self.phi.append([1, 0, 1])
        self.phi.append([0.2, 0.2, 2])
        self.phi.append([0.9, 0, 0.9])
        self.phi.append([0.8, 0.8, 1.6])
        self.phi.append([0, 0.9, 0.9])
        self.phi.append([0.2, 0.2, 0])
        self.phi.append([0, 1, 1.6])
        self.phi.append([0.1, 0, 0.5])
        self.phi = torch.tensor(self.phi)

        # (NS, ) boolean tensor specifying if state is terminal.
        self.is_terminal = torch.tensor([0,0,0,0,1,1,1,1,1,1,1,1,1])

    def execute(self, pi, w):
        """ Execute policy to get reward and final state. """
        state, reward = 0, torch.tensor([0.])
        while not self.is_terminal[state]:
            action = Categorical(pi[state]).sample()
            state = Categorical(self.T[state, action, :]).sample()
            reward += self.phi[state] @ w
        return state, reward

    def q_learning(self, npertask=200, gamma=GAMMA, alpha=ALPHA, eps=EPSILON, beta=BETA):
        """ Vanilla (tabular) Q-learning algorithm. """
        train_tasks = torch.arange(self.Ntrain).repeat(npertask)
        train_tasks = train_tasks[torch.randperm(len(train_tasks))]
        Q = torch.rand(self.NS, self.NA)
        for i in range(self.NS):
            if self.is_terminal[i]:
                Q[i] *= 0.

        for k in train_tasks:
            state = 0
            while not self.is_terminal[state]:
                actions = torch.randint(self.NA, (self.NS, ), dtype=torch.long)
                if stats.uniform.rvs() < 1. - eps:
                    actions = Q.argmax(dim=1)
                action = actions[state]
                new_state = Categorical(self.T[state, action, :]).sample()
                reward = self.phi[new_state] @ self.Wtrain[k]
                new_action = actions[new_state]
                Q[state, action] += alpha * (reward + gamma * Q[new_state, new_action] - Q[state, action])
                state = new_state

        pi = torch.nn.Softmax(dim=1)(beta * Q)
        return Q, pi

    def dqn_learning(self, npertask=200, gamma=GAMMA, alpha=ALPHA, eps=EPSILON, beta=BETA):
        """ Vanilla (tabular) Q-learning algorithm. """
        def compute_all(dqn_model):
            ip1 = encS.repeat(1, self.NA).view(-1, self.NS)
            ip2 = encA.repeat(self.NS, 1)
            out = dqn_model(torch.cat((ip1, ip2), 1)).squeeze().view(self.NS, self.NA)
            return out

        train_tasks = torch.arange(self.Ntrain).repeat(npertask)
        train_tasks = train_tasks[torch.randperm(len(train_tasks))]
        Q = DQN(self.NS + self.NA, 1, zerofy=False)
        optim = torch.optim.Adagrad(Q.parameters(), lr=alpha)
        loss = torch.nn.MSELoss()
        encS = torch.eye(self.NS)
        encA = torch.eye(self.NA)

        for k in train_tasks:
            old_model = compute_all(Q)
            state = 0
            while not self.is_terminal[state]:
                optim.zero_grad()
                current = compute_all(Q)
                actions = torch.randint(self.NA, (self.NS, ), dtype=torch.long)
                if stats.uniform.rvs() < 1. - eps:
                    actions = current.argmax(dim=1)
                action = actions[state]
                current[state,action].backward()
                new_state = Categorical(self.T[state, action, :]).sample()
                reward = (self.phi[new_state] @ self.Wtrain[k])
                #output = loss(current[state,action], reward + gamma * old_model[new_state].max(dim=0)[0])
                for mp in Q.parameters():
                    mp.grad *= alpha * (reward + gamma * old_model[new_state].max(dim=0)[0] - current[state,action])
                old_model = current.detach()
                optim.step()
                state = new_state
        pi = torch.nn.Softmax(dim=1)(beta * compute_all(Q))
        return Q, pi

    def value_iteration(self, w, threshold=0.01, gamma=GAMMA, beta=BETA):
        """ Value iteration algorithm. """
        V = torch.zeros(self.NS)
        pi = torch.zeros(self.NS, self.NA)
        for i in range(self.NS):
            if self.is_terminal[i]:
                V[i] =  self.phi[i] @ w

        delta = torch.tensor([threshold])
        count = 0
        while delta >= threshold:
            delta = torch.tensor([0.])
            count += 1
            for s in range(self.NS):
                curr_state = V[s].clone()
                vals = self.phi @ w
                vals += gamma * V
                Q = self.T[s] @ vals
                pi[s] = torch.nn.Softmax(dim=0)(beta * Q)
                V[s] = Q.max(dim=0)[0]
                delta = max(delta, abs(V[s] - curr_state))
        # print(f'VI converged in {count} iterations.')
        return V, pi

    def value_iteration_4rooms(self, threshold=0.01, gamma=GAMMA, beta=BETA):
        """ Value iteration algorithm. """
        V = torch.zeros(self.NS)
        pi = torch.zeros(self.NS, self.NA)
        for i in range(self.NS):
            if self.is_terminal[i]:
                V[i] = 0
        # print("V initialization:", V)
        delta = torch.tensor([threshold])
        count = 0
        while delta >= threshold:
            delta = torch.tensor([0.])
            count += 1
            for s in range(self.NS):
                oldV = V[s].clone()
                # print("s:",s, "V:",V,"vals", vals)
                Q = torch.zeros(self.NA)
                for a in range(4):
                    if torch.max(self.T[s, a]) == 0.:
                        Q[a] = 0.
                    else:
                        # print("s:",s,"a:",a)
                        s_prime = torch.argmax(self.T[s, a])
                        Q[a] = self.R[s,a] + GAMMA*V[s_prime]
                        # if Q[a] < 0:
                        #     Q[a] = 0    # don't move

                # print("T[s]:", self.T[s], "vals:", vals)
                pi[s] = torch.nn.Softmax(dim=0)(beta * Q)
                V[s] = torch.max(Q)
                # print("s:", s, "Q:", Q, "V:",V[s],"oldV:",oldV)
                delta = max(delta, abs(V[s] - oldV))

            # print("iteration",count,"V:",V)
        # print(f'VI converged in {count} iterations.')
        for i in range(self.NS):
            if self.is_terminal[i]:
                V[i] = 1
        return V, pi

    def uvfa_train(self, npertask=200, epochs=10, bsize=10, gamma=GAMMA, alpha=ALPHA, beta=BETA):
        """ Universal value function approximators. """
        Vtrain = torch.zeros(self.Ntrain, self.NS)
        for widx in range(self.Ntrain):
            Vtrain[widx] = self.value_iteration(self.Wtrain[widx], gamma=gamma, beta=beta)[0]

        UVFN = DQN(self.NS + self.NP, 1)
        optim = torch.optim.Adagrad(UVFN.parameters(), lr=alpha)
        loss = torch.nn.MSELoss()

        train_tasks = torch.arange(self.Ntrain).repeat(npertask)
        X = self.Wtrain[train_tasks].repeat(1, self.NS).view(-1, self.NP)
        X = torch.cat((torch.eye(self.NS).repeat(npertask * self.Ntrain, 1), X), 1)
        Y = Vtrain.flatten().repeat(npertask).unsqueeze(dim=0).t()
        XY = torch.cat((X, Y), 1)
        XY = XY[torch.randperm(len(XY))]
        XY = XY.split(bsize)

        for epoch in range(epochs):
            epoch_loss = 0.
            for i in range(bsize):
                optim.zero_grad()
                preds = UVFN(XY[i][:,:-1])
                target = XY[i][:,-1]
                output = loss(preds.squeeze(), target)
                epoch_loss += output.item()
                output.backward()
                optim.step()
            print(f'Epoch {epoch}: {epoch_loss}')
        return UVFN

    def uvfa_train_4rooms_pl(self, epochs=100, bsize=1, gamma=GAMMA, alpha=ALPHA, beta=BETA, hsize=10):
        """ Universal value function approximators. """

        UVFN = DQN(isize=1, osize=self.NS, hsize=wandb.config['hsize'],
                   do_clamp=wandb.config['do_clamp'], zerofy=wandb.config['zerofy'],
                   lr=wandb.config['learning_rate'])
        # optim = torch.optim.Adam(UVFN.parameters(),lr=wandb.config['learning_rate'])
        loss = torch.nn.MSELoss()
        wandb.watch(UVFN, log='all', log_freq=1)  # Optional
        print(UVFN)
        print(UVFN.parameters())

        # X = states
        # Y = goals
        model = UVFN
        trainer = pl.Trainer(gpus=1)

        train_dataloader = DataLoader(FourRoomsDataset(self))

        # Run learning rate finder
        lr_finder = trainer.tuner.lr_find(model)

        # Plot with
        fig = lr_finder.plot(suggest=True)
        fig.show()

        # Pick point based on plot, or get suggestion
        new_lr = lr_finder.suggestion()

        # update hparams of the model
        model.hparams.lr = new_lr

        # Fit model
        trainer.fit(model, train_dataloader)

        for epoch in range(epochs):
            epoch_loss = 0.
            random_room = random.randint(0, 2)
            goal_cell = self.change_goal_4rooms(random_room)
            X = torch.zeros(self.NS)
            Y = torch.zeros(self.NS)
            Y[goal_cell] = 1.
            XY = torch.cat((X.flatten(), Y.flatten()))
            target, pi = self.value_iteration_4rooms()

            # preds = UVFN(XY).squeeze()
            preds = UVFN(torch.tensor([goal_cell], dtype=torch.float)).squeeze()
            output = loss(preds, target)
            # output = torch.mean(torch.exp(10*target)*(preds-target)**2)
            epoch_loss += output.item()
            output.backward()
            optim.step()

            print(f'Epoch {epoch}: {epoch_loss}')
            wandb.log({"loss": epoch_loss})
            preds_max = torch.max(preds).item()
            print("preds max:",preds_max)
            wandb.log({"predsmax": preds_max})
            if epoch%9 == 0 and epoch>80:
                print(f'Plotting epoch {epoch}: {epoch_loss}')
                # plot_4rooms(target, title='True Epoch'+str(epoch)+' MSE '+str(round(epoch_loss, 4)))
                # plot_4rooms(preds, title='Predicted Epoch'+str(epoch)+' MSE '+str(round(epoch_loss, 4)))
        return UVFN


    def uvfa_train_4rooms(self, epochs=100, bsize=1, gamma=GAMMA, alpha=ALPHA, beta=BETA, hsize=10):
        """ Universal value function approximators. """
        device = torch.device('cuda')
        self.device = device
        UVFN = DQN(isize=self.NS, osize=self.NS, hsize=wandb.config['hsize'],
                   do_clamp=wandb.config['do_clamp'], zerofy=wandb.config['zerofy']).to(device)
        # UVFN = MultivariateRegression(isize=self.NS, osize=self.NS, hsize=wandb.config['hsize'],
        #            do_clamp=wandb.config['do_clamp'], zerofy=wandb.config['zerofy']).to(device)
        # UVFN.load_state_dict(torch.load('val158.5459epoch90loss709.4916.pkl'))
        # optim = torch.optim.Adagrad(UVFN.parameters(), lr=wandb.config['learning_rate'])
        optim = torch.optim.Adam(UVFN.parameters(),lr=wandb.config['learning_rate'])
        loss = torch.nn.MSELoss()
        wandb.watch(UVFN, log='all', log_freq=1)  # Optional
        print(UVFN)
        print(UVFN.parameters())

        # X = states
        # Y = goals

        for epoch in range(epochs):
            UVFN.train()
            epoch_loss = 0.
            UVFN.zero_grad()
            output = 0.
            for batch in range(bsize):
                random_room = random.randint(0, 2)
                goal_cell = self.change_goal_4rooms(random_room)
                # X = torch.zeros(self.NS)
                Y = torch.zeros(self.NS)
                Y[goal_cell] = 1.
                # XY = torch.cat((X.flatten(), Y.flatten()))
                target, pi = self.value_iteration_4rooms()
                # preds = UVFN(XY).squeeze()
                # preds = UVFN(torch.tensor([goal_cell], dtype=torch.float).to(device)).squeeze()
                preds = UVFN(Y.to(device)).to(device).squeeze()
                output = loss(preds.to(device), target.to(device))
                # output = torch.mean(torch.exp(10*target)*(preds-target)**2)
                epoch_loss += output.item()
                output.backward()
            optim.step()

            print(f'Epoch {epoch}: {epoch_loss}')
            wandb.log({"loss": epoch_loss})
            preds_max = torch.max(preds).item()
            print("preds max:",preds_max)
            wandb.log({"predsmax": preds_max})
            if epoch%9 == 0 and epoch>8:
                val_mse = self.uvfa_test_4rooms(UVFN)
                print(f'Plotting epoch {epoch}: {epoch_loss}')
                plot_4rooms(target, title='True Epoch'+str(epoch)+' MSE '+str(round(output.item(), 4)))
                plot_4rooms(preds, title='Predicted Epoch'+str(epoch)+' MSE '+str(round(output.item(), 4)))
                torch.save(UVFN.state_dict(), 'val'+str(round(val_mse, 4))+'epoch'+str(epoch)+'loss'+str(round(epoch_loss, 4))+'.pkl')
        return UVFN

    def uvfa_train_4rooms_1d(self, epochs=100, bsize=1, gamma=GAMMA, alpha=ALPHA, beta=BETA, hsize=10):
        """ Universal value function approximators. """
        device = torch.device('cuda')
        self.device = device
        UVFN = DQN(isize=6, osize=1, hsize=wandb.config['hsize'],
                   do_clamp=wandb.config['do_clamp'], zerofy=wandb.config['zerofy']).to(device)
        # UVFN = MultivariateRegression(isize=self.NS, osize=self.NS, hsize=wandb.config['hsize'],
        #            do_clamp=wandb.config['do_clamp'], zerofy=wandb.config['zerofy']).to(device)
        # UVFN.load_state_dict(torch.load('val158.5459epoch90loss709.4916.pkl'))
        # optim = torch.optim.Adagrad(UVFN.parameters(), lr=wandb.config['learning_rate'])
        optim = torch.optim.Adam(UVFN.parameters(),lr=wandb.config['learning_rate'])
        loss = torch.nn.MSELoss()
        wandb.watch(UVFN, log='all', log_freq=1)  # Optional
        print(UVFN)
        print(UVFN.parameters())

        print("generating all the value functions for the 64 goals")
        allV = torch.zeros(64, self.NS).to(device)    # goal x V on all states for that goal
        for goal in range(64):
            print("goal:",goal)
            room = goal // 16
            self.change_goal_4rooms(room, goal_cell=(goal % 16))
            V, pi = self.value_iteration_4rooms()
            allV[goal, :] = V
        print("training..")

        for epoch in range(epochs):
            UVFN.train()
            epoch_loss = 0.
            UVFN.zero_grad()
            for batch in range(bsize):
                random_room = random.randint(0, 2)
                if random_room == 2:
                    random_room = 3
                goal_cell = random.randint(0, 15)
                x,y=room_goal2xy(random_room, goal_cell)
                x,y,z = xy2polarencoding(x,y)
                GOAL = goal_cell + 16*random_room
                STATE = random.randint(0, self.NS-1)
                state_x, state_y = state2xy(STATE)
                state_x, state_y, state_z = xy2polarencoding(state_x, state_y)
                concat = torch.cat((torch.tensor([x,y,z], dtype=torch.float), torch.tensor([state_x,state_y,state_z], dtype=torch.float)))
                # print("concat:",concat)
                preds = UVFN(concat.to(device)).to(device).squeeze()
                output = loss(preds.to(device), allV[GOAL, STATE].to(device))
                # output = torch.mean(torch.exp(10*target)*(preds-target)**2)
                epoch_loss += output.item()
                output.backward()

            optim.step()

            # print(f'Epoch {epoch}: {epoch_loss}')
            wandb.log({"loss": epoch_loss})
            preds_max = torch.max(preds).item()
            # print("preds max:",preds_max)
            wandb.log({"predsmax": preds_max})
            if (epoch-1)%500 == 0:
                UVFN.eval()
                with torch.no_grad():
                    room = 2
                    goal_cell = random.randint(0, 15)
                    goal = goal_cell + 16 * room
                    targetV = allV[goal, :].to(device)
                    predV = torch.zeros(self.NS)
                    x, y = room_goal2xy(room, goal_cell)
                    x, y, z = xy2polarencoding(x, y)
                    for state in range(self.NS):
                        state_x, state_y = state2xy(state)
                        state_x, state_y, state_z = xy2polarencoding(state_x, state_y)
                        concat = torch.cat(
                            (torch.tensor([x,y,z], dtype=torch.float), torch.tensor([state_x,state_y,state_z], dtype=torch.float)))
                        predss = UVFN(concat.to(device)).squeeze()
                        predV[state] = predss
                    output = loss(targetV.to(device), predV.to(device))
                    print(f'Epoch {epoch} test mse: {str(output.item())}')
                    wandb.log({"test_loss": str(output.item())})
                    # print("preds max:", preds_max)
                    plot_4rooms(targetV, title='Test True Epoch' + str(epoch) + ' MSE ' + str(round(output.item(), 4)))
                    plot_4rooms(predV.squeeze(),
                                title='Test Predicted Epoch' + str(epoch) + ' MSE ' + str(round(output.item(), 4)))
                    print(f'Plotting epoch {epoch}: {epoch_loss}')
                    room = random.randint(0,2)
                    goal_cell = random.randint(0,15)
                    goal = goal_cell + 16*room
                    x, y = room_goal2xy(room, goal_cell)
                    x, y, z = xy2polarencoding(x, y)
                    targetV = allV[goal, :]
                    predV = torch.zeros(self.NS).to(device)
                    for state in range(self.NS):
                        state_x, state_y = state2xy(state)
                        state_x, state_y, state_z = xy2polarencoding(state_x, state_y)
                        concat = torch.cat(
                            (torch.tensor([x,y,z], dtype=torch.float), torch.tensor([state_x, state_y, state_z], dtype=torch.float)))
                        predV[state] = UVFN(concat.to(device)).to(device).squeeze()
                    output = loss(targetV.to(device), predV.to(device))
                    plot_4rooms(targetV, title='True Epoch'+str(epoch)+' MSE '+str(round(output.item(), 4)))
                    plot_4rooms(predV.squeeze(), title='Predicted Epoch'+str(epoch)+' MSE '+str(round(output.item(), 4)))
                    print(f'Epoch {epoch} mse: {str(output.item())}')
                    torch.save(UVFN.state_dict(), 'epoch'+str(epoch)+'loss'+str(round(epoch_loss, 4))+'.pkl')
        return UVFN


    def uvfa_test_4rooms(self, UVFN, examples=10):
        loss = torch.nn.MSELoss()
        total_loss = 0
        for e in range(examples):
            goal_cell = self.change_goal_4rooms(3)
            X = torch.zeros(self.NS)
            Y = torch.zeros(self.NS)
            Y[goal_cell] = 1.
            XY = torch.cat((X.flatten(), Y.flatten()))
            target, pi = self.value_iteration_4rooms()
            UVFN.eval()
            with torch.no_grad():
                # preds = UVFN(XY)
                # preds = UVFN(torch.tensor([goal_cell], dtype=torch.float).to(self.device))
                preds = UVFN(Y.to(self.device)).to(self.device)
            total_loss += loss(preds.squeeze(), target.to(self.device)).item()

        avg_loss = total_loss / examples
        print("Test MSE loss over",examples,"examples:",avg_loss)
        wandb.log({"test_loss":avg_loss})
        plot_4rooms(target, title='MSE '+str(round(avg_loss, 4)))
        plot_4rooms(preds, title='MSE '+str(round(avg_loss, 4)))
        return avg_loss

    def uvfa_predict(self, uvfn, w, gamma=GAMMA, beta=BETA):
        """ Returns optimal policy for a new task. """
        inp = torch.cat((torch.eye(self.NS), w.unsqueeze(dim=0).repeat(self.NS, 1)), 1)
        V = uvfn(inp).squeeze()
        pi = torch.zeros(self.NS, self.NA)
        for s in range(self.NS):
            vals = self.phi @ w
            vals += gamma * V
            Q = self.T[s] @ vals
            pi[s] = torch.nn.Softmax(dim=0)(beta * Q)
        return pi

    def sfgpi_train(self, threshold=0.01, gamma=GAMMA, beta=BETA):
        """ Successor features + general policy iteration. """
        VPtrain = []
        for wtr in self.Wtrain:
            VPtrain.append(self.value_iteration(wtr, gamma=gamma, beta=beta))

        psi = self.phi.unsqueeze(dim=0).repeat(self.Ntrain, 1, 1)
        for widx in range(self.Ntrain):
            delta = torch.tensor([threshold])
            pi = VPtrain[widx][1]
            count = 0
            while delta >= threshold:
                delta = torch.tensor([0.])
                count += 1
                for s in range(self.NS):
                    curr_psi = psi[widx,s].clone()
                    psi[widx,s] = self.phi[s] + gamma * (pi[s] @ self.T[s] @ psi[widx])
                    delta = max(delta, torch.dist(psi[widx,s], curr_psi))
            print(f'SF + GPI converged in {count} iterations.')
        return psi

    def sfgpi_predict(self, psi, w, gamma=GAMMA, beta=BETA):
        """ Returns optimal policy for a new task. """
        vmax = (psi @ w).max(dim=0)[0]
        Q = self.T @ ((self.phi @ w) + gamma * vmax)
        pi = torch.nn.Softmax(dim=1)(beta * Q)
        return pi

    def usfa_train(self, niters=200, nz=10, zsigma=1, threshold=0.01, alpha=ALPHA, gamma=GAMMA, beta=BETA):
        """ Universal successor feature approximators. """
        USFN = DQN(self.NS + self.NP, self.NP, zerofy=False)
        encS = torch.eye(self.NS)
        optim = torch.optim.Adagrad(USFN.parameters(), lr=alpha)
        loss = torch.nn.MSELoss()

        for niter in range(1, niters + 1):
            # Sample training task w ~ M.
            state = 0
            widx = np.random.randint(self.Ntrain)
            wtask = self.Wtrain[widx]

            # Sample policies z ~ D(·|w), and get V* and π* for each z.
            ztasks = MultivariateNormal(wtask, zsigma * torch.eye(self.NP)).sample(torch.Size([nz]))
            zvalues = [self.value_iteration(zt, gamma=gamma, beta=beta) for zt in ztasks]

            # Use "policy evaluation" to get the correct Ψ(s,z) for each z. This is the "SF" part.
            psi = self.phi.unsqueeze(dim=0).repeat(nz, 1, 1) # shape: (nz, NS, 3)
            for zidx in range(nz):
                delta = torch.tensor([threshold])
                pi = zvalues[zidx][1]
                while delta >= threshold:
                    delta = torch.tensor([0.])
                    for s in range(self.NS):
                        curr_psi = psi[zidx,s].clone()
                        psi[zidx,s] = self.phi[s] + gamma * (pi[s] @ self.T[s] @ psi[zidx])
                        delta = max(delta, torch.dist(psi[zidx,s], curr_psi))

            # Use correct Ψ(s,z) to train Ψ^(s,z) neural network approximator. This is the "UVFA" part
            optim.zero_grad()
            ip1 = encS.repeat(nz, 1)
            ip2 = ztasks.repeat(1, self.NS).view(-1, self.NP)
            ip = torch.cat((ip1, ip2), 1)
            preds = USFN(ip) # shape: (NS * nz, 3)
            targets = psi.view(-1, self.NP)
            output = loss(preds, targets)
            if niter % 10 == 0:
                print(f'Epoch {niter}: {output}')
            output.backward()
            optim.step()
        return USFN

    def usfa_predict(self, usfn, w, C=None, beta=BETA, gamma=GAMMA):
        """ Returns optimal policy for a new task. """
        if type(C) == type(None):
            C = self.Wtrain
        NC = len(C)

        # Use learnt Ψ^(s,z) to get π(s). This is the "GPI" part.
        ip1 = torch.eye(self.NS).repeat(1, NC).view(-1, self.NS)
        ip2 = C.repeat(self.NS, 1)
        ip = torch.cat((ip1, ip2), 1)
        vals = (usfn(ip) @ w).split(NC)
        vmax = torch.tensor([vv.max(dim=0)[0] for vv in vals])

        Q = self.T @ ((self.phi @ w) + gamma * vmax)
        pi = torch.nn.Softmax(dim=1)(beta * Q)
        return pi

    def reset_transitions_4rooms(self):
        self.T = torch.zeros(self.NS, self.NA, self.NS)
        for room in range(4):
            # room 0: states 0-15
            # room 1: states 16-31
            # room 2: states 32-47
            # room 3: states 48-63
            for y in range(4):
                for x in range(4):
                    # print("s:", room * 16 + x + y * 4, "x:",x,"y:",y)
                    # action 0: up
                    if not top_row(x, y):
                        self.T[room * 16 + x + y * 4, 0, room * 16 + x + (y - 1) * 4] = 1.

                    # action 1: right
                    if not right_column(x, y):
                        self.T[room * 16 + x + y * 4, 1, room * 16 + x + 1 + y * 4] = 1.

                    # action 2: down
                    if not bottom_row(x, y):
                        self.T[room * 16 + x + y * 4, 2, room * 16 + x + (y + 1) * 4] = 1.

                    # action 3: left
                    if not left_column(x, y):
                        self.T[room * 16 + x + y * 4, 3, room * 16 + x - 1 + y * 4] = 1.

                    # print( "T[s]:", self.T[room*16 + x + y*4])

        # rooms 0<->1: door 0 = state 64, actions right,left
        self.T[64, 1, 20] = 1.
        self.T[20, 3, 64] = 1.
        self.T[64, 3, 7] = 1.
        self.T[7, 1, 64] = 1.
        # rooms 1<->2: door 1 = state 65, actions up, down
        self.T[65, 0, 30] = 1.
        self.T[30, 2, 65] = 1.
        self.T[65, 2, 34] = 1.
        self.T[34, 0, 65] = 1.
        # rooms 2<->3: door 2 = state 66
        self.T[66, 1, 40] = 1.
        self.T[40, 3, 66] = 1.
        self.T[66, 3, 59] = 1.
        self.T[59, 1, 66] = 1.
        # rooms 3<->0: door 3 = state 67
        self.T[67, 0, 13] = 1.
        self.T[13, 2, 67] = 1.
        self.T[67, 2, 49] = 1.
        self.T[49, 0, 67] = 1.



def example():
    """ A working example of an MTRL instance. """
    wtrain = [[1., 0, 0], [0., 1, 0]]
    wtest = [[1., 1, 0], [0., 0, 1]]
    model = MTRL(wtrain, wtest, load_index='0')

    # Q-learning / DQN
    if False:
        Q, pi = model.q_learning()
        for wt in model.Wtest:
            print(pi)
        print('Q-learning works.')

    if True:
        Q, pi = model.dqn_learning(alpha=100.0)
        for wt in model.Wtest:
            print(pi)
        print('DQN Q-learning works.')

    # Value iteration
    if False:
        for wt in model.Wtest:
            V, pi = model.value_iteration(wt)
            print(pi)
        print('Value iteration works.')

    # UVFA
    if False:
        uvfn = model.uvfa_train()
        for wt in model.Wtest:
            pi = model.uvfa_predict(uvfn, wt)
            print(pi)
        print('UVFA works.')

    # SF + GPI
    if False:
        psi = model.sfgpi_train()
        for wt in model.Wtest:
            pi = model.sfgpi_predict(psi, wt)
            print(pi)
        print('SF + GPI works.')

    # USFA
    if True:
        usfn = model.usfa_train(alpha=0.001)
        for wt in model.Wtest:
            pi = model.usfa_predict(usfn, wt, C=wt.unsqueeze(dim=0))
            print(pi)
        print('USFA works.')

def simulate_1f(nsubjects=60):
    wtrain = [[1., -2, 0], [-2, 1, 0], [1, -1, 0], [-1, 1, 0]]
    wtest = [[1., 1, -1], [0, 0, 1]]
    wall = torch.Tensor(wtrain + wtest)

    subjects = []
    all_endstates = torch.zeros(nsubjects, 7, len(wtrain) + len(wtest), dtype=torch.long)
    all_rewards = torch.zeros(nsubjects, 7, len(wtrain) + len(wtest))
    for j in range(nsubjects):
        print(f'[root] SUBJECT {j}')
        model = MTRL(wtrain, wtest, load_index='1f')

        # Training
        print(f'[root] Training...')
        _, qpi = model.q_learning(npertask=100)
        uvfn = model.uvfa_train(npertask=100)
        psi = model.sfgpi_train()
        usfn = model.usfa_train(niters=400, nz=5, zsigma=0.5)

        # Prediction
        print(f'[root] Predicting...')
        for i in range(len(wall)):
            _, pi = model.value_iteration(wall[i])
            all_endstates[j,0,i], all_rewards[j,0,i] = model.execute(pi, wall[i])
            all_endstates[j,1,i], all_rewards[j,1,i] = model.execute(qpi, wall[i])
            pi = model.uvfa_predict(uvfn, wall[i])
            all_endstates[j,2,i], all_rewards[j,2,i] = model.execute(pi, wall[i])
            pi = model.sfgpi_predict(psi, wall[i])
            all_endstates[j,3,i], all_rewards[j,3,i] = model.execute(pi, wall[i])
            pi = model.usfa_predict(usfn, wall[i], model.Wtrain)
            all_endstates[j,4,i], all_rewards[j,4,i] = model.execute(pi, wall[i])
            pi = model.usfa_predict(usfn, wall[i], wall[i].unsqueeze(dim=0))
            all_endstates[j,5,i], all_rewards[j,5,i] = model.execute(pi, wall[i])
            pi = model.usfa_predict(usfn, wall[i], torch.cat((model.Wtrain, wall[i].unsqueeze(dim=0)), 0))
            all_endstates[j,6,i], all_rewards[j,6,i] = model.execute(pi, wall[i])

        subjects.append(model)
    return subjects, all_endstates, all_rewards, wall

def plot_results(all_endstates, all_rewards, wall, nstates=13):
    algos = ['MB', 'MF', 'UVFA', 'SFGPI', 'USFA-M', 'USFA-W', 'USFA-MW']

    # Train -- End states
    plt.figure()
    idx = 1
    for alidx in range(7):
        for widx in range(4):
            vals = all_endstates[:,alidx,widx].bincount(minlength=nstates)[4:]
            plt.subplot(7, 4, idx)
            plt.xticks([5,6,7,8,9,10,11,12,13], size=7)
            plt.bar([5,6,7,8,9,10,11,12,13], vals, width=0.4)
            if widx == 0:
                plt.ylabel(algos[alidx])
            if alidx == 0:
                plt.title(f'Tr: {wall[widx].tolist()}', size=9)
            #if alidx != 6:
            #    plt.tick_params(axis='x', bottom=False, labelbottom=False)
            idx += 1
    plt.tight_layout()
    plt.savefig('tr3.png')

    # Test -- End states
    plt.figure()
    idx = 1
    for alidx in range(7):
        for widx in range(4, 6):
            vals = all_endstates[:,alidx,widx].bincount(minlength=nstates)[4:]
            plt.subplot(7, 2, idx)
            plt.xticks([5,6,7,8,9,10,11,12,13], size=7)
            plt.bar([5,6,7,8,9,10,11,12,13], vals, width=0.4)
            if widx == 4:
                plt.ylabel(algos[alidx])
            if alidx == 0:
                plt.title(f'Te: {wall[widx].tolist()}', size=9)
            #if alidx != 6:
            #    plt.tick_params(axis='x', bottom=False, labelbottom=False)
            idx += 1
    plt.tight_layout()
    plt.savefig('te3.png')


def vector2grid4rooms(V):
    matrix = numpy.zeros((9, 9))
    for v in range(V.shape[0]):
        room = v // 16
        if room <= 3:
            v_inroom = v % 16
            x = v_inroom % 4
            y = v_inroom // 4
            # if x==0 and y==0:
            #     print("v:",v,"V:",V[v])
            if room == 0:
                matrix[y][x] = V[v]
            elif room == 1:
                matrix[y][x+5] = V[v]
            elif room == 2:
                matrix[y+5][x+5] = V[v]
            elif room == 3:
                matrix[y+5][x] = V[v]
        elif v == 64:
            matrix[1][4] = V[v]
        elif v == 65:
            matrix[4][7] = V[v]
            # print("7,4:",matrix[7][4])
        elif v == 66:
            matrix[7][4] = V[v]
        elif v == 67:
            matrix[4][1] = V[v]
    # print(matrix)
    return matrix



def plot_4rooms(V, title=''):
    import matplotlib.pyplot as plt

    min_val, max_val = 0, 9

    fig, ax = plt.subplots()

    V = vector2grid4rooms(V)
    cmap = plt.cm.Blues
    ax.matshow(V, cmap=plt.cm.Blues)

    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            c = V[j, i]
            # ax.text(i, j, str(c), va='center', ha='center')

    # ax.set_xlim(min_val, max_val)
    # ax.set_ylim(min_val, max_val)
    # ax.set_xticks(np.arange(max_val))
    # ax.set_yticks(np.arange(max_val))
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.grid()
    ax.set_title(title)
    plt.show()


wtrain = [[1., 0, 0], [0., 1, 0]]
wtest = [[1., 1, 0], [0., 0, 1]]
model = MTRL(wtrain, wtest, load_index='0')
print("config:",wandb.config)
print("loading 4rooms")
model.load_dataset_4rooms()
print("VI 4rooms")
V, pi = model.value_iteration_4rooms()
print(V)
# plot_4rooms(V)
UVFN = model.uvfa_train_4rooms_1d(epochs=wandb.config['epochs'], hsize=wandb.config['hsize'], bsize=wandb.config['batch_size'])
model.uvfa_test_4rooms(UVFN=UVFN)
# print(pi)


# _, all_endstates, all_rewards, wall = simulate_1f()
# plot_results(all_endstates, all_rewards, wall)
# torch.save(all_endstates, 'endstates.pt')
# torch.save(all_rewards, 'rewards.pt')
