#import gym
import math
import random
import numpy as np
#import matplotlib
#import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import argparse, shutil

import time, os, datetime, json
import copy, argparse, csv, json, datetime, os
from functools import partial
from pathlib import Path
from torch.distributions import Categorical

from models.models import DQN2D, DQN
from models.agent import Agent, AgentSep1D

from buffer import ReplayMemory
from mas import *

class AgentACShare1D(Agent):
    def __init__(self, name, pars, nrenvs=1, job=None, experiment=None):
        Agent.__init__(self,name, pars, nrenvs, job, experiment)
    def build(self):
        self.policy_net = DQN(71, self.pars).to(self.device)
        self.q_net = DQN(71, self.pars).to(self.device)
        self.target_net = DQN(71, self.pars).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        
        if self.pars['momentum']>0:
            self.optimizer = optim.SGD(
                    self.policy_net.parameters(), lr=self.pars['lr'], 
                    momentum=self.pars['momentum'])#
        if self.pars['momentum']<0:
            self.optimizer = optim.Adam(self.policy_net.parameters())
        if self.pars['momentum']<0:
            self.policy_optimizer = optim.Adam(self.policy_net.parameters())
        self.memory = ReplayMemory(10000)
        
        self.eps_threshold = 0.01
        
    def updateTarget(self, i_episode, step=False):
        #soft_update(self.target_net, self.policy_net, tau=0.01)
        if step:
            return
        if np.random.rand()<0.5:
            self.optimize_policy(self.policy_net, self.buf1, self.policy_optimizer)
        else:
            self.optimize_policy(self.policy_net, self.buf1, self.policy_optimizer)
        
        if i_episode % self.TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
            self.eps_threshold -= 0.001
    def saveStates(self, state1, state2, action1,action2, next_state1,next_state2, reward1,reward2):
                    logp1, ent1, logp2, ent2 = self.rem
                    self.memory.push(state2, action2, next_state2, reward2, state1)
                    self.memory.push(state1, action1, next_state1, reward1, state2)
                    self.buf2.append([state2, action2,1, reward2, logp2, ent2])
                    self.buf1.append([state1, action1,1, reward1, logp1, ent1])
    def select_action(self, state, comm, policy_net):
        probs1, _ = policy_net(state, 1, comm)#.cpu().data.numpy()
        m = Categorical(logits=probs1)
        action = m.sample()
        return action.view(1, 1), m.log_prob(action), m.entropy()
    
    def getaction(self, state1, state2, test=False):
        mes = torch.tensor([[0,0,0,0]], device=self.device)
        comm2 = self.policy_net(state2, 0, mes)[self.idC].detach() if np.random.rand()<self.prob else mes
        comm1 = self.policy_net(state1, 0, mes)[self.idC].detach() if np.random.rand()<self.prob else mes
        
        action1, logp1, ent1 = self.select_action(state1,  comm2, self.policy_net)
        action2, logp2, ent2 = self.select_action(state2,  comm1, self.policy_net)
        self.rem =[logp1, ent1, logp2, ent2]
        return action1, action2, [comm1, comm2]
    def optimize_policy(self, policy_net, memory, optimizer):
        policy_loss = 0
        value_loss = 0
        ent = 0
        R = torch.zeros(1, 1, device=self.device)
        GAE = torch.zeros(1, 1, device=self.device)
        saved_r = torch.cat([c[3].float() for c in memory])
        states = torch.cat([c[0].float() for c in memory])
        action_batch = torch.cat([c[1].float() for c in memory]).view(-1,1)
        mes = torch.tensor([[0,0,0,0] for i in memory], device=self.device)
        actionV = self.q_net(states, 0, mes)[0].gather(1, action_batch.long())
        mu = saved_r.mean()
        std = saved_r.std()
        eps = 0.000001       
        #print(memory)
        for i in reversed(range(len(memory)-1)):
                _,_,_,r,log_prob, entr = memory[i]
                ac = actionV[i]
                #Discounted Sum of Future Rewards + reward for the given state
                R = self.GAMMA * R + (r.float() - mu) / (std + eps)#r.float()
                advantage = R - ac
                #value_loss = value_loss +  advantage.pow(2)
                # Generalized Advantage Estimataion(GAE)

                policy_loss += -log_prob *advantage .detach()
                ent += entr*0
                #print('log_prob',log_prob.size(), 'GAE',GAE.size())
        #print(policy_loss)
        optimizer.zero_grad()
        (policy_loss.mean()  + self.eps_threshold*ent).backward()#loss.backward()
        for param in policy_net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        optimizer.step()
        
    def optimize(self):
        self.optimize_model(self.q_net, self.target_net, self.memory, self.optimizer)
        
    def perturb_learning_rate(self, i_episode, nolast=True):
        if nolast:
            new_lr_factor = 10**np.random.normal(scale=1.0)
            new_momentum_delta = np.random.normal(scale=0.1)
            self.eps_threshold += np.random.normal(scale=0.1)
            if self.eps_threshold<0:
                self.eps_threshold = 0.00001
            self.EPS_DECAY += np.random.normal(scale=50.0)
            if self.EPS_DECAY<50:
                self.EPS_DECAY = 50
            if self.prob>=0:
                self.prob += np.random.normal(scale=0.05)-0.025
                self.prob = min(max(0,self.prob),1)
        for param_group in self.optimizer.param_groups:
            if nolast:
                param_group['lr'] *= new_lr_factor
                param_group['momentum'] += new_momentum_delta
            self.momentum =param_group['momentum']
            self.lr = param_group['lr']
        if nolast:
            new_lr_factor = 10**np.random.normal(scale=1.0)
            new_momentum_delta = np.random.normal(scale=0.1)
        for param_group in self.policy_optimizer.param_groups:
            if nolast:
                param_group['lr'] *= new_lr_factor
                param_group['momentum'] += new_momentum_delta
            self.momentum1 =param_group['momentum']
            self.lr1 = param_group['lr']
        with open(os.path.join(self.pars['results_path']+ self.name,'hyper-{}.json').format(i_episode), 'w') as outfile:
            json.dump({'lr':self.lr, 'momentum':self.momentum,
                       'lr1':self.lr1, 'momentum1':self.momentum1,'eps_decay':self.EPS_DECAY,
                       'eps_entropy':self.eps_threshold,
                       'prob':self.prob,'i_episode':i_episode}, outfile)
    def clone(self, agent):
        state_dict = agent.policy_net.state_dict()
        self.policy_net.load_state_dict(state_dict)
        state_dict = agent.policy_optimizer.state_dict()
        self.policy_optimizer.load_state_dict(state_dict)
        
        state_dict = agent.q_net.state_dict()
        self.q_net.load_state_dict(state_dict)
        state_dict = agent.optimizer.state_dict()
        self.optimizer.load_state_dict(state_dict)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.EPS_DECAY = agent.EPS_DECAY
        self.eps_threshold = agent.eps_threshold