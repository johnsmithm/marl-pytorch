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

from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from mas import *
import time, os, datetime, json
import copy, argparse, csv, json, datetime, os
from functools import partial
from pathlib import Path
from torch.distributions import Categorical

from models.models import DQN2D
from models.agent import Agent, AgentSep1D
from models.agentAC import AgentACShare2D

from buffer import ReplayMemory, Memory

class AgentSACShare2D(AgentACShare2D):
    def __init__(self, name, pars, nrenvs=1, job=None, experiment=None):
        Agent.__init__(self,name, pars, nrenvs, job, experiment)
        self.coef_value = 0.5
    def optimize_policy(self, policy_net, memories, optimizer):
        policy_loss = 0
        value_loss = 0
        ent = 0
        nr=0
        for memory in memories:#[memory1, memory2]:
            R = torch.zeros(1, 1, device=self.device)
            #GAE = torch.zeros(1, 1, device=self.device)
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
                    ac = actionV[i]# (actionV[i] - mu) / (std + eps)#actionV[i]#also use mu and std
                    #Discounted Sum of Future Rewards + reward for the given state
                    R = self.GAMMA * R + (r.float() - mu) / (std + eps)
                    advantage = R - ac
                    policy_loss += -log_prob *advantage .detach()
                    value_loss += advantage**2
                    ent += entr#*0
                    nr+=1
                    
        optimizer.zero_grad()
        (policy_loss.mean()/nr + value_loss/nr*self.coef_value  + self.eps_threshold*ent/nr).backward()
        for param in policy_net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        optimizer.step()
    
    def optimize(self):
        pass
        #self.optimize_model(self.q_net, self.target_net, self.memory, self.optimizer)
        