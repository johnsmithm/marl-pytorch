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


from mas import *
import time, os, datetime, json
import copy, argparse, csv, json, datetime, os
from functools import partial
from pathlib import Path
from torch.distributions import Categorical

from models.models import DQN2D
from models.agent import Agent, AgentSep1D

from buffer import ReplayMemory

class AgentSep2D(AgentSep1D):
    def __init__(self, name, pars, nrenvs=1, job=None, experiment=None):
        Agent.__init__(self,name, pars, nrenvs, job, experiment)
    def build(self):
        self.policy_net1 = DQN2D(84,84, self.pars).to(self.device)
        self.target_net1 = DQN2D(84,84, self.pars).to(self.device)
        self.target_net1.load_state_dict(self.policy_net1.state_dict())
        self.target_net1.eval()
        
        self.policy_net2 = DQN2D(84,84, self.pars).to(self.device)
        self.target_net2 = DQN2D(84,84, self.pars).to(self.device)
        self.target_net2.load_state_dict(self.policy_net2.state_dict())
        self.target_net2.eval()
        
        self.optimizer1 = optim.SGD(
                    self.policy_net1.parameters(), lr=self.pars['lr'], 
                    momentum=self.pars['momentum'])#
        self.optimizer2 = optim.SGD(
                    self.policy_net2.parameters(), lr=self.pars['lr'], 
                    momentum=self.pars['momentum'])#
        self.optimizer1 = optim.Adam(self.policy_net1.parameters())
        self.optimizer2 = optim.Adam(self.policy_net2.parameters())
        self.memory2 = ReplayMemory(10000)
        self.memory1 = ReplayMemory(10000)
        
    def getStates(self, env):
        screen1 = env.train_render().transpose((2, 0, 1))
        screen1 = np.ascontiguousarray(screen1, dtype=np.float32) / 255
        return torch.from_numpy(screen1).unsqueeze(0).to(self.device), torch.from_numpy(screen1).unsqueeze(0).to(self.device)
    
class AgentShare2D(Agent):
    def __init__(self, name, pars, nrenvs=1, job=None, experiment=None):
        Agent.__init__(self,name, pars, nrenvs, job, experiment)
    def build(self):
        self.policy_net = DQN2D(84,84, self.pars).to(self.device)
        self.target_net = DQN2D(84,84, self.pars).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()        
        
        self.optimizer = optim.SGD(
                    self.policy_net.parameters(), lr=self.pars['lr'], 
                    momentum=self.pars['momentum'])#
        #self.optimizer = optim.Adam(self.policy_net.parameters())
        self.memory = ReplayMemory(10000)
        
        if self.pars['load'] is not None:
            self.load(self.pars['load'])
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print('loaded')
    def getStates(self, env):
        screen1 = env.train_render(0).transpose((2, 0, 1))
        screen1 = np.ascontiguousarray(screen1, dtype=np.float32) / 255
        screen2 = env.train_render(1).transpose((2, 0, 1))
        screen2 = np.ascontiguousarray(screen2, dtype=np.float32) / 255
        return torch.from_numpy(screen1).unsqueeze(0).to(self.device), torch.from_numpy(screen2).unsqueeze(0).to(self.device) 
    