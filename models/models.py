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

class DQN(nn.Module):

    def __init__(self, inp, pars):
        super(DQN, self).__init__()

        self.h1 = nn.Linear(inp+4, pars['h1']) 
        self.h2 = nn.Linear( pars['h1'], pars['h2']) 
        self.q = nn.Linear( pars['h2'], 4) 
        self.c = nn.Linear( pars['h2'], 4) 
        
    def forward(self, x, agent_index, comm):
        x = torch.cat([x.float(),comm.float()],dim=-1)
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        return self.q(x), self.c(x)
    
           
class DQN2D(nn.Module):

    def __init__(self, h, w, pars):
        super(DQN2D, self).__init__()
        self.conv1 = nn.Conv2d(3, 36, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(36)
        self.conv2 = nn.Conv2d(36, 62, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(62)
        self.conv3 = nn.Conv2d(62, 62, kernel_size=2, stride=2)
        self.bn3 = nn.BatchNorm2d(62)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)),3)
        linear_input_size = 5022# convw * convh * 62
        self.head = nn.Linear(linear_input_size, pars['en']) # 448 or 512
        self.head1 = nn.Linear(pars['en'], 4) # 448 or 512
        
        self.co = nn.Linear(4, pars['en'])
        self.co1 = nn.Linear(pars['en'], 4)
        #self.comm_lookup = nn.Embedding(5, 10)
        #self.agent_lookup1 = nn.Embedding(2, 32)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x, agent_index, comm):
        z_a = F.relu(self.co(comm.view( comm.size(0), 4).float()))#self.comm_lookup(comm.view( comm.size(0)).float())
        #z_a1 = self.agent_lookup1(agent_index)
        x = F.relu(self.bn1(self.conv1(x)))#+z_a.view(-1,16,1,1))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))#+z_a1.view(-1,32,1,1))
        x = F.relu(self.head(x.view(x.size(0), -1)))+z_a
        return self.head1(x), F.relu(self.co1(x))
