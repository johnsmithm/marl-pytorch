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

    def __init__(self, inp, pars, rec=False):
        super(DQN, self).__init__()

        self.h1 = nn.Linear(inp+4, pars['h1']) 
        self.h2 = nn.Linear( pars['h1'], pars['h2']) 
        self.q = nn.Linear( pars['h2'], 4) 
        self.c = nn.Linear( pars['h2'], 4) 
        if rec:
            self.gru = nn.GRU(pars['h2'], pars['h2'])
        
    def forward(self, x, agent_index, comm, h=None, steps=1):
        
        b,s = 1,0
        if len(list(x.size()))>2:
            b = x.size(0)
            s = x.size(1)
            x = x.view(b*s,x.size(2))
        x = torch.cat([x.float(),comm.float()],dim=-1)
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        if h is not None:
            if s==0:
                s=1
                b = x.size(0)
            x = x.view(b,s,-1).permute([1,0,2])
            x, hn = self.gru(x, h.view(1,b,-1))            
            x = x[-1].view(b,-1)
            return self.q(x), self.c(x), hn
        return self.q(x), self.c(x)
    
def where(cond, x_1, x_2):
    return (cond * x_1) + ((1-cond) * x_2)
           
class DQN2D(nn.Module):

    def __init__(self, h, w, pars, rec=False):
        super(DQN2D, self).__init__()
        self.conv1 = nn.Conv2d(3, 36, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(36)
        self.conv2 = nn.Conv2d(36, 62, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(62)
        self.conv3 = nn.Conv2d(62, 62, kernel_size=2, stride=2)
        self.bn3 = nn.BatchNorm2d(62)
        self.pars = pars
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
        if rec:
            self.gru = nn.GRU(pars['en']*2, pars['en'])
        if pars['att'] ==1:
            self.att = nn.Linear(pars['en'], 1)
        #self.comm_lookup = nn.Embedding(5, 10)
        #self.agent_lookup1 = nn.Embedding(2, 32)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x, agent_index, comm, h=None, steps=1):
        #.view( comm.size(0), 4)
        z_a = F.relu(self.co(comm.float()))#self.comm_lookup(comm.view( comm.size(0)).float())
        b,s = 1,0
        if len(list(x.size()))>4:
            b = x.size(0)
            s = x.size(1)
            x = x.view(b*s,x.size(2),x.size(3),x.size(4))
        #z_a1 = self.agent_lookup1(agent_index)
        x = F.relu(self.bn1(self.conv1(x)))#+z_a.view(-1,16,1,1))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))#+z_a1.view(-1,32,1,1))
        
        if h is not None:
            #(seq_len, batch, input_size)
            x = torch.cat([F.relu(self.head(x.view(x.size(0), -1))),z_a],dim=-1)
            if s==0:
                s=1
                b = x.size(0)
            x = x.view(b,s,-1).permute([1,0,2])
            
            x, hn = self.gru(x, h.view(1,b,-1))
            
            x = x[-1].view(b,-1)
            return self.head1(x), self.co1(x), hn
        else:
            x = F.relu(self.head(x.view(x.size(0), -1)))+z_a
        if self.pars['att'] ==1:
            att = F.sigmoid(self.att(x))
            return self.head1(x), self.co1(x)*att,  att
        if self.pars['comma'] == 'no':
            return self.head1(x), self.co1(x)
        c = F.tanh(self.co1(x))
        return self.head1(x), where((c>0).float(), c+1, c-1 )#F.tanh(self.co1(x))
