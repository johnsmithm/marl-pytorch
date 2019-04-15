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
from models.att import MultiHeadAttention, Attention

import time, os, datetime, json
import copy, argparse, csv, json, datetime, os
from functools import partial
from pathlib import Path
from torch.distributions import Categorical



class DRQN(nn.Module):
    def __init__(self, input_shape, pars, num_actions=4, gru_size=512, bidirectional=False, body=None, device=None):
        super(DRQN, self).__init__()
        self.device = device
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.gru_size = gru_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1
        feature_size = 128
        self.body = nn.Linear(input_shape, feature_size)
        self.gru = nn.GRU(feature_size, self.gru_size, num_layers=1, batch_first=True, bidirectional=bidirectional)
        #self.fc1 = nn.Linear(self.body.feature_size(), self.gru_size)
        self.fc2 = nn.Linear(self.gru_size, self.num_actions)
        self.c = nn.Linear(self.gru_size, self.num_actions)
        self.cin = nn.Linear( self.num_actions, feature_size)
        
    def forward(self, x, cin, hx=None):
        batch_size = x.size(0)
        sequence_length = x.size(1)
        
        x = x.view((-1,)+(self.input_shape,))
        
        #format outp for batch first gru
        feats = self.body(x).view(batch_size, sequence_length, -1)
        cfeats = self.cin(cin).view(batch_size, sequence_length, -1)
        hidden = self.init_hidden(batch_size) if hx is None else hx
        out, hidden = self.gru(feats+cfeats, hidden)
        x = self.fc2(out)
        

        return x, hidden, self.c(out)
        #return x

    def init_hidden(self, batch_size):
        return torch.zeros(1*self.num_directions, batch_size, self.gru_size, device=self.device, dtype=torch.float)
    
    def sample_noise(self):
        pass



class DQN(nn.Module):

    def __init__(self, inp, pars, rec=False):
        super(DQN, self).__init__()
        self.pars= pars
        self.h1 = nn.Linear(inp+4, pars['h1']) 
        self.h2 = nn.Linear( pars['h1'], pars['h2']) 
        self.q = nn.Linear( pars['h2'], 4) 
        self.c = nn.Linear( pars['h2'], 4) 
        if rec:
            self.gru = nn.GRU(pars['h2'], pars['h2'], batch_first = True)
        
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
            x = x.view(b,s,-1)#.transpose(1,0)
            x, hn = self.gru(x, h.view(1,b,-1))            
            #x = x[-1].view(b,-1)
            x = x.contiguous().view(-1, self.pars['h2'])
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
        if pars['matt'] ==1:    
            self.matt = MultiHeadAttention(n_head=4, d_model=62, d_k=16, d_v=16)
        if pars['matt'] ==2:    
            self.matt = MultiHeadAttention(n_head=8, d_model=pars['en'], d_k=32, d_v=32)
        if pars['matt'] in [3,4]:    
            self.matt = Attention(encoder_dim=62, decoder_dim=4, attention_dim=60)
            self.matt1 = Attention(encoder_dim=62, decoder_dim=62, attention_dim=60)
            self.matt2 = Attention(encoder_dim=62, decoder_dim=4, attention_dim=30)
            self.head1 = nn.Linear(62, 4) # 448 or 512
            #self.up = nn.Linear(32, 62) # 448 or 512
            self.co1 = nn.Linear(62, 4)
        if pars['matt'] in [5,6]:    
            self.matt = Attention(encoder_dim=62, decoder_dim=4, attention_dim=60)
            self.matt1 = Attention(encoder_dim=62, decoder_dim=62, attention_dim=60)
            self.matt2 = Attention(encoder_dim=62, decoder_dim=62, attention_dim=60)
            self.head1 = nn.Linear(62, 4) # 448 or 512
            #self.up = nn.Linear(32, 62) # 448 or 512
            self.co1 = nn.Linear(62, 4)
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
        x2 = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x2)))#+z_a1.view(-1,32,1,1))
        
        if self.pars['matt'] ==3:
            x1 = x.permute(0,2,3,1).contiguous().view(-1, 81, 62)
            x4, self.attn1 = self.matt(x1, comm.float())
            x3, self.attn2 = self.matt1(x1, x4)
            #print(x2.size())
            x5, self.attn = self.matt2(x2.permute(0,2,3,1).contiguous().view(-1, 19*19, 62), comm.float())
            
            #x5 = self.up(x)
            #print(x.size())
            return self.head1(x5+x3+x4), self.co1(x5+x3+x4)
        if self.pars['matt'] ==4:
            x1 = x.permute(0,2,3,1).contiguous().view(-1, 81, 62)
            x4, self.attn1 = self.matt(x1, comm.float())
            x3, self.attn2 = self.matt1(x1, x4)
            #print(x2.size())
            x5, self.attn = self.matt2(x2.permute(0,2,3,1).contiguous().view(-1, 19*19, 62), comm.float())
            
            #x5 = self.up(x)
            #print(x.size())
            return self.head1(x5+x4), self.co1(x3)
        if self.pars['matt'] ==5:
            x1 = x.permute(0,2,3,1).contiguous().view(-1, 81, 62)
            x4, self.attn1 = self.matt(x1, comm.float())
            x3, self.attn2 = self.matt1(x1, x4)
            #print(x2.size())
            x5, self.attn = self.matt2(x2.permute(0,2,3,1).contiguous().view(-1, 19*19, 62), x4)
            
            #x5 = self.up(x)
            #print(x.size())
            return self.head1(x5+x4), self.co1(x3)
        if self.pars['matt'] ==6:
            x1 = x.permute(0,2,3,1).contiguous().view(-1, 81, 62)
            x4, self.attn1 = self.matt(x1, comm.float())
            x3, self.attn2 = self.matt1(x1,  comm.float())
            #print(x2.size())
            x5, self.attn = self.matt2(x2.permute(0,2,3,1).contiguous().view(-1, 19*19, 62), comm.float())
            
            #x5 = self.up(x)
            #print(x.size())
            return self.head1(x5+x4), self.co1(x3)
        
        if self.pars['matt'] ==1:    
            x1 = x.permute(0,2,3,1).contiguous().view(-1, 81, 62)
            x, self.attn = self.matt(x1, x1, x1)
            #x = F.relu(self.head(x.view(x.size(0), -1)))+z_a
            #print(output.size(), x.size())
            #return self.head1(x), self.co1(output.view(-1, 62))
            #x = output.view(-1, self.pars['en'])
            
        if h is not None:
            #(seq_len, batch, input_size)
            x = torch.cat([F.relu(self.head(x.view(x.size(0), -1))),z_a],dim=-1)
            if s==0:
                s=1
                b = x.size(0)
            x = x.view(b,s,-1).permute([1,0,2])
            
            x, hn = self.gru(x, h.view(1,b,-1))
            
            #x = x[-1].view(b,-1)
            #print(x.size())
            x = x.view(-1, self.pars['en'])
            return self.head1(x), self.co1(x), hn
        else:
            x = F.relu(self.head(x.view(x.size(0), -1)))+z_a
        if self.pars['att'] ==1:
            att = F.sigmoid(self.att(x))
            return self.head1(x), self.co1(x)*att,  att
        if self.pars['matt'] ==2:    
            x1 = x.view(-1, 1,  self.pars['en'])
            output, self.attn = self.matt(x1, x1, x1)
            #print(output.size(), x.size())
            x = output.view(-1, self.pars['en'])
            
            
            
        if self.pars['comma'] == 'no':
            return self.head1(x), self.co1(x)
        c = F.tanh(self.co1(x))
        return self.head1(x), where((c>0).float(), c+1, c-1 )#F.tanh(self.co1(x))
    