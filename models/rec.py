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

from models.models import DQN2D, DRQN
from models.agent import Agent, AgentSep1D

from buffer import ReplayMemory, Memory, RecurrentExperienceReplayMemory
from mas import *

class AgentDRQNShare1D(Agent):
    def __init__(self, name, pars, nrenvs=1, job=None, experiment=None):
        Agent.__init__(self,name, pars, nrenvs, job, experiment)
        self.reset_hx()
    def build(self):        
        self.num_feats = 75
        self.policy_net = DRQN(self.num_feats, self.pars, device=self.device).to(self.device)
        self.target_net = DRQN(self.num_feats, self.pars, device=self.device).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.sequence_length = 10
        if self.pars['momentum']>0:
            self.optimizer = optim.SGD(
                    self.policy_net.parameters(), lr=self.pars['lr'], 
                    momentum=self.pars['momentum'])#
        else:
            self.optimizer = optim.Adam(self.policy_net.parameters())
        self.memory = RecurrentExperienceReplayMemory(10000, self.sequence_length)
        
        self.eps_threshold = 0.01
        self.bufs = [[] for _ in range(len(self.envs)*2)]
        self.me = []
        self.eye= np.eye(4).tolist()
        
    def getT(self, extr, k, p=0.5):
        t = []
        #print(len(extr),extr[0])
        for batch in extr:    
            t.append(batch[k] if np.random.rand()<p else [0,0,0,0.])
        #print(t[-1])
        return t
    def prep_minibatch(self):
        self.batch_size = self.BATCH_SIZE
        transitions, indices, weights = self.memory.sample(self.BATCH_SIZE)

        batch_state, batch_action, batch_reward, batch_next_state, extr = zip(*transitions)

        shape = (self.BATCH_SIZE,self.sequence_length)+(self.num_feats,)
        
        batch_comm  = self.getT(extr,0, self.prob)
        #print(batch_state)
        batch_state = torch.tensor(batch_state, device=self.device, dtype=torch.float).view(shape)
        batch_action = torch.tensor(batch_action, device=self.device, dtype=torch.long).view(self.batch_size, self.sequence_length, -1)
        batch_reward = torch.tensor(batch_reward, device=self.device, dtype=torch.float).view(self.batch_size, self.sequence_length)
        batch_comm = torch.tensor(batch_comm, device=self.device, dtype=torch.float).view(self.batch_size, self.sequence_length, 4)
        #get set of next states for end of each sequence
        batch_next_state = tuple([batch_next_state[i] for i in range(len(batch_next_state)) if (i+1)%(self.sequence_length)==0])

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch_next_state)), device=self.device, dtype=torch.uint8)
        try: #sometimes all next states are false, especially with nstep returns
            non_final_next_states = torch.tensor([s for s in batch_next_state if s is not None], device=self.device, dtype=torch.float).unsqueeze(dim=1)
            non_final_next_states = torch.cat([batch_state[non_final_mask, 1:, :], non_final_next_states], dim=1)
            empty_next_state_values = False
        except:
            empty_next_state_values = True

        return batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights,batch_comm
    
    def compute_loss(self, batch_vars):
        batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights, batch_comm = batch_vars

        #estimate
        cin = torch.zeros(len(batch_state),self.sequence_length,4, device=self.device, dtype=torch.float) 
        current_q_values, _, c = self.policy_net(batch_state, batch_comm)
        current_q_values = current_q_values.gather(2, batch_action).squeeze()
        
        #target
        with torch.no_grad():
            max_next_q_values = torch.zeros((self.batch_size, self.sequence_length), device=self.device, dtype=torch.float)
            max_next_c_values = torch.zeros((self.batch_size, self.sequence_length), device=self.device, dtype=torch.float)
            if not empty_next_state_values:
                max_next, _, c_next = self.target_net(non_final_next_states, cin[:len(non_final_next_states)])
                max_next_q_values[non_final_mask] = max_next.max(dim=2)[0]
            expected_q_values = batch_reward + (self.GAMMA*max_next_q_values)

        diff = (expected_q_values - current_q_values)
        loss = self.huber(diff)
        
        #mask first half of losses
        split = self.sequence_length // 2
        mask = torch.zeros(self.sequence_length, device=self.device, dtype=torch.float)
        mask[split:] = 1.0
        mask = mask.view(1, -1)
        loss *= mask
        
        loss = loss.mean()

        return loss  
    
    def reset_hx(self):
        #self.action_hx = self.model.init_hidden(1)
        self.seq = [np.zeros(self.num_feats) for j in range(self.sequence_length)]
        self.seq1 = [np.zeros(self.num_feats) for j in range(self.sequence_length)]
        self.seqc = [np.zeros(4) for j in range(self.sequence_length)]
        self.seqc1 = [np.zeros(4) for j in range(self.sequence_length)]
        
    def select_action(self, state, comm, policy_net):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) *         math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return policy_net(state, comm)[0][:, -1, :].max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(4)]], device=self.device, dtype=torch.long)
    def getcomm1(self, X, test):         
        
        co = co1 = torch.zeros(1,self.sequence_length,4, device=self.device, dtype=torch.float) 
                
        a, _, c = self.policy_net(X, co)
        a = a[:, -1, :] #select last element of seq
        a = a.max(1)[1]      
        
        return self.eye[a]
    
    def huber(self, x):
        cond = (x.abs() < 1.0).float().detach()
        return 0.5 * x.pow(2) * cond + (x.abs() - 0.5) * (1.0 - cond)    
    def getaction(self, state1, state2, test=False):
        mes = torch.zeros(1,self.sequence_length,4, device=self.device, dtype=torch.float)     
        self.seq.pop(0)
        self.seq.append(state1)
        self.seq1.pop(0)
        self.seq1.append(state2)
        X = torch.tensor([self.seq], device=self.device, dtype=torch.float) 
        X1 = torch.tensor([self.seq1], device=self.device, dtype=torch.float) 
        if True:      
            if test or True:
                comm1 = self.getcomm1(X, test) if 0<self.prob else  [0,0,0,0.]
                comm2 = self.getcomm1(X1, test) if 0<self.prob else  [0,0,0,0.]
            else:
                comm1 = self.getcomm1(X) if np.random.rand()<self.prob else  [0,0,0,0.]
                comm2 = self.getcomm1(X1) if np.random.rand()<self.prob else  [0,0,0,0.]                
            self.seqc.pop(0)
            self.seqc.append(comm1)        
            self.seqc1.pop(0)
            self.seqc1.append(comm2)
            c1 = torch.tensor([self.seqc], device=self.device, dtype=torch.float) 
            c2 = torch.tensor([self.seqc1], device=self.device, dtype=torch.float) 
        if test:
            action1 = self.policy_net(X, c2)[0][:, -1, :].max(1)[1].view(1, 1)
            action2 = self.policy_net(X1, c1)[0][:, -1, :].max(1)[1].view(1, 1)
        else: 
            action1 = self.select_action(X, c2, self.policy_net)
            action2 = self.select_action(X1,  c1, self.policy_net)
        self.rm = [comm1, comm2]
        comm1 = torch.tensor([comm1], device=self.device, dtype=torch.float)
        comm2 = torch.tensor([comm2], device=self.device, dtype=torch.float)
        return action1, action2, [comm1, comm2]
    
    def getStates(self, env):
        screen1 = np.reshape(env.render_env_5x5(0),(-1))#.transpose((2, 0, 1))
        screen2 = np.reshape(env.render_env_5x5(1),(-1))#.transpose((2, 0, 1))
        #print(screen1.shape)
        return screen1,screen2
    #torch.from_numpy(screen1).unsqueeze(0).to(self.device), torch.from_numpy(screen2).unsqueeze(0).to(self.device)
    
    def updateTarget(self, i_episode, step=False):
        #soft_update(self.target_net, self.policy_net, tau=0.01)
        
        if step:
            #self.reset_hx()
            #self.memory.push((None,0,0, None, [[0,0,0,0]]))
            return
        if i_episode % self.TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
    def optimize(self):
        if len(self.memory)<1000:
            return
        b = self.prep_minibatch()
        loss = self.compute_loss(b)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        #self.optimize_model(self.policy_net, self.target_net, self.memory, self.optimizer)
        
    def saveStates(self, state1, state2, action1,action2, next_state1,next_state2, reward1,reward2, env_id, done):
            self.capmem+=2
            if self.pars['ppe']!='1' and False:
                    self.memory.push(state2, action2, next_state2, reward2, state1)
                    self.memory.push(state1, action1, next_state1, reward1, state2)
            else:
                    #model.update(prev_observation, action, reward, observation,[eye[c]], frame_idx)
                    
                    self.memory.push((state1, action1.item(),reward1.item(), None if done else next_state1 , [self.rm[1]]))
                    self.me.append((state2, action2.item(),reward2.item(), None if done else next_state2, [self.rm[0]]))
                    if done:
                        for i in self.me:
                            self.memory.push(i)
                        self.me = []
                        self.reset_hx()
                      
class AgentDRQNShareDVMN1D(AgentDRQNShare1D):
    def __init__(self, name, pars, nrenvs=1, job=None, experiment=None):
        AgentDRQNShare1D.__init__(self,name, pars, nrenvs, job, experiment)
        self.reset_hx()       
    def getcomm1(self, X, test):         
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) *         math.exp(-1. * self.steps_done / self.EPS_DECAY)
        if sample > eps_threshold or test:
            with torch.no_grad():
                co =  torch.zeros(1,self.sequence_length,4, device=self.device, dtype=torch.float) 
                
                a1, _, a = self.policy_net(X, co)
                a = a[:, -1, :] #select last element of seq
                a = a.max(1)[1]      

                return self.eye[a]
        else:
            return self.eye[random.randrange(4)]
           
    def compute_loss(self, batch_vars):
        batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights, batch_comm, batch_comm_v = batch_vars

        #estimate
        cin = torch.zeros(len(batch_state),self.sequence_length,4, device=self.device, dtype=torch.float) 
        _, _, current_c_values = self.policy_net(batch_state, cin)
        current_q_values, _, _ = self.policy_net(batch_state, batch_comm)
        current_q_values = current_q_values.gather(2, batch_action).squeeze()
        #batch_comm = batch_comm.max(dim=2)[1]
        #print(batch_comm)
        current_c_values = current_c_values.gather(2, batch_comm_v).squeeze()
        
        #target
        with torch.no_grad():
            max_next_q_values = torch.zeros((self.batch_size, self.sequence_length), device=self.device, dtype=torch.float)
            max_next_c_values = torch.zeros((self.batch_size, self.sequence_length), device=self.device, dtype=torch.float)
            if not empty_next_state_values:
                max_next, _, c_next = self.target_net(non_final_next_states, cin[:len(non_final_next_states)])
                max_next_q_values[non_final_mask] = max_next.max(dim=2)[0]
                max_next_c_values[non_final_mask] = c_next.max(dim=2)[0]
            expected_q_values = batch_reward + (self.GAMMA*max_next_q_values)
            expected_c_values = batch_reward + (self.GAMMA*max_next_c_values)

        diff = (expected_q_values - current_q_values + expected_c_values - current_c_values)
        loss = self.huber(diff) #+ self.huber(expected_c_values - current_c_values)
        
        #mask first half of losses
        split = self.sequence_length // 2
        mask = torch.zeros(self.sequence_length, device=self.device, dtype=torch.float)
        mask[split:] = 1.0
        mask = mask.view(1, -1)
        loss *= mask
        
        loss = loss.mean()

        return loss  
    
    def prep_minibatch(self):
        self.batch_size = self.BATCH_SIZE
        transitions, indices, weights = self.memory.sample(self.BATCH_SIZE)

        batch_state, batch_action, batch_reward, batch_next_state, extr = zip(*transitions)

        shape = (self.BATCH_SIZE,self.sequence_length)+(self.num_feats,)
        
        batch_comm  = self.getT(extr,0, self.prob)
        batch_comm_v  = self.getT(extr,1,2)
        #print(batch_comm_v)
        batch_state = torch.tensor(batch_state, device=self.device, dtype=torch.float).view(shape)
        batch_action = torch.tensor(batch_action, device=self.device, dtype=torch.long).view(self.batch_size, self.sequence_length, -1)
        batch_reward = torch.tensor(batch_reward, device=self.device, dtype=torch.float).view(self.batch_size, self.sequence_length)
        batch_comm = torch.tensor(batch_comm, device=self.device, dtype=torch.float).view(self.batch_size, self.sequence_length, 4)
        batch_comm_v =torch.tensor(batch_comm_v, device=self.device, dtype=torch.long).view(self.batch_size, self.sequence_length, 1)
        #get set of next states for end of each sequence
        batch_next_state = tuple([batch_next_state[i] for i in range(len(batch_next_state)) if (i+1)%(self.sequence_length)==0])

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch_next_state)), device=self.device, dtype=torch.uint8)
        try: #sometimes all next states are false, especially with nstep returns
            non_final_next_states = torch.tensor([s for s in batch_next_state if s is not None], device=self.device, dtype=torch.float).unsqueeze(dim=1)
            non_final_next_states = torch.cat([batch_state[non_final_mask, 1:, :], non_final_next_states], dim=1)
            empty_next_state_values = False
        except:
            empty_next_state_values = True

        return batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights,batch_comm, batch_comm_v
    def saveStates(self, state1, state2, action1,action2, next_state1,next_state2, reward1,reward2, env_id, done):
            self.capmem+=2
            if self.pars['ppe']!='1' and False:
                    self.memory.push(state2, action2, next_state2, reward2, state1)
                    self.memory.push(state1, action1, next_state1, reward1, state2)
            else:
                    #model.update(prev_observation, action, reward, observation,[eye[c]], frame_idx)
                    
                    self.memory.push((state1, action1.item(),reward1.item(), None if done else next_state1 ,
                                      [self.rm[1],np.argmax(self.rm[0])]))
                    self.me.append((state2, action2.item(),reward2.item(), None if done else next_state2,
                                    [self.rm[0],np.argmax(self.rm[1])]))
                    if done:
                        for i in self.me:
                            self.memory.push(i)
                        self.me = []
                        self.reset_hx()
class AgentDRQNShareDVMNz1D(AgentDRQNShareDVMN1D):
    def __init__(self, name, pars, nrenvs=1, job=None, experiment=None):
        AgentDRQNShareDVMN1D.__init__(self,name, pars, nrenvs, job, experiment)
        self.reset_hx()              
    def compute_loss(self, batch_vars):
        batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights, batch_comm, batch_comm_v, batch_comm_v1 = batch_vars

        #estimate
        cin = torch.zeros(len(batch_state),self.sequence_length,4, device=self.device, dtype=torch.float) 
        _, _, current_c_values2 = self.policy_net(batch_state, cin)
        current_q_values, _, _ = self.policy_net(batch_state, batch_comm)
        current_q_values = current_q_values.gather(2, batch_action).squeeze()
        #batch_comm = batch_comm.max(dim=2)[1]
        #print(batch_comm)
        current_c_values = current_c_values2.gather(2, batch_comm_v).squeeze()
        current_c_values1 = current_c_values2[...,2:].gather(2, batch_comm_v1).squeeze()
        
        #target
        with torch.no_grad():
            max_next_q_values = torch.zeros((self.batch_size, self.sequence_length), device=self.device, dtype=torch.float)
            max_next_c_values = torch.zeros((self.batch_size, self.sequence_length), device=self.device, dtype=torch.float)
            max_next_c_values1 = torch.zeros((self.batch_size, self.sequence_length), device=self.device, dtype=torch.float)
            if not empty_next_state_values:
                max_next, _, c_next = self.target_net(non_final_next_states, cin[:len(non_final_next_states)])
                max_next_q_values[non_final_mask] = max_next.max(dim=2)[0]
                max_next_c_values[non_final_mask] = c_next[...,:2].max(dim=2)[0]
                max_next_c_values1[non_final_mask] = c_next[...,2:].max(dim=2)[0]
            expected_q_values = batch_reward + (self.GAMMA*max_next_q_values)
            expected_c_values = batch_reward + (self.GAMMA*max_next_c_values)
            expected_c_values1 = batch_reward + (self.GAMMA*max_next_c_values1)

        diff = (expected_q_values - current_q_values + expected_c_values - current_c_values+ expected_c_values1- current_c_values1)
        loss = self.huber(diff) #+ self.huber(expected_c_values - current_c_values)
        
        #mask first half of losses
        split = self.sequence_length // 2
        mask = torch.zeros(self.sequence_length, device=self.device, dtype=torch.float)
        mask[split:] = 1.0
        mask = mask.view(1, -1)
        loss *= mask
        
        loss = loss.mean()

        return loss  
    
    def getV(self, q, n):
        o = [0 for i in range(n)]
        for i in range(q.size(-1)//2):
            a = q[...,i*2:i*2+2].max(1)[1] 
            
            o[i] = 1 if a==1 else -1
        return o
         
    def getcomm1(self, X, test):         
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) *         math.exp(-1. * self.steps_done / self.EPS_DECAY)
        if sample > eps_threshold or test:
            with torch.no_grad():
                co =  torch.zeros(1,self.sequence_length,4, device=self.device, dtype=torch.float) 
                
                a1, _, a = self.policy_net(X, co)
                a = a[:, -1, :] #select last element of seq
                return self.getV(a, 4)
        else:
            o = [0 for i in range(4)]
            for i in range(2):
                if np.random.rand()<0.5:
                    o[i] = 1
                else:
                    o[i] = -1
            return o
    def getI(self, c, i):
        return 0 if c[i]<0 else 1
    def prep_minibatch(self):
        self.batch_size = self.BATCH_SIZE
        transitions, indices, weights = self.memory.sample(self.BATCH_SIZE)

        batch_state, batch_action, batch_reward, batch_next_state, extr = zip(*transitions)

        shape = (self.BATCH_SIZE,self.sequence_length)+(self.num_feats,)
        
        batch_comm  = self.getT(extr,0, self.prob)
        batch_comm_v  = self.getT(extr,1,2)
        batch_comm_v1  = self.getT(extr,2,2)
        #print(batch_comm_v)
        batch_state = torch.tensor(batch_state, device=self.device, dtype=torch.float).view(shape)
        batch_action = torch.tensor(batch_action, device=self.device, dtype=torch.long).view(self.batch_size, self.sequence_length, -1)
        batch_reward = torch.tensor(batch_reward, device=self.device, dtype=torch.float).view(self.batch_size, self.sequence_length)
        batch_comm = torch.tensor(batch_comm, device=self.device, dtype=torch.float).view(self.batch_size, self.sequence_length, 4)
        batch_comm_v =torch.tensor(batch_comm_v, device=self.device, dtype=torch.long).view(self.batch_size, self.sequence_length, 1)
        batch_comm_v1 =torch.tensor(batch_comm_v1, device=self.device, dtype=torch.long).view(self.batch_size, self.sequence_length, 1)
        #get set of next states for end of each sequence
        batch_next_state = tuple([batch_next_state[i] for i in range(len(batch_next_state)) if (i+1)%(self.sequence_length)==0])

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch_next_state)), device=self.device, dtype=torch.uint8)
        try: #sometimes all next states are false, especially with nstep returns
            non_final_next_states = torch.tensor([s for s in batch_next_state if s is not None], device=self.device, dtype=torch.float).unsqueeze(dim=1)
            non_final_next_states = torch.cat([batch_state[non_final_mask, 1:, :], non_final_next_states], dim=1)
            empty_next_state_values = False
        except:
            empty_next_state_values = True

        return batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights,batch_comm, batch_comm_v, batch_comm_v1
    def saveStates(self, state1, state2, action1,action2, next_state1,next_state2, reward1,reward2, env_id, done):
            self.capmem+=2
            if self.pars['ppe']!='1' and False:
                    self.memory.push(state2, action2, next_state2, reward2, state1)
                    self.memory.push(state1, action1, next_state1, reward1, state2)
            else:
                    #model.update(prev_observation, action, reward, observation,[eye[c]], frame_idx)
                    
                    self.memory.push((state1, action1.item(),reward1.item(), None if done else next_state1 ,
                                      [self.rm[1], self.getI(self.rm[0],0), self.getI(self.rm[0],1) ]))
                    self.me.append((state2, action2.item(),reward2.item(), None if done else next_state2,
                                    [self.rm[0], self.getI(self.rm[1],0), self.getI(self.rm[1],1)]))
                    if done:
                        for i in self.me:
                            self.memory.push(i)
                        self.me = []
                        self.reset_hx()