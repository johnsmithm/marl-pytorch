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

from models.models import DQN2D, DQN
from models.agent import Agent, AgentSep1D

from buffer import ReplayMemory, Memory

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
        #self.optimizer1 = optim.Adam(self.policy_net1.parameters())
        #self.optimizer2 = optim.Adam(self.policy_net2.parameters())
        self.memory2 = ReplayMemory(10000)
        self.memory1 = ReplayMemory(10000)
        
        if self.pars['ppe'] == '1':
            self.memory1 = Memory(10000)
            self.memory2 = Memory(10000)
    def getStates(self, env):
        screen1 = env.train_render().transpose((2, 0, 1))
        screen1 = np.ascontiguousarray(screen1, dtype=np.float32) / 255
        return torch.from_numpy(screen1).unsqueeze(0).to(self.device), torch.from_numpy(screen1).unsqueeze(0).to(self.device)
    
class AgentShare2D(Agent):
    def __init__(self, name, pars, nrenvs=1, job=None, experiment=None):
        Agent.__init__(self,name, pars, nrenvs, job, experiment)
    def build(self):
        self.policy_net = DQN2D(84,84, self.pars, rec=self.pars['rec']==1).to(self.device)
        self.target_net = DQN2D(84,84, self.pars, rec=self.pars['rec']==1).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()        
        
        if self.pars['momentum']>0:
            self.optimizer = optim.SGD(
                    self.policy_net.parameters(), lr=self.pars['lr'], 
                    momentum=self.pars['momentum'])#
        if self.pars['momentum']<0:
            self.optimizer = optim.Adam(self.policy_net.parameters())
        self.memory = ReplayMemory(10000)
        print(12)
        if self.pars['load'] is not None:
            self.load(self.pars['load'])
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print('loaded')
        if self.pars['ppe'] == '1':
            self.memory = Memory(10000)
    def getStates(self, env):
        screen1 = env.train_render(0).transpose((2, 0, 1))
        screen1 = np.ascontiguousarray(screen1, dtype=np.float32) / 255
        screen2 = env.train_render(1).transpose((2, 0, 1))
        screen2 = np.ascontiguousarray(screen2, dtype=np.float32) / 255
        return torch.from_numpy(screen1).unsqueeze(0).to(self.device), torch.from_numpy(screen2).unsqueeze(0).to(self.device) 


class AgentShareRec2D(AgentShare2D):
    def __init__(self, name, pars, nrenvs=1, job=None, experiment=None):
        Agent.__init__(self,name, pars, nrenvs, job, experiment)
        bs = 1
        self.h2  = torch.zeros(1, bs, pars['en'], device=self.device)
        self.h1  = torch.zeros(1, bs, pars['en'], device=self.device)
        self.rnnB = 10
    def select_action(self, state, comm, policy_net, h):    
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) *         math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        ac,_, h1 = policy_net(state, 1, comm, h=h)
        if sample > eps_threshold:
            with torch.no_grad():
                return ac.max(1)[1].view(1, 1), h1
        else:
            return torch.tensor([[random.randrange(4)]], device=self.device, dtype=torch.long), h1
    
    def getComm(self, mes, policy_net, state1_batch, h):
        return self.policy_net(state1_batch, 1, mes, h=h)[self.idC].detach() if np.random.rand()<self.prob else mes
    
    def getaction(self, state1, state2, test=False):
        mes = torch.tensor([[0,0,0,0]], device=self.device)            
        if test:
            comm2 = self.policy_net(state2, 0, mes, self.h2)[self.idC].detach() if 0<self.prob else mes
            comm1 = self.policy_net(state1, 0, mes, self.h1)[self.idC].detach() if 0<self.prob else mes
            actio1,_,h1 = self.policy_net(state1, 1, comm2, self.h1)
            action1= actio1.max(1)[1].view(1, 1)
            actio2,_,h2 = self.policy_net(state2, 1, comm1, self.h2)
            action2 = actio2.max(1)[1].view(1, 1)
        else:
            comm2 = self.policy_net(state2, 0, mes, self.h2)[self.idC].detach() if np.random.rand()<self.prob else mes
            comm1 = self.policy_net(state1, 0, mes, self.h1)[self.idC].detach() if np.random.rand()<self.prob else mes
            action1,h1 = self.select_action(state1, comm2, self.policy_net, self.h1)
            action2,h2 = self.select_action(state2,  comm1, self.policy_net, self.h2)
            self.rnnS1.append([state1,state2, self.h1.detach(),self.h2.detach(), comm2, action1])
            self.rnnS2.append([state2,state1, self.h2.detach(),self.h1.detach(), comm1, action2])
        
        self.h2  = h2
        self.h1  = h1
        return action1, action2, [comm1, comm2]
    def updateTarget(self, i_episode, step=False):
        #soft_update(self.target_net, self.policy_net, tau=0.01)
        if step:
            return
        #self.optimize_policy(self.policy_net, self.bufs, self.policy_optimizer)        
        bs = 1
        self.h2  = torch.zeros(1, bs, self.pars['en'], device=self.device)
        self.h1  = torch.zeros(1, bs, self.pars['en'], device=self.device)
        if i_episode % self.TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            #self.eps_threshold -= 0.001
    def getT(self, rnns, k):
        t = []
        for s in rnns:
            t.extend([i[k] for i in s])
        return t#bxs
    def getOpStates(self, rnns):        
        s1 = torch.cat(self.getT(rnns, 0)).view(self.BATCH_SIZE,self.rnnB, 3, 84,84)#sxbx 80x80x3
        s2 = torch.cat(self.getT(rnns, 1)).view(self.BATCH_SIZE,self.rnnB, 3, 84,84)#sxbx 80x80x3
        return s1,s2
    def optimize_model(self , policy_net, target_net, memory, optimizer):
        if self.pars['ppe']!='1' and  len(memory) < self.BATCH_SIZE:
            return          
        if self.pars['ppe']=='1' and self.capmem< self.BATCH_SIZE:
            return          
        
        if self.pars['ppe']=='1':
            #state1, action1, next_state1, reward1, state2
            tree_idx, batch, ISWeights_mb = memory.sample(self.BATCH_SIZE)
            non_final_next_states = torch.cat( [i[2] for i in batch])
            state_batch = torch.cat( [i[0] for i in batch])
            action_batch = torch.cat( [i[1] for i in batch])
            reward_batch = torch.cat( [i[3] for i in batch])
            state1_batch = torch.cat( [i[4] for i in batch])
            
        else:
            transitions = memory.sample(self.BATCH_SIZE)
            batch = Transition(*zip(*transitions))
            
            non_final_next_states = torch.cat( batch.next_state)
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)
            state1_batch = torch.cat(batch.agent_index)
        
        rnns =  [i[5] for i in batch]
        h1 = torch.cat([b[0][2] for b in rnns])#bx100
        h2 = torch.cat([b[0][3] for b in rnns])#bx100
        #print([len(b) for b in rnns])
        c = torch.cat([i[4].float()  for b in rnns for i in b]).view(self.BATCH_SIZE*self.rnnB, 4)#sxbx4
        a2 = torch.cat(self.getT(rnns, 5))#.view(self.BATCH_SIZE,self.rnnB).permute([1,0]).view(self.BATCH_SIZE*self.rnnB, 1)#sxbx1
        r2 = torch.cat(self.getT(rnns, 6))#.view(self.BATCH_SIZE,self.rnnB).permute([1,0]).view(self.BATCH_SIZE*self.rnnB)#sxb
        
        #a2 = a1.view(self.BATCH_SIZE,self.rnnB).transpose(1,0)#
        #print(a.size(),r.size())
        a = a2.contiguous().view(-1, 1)
        #r2 = r1.view(self.BATCH_SIZE,self.rnnB).transpose(1,0)
        r = r2.contiguous().view(-1)#sxb
        s1, s2 = self.getOpStates(rnns)
        mes = torch.tensor([[[0,0,0,0] for i in range(self.BATCH_SIZE)] for j in range(self.rnnB)], device=self.device)#sxbx4
        #_, c, _ = policy_net(s2, 1, mes, h=h2)
        probs1, _, hs1 = policy_net(s1, 1, c, h=h1)
        '''
        hs2 = [];hs1 = []
        probs1 = []
        mes = torch.tensor([[0,0,0,0] for i in range(1)], device=self.device)
        for st in rnns:
            s1,s2,h1,h2 = st[0][:4]
            p , c1, h2 = policy_net(s2, 1, mes, h=h2)
            cc = [p,c1]
            c = cc[self.idC] if np.random.rand()<self.prob else mes
            if self.pars['comm'] =='2':
                c = c.detach()
            pr, _, h1 = policy_net(s1, 1, c, h=h1)
            for s in st[1:]:
                s1,s2,_,_ = s[:4]
                p,c1, h2 = policy_net(s2, 1, mes, h=h2)
                cc = [p,c1]
                c = cc[self.idC] if np.random.rand()<self.prob else mes
                if self.pars['comm'] =='2':
                    c = c.detach()
                pr, _, h1 = policy_net(s1, 1, c, h=h1)
            probs1.append(pr)
            hs1.append(h1)
            #hs2.append(h2)
        hs1 = torch.cat(hs1).view(1,len(rnns),-1)
        probs1 = torch.cat(probs1)
        '''
        #print(probs1.size())
        #hs2 = torch.cat(hs2)
        mes = torch.tensor([[0,0,0,0] for i in range(self.BATCH_SIZE)], device=self.device)
        #comm = self.getComm(mes, policy_net,state1_batch, hs2)
        #if self.pars['comm'] =='2':
        #    comm = comm.detach()
        #state_action_values,_,hs11 = policy_net(state_batch, 1, comm, hs1)
        state_action_values= probs1.gather(1, a)

        next_state_values = target_net(non_final_next_states, 1, mes, hs1)[0].max(1)[0].detach()
        expected_state_action_values1 = (next_state_values * self.GAMMA) + reward_batch.float()
        n_v = probs1[self.BATCH_SIZE:].max(1)[0].detach()*self.GAMMA + r[:-self.BATCH_SIZE].float()
        #print(probs1.size(),n_v.size(), expected_state_action_values.size(),state_action_values.size(), r.size(),  probs1[:self.BATCH_SIZE].max(1)[0].size())
        expected_state_action_values = torch.cat([ n_v, expected_state_action_values1])
        #expected_state_action_values[:self.BATCH_SIZE*self.rnnB//2] = 0
        #state_action_values[:self.BATCH_SIZE*self.rnnB//2] *= 0
        loss = F.smooth_l1_loss(state_action_values[self.BATCH_SIZE*self.rnnB//2:],
                                expected_state_action_values.unsqueeze(1)[self.BATCH_SIZE*self.rnnB//2:])
        #loss = weighted_mse_loss(state_action_values, expected_state_action_values.unsqueeze(1), 
        #                         torch.tensor(ISWeights_mb, device=self.device))
        #print(torch.tensor(ISWeights_mb, device=self.device).size())

        if self.pars['ppe']=='1':
            absolute_errors = (state_action_values-expected_state_action_values.unsqueeze(1)).abs().cpu().data.numpy().reshape((-1))
            memory.batch_update(tree_idx, absolute_errors)
        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        optimizer.step()
class AgentShareRec1D(AgentShareRec2D):       
    def __init__(self, name, pars, nrenvs=1, job=None, experiment=None):
        AgentShareRec2D.__init__(self,name, pars, nrenvs, job, experiment)
        self.rnnB = 10
    
    def getOpStates(self, rnns):        
        s1 = torch.cat(self.getT(rnns, 0)).view(self.BATCH_SIZE,self.rnnB, 75)#sxbx 80x80x3
        s2 = torch.cat(self.getT(rnns, 1)).view(self.BATCH_SIZE,self.rnnB, 75)#sxbx 80x80x3
        return s1,s2
    def build(self):
        self.policy_net = DQN(75, self.pars, rec=self.pars['rec']==1).to(self.device)
        self.target_net = DQN(75, self.pars, rec=self.pars['rec']==1).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        if self.pars['momentum']>0:
            self.optimizer = optim.SGD(
                    self.policy_net.parameters(), lr=self.pars['lr'], 
                    momentum=self.pars['momentum'])#
        else:
            self.optimizer = optim.Adam(self.policy_net.parameters())
        self.memory = ReplayMemory(10000)
        if 'ppe' in self.pars:
            self.memory = Memory(10000)
        if self.pars['load'] is not None:
            self.load(self.pars['load'])
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print('loaded')
    def getStates(self, env):
        screen1 = env.render_env_5x5(0).transpose((2, 0, 1))
        screen1 = np.ascontiguousarray(screen1, dtype=np.float32) / 255
        screen2 = env.render_env_5x5(1).transpose((2, 0, 1))
        screen2 = np.ascontiguousarray(screen2, dtype=np.float32) / 255
        return torch.from_numpy(screen1).unsqueeze(0).to(self.device).view(1,-1), torch.from_numpy(screen2).unsqueeze(0).to(self.device) .view(1,-1)
    
class Agent2DDecomposeQ(AgentShare2D):       
    def __init__(self, name, pars, nrenvs=1, job=None, experiment=None):
        AgentShare2D.__init__(self,name, pars, nrenvs, job, experiment)
    
    def optimize_model(self , policy_net, target_net, memory, optimizer):
    
        if self.pars['ppe']!='1' and  len(memory) < self.BATCH_SIZE:
            return          
        if self.pars['ppe']=='1' and self.capmem< self.BATCH_SIZE:
            return          
        
        if self.pars['ppe']=='1':
            #state1, action1, next_state1, reward1, state2
            tree_idx, batch, ISWeights_mb = memory.sample(self.BATCH_SIZE)
            non_final_next_states = torch.cat( [i[2] for i in batch])
            non_final_next_states1 = torch.cat( [i[5] for i in batch])
            state_batch = torch.cat( [i[0] for i in batch])
            action_batch = torch.cat( [i[1] for i in batch])
            action_batch1 = torch.cat( [i[6] for i in batch])
            reward_batch = torch.cat( [i[3].float() for i in batch])
            state1_batch = torch.cat( [i[4] for i in batch])
        else:
            transitions = memory.sample(self.BATCH_SIZE)
            batch = Transition(*zip(*transitions))
            
            non_final_next_states = torch.cat( batch.next_state)
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)
            state1_batch = torch.cat(batch.agent_index)

        mes = torch.tensor([[0,0,0,0] for i in range(self.BATCH_SIZE)], device=self.device)
        
        if self.pars['att'] == 1:
            _,comm1, att1 = policy_net(state1_batch, 1, mes)
            _,comm, att = policy_net(state_batch, 1, mes)
            if np.random.rand()<0.0001:
                print(att.cpu().data.numpy()[:10,0])
        else:
            comm1 = self.getComm(mes, policy_net,state1_batch)
            comm = self.getComm(mes, policy_net,state_batch)
        if self.pars['comm'] =='2':
            comm = comm.detach()
            comm1 = comm1.detach()
            
        q,_ = policy_net(state_batch, 1, comm1)[:2]
        q1,_ = policy_net(state1_batch, 1, comm)[:2]
        state_action_values = q.gather(1, action_batch)
        state_action_values1 = q1.gather(1, action_batch1)

        next_state_values = target_net(non_final_next_states, 1, mes)[0].max(1)[0].detach()
        next_state_values1 = target_net(non_final_next_states1, 1, mes)[0].max(1)[0].detach()
        
        expected_state_action_values = ((next_state_values+next_state_values1) * self.GAMMA) + reward_batch.float()
        
        loss = F.smooth_l1_loss(state_action_values+state_action_values1, expected_state_action_values.unsqueeze(1))
        #loss = weighted_mse_loss(state_action_values, expected_state_action_values.unsqueeze(1), 
        #                         torch.tensor(ISWeights_mb, device=self.device))
        #print(torch.tensor(ISWeights_mb, device=self.device).size())

        if self.pars['ppe']=='1':
            absolute_errors = (state_action_values+state_action_values1-expected_state_action_values.unsqueeze(1)).abs().cpu().data.numpy().reshape((-1))
            memory.batch_update(tree_idx, absolute_errors)
        # Optimize the model
        if self.pars['att'] == 1:
            loss = loss+ att.mean()*0.001
        if self.pars['commr'] ==1:
            comm1 = torch.flip(comm.detach(), [0])
            q1 = policy_net(state_batch, 1, comm1)[0]
            #print(comm.detach(), comm1)
            #F.smooth_l1_loss(comm.detach().float(), comm1.float())# F.smooth_l1_loss(q1,q)#
            dc = 0.1*((comm.detach().float()- comm1.float())**2).mean(-1) #F.kl_div(comm.detach().float(), comm1.float())
            dq = ((q1- q)**2).mean(-1)#F.kl_div(q1, q)
            loss = loss + 0.01*((dc-dq)**2).mean()
            
            if np.random.rand()<0.0005:
                print('difc',dc.cpu().data.numpy()[:10])
                print('difq',dq.cpu().data.numpy()[:10])
                
        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        optimizer.step()
            
    def saveStates(self, state1, state2, action1,action2, next_state1,next_state2, reward1,reward2, env_id):
            self.capmem+=2
            if self.pars['ppe']!='1':
                    self.memory.push(state2, action2, next_state2, reward2, state1)
                    self.memory.push(state1, action1, next_state1, reward1, state2)
            else:
                    self.alpha = 1
                    if self.pars['rec']==1 and len(self.rnnS1)<self.rnnB:#always full steps
                        self.rnnS1 = [self.rnnS1[0]]*(self.rnnB-len(self.rnnS1)+1) + self.rnnS1
                        self.rnnS2 = [self.rnnS2[0]]*(self.rnnB-len(self.rnnS2)+1) + self.rnnS2
                        #print(len(self.rnnS1),111)
                    #print(len(self.rnnS1[-self.rnnB:]))
                    self.memory.store([state1, action1, next_state1, reward1+reward2, state2, next_state2, action2])
class AgentSep2DMessQ(AgentShare2D):       
    def __init__(self, name, pars, nrenvs=1, job=None, experiment=None):
        AgentShare2D.__init__(self,name, pars, nrenvs, job, experiment)
        
    def optimize_model(self , policy_net, target_net, memory, optimizer):
        if self.pars['ppe']!='1' and  len(memory) < self.BATCH_SIZE:
            return          
        if self.pars['ppe']=='1' and self.capmem< self.BATCH_SIZE:
            return          
        
        if self.pars['ppe']=='1':
            #state1, action1, next_state1, reward1, state2
            tree_idx, batch, ISWeights_mb = memory.sample(self.BATCH_SIZE)
            non_final_next_states = torch.cat( [i[2] for i in batch])
            state_batch = torch.cat( [i[0] for i in batch])
            action_batch = torch.cat( [i[1] for i in batch])
            reward_batch = torch.cat( [i[3] for i in batch])
            state1_batch = torch.cat( [i[4] for i in batch])
            mes_batch = torch.cat( [i[5] for i in batch])
            non_final_next_states1 = torch.cat( [i[6] for i in batch])
            comm = torch.cat( [i[7] for i in batch])
        else:
            transitions = memory.sample(self.BATCH_SIZE)
            batch = Transition(*zip(*transitions))
            
            non_final_next_states = torch.cat( batch.next_state)
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)
            state1_batch = torch.cat(batch.agent_index)
        #print(comm.size())
        mes = torch.tensor([[0,0,0,0] for i in range(self.BATCH_SIZE)], device=self.device)  
        _, me = policy_net(state_batch, 1, mes)[:2]
        q,_ = policy_net(state_batch, 1, comm)[:2]
        state_action_values = q.gather(1, action_batch)
        state_mes_values = me.gather(1, mes_batch)
        
        next_state_com = target_net(non_final_next_states1, 1, mes)[1].max(1)[1].detach().view(-1,1)
        comm1 = torch.zeros(self.BATCH_SIZE, 4, device=self.device).scatter_(1, next_state_com, 1)
        #torch.eye(4, device=self.device).index(1, next_state_com).unsqueeze(0)
        next_state_values = target_net(non_final_next_states, 1, comm1)[0].max(1)[0].detach()
        next_state_mess = target_net(non_final_next_states, 1, mes)[1].max(1)[0].detach()
        expected_state_action_values = ((next_state_values+next_state_mess) * self.GAMMA) + reward_batch.float()
        loss = F.smooth_l1_loss(state_action_values+state_mes_values, expected_state_action_values.unsqueeze(1))
        #loss = weighted_mse_loss(state_action_values, expected_state_action_values.unsqueeze(1), 
        #                         torch.tensor(ISWeights_mb, device=self.device))
        #print(torch.tensor(ISWeights_mb, device=self.device).size())

        if self.pars['ppe']=='1':
            absolute_errors = (state_action_values+state_mes_values-expected_state_action_values.unsqueeze(1)).abs().cpu().data.numpy().reshape((-1))
            memory.batch_update(tree_idx, absolute_errors)
        # Optimize the model
        if self.pars['att'] == 1:
            loss = loss+ att.mean()*0.001
        if self.pars['commr'] ==1:
            comm1 = torch.flip(comm.detach(), [0])
            q1 = policy_net(state_batch, 1, comm1)[0]
            #print(comm.detach(), comm1)
            #F.smooth_l1_loss(comm.detach().float(), comm1.float())# F.smooth_l1_loss(q1,q)#
            dc = 0.1*((comm.detach().float()- comm1.float())**2).mean(-1) #F.kl_div(comm.detach().float(), comm1.float())
            dq = ((q1- q)**2).mean(-1)#F.kl_div(q1, q)
            loss = loss + 0.01*((dc-dq)**2).mean()
            
            if np.random.rand()<0.0005:
                print('difc',dc.cpu().data.numpy()[:10])
                print('difq',dq.cpu().data.numpy()[:10])
                
        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        optimizer.step()
        
    def select_action(self, state, comm, policy_net):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) *         math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                q,c= policy_net(state, 1, comm)
                #print(state.size(), comm.size(),c.size(), comm)
                return  q.max(1)[1].view(1, 1), c.max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(4)]], device=self.device, dtype=torch.long), torch.tensor([[random.randrange(4)]], device=self.device, dtype=torch.long)
    
    def getaction(self, state1, state2, test=False):
        mes = torch.tensor([[0,0,0,0]], device=self.device)  
        eye = torch.eye(4, device=self.device)
        #print(eye)
        if test:
            comm2 = self.policy_net(state2, 0, mes)[1].max(1)[1].item()
            comm1 = self.policy_net(state1, 0, mes)[1].max(1)[1].item()
            comm1 = eye[comm1].unsqueeze(0)
            comm2 = eye[comm2].unsqueeze(0)
            action1 = self.policy_net(state1, 1, comm2)[0].max(1)[1].view(1, 1)
            action2 = self.policy_net(state2, 1, comm1)[0].max(1)[1].view(1, 1)
        else:
            
            _, comm1 = self.select_action(state1, mes, self.policy_net)
            _, comm2 = self.select_action(state2,  mes, self.policy_net) 
            self.rem = [comm1, comm2]
            comm1 = eye[comm1.item()].unsqueeze(0)
            comm2 = eye[comm2.item()].unsqueeze(0)
            self.rem.extend([comm1, comm2])
            action1, _ = self.select_action(state1, comm2, self.policy_net)
            action2, _ = self.select_action(state2,  comm1, self.policy_net)
            
            
        return action1, action2, [comm1, comm2]
    
    def saveStates(self, state1, state2, action1,action2, next_state1,next_state2, reward1,reward2, env_id):
            self.capmem+=2
            if self.pars['ppe']!='1':
                    self.memory.push(state2, action2, next_state2, reward2, state1)
                    self.memory.push(state1, action1, next_state1, reward1, state2)
            else:
                    if self.pars['rec']==1 and len(self.rnnS1)<self.rnnB:#always full steps
                        self.rnnS1 = [self.rnnS1[0]]*(self.rnnB-len(self.rnnS1)+1) + self.rnnS1
                        self.rnnS2 = [self.rnnS2[0]]*(self.rnnB-len(self.rnnS2)+1) + self.rnnS2
                        #print(len(self.rnnS1),111)
                    #print(len(self.rnnS1[-self.rnnB:]))
                    self.memory.store([state1, action1, next_state1, reward1+reward2, state2, self.rem[0],
                                       next_state2, self.rem[3]])
                    self.memory.store([state2, action2, next_state2, reward2+reward1, state1, self.rem[1], 
                                       next_state1, self.rem[2]])
                    
class AgentShare2DDeMessQ(AgentSep2DMessQ):       
    def __init__(self, name, pars, nrenvs=1, job=None, experiment=None):
        AgentShare2D.__init__(self,name, pars, nrenvs, job, experiment)
        
    def optimize_model(self , policy_net, target_net, memory, optimizer):
        if self.pars['ppe']!='1' and  len(memory) < self.BATCH_SIZE:
            return          
        if self.pars['ppe']=='1' and self.capmem< self.BATCH_SIZE:
            return          
        
        if self.pars['ppe']=='1':
            #state1, action1, next_state1, reward1, state2
            tree_idx, batch, ISWeights_mb = memory.sample(self.BATCH_SIZE)
            non_final_next_states = torch.cat( [i[2] for i in batch])
            state_batch = torch.cat( [i[0] for i in batch])
            action_batch = torch.cat( [i[1] for i in batch])
            reward_batch = torch.cat( [i[3] for i in batch])
            state1_batch = torch.cat( [i[4] for i in batch])
            mes_batch = torch.cat( [i[5] for i in batch])
            non_final_next_states1 = torch.cat( [i[6] for i in batch])
            comm = torch.cat( [i[7] for i in batch])
            comm1 = torch.cat( [i[8] for i in batch])
            action_batch1 = torch.cat( [i[9] for i in batch])
            mes_batch1 = torch.cat( [i[10] for i in batch])
        else:
            transitions = memory.sample(self.BATCH_SIZE)
            batch = Transition(*zip(*transitions))
            
            non_final_next_states = torch.cat( batch.next_state)
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)
            state1_batch = torch.cat(batch.agent_index)
        #print(comm.size())
        mes = torch.tensor([[0,0,0,0] for i in range(self.BATCH_SIZE)], device=self.device)  .float()
        q, me = policy_net(torch.cat([state_batch,state_batch,state1_batch,state1_batch]), 1, 
                           torch.cat([mes,comm,mes,comm1]))[:2]
        #q,_ = policy_net(state_batch, 1, comm)[:2]
        state_action_values = q[self.BATCH_SIZE:self.BATCH_SIZE*2].gather(1, action_batch)
        state_mes_values = me[:self.BATCH_SIZE].gather(1, mes_batch)
        state_action_values1 = q[self.BATCH_SIZE*3:].gather(1, action_batch1)
        state_mes_values1 = me[self.BATCH_SIZE*2:self.BATCH_SIZE*3].gather(1, mes_batch1)
        
        next_state_com = target_net(torch.cat([non_final_next_states,non_final_next_states1]), 1, 
                                    torch.cat([mes,mes]))[1].max(1)[1].detach().view(-1,1)
        comm1 = torch.zeros(self.BATCH_SIZE*2, 4, device=self.device).scatter_(1, next_state_com, 1)
        #torch.eye(4, device=self.device).index(1, next_state_com).unsqueeze(0)
        next_state_values,next_state_mess = target_net(torch.cat([non_final_next_states1,non_final_next_states,
                                                                 non_final_next_states1,non_final_next_states,
                                                                 ]), 1, 
                                       torch.cat([comm1,mes,mes]))[:2]#.max(1)[0].detach()
        
        next_state_values1 = next_state_values[:self.BATCH_SIZE].max(1)[0].detach()
        next_state_values = next_state_values[self.BATCH_SIZE*1:self.BATCH_SIZE*2].max(1)[0].detach()
        next_state_mess1 = next_state_mess[self.BATCH_SIZE*2:self.BATCH_SIZE*3].max(1)[0].detach()
        next_state_mess = next_state_mess[self.BATCH_SIZE*3:self.BATCH_SIZE*4].max(1)[0].detach()
        
        expected_state_action_values = ((next_state_values+next_state_mess+next_state_values1+next_state_mess1) * self.GAMMA) \
                                        + reward_batch.float()
        loss = F.smooth_l1_loss(state_action_values+state_mes_values+state_action_values1+state_mes_values1,
                                expected_state_action_values.unsqueeze(1))
        #loss = weighted_mse_loss(state_action_values, expected_state_action_values.unsqueeze(1), 
        #                         torch.tensor(ISWeights_mb, device=self.device))
        #print(torch.tensor(ISWeights_mb, device=self.device).size())

        if self.pars['ppe']=='1':
            absolute_errors = (state_action_values+state_mes_values+state_action_values1+state_mes_values1-expected_state_action_values.unsqueeze(1)).abs().cpu().data.numpy().reshape((-1))
            memory.batch_update(tree_idx, absolute_errors)
        # Optimize the model
        if self.pars['att'] == 1:
            loss = loss+ att.mean()*0.001
        if self.pars['commr'] ==1:
            comm1 = torch.flip(comm.detach(), [0])
            q1 = policy_net(state_batch, 1, comm1)[0]
            #print(comm.detach(), comm1)
            #F.smooth_l1_loss(comm.detach().float(), comm1.float())# F.smooth_l1_loss(q1,q)#
            dc = 0.1*((comm.detach().float()- comm1.float())**2).mean(-1) #F.kl_div(comm.detach().float(), comm1.float())
            dq = ((q1- q)**2).mean(-1)#F.kl_div(q1, q)
            loss = loss + 0.01*((dc-dq)**2).mean()
            
            if np.random.rand()<0.0005:
                print('difc',dc.cpu().data.numpy()[:10])
                print('difq',dq.cpu().data.numpy()[:10])
                
        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        optimizer.step()
    
    def saveStates(self, state1, state2, action1,action2, next_state1,next_state2, reward1,reward2, env_id):
            self.capmem+=2
            if self.pars['ppe']!='1':
                    self.memory.push(state2, action2, next_state2, reward2, state1)
                    self.memory.push(state1, action1, next_state1, reward1, state2)
            else:
                    if self.pars['rec']==1 and len(self.rnnS1)<self.rnnB:#always full steps
                        self.rnnS1 = [self.rnnS1[0]]*(self.rnnB-len(self.rnnS1)+1) + self.rnnS1
                        self.rnnS2 = [self.rnnS2[0]]*(self.rnnB-len(self.rnnS2)+1) + self.rnnS2
                        #print(len(self.rnnS1),111)
                    #print(len(self.rnnS1[-self.rnnB:]))
                    self.memory.store([state1, action1, next_state1, reward1+reward2, state2, self.rem[0],
                                       next_state2, self.rem[3], self.rem[2], action2, self.rem[1]])
                    #self.memory.store([state2, action2, next_state2, reward2+reward1, state1, self.rem[1], 
                    #                   next_state1, self.rem[2],self.rem[3]])