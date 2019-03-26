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

class AgentA2CShare2D(AgentACShare2D):
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
        
class AgentPPOShare2D(AgentA2CShare2D):
    def __init__(self, name, pars, nrenvs=1, job=None, experiment=None):
        Agent.__init__(self,name, pars, nrenvs, job, experiment)
        self.coef_value = 0.5
        self.ppo_steps = 5
        self.clip_param = 0.2
        self.max_grad_norm = 0.5
        self.eps_threshold = 0.1
        self.use_clipped_value_loss = True
    def optimize_policy1(self, policy_net, memories, optimizer):
        '''
        vl = []
        mem = []
        for j,memory in enumerate(memories):            
            states = torch.cat([c[0].float() for c in memory])
            state1_batch = torch.cat([c[0].float() for c in memory])
            action_batch = torch.cat([c[1].float() for c in memory]).view(-1,1).detach()
            mes = torch.tensor([[0,0,0,0] for i in memory], device=self.device)
            comm = self.getComm(mes, self.q_net,state1_batch)
            if self.pars['comm'] =='2':
                    comm = comm.detach()                
            actionV = self.policy_net(states, 0, comm)[1].gather(1, action_batch.long()).detach()
            #vl.append(actionV.detach())
            R = torch.zeros(1, 1, device=self.device)
            for i in reversed(range(len(memory))):
                        _,_,_,r,log_prob, entr = memory[i]
                        ac = actionV[i]# (actionV[i] - mu) / (std + eps)#actionV[i]#also use mu and std
                        #Discounted Sum of Future Rewards + reward for the given state
                        R = self.GAMMA * R + (r.float() )#- mu) / (std + eps)
                        mem.append(memory[i]+[R, ac.detach()])
        batch_size = len(mem)
        num_mini_batch = 32
        value_preds_batch = torch.cat([c[7] for c in mem])
        rew = torch.cat([c[6] for c in mem])
        advantages = rew - value_preds_batch
        advantagesmean = advantages.mean()
        advantagesstd = advantages.std()
        for i in range(self.ppo_steps):
            policy_loss = 0
            value_loss = 0
            ent = 0
            nr=0
            
            mini_batch_size = batch_size // num_mini_batch
            sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=False)
            for indices in sampler:
                memory = [mem[i] for i in indices]
                value_preds_batch = torch.cat([c[7] for c in memory])
                rew = torch.cat([c[6] for c in memory])
                states = torch.cat([c[0].float() for c in memory])
                state1_batch = torch.cat([c[0].float() for c in memory])
                action_batch = torch.cat([c[1].float() for c in memory]).view(-1,1).detach()
                old_action_log_probs_batch = torch.cat([c[4] for c in memory]).view(-1,1).detach()
                
                mes = torch.tensor([[0,0,0,0] for i in memory], device=self.device)
                comm = self.getComm(mes, self.q_net,state1_batch)
                if self.pars['comm'] =='2':
                    comm = comm.detach()
                
                actionV = self.policy_net(states, 0, comm)[1].gather(1, action_batch.long())
                comm = self.getComm(mes, self.policy_net,state1_batch)
                if self.pars['comm'] =='2':
                    comm = comm.detach()
                dist = self.policy_net(states, 0, comm)[0]
                m = Categorical(logits=dist)
                #action = m.sample()
                action_log_probs= m.log_prob(action_batch.long())
                entropy =0# m.entropy()               
                
                advantages = rew - value_preds_batch
                adv_targ = (advantages - advantagesmean) / (advantagesstd + 1e-5)
                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ.detach()
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                           1.0 + self.clip_param) * adv_targ.detach()
                action_loss = -torch.min(surr1, surr2).mean()       
                    
                
                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (actionV - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (actionV - rew).pow(2)
                    value_losses_clipped = (value_pred_clipped - rew).pow(2)
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (  actionV-rew).pow(2).mean()   
                #
                optimizer.zero_grad()
                (action_loss +0.5*value_loss   + self.eps_threshold*entropy.mean()).backward()
                for param in policy_net.parameters():
                    if param.grad is not None:
                        param.grad.data.clamp_(-0.5, 0.5)
                optimizer.step()                   
                if True:
                    self.optimizer.zero_grad()
                    (value_loss).backward()
                    for param in self.q_net.parameters():
                        if param.grad is not None:
                            param.grad.data.clamp_(-0.5, 0.5)
                    self.optimizer.step()
        '''
    def optimize_policy(self, policy_net, memories, optimizer):
        mem = []
        for j,memory in enumerate(memories):            
            states = torch.cat([c[0].float() for c in memory])
            state1_batch = torch.cat([c[0].float() for c in memory])
            action_batch = torch.cat([c[1].float() for c in memory]).view(-1,1).detach()
            mes = torch.tensor([[0,0,0,0] for i in memory], device=self.device)
            comm = self.getComm(mes, self.q_net,state1_batch)
            if self.pars['comm'] =='2':
                    comm = comm.detach()                
            actionV = self.q_net(states, 0, comm)[0].gather(1, action_batch.long()).detach()
            #vl.append(actionV.detach())
            R = 0
            for i in reversed(range(len(memory)-1)):
                        _,_,_,r,log_prob, entr = memory[i]
                        ac = actionV[i]# (actionV[i] - mu) / (std + eps)#actionV[i]#also use mu and std
                        #Discounted Sum of Future Rewards + reward for the given state
                        R = self.GAMMA * R + r.item()#- mu) / (std + eps)
                        mem.append(memory[i]+[torch.tensor([[R]] ,device=self.device), ac.detach()])
        batch_size = len(mem)
        num_mini_batch = 32
        value_preds_batch = torch.cat([c[7] for c in mem]).view(-1,1)
        rew = torch.cat([c[6] for c in mem]).view(-1,1)
        advantages = rew - value_preds_batch
        advantagesmean = advantages.mean()
        advantagesstd = advantages.std()
        for i in range(self.ppo_steps):
            mini_batch_size = 32# batch_size // num_mini_batch
            sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=False)
            for indices in sampler:
                memory = [mem[i] for i in indices]
                value_preds_batch = torch.cat([c[7] for c in memory]).view(-1,1)
                rew = torch.cat([c[6] for c in memory]).view(-1,1)
                states = torch.cat([c[0].float() for c in memory])
                state1_batch = torch.cat([c[0].float() for c in memory])
                action_batch = torch.cat([c[1].float() for c in memory]).view(-1,1).detach()
                old_action_log_probs_batch = torch.cat([c[4] for c in memory]).view(-1,1).detach()
                
                mes = torch.tensor([[0,0,0,0] for i in memory], device=self.device)
                comm = self.getComm(mes, self.q_net,state1_batch)
                if self.pars['comm'] =='2':
                    comm = comm.detach()
                
                actionV = self.q_net(states, 0, comm)[0].gather(1, action_batch.long())
                comm = self.getComm(mes, self.policy_net,state1_batch)
                if self.pars['comm'] =='2':
                    comm = comm.detach()
                dist = self.policy_net(states, 0, comm)[0]
                m = Categorical(logits=dist)
                #action = m.sample()
                action_log_probs= m.log_prob(action_batch.long())
                entropy = m.entropy().mean()  
                
                advantages = rew - value_preds_batch
                adv_targ = (advantages - advantagesmean) / (advantagesstd + 1e-5)
                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ.detach()
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                           1.0 + self.clip_param) * adv_targ.detach()
                action_loss = -torch.min(surr1, surr2).mean()       
                #+0.5*value_loss
                optimizer.zero_grad()
                (action_loss    - self.eps_threshold*entropy).backward()
                for param in policy_net.parameters():
                    if param.grad is not None:
                        param.grad.data.clamp_(-0.5, 0.5)
                optimizer.step()      
                
                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (actionV - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (actionV - rew).pow(2)
                    value_losses_clipped = (value_pred_clipped - rew).pow(2)
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (  actionV-rew).pow(2).mean()   
                                 
                if True:
                    self.optimizer.zero_grad()
                    (value_loss).backward()
                    for param in self.q_net.parameters():
                        if param.grad is not None:
                            param.grad.data.clamp_(-0.5, 0.5)
                    self.optimizer.step()
                
    def select_action(self, state, comm, policy_net):
        probs1, _ = policy_net(state, 1, comm)#.cpu().data.numpy()
        m = Categorical(logits=probs1)
        action = m.sample()
        return action.view(1, 1), m.log_prob(action), m.entropy()
    
    def getComm(self, mes, policy_net, state1_batch):
        return policy_net(state1_batch, 1, mes)[self.idC].detach() if np.random.rand()<self.prob else mes
    
    def getaction(self, state1, state2, test=False):
        mes = torch.tensor([[0,0,0,0]], device=self.device)
        #maybe error
        comm2 = self.policy_net(state2, 0, mes)[self.idC] if (test and  0<self.prob) or np.random.rand()<self.prob else mes
        comm1 = self.policy_net(state1, 0, mes)[self.idC] if (test and  0<self.prob) or np.random.rand()<self.prob else mes
        
        action1, logp1, ent1 = self.select_action(state1,  comm2, self.policy_net)
        action2, logp2, ent2 = self.select_action(state2,  comm1, self.policy_net)
        self.rem =[logp1, ent1, logp2, ent2]
        return action1, action2, [comm1, comm2]
    
    def saveStates(self, state1, state2, action1,action2, next_state1,next_state2, reward1,reward2, env_id):
                    logp1, ent1, logp2, ent2 = self.rem
                    if self.pars['ppe']!='1':
                        self.memory.push(state2, action2, next_state2, reward2, state1)
                        self.memory.push(state1, action1, next_state1, reward1, state2)
                    else:
                        self.memory.store([state1, action1, next_state1, reward1, state2])
                        self.memory.store([state2, action2, next_state2, reward2, state1])
                    #self.buf2.append([state2, action2,1, reward2, logp2, ent2])
                    #self.buf1.append([state1, action1,1, reward1, logp1, ent1])
                    
                    self.bufs[2*env_id  ].append([state2, action2,1, reward2, logp2, ent2])
                    self.bufs[2*env_id+1].append([state1, action1,1, reward1, logp1, ent1])