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

from models.models import DQN
from mas import *

from buffer import ReplayMemory,save_episode_and_reward_to_csv, Transition


class Agent:
    def __init__(self, name, pars, nrenvs=1, job=None, experiment=None):
        self.job = job
        self.name = name
        self.experiment = experiment
        self.pars = pars
        self.envs = [GameEnv(pars['subhid']) for i in range(nrenvs)]
        for env in self.envs:
            env.reset()
        self.BATCH_SIZE = pars['bs']
        self.GAMMA = 0.999
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.alpha = pars['alpha']
        self.EPS_DECAY = 200
        self.TARGET_UPDATE = pars['tg']
        self.nrf = pars['nrf']            

        self.prob = 0.5
        self.idC = 0
        if pars['comm'] =='0':
            self.prob = -1
        if pars['comm'] =='1':
            self.idC = 1
    
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.build()
        
        if pars['results_path']:
            result_path =  pars['results_path']+name
            if not os.path.exists(result_path):
                os.makedirs(result_path)            
            
        result_path = result_path + '/results_' + str(0) + '.csv'
        self.result_out = open(result_path, 'w')
        csv_meta = '#' + json.dumps(pars) + '\n'    
        self.result_out.write(csv_meta)    
        self.writer = csv.DictWriter(self.result_out, fieldnames=['episode', 'reward'])
        self.writer.writeheader()

        self.steps_done = 0
        self.num_episodes = pars['numep']
        self.lr = pars['lr']
        self.momentum = pars['momentum']
    def build(self):
        self.policy_net = DQN(71, self.pars).to(self.device)
        self.target_net = DQN(71, self.pars, self.pars).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        if self.pars['momentum']>0:
            self.optimizer = optim.SGD(
                    self.policy_net.parameters(), lr=self.pars['lr'], 
                    momentum=self.pars['momentum'])#
        if self.pars['momentum']<0:
            self.optimizer = optim.Adam(self.policy_net.parameters())
        self.memory = ReplayMemory(10000)
    def optimize_model(self , policy_net, target_net, memory, optimizer):
        if len(memory) < self.BATCH_SIZE:
            return
        transitions = memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        
        non_final_next_states = torch.cat( batch.next_state)
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        state1_batch = torch.cat(batch.agent_index)

        mes = torch.tensor([[0,0,0,0] for i in range(self.BATCH_SIZE)], device=self.device)
        comm = policy_net(state1_batch, 1, mes)[self.idC] if np.random.rand()<self.prob else mes
        if self.pars['comm'] =='2':
            comm = comm.detach()
        state_action_values = policy_net(state_batch, 1, comm)[0].gather(1, action_batch)

        next_state_values = target_net(non_final_next_states, 1, mes)[0].max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch.float()
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
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
                return policy_net(state, 1, comm)[0].max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(4)]], device=self.device, dtype=torch.long)
    
    def getaction(self, state1, state2, test=False):
        mes = torch.tensor([[0,0,0,0]], device=self.device)            
        if test:
            comm2 = self.policy_net(state2, 0, mes)[self.idC].detach() 
            comm1 = self.policy_net(state1, 0, mes)[self.idC].detach() 
            action1 = self.policy_net(state1, 1, comm2)[0].max(1)[1].view(1, 1)
            action2 = self.policy_net(state2, 1, comm1)[0].max(1)[1].view(1, 1)
        else:
            comm2 = self.policy_net(state2, 0, mes)[self.idC].detach() if np.random.rand()<self.prob else mes
            comm1 = self.policy_net(state1, 0, mes)[self.idC].detach() if np.random.rand()<self.prob else mes
            action1 = self.select_action(state1, comm2, self.policy_net)
            action2 = self.select_action(state2,  comm1, self.policy_net)
        return action1, action2, [comm1, comm2]
    def getStates(self, env):
        screen1 = env.render_env_1d(0)#.transpose((2, 0, 1))
        screen2 = env.render_env_1d(1)#.transpose((2, 0, 1))
        return torch.from_numpy(screen1).unsqueeze(0).to(self.device), torch.from_numpy(screen2).unsqueeze(0).to(self.device)
    
    def saveStates(self, state1, state2, action1,action2, next_state1,next_state2, reward1,reward2):
        
                    self.memory.push(state2, action2, next_state2, reward2, state1)
                    self.memory.push(state1, action1, next_state1, reward1, state2)
    def optimize(self):
        self.optimize_model(self.policy_net, self.target_net, self.memory, self.optimizer)
    def train(self, num_episodes): 
        
        for i_episode in range(num_episodes):
            if self.job is not None and self.job.stopEx:
                return
            
            for env in self.envs:
                env.reset()
                self.buf1 = []
                self.buf2 = []
                state1,state2 = self.getStates(env)
                rt = 0; ac=[]
                start_time = time.time()
                buf1= [];buf2= []; ep1=[]
                for t in range(self.pars['epsteps']):
                
                    action1, action2, _ = self.getaction(state1,state2)
                    reward1, reward2 = env.move(action1.item(), action2.item())#multi envs??
                    rt+=reward1+reward2;
                    ac.append(str(action1.item()))

                    reward1 = torch.tensor([reward1*self.alpha+reward2*(1-self.alpha)], device=self.device)
                    reward2 = torch.tensor([reward2*self.alpha+reward1*(1-self.alpha)], device=self.device)

                    next_state1, next_state2 = self.getStates(env)
                    self.saveStates(state1, state2, action1,action2, next_state1,next_state2, reward1,reward2)

                    state1 = next_state1
                    state2 = next_state2

                    self.optimize()
                    self.updateTarget(i_episode, step=True)
            if i_episode%10==0:
                print('ep',i_episode, 'reward train',rt, 'time', time.time() - start_time, ','.join(ac[:20]))            
            self.updateTarget(i_episode)
            
    def test(self, tries=3, log=True, i_episode=-1):
        rs = []
        rt1=0
        ep=[]
        #self.policy_net.eval()
        for i in range(tries):
            ep = []
            rt1=0
            self.envs[0].reset()
            for t in range(100):#sep function
                    state1,state2 = self.getStates(self.envs[0])
                    action1, action2, r = self.getaction(state1,state2, test=True)
                    comm1,comm2 = r
                    reward1, reward2 = self.envs[0].move(action1.item(), action2.item())
                    ep.append([self.envs[0].render_env(), [action1.item(), action2.item()], [reward1, reward2],
                               [comm1.cpu().data.numpy()[0].tolist(), comm1.cpu().data.numpy()[0].tolist()],
                               [comm1.max(1)[1].item(), comm1.max(1)[1].item()]])
                    rt1+=reward1+reward2
            rs.append(rt1)
        rm = np.mean(rs)
        if log and i_episode>0:
            if self.job is not None:
                    self.job.log({'reward'+self.name:rm,'ep':i_episode})
            if self.experiment is not None:
                    self.experiment.set_step(i_episode)
                    self.experiment.log_metric("reward"+self.name, rm)
            save_episode_and_reward_to_csv(self.result_out, self.writer, i_episode, rt1, ep, self.name, self.pars)
        
        print( 'reward test', rm, rs)
        #self.policy_net.train()
        return rm
                      
    def updateTarget(self, i_episode, step=False):
        #soft_update(self.target_net, self.policy_net, tau=0.01)
        if step:
            return
        if i_episode % self.TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
    def save(self):
        torch.save(self.policy_net.state_dict(), pars['results_path']+self.name+'/model')
    def load(self, PATH):
        self.policy_net.load_state_dict(torch.load(PATH))        
    
    def perturb_learning_rate(self, i_episode, nolast=True):
        if nolast:
            new_lr_factor = 10**np.random.normal(scale=1.0)
            new_momentum_delta = np.random.normal(scale=0.1)
            self.EPS_DECAY += np.random.normal(scale=50.0)
            if self.prob>=0:
                self.prob += np.random.normal(scale=0.05)-0.025
                self.prob = min(max(0,self.prob),1)
        for param_group in self.optimizer.param_groups:
            if nolast:
                param_group['lr'] *= new_lr_factor
                param_group['momentum'] += new_momentum_delta
            self.momentum =param_group['momentum']
            self.lr = param_group['lr']
        with open(os.path.join(self.pars['results_path']+ self.name,'hyper-{}.json').format(i_episode), 'w') as outfile:
            json.dump({'lr':self.lr, 'momentum':self.momentum,'eps_decay':self.EPS_DECAY,
                       'prob':self.prob,'i_episode':i_episode}, outfile)
    def clone(self, agent):
        state_dict = agent.policy_net.state_dict()
        self.policy_net.load_state_dict(state_dict)
        state_dict = agent.optimizer.state_dict()
        self.optimizer.load_state_dict(state_dict)
        self.EPS_DECAY = agent.EPS_DECAY
        self.prob = agent.prob
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
class AgentSep1D(Agent):
    def __init__(self, name, pars, nrenvs=1, job=None, experiment=None):
        Agent.__init__(self,name, pars, nrenvs, job, experiment)
    def build(self):
        self.policy_net1 = DQN(71, self.pars).to(self.device)
        self.target_net1 = DQN(71, self.pars).to(self.device)
        self.target_net1.load_state_dict(self.policy_net1.state_dict())
        self.target_net1.eval()
        
        self.policy_net2 = DQN(71, self.pars).to(self.device)
        self.target_net2 = DQN(71, self.pars).to(self.device)
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
    def getaction(self, state1, state2, test=False):
        mes = torch.tensor([[0,0,0,0]], device=self.device)
        comm2 = self.policy_net1(state2, 0, mes)[self.idC].detach() if np.random.rand()<self.prob else mes
        comm1 = self.policy_net2(state1, 0, mes)[self.idC].detach() if np.random.rand()<self.prob else mes
        if test:
            action1 = self.policy_net1(state1, 1, comm2)[0].max(1)[1].view(1, 1)
            action2 = self.policy_net2(state2, 1, comm1)[0].max(1)[1].view(1, 1)
        else:
            action1 = self.select_action(state1, comm2, self.policy_net1)
            action2 = self.select_action(state2,  comm1, self.policy_net2)
        return action1, action2, [comm1, comm2]
    def getStates(self, env):
        screen1 = env.render_env_1d()#.transpose((2, 0, 1))
        return torch.from_numpy(screen1).unsqueeze(0).to(self.device), torch.from_numpy(screen1).unsqueeze(0).to(self.device)
    
    def saveStates(self, state1, state2, action1,action2, next_state1,next_state2, reward1,reward2):
        
                    self.memory2.push(state2, action2, next_state2, reward2, state1)
                    self.memory1.push(state1, action1, next_state1, reward1, state2)
    def optimize(self):
        self.optimize_model(self.policy_net1, self.target_net1, self.memory1, self.optimizer1)
        self.optimize_model(self.policy_net2, self.target_net2, self.memory2, self.optimizer2)
    
    def updateTarget(self, i_episode, step=False):
        #soft_update(self.target_net, self.policy_net, tau=0.01)
        if step:
            return
        if i_episode % self.TARGET_UPDATE == 0:
            self.target_net1.load_state_dict(self.policy_net1.state_dict())
            self.target_net2.load_state_dict(self.policy_net2.state_dict())
    def save(self):
        torch.save(self.policy_net1.state_dict(), pars['results_path']+self.name+'/model1')
        torch.save(self.policy_net2.state_dict(), pars['results_path']+self.name+'/model2')
       
    def perturb_learning_rate(self, i_episode, nolast=True):
        if nolast:
            new_lr_factor = 10**np.random.normal(scale=1.0)
            new_momentum_delta = np.random.normal(scale=0.1)
            self.EPS_DECAY += np.random.normal(scale=50.0)
            if self.prob>=0:
                self.prob += np.random.normal(scale=0.05)-0.025
                self.prob = min(max(0,self.prob),1)
        for param_group in self.optimizer1.param_groups:
            if nolast:
                param_group['lr'] *= new_lr_factor
                param_group['momentum'] += new_momentum_delta
            self.momentum1 =param_group['momentum']
            self.lr1 = param_group['lr']
        if nolast:
            new_lr_factor = 10**np.random.normal(scale=1.0)
            new_momentum_delta = np.random.normal(scale=0.1)
            self.EPS_DECAY += np.random.normal(scale=50.0)
            if self.prob>=0:
                self.prob += np.random.normal(scale=0.05)-0.025
                self.prob = min(max(0,self.prob),1)
        for param_group in self.optimizer2.param_groups:
            if nolast:
                param_group['lr'] *= new_lr_factor
                param_group['momentum'] += new_momentum_delta
            self.momentum2 =param_group['momentum']
            self.lr2 = param_group['lr']
        with open(os.path.join(self.pars['results_path']+ self.name,'hyper-{}.json').format(i_episode), 'w') as outfile:
            json.dump({'lr1':self.lr1, 'momentum1':self.momentum1,'lr2':self.lr2, 'momentum2':self.momentum2,
                       'eps_decay':self.EPS_DECAY,
                       'prob':self.prob,'i_episode':i_episode}, outfile)
    def clone(self, agent):
        state_dict = agent.policy_net1.state_dict()
        self.policy_net1.load_state_dict(state_dict)
        state_dict = agent.optimizer1.state_dict()
        self.optimizer1.load_state_dict(state_dict)
        state_dict = agent.policy_net2.state_dict()
        self.policy_net2.load_state_dict(state_dict)
        state_dict = agent.optimizer2.state_dict()
        self.optimizer2.load_state_dict(state_dict)
        self.target_net1.load_state_dict(self.policy_net1.state_dict())
        self.target_net2.load_state_dict(self.policy_net2.state_dict())
        self.EPS_DECAY = agent.EPS_DECAY
 
