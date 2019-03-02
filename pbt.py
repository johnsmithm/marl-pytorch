from comet_ml import Experiment


import gym
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


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'agent_index'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if np.random.rand()>0.8:
            self.memory = self.memory[-self.capacity:]
        #return random.sample(self.memory, batch_size)
        x = random.sample(self.memory, batch_size//2)
        x1 = []
        i = np.random.randint(max(0,len(self.memory)-batch_size*2)) if len(self.memory)-batch_size*2>1 else 0
        n = 0
        while len(x1)<batch_size//2 and i<len(self.memory) and n<100:
            n+=1
            if self.memory[i].reward>0.001:
                x1.append(self.memory[i])
            i+=1   
        i = np.random.randint(max(0,len(self.memory)-batch_size*2)) if len(self.memory)-batch_size*2>1 else 0
        while len(x1)<batch_size//2 :
                x1.append(self.memory[i])
                i+=1
        #print(x+x1)
        return x+x1

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, inp):
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

def getA():
    parser = argparse.ArgumentParser(description='Test a network')
    parser.add_argument('--bs', dest='bs',
                help='file to load the db state', default=10, type=int)
    parser.add_argument('--tg', dest='tg',
                help='file to load the db state', default=10, type=int)
    parser.add_argument('--numep', dest='numep',
                help='file to load the db state', default=50, type=int)
    parser.add_argument('--epsteps', dest='epsteps',
                help='file to load the db state', default=100, type=int)
    parser.add_argument('--show', dest='show',
                help='file to load the db state', default=10, type=int)
    parser.add_argument('--nrf', dest='nrf',
                help='file to load the db state', default=10, type=int)
    parser.add_argument('--subhid', dest='subhid',
                help='file to load the db state', default=0.6, type=float)
    parser.add_argument('--alpha', dest='alpha',
                help='file to load the db state', default=1, type=float)
    
    parser.add_argument('-r', '--results_path', type=str, help='path to results directory', default='logs/t3')
    parser.add_argument('-d', '--debug', type=str, help='path to results directory', default='0')
    parser.add_argument('-c', '--comment', type=str, help='path to results directory', default='')
    parser.add_argument('-n', '--name', type=str, help='path to results directory', default='t3')
    parser.add_argument('-w', '--comm', type=str, help='0-no,2-Q,1-encoded', default='0')
    parser.add_argument('-e', '--en', type=int, help='0-no,1-Q,2-encoded', default=100)
    parser.add_argument('-e1', '--h1', type=int, help='0-no,1-Q,2-encoded', default=100)
    parser.add_argument('-e2', '--h2', type=int, help='0-no,1-Q,2-encoded', default=100)
    parser.add_argument('-e12', '--lr', type=float, help='0-no,1-Q,2-encoded', default=0.01)
    parser.add_argument('-e21', '--momentum', type=float, help='0-no,1-Q,2-encoded', default=0.5)
    parser.add_argument('-e212', '--epochs', type=int, help='0-no,1-Q,2-encoded', default=5)
    parser.add_argument('-e2122', '--workers', type=int, help='0-no,1-Q,2-encoded', default=5)
    args = parser.parse_args()
    
    return args

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
    def build(self):
        self.policy_net = DQN(43).to(self.device)
        self.target_net = DQN(43).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.SGD(
                    self.policy_net.parameters(), lr=self.pars['lr'], 
                    momentum=self.pars['momentum'])#
        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.memory = ReplayMemory(10000)
    def optimize_model(self ):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        
        non_final_next_states = torch.cat( batch.next_state)
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        state1_batch = torch.cat(batch.agent_index)

        mes = torch.tensor([[0,0,0,0] for i in range(self.BATCH_SIZE)], device=self.device)
        comm = self.policy_net(state1_batch, 1, mes)[self.idC] if np.random.rand()<self.prob else mes
        if self.pars['comm'] =='2':
            comm = comm.detach()
        state_action_values = self.policy_net(state_batch, 1, comm.detach())[0].gather(1, action_batch)

        next_state_values = self.target_net(non_final_next_states, 1, mes)[0].max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch.float()
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
    def select_action(self, state, comm):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) *         math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state, 1, comm)[0].max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(4)]], device=self.device, dtype=torch.long)
    
    def getaction(self, state1, state2, test=False):
        mes = torch.tensor([[0,0,0,0]], device=self.device)
        comm2 = self.policy_net(state2, 0, mes)[self.idC].detach() if np.random.rand()<self.prob else mes
        comm1 = self.policy_net(state1, 0, mes)[self.idC].detach() if np.random.rand()<self.prob else mes
        if test:
            action1 = self.policy_net(state1, 1, comm2)[0].max(1)[1].view(1, 1)
            action2 = self.policy_net(state2, 1, comm1)[0].max(1)[1].view(1, 1)
        else:
            action1 = self.select_action(state1, comm2)
            action2 = self.select_action(state2,  comm1)
        return action1, action2, [comm1, comm2]
    def getStates(self, env):
        screen1 = env.render_env_1d(0)#.transpose((2, 0, 1))
        screen2 = env.render_env_1d(1)#.transpose((2, 0, 1))
        return torch.from_numpy(screen1).unsqueeze(0).to(self.device), torch.from_numpy(screen2).unsqueeze(0).to(self.device)
    
    def train(self, num_episodes): 
        
        for i_episode in range(num_episodes):
            if job is not None and job.stopEx:
                return
            
            for env in self.envs:
                env.reset()
                
                state1,state2 = self.getStates(env)
                rt = 0; ac=[]
                start_time = time.time()
                buf1= [];buf2= []; ep1=[]
                for t in range(self.pars['epsteps']):
                
                    action1, action2, _ = self.getaction(state1,state2)
                    reward1, reward2 = env.move(action1.item(), action2.item())#multi envs??
                    rt+=reward1+reward2;
                    ac.append(str(action1.item()))

                    reward1 = torch.tensor([reward1*pars['alpha']+reward2*(1-pars['alpha'])], device=self.device)
                    reward2 = torch.tensor([reward2*pars['alpha']+reward1*(1-pars['alpha'])], device=self.device)

                    next_state1, next_state2 = self.getStates(env)
                    self.memory.push(state2, action2, next_state2, reward2, state1)
                    self.memory.push(state1, action1, next_state1, reward1, state2)

                    state1 = next_state1
                    state2 = next_state2

                    self.optimize_model()
            if i_episode%10==0:
                print('ep',i_episode, 'reward train',rt, 'time', time.time() - start_time, ','.join(ac[:20]))            
            self.updateTarget(i_episode)
            
    def test(self, tries=3, log=True, i_episode=-1):
        rs = []
        rt1=0
        ep=[]
        self.policy_net.eval()
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
            save_episode_and_reward_to_csv(self.result_out, self.writer, i_episode, rt1, ep, self.name)
        
        print( 'reward test', rm, rs)
        self.policy_net.train()
        return rm
                      
    def updateTarget(self, i_episode):
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
        lr = 0
        momentum = 0
        for param_group in self.optimizer.param_groups:
            if nolast:
                param_group['lr'] *= new_lr_factor
                param_group['momentum'] += new_momentum_delta
            momentum =param_group['momentum']
            lr = param_group['lr']
        with open(os.path.join(pars['results_path']+ self.name,'hyper-{}.json').format(i_episode), 'w') as outfile:
            json.dump({'lr':lr, 'momentum':momentum,'eps_decay':self.EPS_DECAY,
                       'prob':self.prob,'i_episode':i_episode}, outfile)
    def clone(self, agent):
        optimizer = optim.SGD(
            self.policy_net.parameters(), lr=self.pars['lr'], momentum=self.pars['momentum'])
        state_dict = agent.policy_net.state_dict()
        self.policy_net.load_state_dict(state_dict)
        state_dict = agent.optimizer.state_dict()
        self.optimizer.load_state_dict(state_dict)
        self.EPS_DECAY = agent.EPS_DECAY
        
def save_episode_and_reward_to_csv(file, writer, e, r, ep, name):
    #[env.render_env(), action.item(), reward]
    #print(ep.step_records[27].agent_inputs[0]['s_t'].data.numpy())
    data = {'eps':[i[0].tolist() for i in ep]}
    data ['a']=[i[1] for i in ep]
    data ['r']=[i[2] for i in ep]
    data ['co']=[i[3] for i in ep]
    data ['ac']=[i[4] for i in ep]
    #print(data ['r'])
    data['rewardT'] =r# sum(data ['r'])
    #episode.step_records[step].r_t
    with open(os.path.join(pars['results_path']+ name,'ep:{}.json').format(e), 'w') as outfile:
        json.dump(data, outfile)
    #episode.step_records[step].agent_inputs.append(agent_inputs)#s_t
    writer.writerow({'episode': e, 'reward': r})
    file.flush()
           
import numpy as np

class DeviationPbtAdvisor:
    def __init__(self, max_lower_deviation=1):
        self.max_lower_deviation = max_lower_deviation
        
    def advise(self, performance):
        performance = np.array(performance)
        
        stddev = np.std(performance, ddof=1)
        mean = np.mean(performance)
        
        # If we call this too often in a row without training in-between,
        # it will just copy the best performer everywhere.
        underperformers = performance < mean - self.max_lower_deviation * stddev
        indices = np.transpose(np.nonzero(underperformers))
        best_performer = np.unravel_index(np.argmax(performance),
                                          dims=performance.shape)

        return {tuple(index): tuple(best_performer) for index in indices}

def pbt(pars, nrenvs=1, job=None, experiment=None, num_workers = 5):
    pbt_advisor = DeviationPbtAdvisor(0)
    
    workers = [Agent("{}".format(i), pars, nrenvs=nrenvs, job=job, experiment=experiment) for i in range(num_workers)]
    for worker in workers:
        worker.perturb_learning_rate(0)

    for epoch in range(1, pars['epochs'] + 1):
        performances = []
        for idx, worker in enumerate(workers):
            print('Worker %s' % idx)
            worker.train(pars['numep'])
            performance = worker.test(i_episode=pars['numep']*epoch)
            performances.append(performance)

        copy_actions = pbt_advisor.advise(performance=performances)
        for dst, src in copy_actions.items():
            workers[dst[0]].clone(workers[src[0]])
            workers[dst[0]].perturb_learning_rate(i_episode=pars['numep']*epoch)
            print('Copying %s to %s' % (src, dst))
    for idx, worker in enumerate(workers):
        worker.save()
        worker.result_out.close()
        worker.perturb_learning_rate(i_episode='last', nolast=False)

def train(agent, pars):
    for epoch in range(1, pars['epochs'] + 1):
        if epoch%4==0:
            pars['epsteps'] = min(100, pars['epsteps']+10)
            agent.pars['epsteps'] = pars['epsteps']
        agent.train(pars['numep'])
        print('epoch', epoch)
        agent.test(i_episode=pars['numep']*epoch)
    agent.save()
    agent.result_out.close()
        
if __name__ == '__main__': 
    from htuneml1 import Job
    job = Job('5c61b674203efd001a65d4b1')
    
    if True:
        args = getA()
        pars = vars(args)    
        print(pars)
        pars['results_path'] += pars['name']
        env = None
        
        experiment = Experiment(api_key="ubnNI8IwcycXWmKD7eT7YlP4J", auto_output_logging=None,auto_metric_logging=False,
                        disabled=pars['debug'] == '1',
                        project_name="general", workspace="ionmosnoi")
        
        experiment.log_parameters(pars)

        job.setName(pars['name'])
        if pars['debug'] == '1':
            job.debug()
        else:
            job=None#job.makeExperiment(pars['name'], pars)
        if False:
            pbt(pars, nrenvs=1, job=job, experiment=experiment, num_workers = pars['workers'])
        else:
            agent = Agent('1', pars, nrenvs=1, job=job, experiment=experiment)
            train(agent, pars)
    else:
        job.waitTask(main)