from comet_ml import Experiment


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

from models.agent import Agent 
from models.agent2d import AgentShare2D, AgentSep2D, AgentShareRec2D, AgentShareRec1D, Agent2DDecomposeQ, AgentSep2DMessQ
from models.agentAC import AgentACShare1D, AgentACShareRec2D
from models.ppo import AgentPPOShare2D

import time, os, datetime, json
import copy, argparse, csv, json, datetime, os
from functools import partial
from pathlib import Path
from torch.distributions import Categorical
import numpy as np

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
    parser.add_argument('-h1', '--h1', type=int, help='0-no,1-Q,2-encoded', default=100)
    parser.add_argument('-h2', '--h2', type=int, help='0-no,1-Q,2-encoded', default=100)
    parser.add_argument('-lr', '--lr', type=float, help='0-no,1-Q,2-encoded', default=0.01)
    parser.add_argument('-mo', '--momentum', type=float, help='0-no,1-Q,2-encoded', default=0.5)
    parser.add_argument('-ep', '--epochs', type=int, help='0-no,1-Q,2-encoded', default=5)
    parser.add_argument('-wk', '--workers', type=int, help='0-no,1-Q,2-encoded', default=5)
    parser.add_argument('-m', '--model', type=str, help='sep2d,share2d,sep1d,share1d', default='share2d')
    parser.add_argument('-l', '--load', type=str, help='sep2d,share2d,sep1d,share1d,', default=None)
    parser.add_argument('-pp', '--ppe', type=str, help='use priority replay buffer,', default='0')
    parser.add_argument('-envs', '--envs', type=int, help='use priority replay buffer,', default=1)
    parser.add_argument('-at', '--att', type=int, help='use priority replay buffer,', default=10)
    parser.add_argument('-rec', '--rec', type=int, help='use recurent nets', default=0)
    parser.add_argument('-pr', '--pretrain', type=str, help='use one demonstration', default=None)
    parser.add_argument('-cr', '--commr', type=int, help='use one demonstration', default=2)
    parser.add_argument('-cra', '--comma', type=str, help='use one demonstration', default='no')
    args = parser.parse_args()
    
    return args   
        


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
    
    workers = [getAgent("{}".format(i), pars, nrenvs=nrenvs, job=job, experiment=experiment) for i in range(num_workers)]
    for worker in workers:
        worker.perturb_learning_rate(0)

    for epoch in range(1, pars['epochs'] + 1):
        performances = []
        for idx, worker in enumerate(workers):
            print('epoch:',epoch,',Worker %s' % idx, worker.pars['epsteps'])
            worker.train(pars['numep'])
            performance = worker.test(i_episode=pars['numep']*epoch)
            performances.append(performance)
            if epoch%3==0 and idx==0:
                worker.pars['epsteps'] = min(100, worker.pars['epsteps']+10)

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
        if epoch%3==0:
            pars['epsteps'] = min(100, pars['epsteps']+10)
            agent.pars['epsteps'] = pars['epsteps']
        agent.train(pars['numep'])
        print('epoch', epoch, agent.pars['epsteps'])
        agent.test(i_episode=pars['numep']*epoch)
    agent.save()
    agent.result_out.close()

def getAgent(name, pars, nrenvs, job, experiment):
    cl = {'share2d':AgentShare2D, 'share1d':Agent, 'share2dacrec':AgentACShareRec2D,
          'share1drec':AgentShareRec1D, 'share2ddeQ':Agent2DDecomposeQ,
          'share2dmq':AgentSep2DMessQ,
          'sep2d':AgentSep2D,'ppo':AgentPPOShare2D, 'shared2drec':AgentShareRec2D,
          'share1dac':AgentACShare1D,'share2dac':AgentShare2D}
    return cl[pars['model']](name, pars, nrenvs=nrenvs, job=job, experiment=experiment)

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
        if pars['workers']>1:
            pbt(pars, nrenvs=pars['envs'], job=job, experiment=experiment, num_workers = pars['workers'])
        else:
            agent = getAgent('1', pars, nrenvs=pars['envs'], job=job, experiment=experiment)
            train(agent, pars)
    else:
        job.waitTask(main)