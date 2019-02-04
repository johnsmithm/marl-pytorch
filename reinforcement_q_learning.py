
# coding: utf-8

# In[3]:




# 
# Reinforcement Learning (DQN) Tutorial
# =====================================
# **Author**: `Adam Paszke <https://github.com/apaszke>`_
# 
# 
# This tutorial shows how to use PyTorch to train a Deep Q Learning (DQN) agent
# on the CartPole-v0 task from the `OpenAI Gym <https://gym.openai.com/>`__.
# 
# **Task**
# 
# The agent has to decide between two actions - moving the cart left or
# right - so that the pole attached to it stays upright. You can find an
# official leaderboard with various algorithms and visualizations at the
# `Gym website <https://gym.openai.com/envs/CartPole-v0>`__.
# 
# .. figure:: /_static/img/cartpole.gif
#    :alt: cartpole
# 
#    cartpole
# 
# As the agent observes the current state of the environment and chooses
# an action, the environment *transitions* to a new state, and also
# returns a reward that indicates the consequences of the action. In this
# task, rewards are +1 for every incremental timestep and the environment
# terminates if the pole falls over too far or the cart moves more then 2.4
# units away from center. This means better performing scenarios will run
# for longer duration, accumulating larger return.
# 
# The CartPole task is designed so that the inputs to the agent are 4 real
# values representing the environment state (position, velocity, etc.).
# However, neural networks can solve the task purely by looking at the
# scene, so we'll use a patch of the screen centered on the cart as an
# input. Because of this, our results aren't directly comparable to the
# ones from the official leaderboard - our task is much harder.
# Unfortunately this does slow down the training, because we have to
# render all the frames.
# 
# Strictly speaking, we will present the state as the difference between
# the current screen patch and the previous one. This will allow the agent
# to take the velocity of the pole into account from one image.
# 
# **Packages**
# 
# 
# First, let's import needed packages. Firstly, we need
# `gym <https://gym.openai.com/docs>`__ for the environment
# (Install using `pip install gym`).
# We'll also use the following from PyTorch:
# 
# -  neural networks (``torch.nn``)
# -  optimization (``torch.optim``)
# -  automatic differentiation (``torch.autograd``)
# -  utilities for vision tasks (``torchvision`` - `a separate
#    package <https://github.com/pytorch/vision>`__).
# 
# 
# 

# In[2]:


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

env = GameEnv()
env.reset()

temp = env.render_env()

#env = gym.make('CartPole-v0').unwrapped



# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Replay Memory
# -------------
# 
# We'll be using experience replay memory for training our DQN. It stores
# the transitions that the agent observes, allowing us to reuse this data
# later. By sampling from it randomly, the transitions that build up a
# batch are decorrelated. It has been shown that this greatly stabilizes
# and improves the DQN training procedure.
# 
# For this, we're going to need two classses:
# 
# -  ``Transition`` - a named tuple representing a single transition in
#    our environment. It maps essentially maps (state, action) pairs
#    to their (next_state, reward) result, with the state being the
#    screen difference image as described later on.
# -  ``ReplayMemory`` - a cyclic buffer of bounded size that holds the
#    transitions observed recently. It also implements a ``.sample()``
#    method for selecting a random batch of transitions for training.
# 
# 
# 

# In[15]:


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
        if np.random.rand()>0.05:
            self.memory = self.memory[-self.capacity:]
        x = random.sample(self.memory, batch_size//2)
        x1 = []
        i = np.random.randint(max(0,len(self.memory)-batch_size*2))
        n = 0
        while len(x1)<batch_size//2 and i<len(self.memory) and n<100:
            n+=1
            if self.memory[i].reward>0:
                x1.append(self.memory[i])
            i+=1   
        i = np.random.randint(max(0,len(self.memory)-batch_size*2))
        while len(x1)<batch_size//2 :
                x1.append(self.memory[i])
                i+=1
        #print(x+x1)
        return x+x1

    def __len__(self):
        return len(self.memory)


# Now, let's define our model. But first, let quickly recap what a DQN is.
# 
# DQN algorithm
# -------------
# 
# Our environment is deterministic, so all equations presented here are
# also formulated deterministically for the sake of simplicity. In the
# reinforcement learning literature, they would also contain expectations
# over stochastic transitions in the environment.
# 
# Our aim will be to train a policy that tries to maximize the discounted,
# cumulative reward
# $R_{t_0} = \sum_{t=t_0}^{\infty} \gamma^{t - t_0} r_t$, where
# $R_{t_0}$ is also known as the *return*. The discount,
# $\gamma$, should be a constant between $0$ and $1$
# that ensures the sum converges. It makes rewards from the uncertain far
# future less important for our agent than the ones in the near future
# that it can be fairly confident about.
# 
# The main idea behind Q-learning is that if we had a function
# $Q^*: State \times Action \rightarrow \mathbb{R}$, that could tell
# us what our return would be, if we were to take an action in a given
# state, then we could easily construct a policy that maximizes our
# rewards:
# 
# \begin{align}\pi^*(s) = \arg\!\max_a \ Q^*(s, a)\end{align}
# 
# However, we don't know everything about the world, so we don't have
# access to $Q^*$. But, since neural networks are universal function
# approximators, we can simply create one and train it to resemble
# $Q^*$.
# 
# For our training update rule, we'll use a fact that every $Q$
# function for some policy obeys the Bellman equation:
# 
# \begin{align}Q^{\pi}(s, a) = r + \gamma Q^{\pi}(s', \pi(s'))\end{align}
# 
# The difference between the two sides of the equality is known as the
# temporal difference error, $\delta$:
# 
# \begin{align}\delta = Q(s, a) - (r + \gamma \max_a Q(s', a))\end{align}
# 
# To minimise this error, we will use the `Huber
# loss <https://en.wikipedia.org/wiki/Huber_loss>`__. The Huber loss acts
# like the mean squared error when the error is small, but like the mean
# absolute error when the error is large - this makes it more robust to
# outliers when the estimates of $Q$ are very noisy. We calculate
# this over a batch of transitions, $B$, sampled from the replay
# memory:
# 
# \begin{align}\mathcal{L} = \frac{1}{|B|}\sum_{(s, a, s', r) \ \in \ B} \mathcal{L}(\delta)\end{align}
# 
# \begin{align}\text{where} \quad \mathcal{L}(\delta) = \begin{cases}
#      \frac{1}{2}{\delta^2}  & \text{for } |\delta| \le 1, \\
#      |\delta| - \frac{1}{2} & \text{otherwise.}
#    \end{cases}\end{align}
# 
# Q-network
# ^^^^^^^^^
# 
# Our model will be a convolutional neural network that takes in the
# difference between the current and previous screen patches. It has two
# outputs, representing $Q(s, \mathrm{left})$ and
# $Q(s, \mathrm{right})$ (where $s$ is the input to the
# network). In effect, the network is trying to predict the *expected return* of
# taking each action given the current input.
# 
# 
# 

# In[16]:


class DQN(nn.Module):

    def __init__(self, h, w):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, 8) # 448 or 512
        
        self.agent_lookup = nn.Embedding(2, linear_input_size)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x, agent_index):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        z_a = self.agent_lookup(agent_index)
        return self.head(x.view(x.size(0), -1)+z_a)


# Input extraction
# ^^^^^^^^^^^^^^^^
# 
# The code below are utilities for extracting and processing rendered
# images from the environment. It uses the ``torchvision`` package, which
# makes it easy to compose image transforms. Once you run the cell it will
# display an example patch that it extracted.
# 
# 
# 

# In[13]:


resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])


def get_cart_location(screen_width):
    world_width = 84#env.x_threshold * 2
    scale = screen_width / world_width
    return int( screen_width / 2.0)  # MIDDLE OF CART

def get_screen():
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.train_render().transpose((2, 0, 1))#env.render(mode='rgb_array').transpose((2, 0, 1))
    # Cart is in the lower half, so strip off the top and bottom of the screen
    #return screen
    _, screen_height, screen_width = screen.shape
    #screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(screen_width)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    # Strip off the edges, so that we have a square image centered on a cart
    #screen = screen[:, :, slice_range]
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return screen.unsqueeze(0).to(device)


env.reset()

e = get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy()
print(e.shape, np.amax(e), np.amin(e))
#plt.show()


# Training
# --------
# 
# Hyperparameters and utilities
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# This cell instantiates our model and its optimizer, and defines some
# utilities:
# 
# -  ``select_action`` - will select an action accordingly to an epsilon
#    greedy policy. Simply put, we'll sometimes use our model for choosing
#    the action, and sometimes we'll just sample one uniformly. The
#    probability of choosing a random action will start at ``EPS_START``
#    and will decay exponentially towards ``EPS_END``. ``EPS_DECAY``
#    controls the rate of the decay.
# -  ``plot_durations`` - a helper for plotting the durations of episodes,
#    along with an average over the last 100 episodes (the measure used in
#    the official evaluations). The plot will be underneath the cell
#    containing the main training loop, and will update after every
#    episode.
# 
# 
# 

# In[17]:


if True:

    BATCH_SIZE = 10
    GAMMA = 0.999
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 200
    TARGET_UPDATE = 10
    
    steps_done = 0
    memory = None
    policy_net, target_net, optimizer = [None]*3

def select_action(state, id):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) *         math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state, torch.tensor([id], device=device)).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(8)]], device=device, dtype=torch.long)


episode_durations = []


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


# Training loop
# ^^^^^^^^^^^^^
# 
# Finally, the code for training our model.
# 
# Here, you can find an ``optimize_model`` function that performs a
# single step of the optimization. It first samples a batch, concatenates
# all the tensors into a single one, computes $Q(s_t, a_t)$ and
# $V(s_{t+1}) = \max_a Q(s_{t+1}, a)$, and combines them into our
# loss. By defition we set $V(s) = 0$ if $s$ is a terminal
# state. We also use a target network to compute $V(s_{t+1})$ for
# added stability. The target network has its weights kept frozen most of
# the time, but is updated with the policy network's weights every so often.
# This is usually a set number of steps but we shall use episodes for
# simplicity.
# 
# 
# 

# In[23]:


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    agent_index = torch.cat(batch.agent_index)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch, agent_index).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states, agent_index).max(1)[0].detach()
    # Compute the expected Q values
    #print(next_state_values, reward_batch)
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch.float()

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


# Below, you can find the main training loop. At the beginning we reset
# the environment and initialize the ``state`` Tensor. Then, we sample
# an action, execute it, observe the next screen and the reward (always
# 1), and optimize our model once. When the episode ends (our model
# fails), we restart the loop.
# 
# Below, `num_episodes` is set small. You should download
# the notebook and run lot more epsiodes, such as 300+ for meaningful
# duration improvements.
# 
# 
# 

# In[ ]:


def getA():
    parser = argparse.ArgumentParser(description='Test a network')
    parser.add_argument('--bs', dest='bs',
                help='file to load the db state', default=10, type=int)
    parser.add_argument('--tg', dest='tg',
                help='file to load the db state', default=10, type=int)
    parser.add_argument('--numep', dest='numep',
                help='file to load the db state', default=500, type=int)
    parser.add_argument('--epsteps', dest='epsteps',
                help='file to load the db state', default=100, type=int)
    parser.add_argument('--show', dest='show',
                help='file to load the db state', default=10, type=int)
    parser.add_argument('--nrf', dest='nrf',
                help='file to load the db state', default=10, type=int)
    
    parser.add_argument('-r', '--results_path', type=str, help='path to results directory', default='logs/t3')
    args = parser.parse_args()
    
    return args


import time, os, datetime, json
import copy, argparse, csv, json, datetime, os
from functools import partial
from pathlib import Path

def save_episode_and_reward_to_csv(file, writer, e, r, ep):
    #[env.render_env(), action.item(), reward]
    #print(ep.step_records[27].agent_inputs[0]['s_t'].data.numpy())
    data = {'eps':[i[0].tolist() for i in ep]}
    data ['a']=[i[1] for i in ep]
    data ['r']=[i[2] for i in ep]
    #print(data ['r'])
    data['rewardT'] =r# sum(data ['r'])
    #episode.step_records[step].r_t
    with open(os.path.join(pars['results_path'],'ep:{}.json').format(e), 'w') as outfile:
        json.dump(data, outfile)
    #episode.step_records[step].agent_inputs.append(agent_inputs)#s_t
    writer.writerow({'episode': e, 'reward': r})
    file.flush()

def main(pars):
    global memory, BATCH_SIZE, TARGET_UPDATE, policy_net, target_net, optimizer
    BATCH_SIZE = pars['bs']
    GAMMA = 0.999
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 200
    TARGET_UPDATE = pars['tg']
    nrf = pars['nrf']

    # Get screen size so that we can initialize layers correctly based on shape
    # returned from AI gym. Typical dimensions at this point are close to 3x40x90
    # which is the result of a clamped and down-scaled render buffer in get_screen()
    init_screen = get_screen()
    _, _, screen_height, screen_width = init_screen.shape

    policy_net = DQN(screen_height, screen_width).to(device)
    target_net = DQN(screen_height, screen_width).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    #RMSprop
    optimizer = optim.Adam(policy_net.parameters())
    memory = ReplayMemory(10000)

    if args.results_path:
        if not os.path.exists(args.results_path):
            os.makedirs(args.results_path)
        result_path =  args.results_path
    result_path = result_path + '/results_' + str(0) + '.csv'
    result_out = open(result_path, 'w')
    csv_meta = '#' + json.dumps(pars) + '\n'    
    result_out.write(csv_meta)    
    writer = csv.DictWriter(result_out, fieldnames=['episode', 'reward'])
    writer.writeheader()

    steps_done = 0
    num_episodes = pars['numep']
    done = False
    for i_episode in range(num_episodes):
        # Initialize the environment and state
        env.reset()
        last_screen = get_screen()
        current_screen = get_screen()
        state = get_screen()# current_screen - last_screen
        rt = 0; ac=[]
        start_time = time.time()
        buf= []; retime=[]; ep1=[]
        for t in range(pars['epsteps']):# count():
            # Select and perform an action
            action1 = select_action(state, 0)
            #_, reward, done, _ = env.step(action.item())
            action2 = select_action(state, 1)
            reward1, reward2 = env.move(action1.item(), action2.item())
            rt+=reward1+reward2;
            ac.append(str(action1.item()))

            #ep1.append([env.render_env(), action.item(), reward])
            reward1 = torch.tensor([reward1], device=device)
            reward2 = torch.tensor([reward2], device=device)
            id1 = torch.tensor([0], device=device)
            id2 = torch.tensor([1], device=device)

            retime.append(0)#reward)
            # Store the transition in memory
            #memory.push(state, action, next_state, reward)
            next_state = get_screen()#
            buf.append([state, action1, next_state, reward1, id1])
            buf.append([state, action2, next_state, reward2, id2])
            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            optimize_model()
            if done:
                episode_durations.append(t + 1)
                plot_durations()
                break
        
        for i,it in enumerate(buf[:]):
            memory.push(it[0], it[1], it[2], it[3], it[4])#+sum(retime[i+1:i+nrf])/len(retime[i+1:i+nrf]))
        if rt>5 and False:
            for j in range(3):
                for i,it in enumerate(buf[:]):
                    memory.push(it[0], it[1], it[2], it[3], it[4])#+sum(retime[i+1:i+nrf])/len(retime[i+1:i+nrf]))
        print('ep',i_episode, 'reward train',rt, 'time', time.time() - start_time, ','.join(ac[:20]))
        if i_episode%pars['show']==0:
            ep = []
            rt1=0
            env.reset()
            for t in range(pars['epsteps']):# count():
                # Select and perform an action
                state = get_screen()
                action1 = policy_net(state, torch.tensor([0], device=device)).max(1)[1].view(1, 1)
                action2 = policy_net(state, torch.tensor([1], device=device)).max(1)[1].view(1, 1)
                #_, reward, done, _ = env.step(action.item())
                #action2 = np.random.randint(8)
                reward1, reward2 = env.move(action1.item(), action2.item())
                ep.append([env.render_env(), [action1.item(), action2.item()], [reward1, reward2]])
                #print(env.render_env().shape)
                rt1+=reward1+reward2;
            save_episode_and_reward_to_csv(result_out, writer, i_episode, rt1, ep)
            print( 'reward test', rt1)
        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
    result_out.close()
    print('Complete')


# Here is the diagram that illustrates the overall resulting data flow.
# 
# .. figure:: /_static/img/reinforcement_learning_diagram.jpg
# 
# Actions are chosen either randomly or based on a policy, getting the next
# step sample from the gym environment. We record the results in the
# replay memory and also run optimization step on every iteration.
# Optimization picks a random batch from the replay memory to do training of the
# new policy. "Older" target_net is also used in optimization to compute the
# expected Q values; it is updated occasionally to keep it current.
# 
# 
# 

if __name__ == '__main__': 

    args = getA()
    pars = vars(args)    
    print(pars)
    main(pars)