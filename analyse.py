import numpy as np
import json
import matplotlib.animation as animation
import scipy.misc
import cv2, os

from models.models import DQN2D, DQN
from mas import *
import torch
from torch.distributions import Categorical
env = GameEnv(0.6)
env.reset()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = DQN2D(84,84, {'en':100}).to(device)
PATH = 'logs1/pbtacshared2dcommEnAlphav10/model'
PATH = 'logs1/pbtdqnsharedCommEnLv30/model'
policy_net.load_state_dict(torch.load(PATH, map_location= 'cuda' if torch.cuda.is_available() else 'cpu')) 
def getStates( env):
        screen1 = env.train_render(0).transpose((2, 0, 1))
        screen1 = np.ascontiguousarray(screen1, dtype=np.float32) / 255
        screen2 = env.train_render(1).transpose((2, 0, 1))
        screen2 = np.ascontiguousarray(screen2, dtype=np.float32) / 255
        return torch.from_numpy(screen1).unsqueeze(0).to(device),\
    torch.from_numpy(screen2).unsqueeze(0).to(device) 
def select_action( state, comm, policy_net):
        return policy_net(state, 1, comm)[0].max(1)[1].view(1, 1),1,1
        probs1, _ = policy_net(state, 1, comm)#.cpu().data.numpy()
        m = Categorical(logits=probs1)
        action = m.sample()
        return action.view(1, 1), m.log_prob(action), m.entropy()
def getaction( state1, state2, test=False):
        mes = torch.tensor([[0,0,0,0]], device=device)
        #maybe error
        idC = 1
        comm2 = policy_net(state2, 0, mes)[idC] 
        #if (test and  0<self.prob) or np.random.rand()<self.prob else mes
        comm1 = policy_net(state1, 0, mes)[idC] 
        #if (test and  0<self.prob) or np.random.rand()<self.prob else mes
        
        action1, logp1, ent1 = select_action(state1,  comm2, policy_net)
        action2, logp2, ent2 = select_action(state2,  comm1, policy_net)
        rem =[logp1, ent1, logp2, ent2]
        return action1, action2, [comm1, comm2]
db = []
for n in range(40):
    rt = 0
    env.reset()
    for t in range(200):#sep function
                        state1,state2 = getStates(env)
                        action1, action2, r = getaction(state1,state2, test=True)
                        comm1,comm2 = r
                        reward1, reward2 = env.move(action1.item(), action2.item())
                        rt+= reward1+ reward2
                        if t%10==0:
                            print(comm1.cpu().data.numpy(),comm2.cpu().data.numpy(), action1.item(), action2.item())
                        db.append([comm1.cpu().data.numpy()[0].tolist(),comm2.cpu().data.numpy()[0].tolist(), 
                                   action1.item(), action2.item(), reward1, reward2,
                                  env.agent2.x-env.agent1.x, env.agent2.y-env.agent1.y])
    print(n, 'total',rt)
with open(os.path.join('logs1/dbCommEL','pbtdqnsharedCommEnLv30.json'), 'w') as outfile:
        json.dump(db, outfile)