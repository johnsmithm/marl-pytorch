#!/usr/bin/env python3
# encoding=utf-8


import numpy as np
import scipy.misc


class AgentObj:
    def __init__(self, coordinates, type, name, direction=0, mark=0, hidden=0):
        self.x = coordinates[0]
        self.y = coordinates[1]
        #0: r, 1: g, 3: b
        self.type = type
        self.name = name
        self.hidden = hidden

        # 0: right, 1:top 2: left. 3: bottom
        self.direction = direction
        self.mark = mark

    def is_hidden(self):
        #return False
        return self.hidden > 0

    def add_mark(self, agent_hidden):
        self.mark += 1
        if self.mark >= 2:
            self.mark = 0
            self.hidden = agent_hidden
        return self.mark

    def sub_hidden(self):
        self.hidden -= 1
        self.hidden = 0 if self.hidden <=0 else self.hidden
        return self.hidden

    def turn_left(self, **kwargs):
        self.direction = (self.direction + 1) % 4
        return self.direction

    def turn_right(self, **kwargs):
        self.direction = (self.direction - 1 + 4) % 4
        return self.direction

    def move_forward_delta(self):
        if self.direction == 0:
            delta_x, delta_y = 1, 0
        elif self.direction == 1:
            delta_x, delta_y = 0, -1
        elif self.direction == 2:
            delta_x, delta_y = -1, 0
        elif self.direction == 3:
            delta_x, delta_y = 0, 1
        else:
            assert self.direction in range(4), 'wrong direction'

        return delta_x, delta_y

    def move_left_delta(self):
        if self.direction == 0:
            delta_x, delta_y = 0, -1
        elif self.direction == 1:
            delta_x, delta_y = -1, 0
        elif self.direction == 2:
            delta_x, delta_y = 0, 1
        elif self.direction == 3:
            delta_x, delta_y = 1, 0
        else:
            assert self.direction in range(4), 'wrong direction'

        return delta_x, delta_y

    def move_forward(self, env_x_size, env_y_size):
        delta_x, delta_y = self.move_forward_delta()

        self.x = self.x + delta_x if self.x + delta_x >=0 and self.x + delta_x <= env_x_size - 1 else self.x
        self.y = self.y + delta_y if self.y + delta_y >=0 and self.y + delta_y <= env_y_size - 1 else self.y
        return self.x, self.y

    def move_backward(self, env_x_size, env_y_size):
        forward_delta_x, forward_delta_y = self.move_forward_delta()
        delta_x, delta_y = -forward_delta_x, -forward_delta_y

        self.x = self.x + delta_x if self.x + delta_x >= 0 and self.x + delta_x <= env_x_size - 1 else self.x
        self.y = self.y + delta_y if self.y + delta_y >= 0 and self.y + delta_y <= env_y_size - 1 else self.y
        return self.x, self.y

    def move_left(self, env_x_size, env_y_size):
        delta_x, delta_y = self.move_left_delta()

        self.x = self.x + delta_x if self.x + delta_x >= 0 and self.x + delta_x <= env_x_size - 1 else self.x
        self.y = self.y + delta_y if self.y + delta_y >= 0 and self.y + delta_y <= env_y_size - 1 else self.y
        return self.x, self.y

    def move_right(self, env_x_size, env_y_size):
        left_delta_x, left_delta_y = self.move_left_delta()
        delta_x, delta_y = -left_delta_x, -left_delta_y

        self.x = self.x + delta_x if self.x + delta_x >= 0 and self.x + delta_x <= env_x_size - 1 else self.x
        self.y = self.y + delta_y if self.y + delta_y >= 0 and self.y + delta_y <= env_y_size - 1 else self.y
        return self.x, self.y

    def stay(self, **kwargs):
        pass

    def beam(self, env_x_size, env_y_size):
        if self.direction == 0:
            beam_set = [(i + 1, self.y) for i in range(self.x, env_x_size - 1)]
        elif self.direction == 1:
            beam_set = [(self.x, i - 1) for i in range(self.y, 0, -1)]
        elif self.direction == 2:
            beam_set = [(i - 1, self.y) for i in range(self.x, 0, -1)]
        elif self.direction == 3:
            beam_set = [(self.x, i + 1) for i in range(self.y, env_y_size - 1)]
        else:
            assert self.direction in range(4), 'wrong direction'
        return beam_set


class FoodObj:
    def __init__(self, coordinates, type=1, hidden=0, reward=1):
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.type = type
        self.hidden = hidden
        self.reward = reward

    def is_hidden(self):
        return self.hidden > 0

    def eat(self, food_hidden):
        self.hidden = food_hidden
        return self.reward

    def sub_hidden(self):
        self.hidden -= 1
        self.hidden = 0 if self.hidden <= 0 else self.hidden
        return self.hidden



class GameEnv:
    def __init__(self, subhid=0.5, widht=31, hight=11, agent_hidden=5, food_hidden=4):
        self.size_x = widht
        self.size_y = hight
        self.objects = []
        self.agent_hidden = agent_hidden
        self.food_hidden = food_hidden
        self.subhid=subhid

        # 0: forward, 1: backward, 2: left, 3: right
        # 4: trun lelf, 5:turn right, 6: beam, 7: stay
        self.action_num = 8

        self.reset()

    def reset(self):
        self.agent1 = AgentObj(coordinates=(0, 5), type=2, name='agent1')
        self.agent2 = AgentObj(coordinates=(30, 5), type=0, name='agent2', direction=2)
        self.agent1_actions = [self.agent1.move_forward, self.agent1.move_backward, self.agent1.move_left, self.agent1.move_right,
                               self.agent1.turn_left, self.agent1.turn_right, self.agent1.beam, self.agent1.stay]
        self.agent2_actions = [self.agent2.move_forward, self.agent2.move_backward, self.agent2.move_left, self.agent2.move_right,
                               self.agent2.turn_left, self.agent2.turn_right, self.agent2.beam, self.agent2.stay]
        self.agent1_beam_set = []
        self.agent2_beam_set = []

        self.food_objects = []
        a1 = np.zeros((5,5),dtype=np.int32)-1
        self.m1 = []
        self.m2 = []
        for x in range(13, 18):
            delta = x - 13 if x -13 < 17 - x else 17 -x
            self.food_objects.append(FoodObj(coordinates=(x, 5)))
            a1[x-13,5-3] = len(self.food_objects)-1
            for i in range(delta):
                self.food_objects.append(FoodObj(coordinates=(x, 4 - i)))
                a1[x-13,4-i-3] = len(self.food_objects)-1
                self.food_objects.append(FoodObj(coordinates=(x, 6 + i)))
                a1[x-13,6+i-3] = len(self.food_objects)-1
        for i in a1:
            for j in i:
                if j >=0:
                    self.m1.append(j)
        for i in a1[::-1]:
            for j in i[::-1]:
                if j >=0:
                    self.m2.append(j)
        #print(self.m1, self.m2)
    def getState(self):
        a = [self.agent1.x, self.agent1.y, self.agent2.x, self.agent2.y]
        for food in self.food_objects:
            a.append(food.hidden )
        return a
    def getFrom(self, a):
        
        self.agent1 = AgentObj(coordinates=(a[0], a[1]), type=2, name='agent1')
        self.agent2 = AgentObj(coordinates=(a[2], a[3]), type=0, name='agent2', direction=2)
        self.agent1_actions = [self.agent1.move_forward, self.agent1.move_backward, self.agent1.move_left, self.agent1.move_right,
                               self.agent1.turn_left, self.agent1.turn_right, self.agent1.beam, self.agent1.stay]
        self.agent2_actions = [self.agent2.move_forward, self.agent2.move_backward, self.agent2.move_left, self.agent2.move_right,
                               self.agent2.turn_left, self.agent2.turn_right, self.agent2.beam, self.agent2.stay]
        for i,h in enumerate(a[4:]):
            self.food_objects[i].hidden = h
                
    def move(self, agent1_action, agent2_action):
        assert agent1_action in range(8), 'agent1 take wrong action'
        assert agent2_action in range(8), 'agent2 take wrong action'

        agent1_old_x, agent1_old_y = self.agent1.x, self.agent1.y
        agent2_old_x, agent2_old_y = self.agent2.x, self.agent2.y

        #self.agent1.sub_hidden()
        #self.agent2.sub_hidden()

        self.agent1_beam_set = []
        self.agent2_beam_set = []
        if not self.agent1.is_hidden():
            agent1_action_return = self.agent1_actions[agent1_action](env_x_size=self.size_x, env_y_size=self.size_y)
            self.agent1_beam_set = [] if agent1_action != 6 else agent1_action_return
        if not self.agent2.is_hidden():
            agent2_action_return = self.agent2_actions[agent2_action](env_x_size=self.size_x, env_y_size=self.size_y)
            self.agent2_beam_set = [] if agent2_action != 6 else agent2_action_return

        if not self.agent1.is_hidden() and not self.agent2.is_hidden() and\
                ((self.agent1.x == self.agent2.x and self.agent1.y == self.agent2.y) or
                     (self.agent1.x == agent2_old_x and self.agent1.y == agent2_old_y and
                              self.agent2.x == agent1_old_x and self.agent2.y == agent1_old_y)):

            self.agent1.x, self.agent1.y = agent1_old_x, agent1_old_y
            self.agent2.x, self.agent2.y = agent2_old_x, agent2_old_y

        agent1_reward = 0
        agent2_reward = 0
        for food in self.food_objects:
            if food.is_hidden() and np.random.rand()<self.subhid:
                food.sub_hidden()
            if not food.is_hidden():
                if not self.agent1.is_hidden() and food.x == self.agent1.x and food.y == self.agent1.y:
                    agent1_reward = food.eat(self.food_hidden)
                elif not self.agent2.is_hidden() and  food.x == self.agent2.x and food.y == self.agent2.y:
                    agent2_reward = food.eat(self.food_hidden)

        if (self.agent1.x, self.agent1.y) in self.agent2_beam_set:
            self.agent1.add_mark(self.agent_hidden)
        if (self.agent2.x, self.agent2.y) in self.agent1_beam_set:
            self.agent2.add_mark(self.agent_hidden)

        return agent1_reward, agent2_reward

    def contribute_metrix(self, ag=None):
        a = np.ones([self.size_y + 2, self.size_x + 2, 3])
        a[1:-1, 1:-1, :] = 0

        for x, y in self.agent1_beam_set:
            a[y + 1, x + 1, 0] = 0.5
            a[y + 1, x + 1, 1] = 0.5
            a[y + 1, x + 1, 2] = 0.5
        for x, y in self.agent2_beam_set:
            a[y + 1, x + 1, 0] = 0.5
            a[y + 1, x + 1, 1] = 0.5
            a[y + 1, x + 1, 2] = 0.5

        for food in self.food_objects:
            if not food.is_hidden():
                for i in range(3):
                    a[food.y + 1, food.x + 1, i] = 1 if i == food.type else 0

        for i in range(3):
            if not self.agent1.is_hidden() or True:
                delta_x, delta_y = self.agent1.move_forward_delta()
                a[self.agent1.y + 1 + delta_y, self.agent1.x + 1 + delta_x, i] = 0.5
            if not self.agent2.is_hidden() or True:
                delta_x, delta_y = self.agent2.move_forward_delta()
                a[self.agent2.y + 1 + delta_y, self.agent2.x + 1 + delta_x, i] = 0.5
            if not self.agent1.is_hidden() or True:
                if ag is not None and ag==1 and True:
                    a[self.agent1.y + 1, self.agent1.x + 1, i] = 1 if i == self.agent2.type else 0
                else:
                    a[self.agent1.y + 1, self.agent1.x + 1, i] = 1 if i == self.agent1.type else 0
            if not self.agent2.is_hidden() or True:
                if ag is not None and ag==1 and True:
                    a[self.agent2.y + 1, self.agent2.x + 1, i] = 1 if i == self.agent1.type else 0
                else:
                    a[self.agent2.y + 1, self.agent2.x + 1, i] = 1 if i == self.agent2.type else 0
                
            #0 and 2
            if ag is not None and i == 1 and False:
                if ag == 0:
                    a[self.agent1.y + 1, self.agent1.x + 1, i] = 1
                else:
                    a[self.agent2.y + 1, self.agent2.x + 1, i] = 1
            #print(self.agent1.is_hidden(),self.agent2.is_hidden(), self.agent2.y + 1, self.agent2.x + 1, self.agent1.y + 1, self.agent1.x + 1)
            if ag is not None and ag==1 and False:
                a[:,:,0] += a[:,:,2]
                a[:,:,2] = a[:,:,0] - a[:,:,2] 
                a[:,:,0] = a[:,:,0] - a[:,:,2] 
                #a[0,0]=1
                #print(1)
            #print(a[:,-10:,0])
            #print(11)
        if ag is not None and ag==1 and True:
                return  a[::-1,::-1,:]
        return a

    def render_env_1d(self, ag=None):
        a = []
        dy = 0
        dx = 0
        vx = 15.5
        vy = 5.5
        if ag is not None and ag==1:
            dx=30
            dy=10
            a.extend([abs(dy-self.agent2.y)-vy , abs(dx-self.agent2.x)-vx, abs(dy-self.agent1.y)-vy, abs(dx-self.agent1.x)-vx])
        else:            
            a.extend([self.agent1.y -vy, self.agent1.x-vx, self.agent2.y-vy , self.agent2.x-vx])
        b = np.array(a[:])
        c = np.zeros((4))+100
        a.append(a[2]-a[0])
        a.append(a[3]-a[1])
        m = self.m2 if  ag is not None and ag==1 else self.m1
        #for food in self.food_objects:
        for i in m:
            food= self.food_objects[i]
            #print(food.x,food.y)
            a.extend([abs(dy-food.y)-vy , abs(dx-food.x) -vx, 1000 if food.is_hidden() else -1000] )
            if False and not food.is_hidden():
                b1 = np.abs(b-np.array((a[-3:-1]+a[-3:-1])))
                c = np.where(b1<np.abs(c), b-(a[-3:-1]+a[-3:-1]), c)
            a.append((food.y-a[0]) if not food.is_hidden() else 1000)
            a.append((food.x-a[1]) if not food.is_hidden() else 1000)
            a.append((food.y-a[2]) if not food.is_hidden() else 1000)
            a.append((food.x-a[3]) if not food.is_hidden() else 1000)
        #a.extend(c.tolist())
        return np.array(a)
    def render_env_5x5(self, ag=None):
        a = self.contribute_metrix(ag=ag)
        if ag is not None and ag==1:
            x = max(0,30-self.agent2.x-1)
            y = max(0,10-self.agent2.y-1)
            
        else:          
            
            x = max(0,self.agent1.x-1)
            y = max(0,self.agent1.y-1)
        if True:
            #print(x,y)
            b = a[y:y+5,x:x+5]
            
            if b.shape[0] != 5 or b.shape[1]!=5:
                #print(b.shape, a.shape, x,y, ag)
                b = np.pad(b, ((0, 5-b.shape[0]), (0, 5-b.shape[1]),(0,0)), 'constant')
            return b
            
    def render_env(self, ag=None):
        a = self.contribute_metrix(ag=ag)
        sc = 1
        b = scipy.misc.imresize(a[:, :, 0], [sc * self.size_y, sc * self.size_x, 1], interp='cubic')
        c = scipy.misc.imresize(a[:, :, 1], [sc * self.size_y, sc * self.size_x, 1], interp='cubic')
        d = scipy.misc.imresize(a[:, :, 2], [sc * self.size_y, sc * self.size_x, 1], interp='cubic')

        #a = np.stack([b, c, d], axis=2)
        return a

    def train_render(self, ag=None):
        a = self.contribute_metrix(ag)

        b = scipy.misc.imresize(a[:, :, 0], [84, 84, 1], interp='nearest')
        c = scipy.misc.imresize(a[:, :, 1], [84, 84, 1], interp='nearest')
        d = scipy.misc.imresize(a[:, :, 2], [84, 84, 1], interp='nearest')

        a = np.stack([b, c, d], axis=2)
        return a
