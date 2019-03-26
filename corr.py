import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


with open('logs1/dbCommEL/pbtdqnsharedCommEnLv30.json') as f:
                data = json.load(f)
X = np.array([i[0] for i in data]+[i[1] for i in data])
#Y = np.array([i[4] for i in data]+[i[5] for i in data])#reward
#Y = np.array([i[2] for i in data]+[i[3] for i in data])#action
Y = np.array([1 if abs(i[6])+abs(i[7])>3 else 0 for i in data]+[1 if abs(i[6])+abs(i[7])>3 else 0 for i in data])#distance 
train_X, test_X, train_y, test_y = train_test_split(X,
                                                    Y, test_size=0.3)
train_X = Variable(torch.Tensor(train_X).float())
test_X = Variable(torch.Tensor(test_X).float())
train_y = Variable(torch.Tensor(train_y).long())
test_y = Variable(torch.Tensor(test_y).long())


class Net(nn.Module):
    # define nn
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.relu(self.fc3(X))
        X = self.softmax(X)

        return X

net = Net()

criterion = nn.CrossEntropyLoss()# cross entropy loss

optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

for epoch in range(8000):
    optimizer.zero_grad()
    out = net(train_X)
    loss = criterion(out, train_y)
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print ('number of epoch', epoch, 'loss', loss.item())
    if epoch % 1000 == 0:
        predict_out = net(test_X)
        _, predict_y = torch.max(predict_out, 1)

        print ('prediction accuracy', accuracy_score(test_y.data, predict_y.data))

print ('macro precision', precision_score(test_y.data, predict_y.data, average='macro'))
print ('micro precision', precision_score(test_y.data, predict_y.data, average='micro'))
print ('macro recall', recall_score(test_y.data, predict_y.data, average='macro'))
print ('micro recall', recall_score(test_y.data, predict_y.data, average='micro'))