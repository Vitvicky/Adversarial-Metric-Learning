import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataset import *


# generate adversarial samples
class GeneratorNet(torch.nn.Module):
    def __init__(self):
        super(GeneratorNet, self).__init__()
        n_features = 100
        n_out = 784

        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.LeakyReLU(0.2)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2)
        )

        self.out = nn.Sequential(
            nn.Linear(1024, n_out),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x


class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        # self.fc1 = nn.Linear(784, 300)
        # self.relu = nn.ReLU()
        # self.fc2 = nn.Linear(300, 100)
        self.fc = nn.Sequential(nn.Linear(784, 512),
                                nn.ReLU(),
                                nn.Linear(512, 256),
                                nn.ReLU(),
                                nn.Linear(256, 100)
                                )

    def forward(self, x):
        out = self.fc(x)
        # out = self.relu(out)
        # out = self.fc2(out)
        # out = self.relu(out)
        # out = self.l2_norm(out)
        # alpha = 10
        # out = out * alpha

        return out

    def get_embedding(self, x):
        return self.forward(x)

    def l2_norm(self, input):
        input_size = input.size()

        temp = torch.pow(input, 2)

        if len(input.size()) > 1:
            normp = torch.sum(temp, 1).add_(1e-10)
        else:
            normp = torch.sum(temp).add_(1e-10)
        norm = torch.sqrt(normp)

        if len(input.size()) > 1:
            _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        else:
            _output = torch.div(input, norm.expand_as(input))

        output = _output.view(input_size)

        return output


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        # dist_a = F.pairwise_distance(output1, output2, 2)
        # dist_b = F.pairwise_distance(output1, output3, 2)
        return output1, output2, output3
        # return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)


