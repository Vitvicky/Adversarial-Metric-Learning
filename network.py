import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size() # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image


class Unflatten(nn.Module):
    """
    An Unflatten module receives an input of shape (N, C*H*W) and reshapes it
    to produce an output of shape (N, C, H, W).
    """
    def __init__(self, N=-1, C=128, H=7, W=7):
        super(Unflatten, self).__init__()
        self.N = N
        self.C = C
        self.H = H
        self.W = W

    def forward(self, x):
        return x.view(self.N, self.C, self.H, self.W)


def initialize_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
        init.xavier_uniform(m.weight.data)

# def initialize_weights(model):
#     for m in model.modules():
#         if isinstance(m, nn.Conv2d):
#             m.weight.data.normal_(0, 0.02)
#             m.bias.data.zero_()
#         elif isinstance(m, nn.ConvTranspose2d):
#             m.weight.data.normal_(0, 0.02)
#             m.bias.data.zero_()
#         elif isinstance(m, nn.Linear):
#             m.weight.data.normal_(0, 0.02)
#             m.bias.data.zero_()


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


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.fc = nn.Sequential(nn.Linear(784, 512),
                                nn.ReLU(),
                                nn.Linear(512, 256),
                                nn.ReLU(),
                                nn.Linear(256, 100)
                                )

    def forward(self, x):
        out = self.fc(x)

        return out

    def get_embedding(self, x):
        return self.forward(x)


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        out1 = self.embedding_net(x1)
        out2 = self.embedding_net(x2)
        out3 = self.embedding_net(x3)
        # dist_a = F.pairwise_distance(output1, output2, 2)
        # dist_b = F.pairwise_distance(output1, output3, 2)
        return out1, out2, out3

    # def get_embedding(self, x):
    #     return self.embedding_net(x)
