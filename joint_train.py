import torch
import numpy as np
from network import *
from loss import *
from network import *
from loss import *
import torch.optim as optim
from torch.autograd.variable import Variable


generate = GeneratorNet()
generate.cuda()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_generator_input_sampler():
    return lambda m, n: torch.rand(m, n)


def noise(size):
    n = Variable(torch.randn(size, 784)).cuda()
    return n


def train(train_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()

    # switch to train mode
    model.train()
    for batch_idx, (data1, data2, data3) in enumerate(train_loader):
        data1, data2, data3 = data1.cuda(), data2.cuda(), data3.cuda()
        N = data1.size(0)
        # print("N is: ", N)
        # compute output
        embedded_x, embedded_y, embedded_z = model(data1, data2, data3)
        noise_data = noise(N)
        # print("noise data is: ", noise_data)
        fake_data = generate(noise_data)
        # print(fake_data)
        # loss_triplet = criterion(embedded_x, embedded_y, embedded_z)
        loss = criterion(embedded_x, fake_data, embedded_y, embedded_z)
        losses.update(loss.data[0], data1.size(0))
        #
        # # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 5 == 0:
            print('Train Epoch: {} [{}/{}]\t'
                  'Loss: {:.4f} ({:.4f})'.format(
                epoch, batch_idx * len(data1), len(train_loader.dataset),
                losses.val, losses.avg))


if __name__ == "__main__":
    dataset_path = '/home/wzy/Coding/Data/metric_learning/mnist_normal.csv'
    dataset, classes = read_dataset(dataset_path)
    class_count = len(classes)
    test_data = dataset[:100]
    margin = 1
    lambda1 = 0.1
    lambda2 = 0.5
    epochs = 5

    triplet_dataset = TripletMNIST(test_data, 200)
    net = EmbeddingNet()
    model = TripletNet(net)
    model.cuda()

    # criterion_triplet = TripletLoss(margin)
    criterion = generateLoss(margin, lambda1, lambda2)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    train_dataloader = DataLoader(dataset=triplet_dataset, shuffle=True, batch_size=10)

    for epoch in range(1, epochs + 1):
        # train for one epoch
        train(train_dataloader, model, criterion, optimizer, epoch)
