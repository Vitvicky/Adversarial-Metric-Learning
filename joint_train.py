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


def extract(v):
    return v.data.storage().tolist()


def stats(d):
    return [np.mean(d), np.std(d)]


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


# use real samples' pair/triplet to get pre-train model
def pre_train(pre_train_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()

    # switch to train mode
    model.train()
    for batch_idx, (data1, data2, data3) in enumerate(pre_train_loader):
        data1, data2, data3 = data1.cuda(), data2.cuda(), data3.cuda()
        embedded_x, embedded_y, embedded_z = model(data1, data2, data3)
        loss = criterion(embedded_x, embedded_y, embedded_z)
        losses.update(loss.data[0], data1.size(0))
        #
        # # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 5 == 0:
            print('Pre-Train Epoch: {} [{}/{}]\t'
                  'Loss: {:.4f} ({:.4f})'.format(epoch, batch_idx * len(data1),
                   len(pre_train_loader.dataset), losses.val, losses.avg))


def train_metric(data1, data2, fake_data, metric_criterion, metric_optimizer):
    metric_optimizer.zero_grad()
    loss = metric_criterion(data1, data2, fake_data)
    loss.backward(retain_graph=True)
    metric_optimizer.step()
    return loss


def train_generator(data1, data2, data3, fake_data, generator_criterion, generator_optimizer):
    generator_optimizer.zero_grad()
    loss = generator_criterion(data1, fake_data, data2, data3)
    loss.backward(retain_graph=True)
    generator_optimizer.step()
    return loss


def train(train_loader, model, criterion1, criterion2, optimizer1, optimizer2, epoch):
    loss_adv = AverageMeter()
    loss_metric = AverageMeter()
    lambda_all = 1.0
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

        # train metric on real triplet
        optimizer1.zero_grad()
        # loss_triplet1 = criterion1(embedded_x, embedded_y, embedded_z)
        # loss_triplet1.backward(retain_graph=True)

        # train on adversarial triplet
        loss_triplet = criterion1(embedded_x, embedded_y, fake_data)
        # loss_triplet.backward(retain_graph=True)
        loss_metric.update(loss_triplet.data[0], embedded_x.size(0))
        # optimizer1.step()

        # train generator
        optimizer2.zero_grad()
        loss_generate = criterion2(embedded_x, fake_data, embedded_y, embedded_z)
        loss_adv.update(loss_generate.data[0], embedded_x.size(0))

        loss = loss_triplet + loss_generate
        loss.backward(retain_graph=True)
        optimizer1.step()
        loss.backward(retain_graph=True)
        optimizer2.step()
        # joint train
        # loss = loss_triplet + lambda_over * loss_generate

        if batch_idx % 5 == 0:
            print('Train Epoch: {} [{}/{}]\t'
                  'metric & gen Loss: {:.4f} & {:.4f}\t'.format(
                epoch, batch_idx * len(data1), len(train_loader.dataset), loss_metric.avg, loss_adv.avg))


if __name__ == "__main__":
    dataset_path = '/home/wzy/Coding/Data/metric_learning/mnist_normal.csv'
    dataset, classes = read_dataset(dataset_path)
    class_count = len(classes)
    pre_train_data = dataset[:1000]
    train_data = dataset[1000:2000]
    margin = 1
    lambda1 = 1
    lambda2 = 50
    pre_epochs = 10
    # often setting to more than 10000
    train_epochs = 100

    pre_train_dataset = TripletMNIST(pre_train_data, 2000)
    train_dataset = TripletMNIST(train_data, 2000)
    net = EmbeddingNet()
    model = TripletNet(net)
    model.cuda()

    criterion_triplet = TripletLoss(margin)
    # criterion = generateLoss(margin, lambda1, lambda2)
    criterion_g = generateLoss(margin, lambda1, lambda2)
    # optimizer_triplet = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer_triplet = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer_g = optim.Adam(generate.parameters(), lr=0.0002)
    pre_dataloader = DataLoader(dataset=pre_train_dataset, shuffle=True, batch_size=64)
    train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=64)

    # first, do pre-train
    print("start pre-train")
    for epoch in range(1, pre_epochs + 1):
        # train for one epoch
        pre_train(pre_dataloader, model, criterion_triplet, optimizer_triplet, epoch)

    # start joint train g and metric
    print("start train metric and adversarial")
    for epoch in range(1, train_epochs + 1):
        train(train_dataloader, model, criterion_triplet, criterion_g, optimizer_triplet, optimizer_g, epoch)
