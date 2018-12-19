import torch
import numpy as np
from network import *
from loss import *
from network import *
from loss import *
import torch.optim as optim
from torch.autograd.variable import Variable
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import svm
from torch.optim import lr_scheduler

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
def pre_train(pre_train_loader, model, criterion, optimizer, cuda, log_interval, metrics):
    # switch to train mode
    model.train()
    print("length of data loader: ")
    losses = []
    total_loss = 0

    for batch_idx, (data, target) in enumerate(pre_train_loader):
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda() for d in data)

        optimizer.zero_grad()
        outputs = model(*data)
        if type(outputs) not in (tuple, list):
            outputs = (outputs,)
        loss_inputs = outputs

        loss_outputs = criterion(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        for metric in metrics:
            metric(outputs, target, loss_outputs)

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data[0]), len(pre_train_loader.dataset),
                100. * batch_idx / len(pre_train_loader), np.mean(losses))
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            print(message)
            losses = []

    total_loss /= (batch_idx + 1)
    return total_loss, metrics


def train_metric(data1, data2, fake_data, metric_criterion, metric_optimizer):
    metric_optimizer.zero_grad()
    loss = metric_criterion(data1, data2, fake_data)
    loss.backward()
    metric_optimizer.step()
    return loss


def train_generator(data1, data2, data3, fake_data, generator_criterion, generator_optimizer):
    generator_optimizer.zero_grad()
    loss = generator_criterion(data1, fake_data, data2, data3)
    loss.backward()
    generator_optimizer.step()
    return loss


def train(train_loader, model, criterion_metric, criterion_gen, optimizer_metric, optimizer_gen, epoch):
    losses_metric = AverageMeter()
    losses_gen = AverageMeter()

    # switch to train mode
    model.train()
    for batch_idx, (data1, data2, data3) in enumerate(train_loader):
        data1, data2, data3 = data1.cuda(), data2.cuda(), data3.cuda()
        N = data1.size(0)
        # print("N is: ", N)
        # compute output
        embedded_x, embedded_y, embedded_z = model(data1, data2, data3)
        noise_data = noise(N)
        fake_data = generate(noise_data)

        # train metric on real triplet
        metric_loss = train_metric(embedded_x, embedded_y, fake_data.data, criterion_metric, optimizer_metric)

        # train generator
        generator_loss = train_generator(embedded_x.data, embedded_y.data, embedded_z.data,
                                         fake_data, criterion_gen, optimizer_gen)

        # loss = generator_loss + 0.1 * metric_loss
        losses_metric.update(metric_loss.data[0], data1.size(0))
        losses_gen.update(generator_loss.data[0], data1.size(0))

        if batch_idx % 5 == 0:
            print('Train Epoch: {} [{}/{}]\t'
                  'metric & gen Loss: {:.4f} & {:.4f}\t'.format(
                epoch, batch_idx * len(data1), len(train_loader.dataset), losses_metric.avg, losses_gen.avg))
            # print("%s: Metric: %s Generator: " % (epoch, extract(metric_loss)[0], extract(generator_loss)[0]))


def svm_test(X, Y, split):
    svc = svm.SVC(kernel='linear', C=32, gamma=0.1)
    train_x = X[0:split]
    train_y = Y[0:split]

    test_x = X[split:]
    test_y = Y[split:]

    svc.fit(train_x, train_y)
    predictions = svc.predict(test_x)
    accuracy = accuracy_score(test_y, predictions)
    # neigh = KNeighborsClassifier(n_neighbors=10)
    # neigh.fit(train_x, train_y)
    # predictions = neigh.predict(test_x)
    # accuracy = accuracy_score(test_y, predictions)
    return accuracy

if __name__ == "__main__":
    dataset_path = '/home/wzy/Coding/Data/metric_learning/fashion-mnist.csv'
    dataset, classes = read_dataset(dataset_path)
    class_count = len(classes)
    split = 5000
    pre_train_split = split/2
    pre_train_data = dataset[:5000]
    train_data = dataset[3000:10000]
    test_data = dataset
    margin = 0.5
    lambda1 = 1
    lambda2 = 50
    pre_epochs = 120
    # often setting to more than 10000
    train_epochs = 20000

    pre_train_dataset = TripletDataSet(pre_train_data)
    train_dataset = TripletDataSet(train_data)
    test_dataset = Test_Dataset(test_data)
    net = EmbeddingNet()
    model = TripletNet(net)
    model.cuda()

    criterion_triplet = TripletLoss(margin)
    # criterion = generateLoss(margin, lambda1, lambda2)
    criterion_g = generateLoss(margin, lambda1, lambda2)
    # optimizer_triplet = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # optimizer_triplet = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
    optimizer_triplet = optim.Adam(model.parameters(), lr=0.001)
    scheduler = lr_scheduler.StepLR(optimizer_triplet, 10, gamma=0.1, last_epoch=-1)
    optimizer_g = optim.Adam(generate.parameters(), lr=0.0005)
    pre_dataloader = DataLoader(dataset=pre_train_dataset, shuffle=True, batch_size=128)
    train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=64)
    test_dataloader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=1)

    # first, do pre-train
    print("start pre-train")
    start_epoch = 0
    metrics = []
    log_interval = 200
    # train for one epoch
    for epoch in range(0, start_epoch):
        scheduler.step()
    for epoch in range(start_epoch, pre_epochs):
        scheduler.step()
        # pre_train(pre_dataloader, model, criterion_triplet, optimizer_triplet, epoch)
        train_loss, metrics = pre_train(pre_dataloader, model, criterion_triplet,
                                        optimizer_triplet, cuda, log_interval, metrics)
        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, pre_epochs, train_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        print(message)

    # start joint train g and metric
    # print("start train metric and adversarial")
    # for epoch in range(1, train_epochs + 1):
    #     train(train_dataloader, model, criterion_triplet, criterion_g, optimizer_triplet, optimizer_g, epoch)

    # start test
    X = []
    Y = []
    for s in test_dataloader:
        # x = s[0]
        x = s[0].cuda()
        x1, x2, x3 = model(x, x, x)
        X.append(x1.data.cpu().squeeze().numpy())
        # print(int(s[1][0]))
        y = int(s[1][0])
        Y.append(y)

    X = np.array(X)
    print(len(X))
    Y = np.array(Y)
    svm_accuracy = svm_test(X, Y, split)
    print("acc is", svm_accuracy)
