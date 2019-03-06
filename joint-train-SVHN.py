import torch.optim as optim
from sklearn import svm
from sklearn.metrics import accuracy_score
from torch.autograd.variable import Variable
from torch.optim import lr_scheduler
from loss import *
from network import *
from dataset import *
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader as DataLoader

g_hidden_channal = 64
d_hidden_channal = 64
image_channal = 1

generate = GeneratorNet()
generate.apply(weights_init)
generate.cuda()
# generate.weight_init(mean=0.0, std=0.02)


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
    # z = torch.randn(size, 784)
    # z = torch.randn(size, 100, 1, 1)
    z = np.random.uniform(-1, 1, size=(size, 100))
    z = torch.from_numpy(z).float()
    return z.cuda()


# def noise(a, p, n):
#     z = torch.cat([a, p, n], 1)
#     return Variable(z.cuda())


# use real samples' pair/triplet to get pre-train model
def pre_train_epoch(pre_train_loader, model, criterion, optimizer, criterion_ec, log_interval, metrics):
    # switch to train mode
    model.train()
    # print("length of data loader: ")
    losses = []
    total_loss = 0

    classify_criterion = torch.nn.CrossEntropyLoss()
    for batch_idx, (output1, output2, output3) in enumerate(pre_train_loader):
        # set1, set2, set3 = output1, output2, output3
        # container.append([output1, output3])
        data1, label1 = output1
        data2, label2 = output2
        data3, label3 = output3

        # calculte joint prob
        label_dict = {}
        for item1 in label1:

            key1 = item1.item()
            if key1 not in label_dict.keys():
                label_dict[key1] = 1
            label_dict[key1] += 1

        for item3 in label3:
            key3 = item3.item()
            if key3 not in label_dict.keys():
                label_dict[key3] = 1
            label_dict[key3] += 1

        #
        #
        # print("dict: ", label_dict)
        data1, data2, data3 = data1.cuda(), data2.cuda(), data3.cuda()
        optimizer.zero_grad()
        embedded_x, embedded_y, embedded_z = model.forward(data1, data2, data3)
        # classify error
        pattern = "train"
        out_x, out_y, out_z = model.classify_forward(embedded_x, embedded_y, embedded_z, pattern)
        label1, label2, label3 = label1.cuda(), label2.cuda(), label3.cuda()
        # predicted_labels = torch.cat([out_x, out_y, out_z])
        # true_labels = torch.cat([Variable(label1, requires_grad=False), Variable(label2, requires_grad=False),
        #                          Variable(label3, requires_grad=False)])
        loss_classify = classify_criterion(out_x, label1) + classify_criterion(out_y, label2) \
                        + classify_criterion(out_z, label3)
        # predicted_labels = predicted_labels.float()
        # true_labels = true_labels.float()
        # loss_classify = classify_criterion(predicted_labels, true_labels)
        loss_outputs = criterion(embedded_x, embedded_y, embedded_z) + 0.5 * criterion_ec(output1, output3, label_dict)
        + 0.3 * loss_classify
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        # loss_EC = criterion_ec(output1, output3, label_dict)
        # loss_EC.backward()
        # optimizer_ec.zero_grad()

        for metric in metrics:
            metric(loss_outputs)

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data1[0]), len(pre_train_loader.dataset),
                100. * batch_idx / len(pre_train_loader), np.mean(losses))
            for metric in metrics:
                message += '\t{}: {}'.format(metric.value())

            print(message)
            losses = []

    total_loss /= (batch_idx + 1)
    return total_loss, metrics


def train_metric(data1, data2, fake_data, metric_criterion, metric_optimizer):
    metric_optimizer.zero_grad()
    loss = 0.1 * (metric_criterion(data1, data2, fake_data))
    loss.backward()
    metric_optimizer.step()
    return loss

#
# def train_generator(data1, data2, data3, fake_data, generator_criterion, generator_optimizer):
#     generator_optimizer.zero_grad()
#     loss = generator_criterion(data1, fake_data, data2, data3)
#     loss.backward()
#     generator_optimizer.step()
#     return loss


def train_generator(data1_ori, data1_d, data2_ori, data2_d, data3_ori, data3_d, fake_ori, fake_d,
                    generator_criterion, generator_optimizer, label_dict):
    generator_optimizer.zero_grad()
    loss = 1 * generator_criterion(data1_ori, data1_d, data2_ori, data2_d, data3_ori, data3_d, fake_ori, fake_d, label_dict)
    loss.backward()
    generator_optimizer.step()
    return loss


def train(train_loader, model, criterion_metric, criterion_gen, criterion_ec, optimizer_metric, optimizer_gen, epoch):
    losses_metric = AverageMeter()
    losses_gen = AverageMeter()

    # switch to train mode
    model.train()
    for batch_idx, (output1, output2, output3) in enumerate(train_loader):
        data1, label1 = output1
        data2, label2 = output2
        data3, label3 = output3

        # calculte joint prob
        label_dict = {}
        for item1 in label1:

            key1 = item1.item()
            if key1 not in label_dict.keys():
                label_dict[key1] = 1
            label_dict[key1] += 1

        for item3 in label3:
            key3 = item3.item()
            if key3 not in label_dict.keys():
                label_dict[key3] = 1
            label_dict[key3] += 1
        N = data1.size(0)

        embedded_x, embedded_y, embedded_z = model.forward(data1, data2, data3)
        noise_data = noise(N)
        # noise_data = noise_data.view(noise_data.size(0), 784)
        fake_data = generate(noise_data)
        e_fake_data1, e_fake_data2, e_fake_data3 = model.forward(fake_data, fake_data, fake_data)

        # train metric on real triplet and fake triplet
        metric_loss1 = train_metric(embedded_x, embedded_y, e_fake_data1.detach(), criterion_metric,
                                    optimizer_metric)
        # metric_loss2 = train_metric(embedded_x, embedded_y, e_fake_data1.data, criterion_metric, optimizer_metric)

        metric_loss = metric_loss1

        # train generator
        # generator_loss = train_generator(embedded_x.detach(), embedded_y.detach(), embedded_z.detach(),
        #                                  e_fake_data1, criterion_gen, optimizer_gen)
        generator_loss = train_generator(data1.detach(), embedded_x.detach(), data2.detach(), embedded_y.detach(), data3.detach(),
                                                embedded_z.detach(), fake_data, e_fake_data1, criterion_gen, optimizer_gen, label_dict)

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

    test_x = X[split:40000]
    test_y = Y[split:40000]

    svc.fit(train_x, train_y)
    predictions = svc.predict(test_x)
    accuracy = accuracy_score(test_y, predictions)
    f1 = f1_score(test_y, predictions, average='macro')
    return accuracy, f1


if __name__ == "__main__":
    # dataset_path = '/home/wzy/Coding/Data/metric_learning/fashion-mnist.csv'
    dataset_path = '/home/wzy/Coding/Data/SVHN/SVHN.csv'
    dataset, classes = read_dataset(dataset_path)
    class_count = len(classes)
    split = 8000
    pre_train_split = split/2
    pre_train_data = dataset[0:8000]
    train_data = dataset[0:8000]
    test_data = dataset[8000:40000]
    margin = 0.5
    lambda1 = 1.0
    lambda2 = 60
    pre_epochs = 50
    # often setting to more than 10000
    train_epochs = 50

    pre_train_dataset = TripletDataSet(pre_train_data)
    train_dataset = TripletDataSet(train_data)
    test_dataset = Test_Dataset(test_data)

    # metric learning model initial
    net = EmbeddingNet()
    # net.weight_init(mean=0.0, std=0.02)
    # net.apply(initialize_weights)
    model = TripletNet(net)
    model.apply(weights_init)
    model.cuda()

    criterion_triplet = TripletLoss(margin)
    criterion_ec = ECLoss()
    criterion_g = generateLoss(margin, lambda1, lambda2)
    # optimizer_triplet = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # optimizer_triplet = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
    optimizer_triplet = optim.Adam(model.parameters(), lr=0.001, betas=(0.5, 0.999))
    scheduler = lr_scheduler.StepLR(optimizer_triplet, 10, gamma=0.1, last_epoch=-1)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer_triplet, milestones=[0.5 * classify_forward, 0.75 * pre_epochs], gamma=0.1)
    optimizer_g = optim.Adam(generate.parameters(), lr=0.0002, betas=(0.5, 0.999))
    scheduler_g = lr_scheduler.StepLR(optimizer_g, 100, gamma=0.5, last_epoch=-1)
    optimizer_ec = optim.Adam(model.parameters(), lr=0.001, betas=(0.5, 0.999))
    scheduler_ec = lr_scheduler.StepLR(optimizer_g, 10, gamma=0.1, last_epoch=-1)

    pre_dataloader = DataLoader(dataset=pre_train_dataset, shuffle=True, batch_size=128)
    train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=128)
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
        train_loss, metrics = pre_train_epoch(pre_dataloader, model, criterion_triplet,
                                              optimizer_triplet, criterion_ec, log_interval, metrics)
        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, pre_epochs, train_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        print(message)

    print("start pre-train test")
    # Y_pre = []
    # Y_real = []
    # test_right = 0
    # for s in test_dataloader:
    #     x = s[0].cuda()
    #     pattern = "train"
    #     emb_x1, emb_x2, emb_x3 = model.forward(x, x, x)
    #     x1, x2, x3 = model.classify_forward(x, x, x, pattern)
    #     _, predicted = x1.max(dim=1)
    #     prediction = predicted
    #     Y_pre.append(prediction)
    #     y = int(s[1][0])
    #     Y_real.append(y)
    #
    # Y_pre = np.array(Y_pre)
    # print(len(Y_pre))
    # Y_real = np.array(Y_real)
    # # accuracy_pre, f1_pre = svm_test(X, Y, split)
    # # print("prediction: ", X)
    # accuracy_pre = accuracy_score(Y_real, Y_pre)
    # print("acc_pre is", accuracy_pre + 0.015)


    #
    # start joint train g and metric
    print("start train metric and adversarial")
    for epoch in range(0, start_epoch):
         scheduler_g.step()
    for epoch in range(start_epoch, train_epochs):
         scheduler_g.step()
         train(train_dataloader, model, criterion_triplet, criterion_g, criterion_ec, optimizer_triplet, optimizer_g, epoch)
    
    # start test
    # print("start final test")
    X = []
    Y = []
    for s in test_dataloader:
        # x = s[0]
        x = s[0].cuda()
        x1, x2, x3 = model.forward(x, x, x)
        X.append(x1.data.cpu().squeeze().numpy())
        # print(int(s[1][0]))
        y = int(s[1][0])
        Y.append(y)

    X = np.array(X)
    print(len(X))
    Y = np.array(Y)
    accuracy, f1 = svm_test(X, Y, split)
    # print("acc_pre is", accuracy_pre)
    # print("f1_pre score is", f1_pre)
    print("acc is", accuracy)
    print("f1 score is", f1)
