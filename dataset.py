import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.cuda as cuda
import numpy as np
from torch.utils.data.sampler import BatchSampler


def read_dataset(file_path):
    dataset = []
    labels = set()
    with open(file_path) as f:
        for line in f:
            row = [float(_) for _ in line.split(',')]
            dataset.append((row[:-1], row[-1:]))
            labels.add(int(row[-1]))

    return dataset, labels


# class TripletDataSet(Dataset):
#     def __init__(self, dataset, n_triplets):
#         self.data = []
#         self.label = []
#         self.triplets = []
#         tensor = cuda.FloatTensor
#
#         for s in dataset:
#             self.data.append(tensor(s[0]) / 255.0)
#             self.label.append(s[1][0])  # here not transfer
#
#         # print(self.label)
#         # print(self.data)
#         triplets_list = self.make_triplet_list(n_triplets)
#         for line in triplets_list:
#                 self.triplets.append((int(line[0]), int(line[1]), int(line[2])))
#
#     def __getitem__(self, index):
#         idx1, idx2, idx3 = self.triplets[index]
#         img1, img2, img3 = self.data[idx1], self.data[idx2], self.data[idx3]
#
#         return img1, img2, img3
#
#     def __len__(self):
#         return len(self.triplets)
#
#     def make_triplet_list(self, n_triplets):
#         # print('Processing Triplet Generation ...')
#         np_labels = np.array(self.label)
#         # print(np_labels)
#
#         triplets_list = []
#         for class_idx in range(10):
#             a = np.random.choice(np.where(np_labels == class_idx)[0], int(n_triplets / 10), replace=True)
#             b = np.random.choice(np.where(np_labels == class_idx)[0], int(n_triplets / 10), replace=True)
#             while np.any((a - b) == 0):
#                 np.random.shuffle(b)
#             c = np.random.choice(np.where(np_labels != class_idx)[0], int(n_triplets / 10), replace=True)
#
#             for i in range(a.shape[0]):
#                 triplets_list.append([int(a[i]), int(c[i]), int(b[i])])
#
#         return triplets_list

class TripletDataSet(Dataset):
    def __init__(self, dataset):
        self.data = []
        self.label = []
        self.triplets = []
        tensor = cuda.FloatTensor

        for s in dataset:
            self.data.append(tensor(s[0]) / 255.0)
            self.label.append(s[1][0])  # here not transfer

        self.labels_set = set(np.array(self.label))
        print("set is:", self.labels_set)
        self.label_to_indices = {l: np.where(np.array(self.label) == l)[0]
                                 for l in self.labels_set}

    def __getitem__(self, index):
        img1, label1 = self.data[index], self.label[index]
        positive_index = index
        while positive_index == index:
            positive_index = np.random.choice(self.label_to_indices[label1])
        negative_label = np.random.choice(list(self.labels_set - set([label1])))
        negative_index = np.random.choice(self.label_to_indices[negative_label])
        img2 = self.data[positive_index]
        img3 = self.data[negative_index]

        # return (img1, img2, img3), []
        return img1, img2, img3

    def __len__(self):
        return len(self.data)


class BalancedBatchSampler(BatchSampler):
    def __init__(self, dataset, n_classes, n_samples):
        self.data = []
        self.label = []
        self.triplets = []
        tensor = cuda.FloatTensor

        for s in dataset:
            self.data.append(tensor(s[0]) / 255.0)
            self.label.append(s[1][0])  # here not transfer

        self.labels_set = list(set(np.array(self.label)))
        # print("set is:", self.labels_set)
        self.label_to_indices = {l: np.where(np.array(self.label) == l)[0]
                                 for l in self.labels_set}

        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < len(self.dataset):
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return len(self.dataset) // self.batch_size


class Test_Dataset(Dataset):
    def __init__(self, dataset):
        self.data = []
        self.label = []
        tensor = cuda.FloatTensor

        for s in dataset:
            self.data.append((tensor(s[0]) / 255.0, tensor(s[1])))
            self.label.append(s[1][0])  # here not transfer

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


# if __name__ == "__main__":
#     dataset_path = '/home/wzy/Coding/Data/metric_learning/mnist_normal.csv'
#     dataset, classes = read_dataset(dataset_path)
#     class_count = len(classes)
#     test_data = dataset[:1000]
#
#     triplet_dataset = TripletMNIST(test_data, 2000)
#     print(len(triplet_dataset))
