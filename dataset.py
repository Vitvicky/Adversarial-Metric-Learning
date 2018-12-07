import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.cuda as cuda
import numpy as np


def read_dataset(file_path):
    dataset = []
    labels = set()
    with open(file_path) as f:
        for line in f:
            row = [float(_) for _ in line.split(',')]
            dataset.append((row[:-1], row[-1:]))
            labels.add(int(row[-1]))

    return dataset, labels


class TripletMNIST(Dataset):
    def __init__(self, dataset, n_triplets):
        self.data = []
        self.label = []
        self.triplets = []
        tensor = cuda.FloatTensor

        for s in dataset:
            self.data.append(tensor(s[0]))
            self.label.append(s[1][0])  # here not transfer

        # print(self.label)
        # print(self.data)
        triplets_list = self.make_triplet_list(n_triplets)
        for line in triplets_list:
                self.triplets.append((int(line[0]), int(line[1]), int(line[2])))

    def __getitem__(self, index):
        idx1, idx2, idx3 = self.triplets[index]
        img1, img2, img3 = self.data[idx1], self.data[idx2], self.data[idx3]

        return img1, img2, img3

    def __len__(self):
        return len(self.triplets)

    def make_triplet_list(self, n_triplets):
        print('Processing Triplet Generation ...')
        np_labels = np.array(self.label)
        # print(np_labels)

        triplets_list = []
        for class_idx in range(10):
            a = np.random.choice(np.where(np_labels == class_idx)[0], int(n_triplets / 10), replace=True)
            b = np.random.choice(np.where(np_labels == class_idx)[0], int(n_triplets / 10), replace=True)
            while np.any((a - b) == 0):
                np.random.shuffle(b)
            c = np.random.choice(np.where(np_labels != class_idx)[0], int(n_triplets / 10), replace=True)

            for i in range(a.shape[0]):
                triplets_list.append([int(a[i]), int(c[i]), int(b[i])])

        return triplets_list


if __name__ == "__main__":
    dataset_path = 'mnist_normal.csv'
    dataset, classes = read_dataset(dataset_path)
    class_count = len(classes)
    test_data = dataset[:100]

    triplet_dataset = TripletMNIST(test_data, 200)
    print(triplet_dataset)
