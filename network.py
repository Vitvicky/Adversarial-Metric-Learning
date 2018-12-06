import torch
import torch.nn as nn
import torch.nn.functional as F
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
	
	
# generate adversarial samples
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, f):
        super(Generator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.f = f

    def forward(self, x):
        x = self.map1(x)
        x = self.f(x)
        x = self.map2(x)
        x = self.f(x)
        x = self.map3(x)
        return x	
	
	
class TripletMNIST(Dataset):
	def __init__(self, dataset, n_triplets=10000):
        self.data = []
		self.label = []
		self.triplets = []
        tensor = cuda.FloatTensor
        
        for s in dataset:
            self.data.append(s[0]/255))
			self.label.append(s[1]) # here not transfer
		
		self.labels_set = set(np.array(self.label))
		triplets_list = self.make_triplet_list(n_triplets)
		for line in triplets_list:
			self.triplets.append((int(line.split()[0]), int(line.split()[1]), int(line.split()[2])))
	
	
	def __getitem__(self, index):
        idx1, idx2, idx3 = self.triplets[index]
        img1, img2, img3 = self.data[idx1], self.data[idx2], self.data[idx3]
    
        return img1, img2, img3

		
    def __len__(self):
        
		return len(self.triplets)
			
			
	def make_triplet_list(self, ntriplets):
        print('Processing Triplet Generation ...')
        np_labels = np.array(self.label)

        triplets_list = []
        for class_idx in range(10):
            a = np.random.choice(np.where(np_labels==class_idx)[0], int(ntriplets/10), replace=True)
            b = np.random.choice(np.where(np_labels==class_idx)[0], int(ntriplets/10), replace=True)
            while np.any((a-b)==0):
                np.random.shuffle(b)
            c = np.random.choice(np.where(np_labels!=class_idx)[0], int(ntriplets/10), replace=True)

            for i in range(a.shape[0]):
                triplets_list.append([int(a[i]), int(c[i]), int(b[i])])           


	
	

class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()

        self.fc = nn.Sequential(nn.Linear(64 * 4 * 4, 100), # change here
                                nn.PReLU(),
                                nn.Linear(500, 100)
                                )

    def forward(self, x):
        # output = self.convnet(x)
        # output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)
		

class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)