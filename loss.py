import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, generate, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - generate).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_generate + self.margin)
        return losses.mean() if size_average else losses.sum()
		
	
class OnlineTripletLoss(nn.Module):

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):

        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean(), len(triplets)
		
		
		
class generateLoss(nn.Module):

    def __init__(self, margin, lambda1, lambda2):
        super(generateLoss, self).__init__()
        self.margin = margin
		self.lambda1 = lambda1
		self.lambda2 = lambda2
		

    def forward(self, anchor, generate, positive, negative, size_average=True):
		# pdist = nn.PairwiseDistance(p=2)
        distance_1 = (generate - anchor).pow(2).sum(1)  # .pow(.5)
        distance_2 = (generate - negative).pow(2).sum(1)  # .pow(.5)
		
		distance_adv = F.pairwise_distance(generate, anchor, 2) - F.pairwise_distance(positive, anchor, 2) - self.margin
		
		
        gen_loss = distance_1 + self.lambda1*distance_2 + self.lambda2*F.relu(distance_adv)
        return gen_loss.mean() if size_average else losses.sum()
		