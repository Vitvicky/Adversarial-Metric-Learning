import torch
import torch.nn as nn
import torch.nn.functional as F


class PairwiseDistance(torch.nn.Module):
    def __init__(self, p):
        super(PairwiseDistance, self).__init__()
        self.norm = p

    def forward(self, x1, x2, eps=1e-6):
        assert x1.size() == x2.size()
        diff = torch.abs(x1 - x2)
        out = torch.pow(diff + eps, self.norm).sum(dim=1, keepdim=True)

        return torch.pow(out, 1. / self.norm)


class NormalDistance(torch.nn.Module):
    def __init__(self):
        super(NormalDistance, self).__init__()

    def forward(self, x1, x2, eps=1e-6):
        assert x1.size() == x2.size()
        diff = torch.abs(x1 - x2)
        out = torch.pow(diff + eps, 2).sum(dim=1, keepdim=True)

        return out


class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.pdist = NormalDistance()

    def forward(self, anchor, positive, generate, size_average=True):
        d_p = self.pdist.forward(anchor, positive)
        d_n = self.pdist.forward(anchor, generate)
        tmp1 = torch.log(d_p + 1)
        tmp2 = torch.log(d_n + 1)
        dist_hinge = torch.clamp(tmp1 + self.margin - tmp2, min=0.0)
        loss = torch.mean(dist_hinge)
        return loss

        # distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        # distance_negative = (anchor - generate).pow(2).sum(1)  # .pow(.5)
        # losses = F.relu(distance_positive - distance_negative + self.margin)
        # return losses.mean() if size_average else losses.sum()


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
        self.pdist = PairwiseDistance(2)
        self.dist = NormalDistance()

    def forward(self, anchor_ori, anchor_d, positive_ori, pos_d, negative_ori, neg_d, generate_ori, gen_d, class_statis, size_average=True):
        generate_ori = generate_ori.view(generate_ori.size(0), 3072)
        anchor_ori = anchor_ori.view(anchor_ori.size(0), 3072)
        negative_ori = negative_ori.view(negative_ori.size(0), 3072)
        distance_1 = self.dist(generate_ori, anchor_ori)  # .pow(.5)
        # distance_2 = self.dist(generate_ori, negative_ori)  # .pow(.5)

        # confusion
        # anchor = anchor_set[0]
        # negative = negative_set[0]
        # anchor = anchor.cuda()
        # negative = negative.cuda()
        # anchor_label = anchor_set[1]
        # negative_label = negative_set[1]
        # distance_2 = 0
        # for i in range(0, len(anchor)):
        #     anchor_item = anchor[i]
        #     negative_item = negative[i]
        #     joint_prob_a = 1 / class_statis[anchor_label[i].item()]
        #     joint_prob_n = 1 / class_statis[negative_label[i].item()]
        #     distance_2 += joint_prob_a * joint_prob_n * self.pdist(anchor_item, negative_item)

        distance_2 = 0.01 * self.dist.forward(anchor_ori, negative_ori)  # .pow(.5)
        tmp_d1 = torch.log(distance_1 + 0.5)
        tmp_d2 = torch.log(distance_2 + 0.5) + 1

        d_p = self.dist.forward(pos_d, anchor_d)
        d_n = self.dist.forward(gen_d, anchor_d)
        tmp1 = torch.log(d_p + 0.5)
        tmp2 = torch.log(d_n + 0.5) + 1
        dist_hinge = torch.clamp(tmp2 - self.margin - tmp1, min=0.0)
        distance_adv = dist_hinge

        # gen_loss = distance_1 + self.lambda1 * distance_2 + self.lambda2 * F.relu(distance_adv)
        gen_loss = tmp_d1 + self.lambda1 * tmp_d2 + self.lambda2 * distance_adv
        return gen_loss.mean()

# class generateLoss(nn.Module):
#     def __init__(self, margin, lambda1, lambda2):
#         super(generateLoss, self).__init__()
#         self.margin = margin
#         self.lambda1 = lambda1
#         self.lambda2 = lambda2
#         self.pdist = PairwiseDistance(2)
#
#     def forward(self, anchor, generate, positive, negative, size_average=True):
#         # pdist = nn.PairwiseDistance(p=2)
#         distance_1 = (generate - anchor).pow(2).sum(1)  # .pow(.5)
#         distance_2 = (generate - negative).pow(2).sum(1)  # .pow(.5)
#
#         # distance_adv = F.pairwise_distance(generate, anchor, 2)-F.pairwise_distance(positive, anchor, 2)-self.margin
#         d_p = self.pdist.forward(generate, anchor)
#         d_n = self.pdist.forward(positive, anchor)
#         tmp1 = torch.log(d_p + 1)
#         tmp2 = torch.log(d_n + 1)
#         dist_hinge = torch.clamp(tmp2 - self.margin - tmp1, min=0.0)
#         distance_adv = dist_hinge
#
#         # gen_loss = distance_1 + self.lambda1 * distance_2 + self.lambda2 * F.relu(distance_adv)
#         gen_loss = distance_1 + self.lambda1 * distance_2 + self.lambda2 * distance_adv
#         return gen_loss.mean()


class ECLoss(nn.Module):
    def __init__(self):
        super(ECLoss, self).__init__()
        # self.margin = margin
        self.pdist = NormalDistance()

    def forward(self, anchor_set, negative_set, class_statis, size_average=True):
        anchor = anchor_set[0]
        negative = negative_set[0]
        anchor = anchor.cuda()
        negative = negative.cuda()
        anchor_label = anchor_set[1]
        negative_label = negative_set[1]
        loss = 0
        # joint_prob_a = 1/class_statis[anchor_label]
        # joint_prob_n = 1/class_statis[negative_label]
        # loss = joint_prob_a * joint_prob_n * (self.pdist(anchor, negative))
        for i in range(0, len(anchor)):
            anchor_item = anchor[i]
            negative_item = negative[i]
            joint_prob_a = 1/class_statis[anchor_label[i].item()]
            joint_prob_n = 1 / class_statis[negative_label[i].item()]
            loss += joint_prob_a * joint_prob_n * self.pdist(anchor_item, negative_item)

        return loss.mean()
