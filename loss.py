import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class NormalDistance(torch.nn.Module):
    def __init__(self):
        super(NormalDistance, self).__init__()

    def forward(self, x1, x2, eps=1e-6):
        assert x1.size() == x2.size()
        diff = torch.abs(x1 - x2)
        out = torch.pow(diff, 2).sum(dim=1, keepdim=True)

        return out


class PairwiseDistance(torch.nn.Module):
    def __init__(self, p):
        super(PairwiseDistance, self).__init__()
        self.norm = p

    def forward(self, x1, x2, eps=1e-6):
        assert x1.size() == x2.size()
        diff = torch.abs(x1 - x2)
        out = torch.pow(diff + eps, self.norm).sum(dim=1, keepdim=True)

        return torch.pow(out, 1. / self.norm)


class CosineSimilarity(torch.nn.Module):
    def __init__(self, dim=1, eps=1e-8):
        super(CosineSimilarity, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x1, x2):
        return F.cosine_similarity(x1, x2, self.dim, self.eps)


class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.pdist = PairwiseDistance()

    def forward(self, anchor, positive, generate, size_average=True):
        d_p = self.pdist.forward(anchor, positive)
        d_n = self.pdist.forward(anchor, generate)
        tmp1 = torch.log(d_p + 1)
        tmp2 = torch.log(d_n + 1)
        dist_hinge = torch.clamp(tmp1 + self.margin - tmp2, min=0.0)
        loss = torch.mean(dist_hinge)
        return loss


class ContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()


class BinomialLoss(nn.Module):
    def __init__(self, alpha=50, margin=0.5, beta=0.5, penalty=1):
        super(BinomialLoss, self).__init__()
        self.pdist = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.penalty = penalty
        self.alpha = alpha
        self.beta = beta
        self.margin = margin

    def forward(self, output1, output2, target, size_average=True):
        # output1_T = output1.t()
        # distances = output1_T * output2 / (torch.norm(output1, p=1) * torch.norm(output2, p=1))  # squared distances
        losses = 0
        c = 0
        for i in range(0, len(target)):
            c += 1
            output1_item = output1[i]
            output2_item = output2[i]
            output1_vec = output1_item.unsqueeze(0)
            output2_vec = output2_item.unsqueeze(0)

            # distances_item = (output1_vec - output2_vec).pow(2).sum(1)
            distances_item = self.pdist(output1_vec, output2_vec)
            # print("distances_item: ", distances_item)
            # print("target: ", target[i].item())
            pos_loss = 0
            neg_loss = 0
            num_pos = 0
            num_neg = 0
            if target[i].item() == 1:
                # self.penalty = 1
                # print(torch.log(1 + torch.exp(-2 * (distances_item - self.margin))))
                # pos_loss = math.log(1 + math.exp(-2 * (distances_item - self.margin)))
                pos_loss += 2.0/self.beta * math.log(1 + math.exp(-self.beta*(distances_item - 0.5)))
                num_pos += 1

            else:
                # self.penalty = 25
                # neg_loss = 0.1 * math.log(1 + math.exp(self.alpha * (distances_item - self.margin)))
                neg_loss += 2.0/self.alpha * math.log(1 + math.exp(self.alpha*(distances_item - 2.0)))
                num_neg += 1

            # losses += pos_loss + neg_loss
            # print("losses: ", losses)
            # print("=====================")
        losses = pos_loss/num_pos + neg_loss/num_neg
        return losses


# class EC_Pair_Loss(nn.Module):
#     def __init__(self):
#         super(EC_Pair_Loss, self).__init__()
#         self.pdist = PairwiseDistance(2)
#
#     def forward(self, input1, input2, target, pos_num, neg_num, size_average=True):
#         loss = 0
#         w = 0
#         for i in range(0, len(target)):
#             anchor_item = input1[i].view(1, -1)
#             comp_item = input2[i].view(1, -1)
#             # print("size: ", anchor_item.shape)
#             if target[i].item() == 1:
#                 w = 1
#             else:
#                 w = 1/neg_num
#
#             loss += w * self.pdist(anchor_item, comp_item)
#
#         loss_ec = math.log(1 + loss)
#         return loss_ec

class EC_Pair_Loss(nn.Module):
    def __init__(self):
        super(EC_Pair_Loss, self).__init__()
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
            anchor_item = anchor[i].view(1, -1)
            negative_item = negative[i].view(1, -1)
            if anchor_label[i].item() != negative_label[i].item():
                joint_prob_a = 1/class_statis[anchor_label[i].item()]
                joint_prob_n = 1 / class_statis[negative_label[i].item()]
                loss += joint_prob_a * joint_prob_n * self.pdist(anchor_item, negative_item)

        # math.log(1 + loss)
        return math.log(1+loss.mean())
