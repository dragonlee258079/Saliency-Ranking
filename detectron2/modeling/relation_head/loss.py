import torch


class RelationLossComputation(object):
    def prepare_targets(self, target):
        rank_labels = target.gt_ranks
        return rank_labels

    def rank_convert(self, rank_labels):
        _, order = torch.sort(rank_labels)
        _, order_ = torch.sort(order)
        return order_

    def loss_compute(self, rank_labels, saliency_score):
        N = len(rank_labels)
        saliency_score = saliency_score.reshape(-1)
        S1, S2 = torch.meshgrid((saliency_score, saliency_score))
        S = -S1 + S2
        R1, R2 = torch.meshgrid((rank_labels, rank_labels))
        R = (R1 - R2).cuda()
        R[R > 0] = 1
        R[R < 0] = -1
        S = S * R
        S = torch.log(1+torch.exp(S))
        S[R == 0] = 0
        S = torch.triu(S, 1)
        B = torch.abs((R1 - R2).cuda().float())
        Wr_m = torch.sum(torch.arange(1, N) * torch.arange(N - 1, 0, -1)).float()
        B = B / Wr_m
        S = S * B
        relation_loss = torch.sum(S)
        return relation_loss

    def __call__(self, gt_ranks, saliency_scores):
        relation_losses = 0
        for gt_ranks_per_image, score_per_image in zip(gt_ranks, saliency_scores):
            # rank_labels = self.prepare_targets(target_per_image)
            relation_loss = self.loss_compute(gt_ranks_per_image, score_per_image)
            relation_losses += relation_loss
        return relation_losses/len(gt_ranks)


def make_relation_loss_evalutor():
    return RelationLossComputation()
