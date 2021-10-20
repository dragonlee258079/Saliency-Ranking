import torch
from torch import nn
from torch.nn import functional as F
from detectron2.layers import Linear
import fvcore.nn.weight_init as weight_init


class SaliencyPredictor(nn.Module):
    def __init__(self, cfg):
        super(SaliencyPredictor, self).__init__()
        in_channels = cfg.MODEL.RELATION_HEAD.MLP_HEAD_DIM
        self.saliency_score = Linear(in_channels, 1)
        weight_init.c2_xavier_fill(self.saliency_score)

    def forward(self, x):
        y = []
        for x_ in x:
            x_ = self.saliency_score(x_)
            y.append(x_)
        return y


def make_saliency_predictor(cfg):
    return SaliencyPredictor(cfg)
