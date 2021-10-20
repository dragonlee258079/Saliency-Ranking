# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import Linear, ShapeSpec
from detectron2.utils.registry import Registry

PERSON_HEAD_REGISTRY = Registry("PERSON_HEAD")
PERSON_HEAD_REGISTRY.__doc__ = """ """


@PERSON_HEAD_REGISTRY.register()
class FastRCNNFCHead(nn.Module):
    def __init__(self, cfg, input_shape: ShapeSpec):
        super().__init__()

        fc_dim = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        input_size = input_shape.channels * input_shape.height * input_shape.width

        self.fc1 = Linear(input_size, fc_dim)
        weight_init.c2_xavier_fill(self.fc1)

        self.fc2 = Linear(fc_dim, fc_dim)
        weight_init.c2_xavier_fill(self.fc2)

        self._output_size = fc_dim

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

    @property
    def output_size(self):
        return self._output_size


def build_person_head(cfg, input_shape):
    """
    Build a person head defined by 'cfg.MODEL.ROI_BOX_HEAD.PERSON_HEAD'
    """
    name = cfg.MODEL.ROI_BOX_HEAD.PERSON_HEAD
    return PERSON_HEAD_REGISTRY.get(name)(cfg, input_shape)
