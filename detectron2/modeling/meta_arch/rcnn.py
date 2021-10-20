# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from torch import nn

from detectron2.structures import ImageList
from detectron2.structures import Instances
import math

from ..backbone import build_backbone
from ..backbone import build_bottom_up_fuse
from ..relation_head import build_relation_head
from ..backbone.panet_fpn import PAnetFPN
from ..proposal_generator import build_proposal_generator
from ..roi_heads import build_roi_heads
from .build import META_ARCH_REGISTRY
__all__ = ["RankSaliencyNetwork"]


@META_ARCH_REGISTRY.register()
class RankSaliencyNetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = build_backbone(cfg)
        self.bottom_up_fuse = build_bottom_up_fuse(cfg)
        self.panet_fpn = PAnetFPN()
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
        self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())
        self.relation_head = build_relation_head(cfg, self.bottom_up_fuse.out_channels)

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(num_channels, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def preprocess_image(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def forward(self, batched_inputs):
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

        features_res, features_fpn = self.backbone(images.tensor)
        features_relation = self.bottom_up_fuse(features_res)
        features = self.panet_fpn(features_fpn)

        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        results, detector_losses, person_features = self.roi_heads(images, features, proposals, gt_instances)
        _, relation_loss = self.relation_head(features_relation, results, person_features)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        losses.update(relation_loss)
        return losses

    def inference(self, batched_inputs):
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        # gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

        features_res, features_fpn = self.backbone(images.tensor)
        features_relation = self.bottom_up_fuse(features_res)
        features = self.panet_fpn(features_fpn)

        proposals, _ = self.proposal_generator(images, features, None)
        results, _, person_features = self.roi_heads(images, features, proposals)
        salienct_rank, _ = self.relation_head(features_relation, results, person_features)

        return {'roi_results': results,
                'rank_result': salienct_rank}


def generate_gt_proposals_single_image(gt_boxes, device, image_size):
    """
    Augment `proposals` with ground-truth boxes from `gt_boxes`.

    Args:
        Same as `add_ground_truth_to_proposals`, but with gt_boxes and proposals
        per image.

    Returns:
        Same as `add_ground_truth_to_proposals`, but for only one image.
    """
    # Concatenating gt_boxes with proposals requires them to have the same fields
    # Assign all ground-truth boxes an objectness logit corresponding to P(object) \approx 1.
    gt_logit_value = math.log((1.0 - 1e-10) / (1 - (1.0 - 1e-10)))

    gt_logits = gt_logit_value * torch.ones(len(gt_boxes), device=device)
    gt_proposal = Instances(image_size)

    gt_proposal.proposal_boxes = gt_boxes
    gt_proposal.objectness_logits = gt_logits
    gt_proposal.is_gt = torch.ones(len(gt_boxes), device=device)
    return gt_proposal
