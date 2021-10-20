import torch
from .relation_features_extractor import make_relation_features_combine
from .relation_module import make_relation_module
from .saliency_predictors import make_saliency_predictor
from .loss import make_relation_loss_evalutor
from detectron2.utils.registry import Registry

RELATION_REGISTRY = Registry("RELATION_HEAD")


def build_relation_head(cfg, in_channels):
    relation_head_name = cfg.MODEL.RELATION_HEAD.NAME
    relation_head = RELATION_REGISTRY.get(relation_head_name)(cfg, in_channels)
    return relation_head


@RELATION_REGISTRY.register()
class RelationHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(RelationHead, self).__init__()
        self.relation_feature_extractor = make_relation_features_combine(cfg, in_channels)
        relation_feature_dim = cfg.MODEL.RELATION_HEAD.MLP_HEAD_DIM
        self.relation_process = make_relation_module(cfg, relation_feature_dim, 'RelationModule')
        self.saliency_predictor = make_saliency_predictor(cfg)
        self.loss_evalutor = make_relation_loss_evalutor()

    def forward(self, feature, proposals, person_features):
        if self.training:
            image_sizes = [p.image_size for p in proposals]
            boxes, person_features, person_probs, gt_ranks = select_from_proposals(proposals, person_features)
            features_pre_relation = self.relation_feature_extractor(feature, boxes, image_sizes)
            features_post_relation = self.relation_process(features_pre_relation, person_probs, person_features)
            saliency_score = self.saliency_predictor(features_post_relation)
            relation_loss = self.loss_evalutor(gt_ranks, saliency_score)
            return saliency_score, dict(relation_loss=relation_loss)
        else:
            image_sizes = [p.image_size for p in proposals]
            boxes = [p.pred_boxes for p in proposals]
            person_probs = [p.person_probs.view(-1, 1) for p in proposals]
            features_pre_relation = self.relation_feature_extractor(feature, boxes, image_sizes)
            features_post_relation = self.relation_process(features_pre_relation, person_probs, person_features)
            saliency_score = self.saliency_predictor(features_post_relation)
        return saliency_score, {}


def select_from_proposals(proposals, person_features):
    boxes = []
    result_features = []
    person_probs = []
    gt_ranks = []
    for proposals_per_img, person_features_per_img in zip(proposals, person_features):
        gt_index = torch.nonzero(proposals_per_img.is_gt == 1)
        boxes.append(proposals_per_img.gt_boxes[gt_index.squeeze()])
        result_features.append(person_features_per_img[gt_index.squeeze()])
        person_probs.append(proposals_per_img.person_prob[gt_index])
        gt_ranks.append(proposals_per_img.gt_ranks[gt_index.squeeze()])
    return boxes, result_features, person_probs, gt_ranks
