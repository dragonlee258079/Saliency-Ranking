import math
import torch
from torch import nn
from torch.nn import functional as F
import fvcore.nn.weight_init as weight_init
from detectron2.layers import ROIAlign, Linear, cat
from detectron2.structures import Boxes, Instances
from detectron2.utils.registry import Registry

RELATION_EXTRACTOR_REGISTRY = Registry("RELATION_EXTRACTOR")


@RELATION_EXTRACTOR_REGISTRY.register()
class FeatureExtractorRoIAlign(nn.Module):
    def __init__(self, cfg, in_channels):
        super(FeatureExtractorRoIAlign, self).__init__()
        resolution = cfg.MODEL.RELATION_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.RELATION_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.RELATION_HEAD.POOLER_SAMPLING_RATIO

        self.pooler = ROIAlign(
                output_size=(resolution, resolution),
                spatial_scale=scales[0],
                sampling_ratio=sampling_ratio,
                aligned=True
            )

        input_size = in_channels * resolution ** 2
        representation_size = cfg.MODEL.RELATION_HEAD.MLP_HEAD_DIM

        self.fc6 = Linear(input_size, representation_size)
        self.fc7 = Linear(representation_size, representation_size)
        nn.init.kaiming_uniform_(self.fc6.weight, a=0, nonlinearity='relu')
        nn.init.constant_(self.fc6.bias, 0)
        nn.init.kaiming_uniform_(self.fc7.weight, a=0, nonlinearity='relu')
        nn.init.constant_(self.fc7.bias, 0)

    def forward(self, x, proposals=None):
        pooler_fmt_boxes = convert_boxes_to_pooler_format(proposals)
        x = self.pooler(x, pooler_fmt_boxes)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        return x


@RELATION_EXTRACTOR_REGISTRY.register()
class FeatureExtractorMaxPooling(nn.Module):
    def __init__(self, cfg, in_channels):
        super(FeatureExtractorMaxPooling, self).__init__()

        resolution = cfg.MODEL.RELATION_HEAD.POOLER_RESOLUTION
        self.adaptive_pooling = nn.AdaptiveAvgPool2d(resolution)

    def forward(self, x, proposals=None):
        feature_num = x.shape[0]
        x = self.adaptive_pooling(x)
        x = x.split([1]*feature_num)
        return x


@RELATION_EXTRACTOR_REGISTRY.register()
class FeatureExtractorPosition(nn.Module):
    def __init__(self, cfg, in_channels):
        super(FeatureExtractorPosition, self).__init__()
        representation_size = cfg.MODEL.RELATION_HEAD.MLP_HEAD_DIM

        self.fc6 = Linear(representation_size, representation_size)
        self.fc7 = Linear(representation_size, representation_size)
        nn.init.kaiming_uniform_(self.fc6.weight, a=0, nonlinearity='relu')
        nn.init.constant_(self.fc6.bias, 0)
        nn.init.kaiming_uniform_(self.fc7.weight, a=0, nonlinearity='relu')
        nn.init.constant_(self.fc7.bias, 0)
        self.out_channels = representation_size

    def forward(self, proposals, image_sizes):
        x = []
        for proposal, image_size in zip(proposals, image_sizes):
            x_ = position_emdedding(proposal, dim_geo=self.out_channels, image_size=image_size)
            x_ = x_.view(x_.size(0), -1)
            x_ = F.relu(self.fc6(x_))
            x_ = F.relu(self.fc7(x_))
            x.append(x_)
        return x


class RelationFeatureCombine(nn.Module):
    def __init__(self, cfg, in_channels):
        super(RelationFeatureCombine, self).__init__()
        self.expand_ratio = cfg.MODEL.RELATION_HEAD.POOLER_EXPAND_RATIO

        self.feature_extractor_origin = make_relation_features_extractor(cfg, 'FeatureExtractorRoIAlign', in_channels)
        self.feature_extractor_expand = make_relation_features_extractor(cfg, 'FeatureExtractorRoIAlign', in_channels)
        self.feature_extractor_full = make_relation_features_extractor(cfg, 'FeatureExtractorMaxPooling', in_channels)
        self.feature_extractor_position = make_relation_features_extractor(cfg, 'FeatureExtractorPosition', in_channels)

        # representation_size = cfg.MODEL.RELATION_HEAD.MLP_HEAD_DIM
        # head_nums = cfg.MODEL.RELATION_HEAD.FEATURE_NUM
        # self.linear = Linear(representation_size * 2, representation_size)
        # nn.init.normal_(self.linear.weight, mean=0, std=0.01)
        # nn.init.constant_(self.linear.bias, 0)
        # self.linear = Linear(representation_size*head_nums, representation_size)
        # nn.init.normal_(self.linear.weight, mean=0, std=0.01)
        # nn.init.constant_(self.linear.bias, 0)

    def forward(self, feature, proposals, image_sizes):
        split_list = [len(proposal) for proposal in proposals]
        origin_feature = _split(self.feature_extractor_origin(feature, proposals), split_list)
        expanded_proposals = bbox_expand(proposals, image_sizes, self.expand_ratio)
        expanded_feature = _split(self.feature_extractor_expand(feature, expanded_proposals), split_list)
        full_feature = self.feature_extractor_full(feature, proposals)
        # position_feature = self.feature_extractor_position(proposals, image_sizes)
        relation_feature = []
        # for o, e, f, p in zip(origin_feature, expanded_feature, full_feature, position_feature):
        #     r_f = torch.cat((o, e, f, p), dim=1)
        #     r_f = self.linear(r_f)
        #     relation_feature.append(r_f)
        # return relation_feature
        for o, e, f in zip(origin_feature, expanded_feature, full_feature):
            r_f = [o, e, f]
            relation_feature.append(r_f)
        return relation_feature


def _split(feature, list):
    return feature.split(list)


def position_emdedding(geo_feature, dim_geo, image_size, wave_len=1000):
    boxes = geo_feature.tensor
    x_min, y_min, x_max, y_max = torch.chunk(boxes, 4, dim=1)
    img_h, img_w = image_size
    cx = ((x_min + x_max) * 0.5)/img_w
    cy = ((y_min + y_max) * 0.5)/img_h
    w = ((x_max - x_min) + 1.)/img_w
    h = ((y_max - y_min) + 1.)/img_h
    position_mat = torch.cat((cx, cy, w, h), -1)
    feature_range = torch.arange(dim_geo/8).cuda()
    dim_mat = feature_range/(dim_geo/8)
    dim_mat = 1./(torch.pow(wave_len, dim_mat))
    dim_mat = dim_mat.view(1, 1, -1)
    size = position_mat.size()
    position_mat = position_mat.view(size[0], size[1], 1)

    mul_mat = position_mat * dim_mat
    mul_mat = mul_mat.view(size[0], -1)
    sin_mat = torch.sin(mul_mat)
    cos_mat = torch.cos(mul_mat)
    embedding = torch.cat((sin_mat, cos_mat), -1)
    return embedding


def bbox_expand(proposals, image_sizes, ratio):
    expanded_proposals = []
    for proposals_per_img, image_size in zip(proposals, image_sizes):
        boxes = proposals_per_img.tensor
        xmin, ymin, xmax, ymax = torch.chunk(boxes, 4, dim=1)
        w = xmax - xmin + 1
        h = ymax - ymin + 1
        x_ctr = xmin + 0.5 * (w - 1)
        y_ctr = ymin + 0.5 * (h - 1)
        ws = w * math.sqrt(ratio)
        hs = h * math.sqrt(ratio)
        expanded_bbox = torch.stack(
            (
                x_ctr - 0.5 * (ws - 1),
                y_ctr - 0.5 * (hs - 1),
                x_ctr + 0.5 * (ws - 1),
                y_ctr + 0.5 * (hs - 1)
            ),
            dim=1
        )
        expanded_bbox = expanded_bbox.reshape(-1, 4)
        expanded_bbox = Boxes(expanded_bbox)
        expanded_bbox.clip(image_size)
        expanded_proposals.append(expanded_bbox)
    return expanded_proposals


def convert_boxes_to_pooler_format(box_lists):
    def fmt_box_list(box_tensor, batch_index):
        repeated_index = torch.full(
            (len(box_tensor), 1), batch_index, dtype=box_tensor.dtype, device=box_tensor.device
        )
        return cat((repeated_index, box_tensor), dim=1)

    pooler_fmt_boxes = cat(
        [fmt_box_list(box_list.tensor, i) for i, box_list in enumerate(box_lists)], dim=0
    )

    return pooler_fmt_boxes


def make_relation_features_extractor(cfg, method, in_channels):
    feature_extractor = RELATION_EXTRACTOR_REGISTRY.get(method)(cfg, in_channels)
    return feature_extractor


def make_relation_features_combine(cfg, in_channels):
    return RelationFeatureCombine(cfg, in_channels)
