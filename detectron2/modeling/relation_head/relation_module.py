import torch
from torch import nn
from detectron2.layers import Linear
import fvcore.nn.weight_init as weight_init
from detectron2.utils.registry import Registry
from torch.nn import functional as F

RELATION_MODULE_REGISTER = Registry("REALTION_MODULE")


class RelationBetweenMulti(nn.Module):
    def __init__(self, unit_nums, in_channels):
        super(RelationBetweenMulti, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = in_channels // unit_nums
        if self.inter_channels == 0:
            self.inter_channels = 1

        self.g = Linear(self.in_channels, self.inter_channels)
        nn.init.normal_(self.g.weight, mean=0, std=0.01)
        nn.init.constant_(self.g.bias, 0)

        # 2019/10/23
        # self.W = nn.Linear(self.inter_channels, self.inter_channels)
        # nn.init.normal_(self.W.weight, mean=0, std=0.01)
        # nn.init.constant_(self.W.bias, 0)

        self.theta = Linear(self.in_channels, self.inter_channels)
        nn.init.normal_(self.theta.weight, mean=0, std=0.01)
        nn.init.constant_(self.theta.bias, 0)

        self.phi = Linear(self.in_channels, self.inter_channels)
        nn.init.normal_(self.phi.weight, mean=0, std=0.01)
        nn.init.constant_(self.phi.bias, 0)

        self.concat_project = nn.Sequential(
            nn.Conv2d(self.inter_channels * 2, 1, 1, 1, 0, bias=False),
            nn.ReLU()
        )
        nn.init.normal_(self.concat_project[0].weight, mean=0, std=0.01)

    def forward(self, x):
        g_x = self.g(x)

        theta_x = self.theta(x)
        theta_x = theta_x.permute(1, 0)
        N = theta_x.size(1)
        C = theta_x.size(0)
        theta_x = theta_x.view(C, N, 1)
        theta_x = theta_x.repeat(1, 1, N)

        phi_x = self.phi(x)
        phi_x = phi_x.permute(1, 0)
        phi_x = phi_x.view(C, 1, N)
        phi_x = phi_x.repeat(1, N, 1)

        concat_feature = torch.cat((theta_x, phi_x), dim=0)
        concat_feature = concat_feature.view(1, *concat_feature.size()[:])
        f = self.concat_project(concat_feature)
        f = f.view(N, N)
        f_dic_C = f / N

        z = torch.matmul(f_dic_C, g_x)

        return z


class RelationBetweenPair(nn.Module):
    def __init__(self, unit_nums, in_channels):
        super(RelationBetweenPair, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = in_channels // unit_nums
        if self.inter_channels == 0:
            self.inter_channels = 1

        self.g = Linear(self.in_channels, self.inter_channels)
        nn.init.normal_(self.g.weight, mean=0, std=0.01)
        nn.init.constant_(self.g.bias, 0)

        # 2019/10/23
        # self.W = nn.Linear(self.inter_channels, self.inter_channels)
        # nn.init.normal_(self.W.weight, mean=0, std=0.01)
        # nn.init.constant_(self.W.bias, 0)

        self.theta = Linear(self.in_channels, self.inter_channels)
        nn.init.normal_(self.theta.weight, mean=0, std=0.01)
        nn.init.constant_(self.theta.bias, 0)

        self.phi = Linear(self.in_channels, self.inter_channels)
        nn.init.normal_(self.phi.weight, mean=0, std=0.01)
        nn.init.constant_(self.phi.bias, 0)

        self.concat_project = nn.Sequential(
            nn.Linear(self.inter_channels * 2, 1, bias=False),
            nn.ReLU()
        )
        nn.init.normal_(self.concat_project[0].weight, mean=0, std=0.01)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, y):
        g_y = self.g(y)

        theta_x = self.theta(x)
        phi_y = self.phi(y)

        concat_feature = torch.cat((theta_x, phi_y), dim=1)
        f = self.concat_project(concat_feature)

        z = self.gamma * f * g_y

        return z


class RelationOnSpatial(nn.Module):
    def __init__(self, unit_nums, in_channels):
        super(RelationOnSpatial, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = in_channels // unit_nums
        if self.inter_channels == 0:
            self.inter_channels = 1

        self.g = nn.Conv2d(256, self.inter_channels, kernel_size=3, stride=1, padding=1)
        nn.init.normal_(self.g.weight, mean=0, std=0.01)
        nn.init.constant_(self.g.bias, 0)

        # 2019/10/23
        # self.W = nn.Linear(self.inter_channels, self.inter_channels)
        # nn.init.normal_(self.W.weight, mean=0, std=0.01)
        # nn.init.constant_(self.W.bias, 0)

        self.phi = nn.Conv2d(256, self.inter_channels, kernel_size=3, stride=1, padding=1)
        nn.init.normal_(self.phi.weight, mean=0, std=0.01)
        nn.init.constant_(self.phi.bias, 0)

        self.theta = Linear(self.in_channels, self.inter_channels)
        nn.init.normal_(self.theta.weight, mean=0, std=0.01)
        nn.init.constant_(self.theta.bias, 0)

        self.concat_project = nn.Conv2d(self.inter_channels*2, 1, 1, 1, 0, bias=False)
        nn.init.normal_(self.concat_project.weight, mean=0, std=0.01)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, y):
        g_y = self.g(y)
        theta_x = self.theta(x)

        N = theta_x.size(0)
        C = theta_x.size(1)

        phi_y = self.phi(y)
        phi_y = phi_y.repeat(N, 1, 1, 1)

        resolution = phi_y.size(2)

        theta_x = theta_x.view(N, C, 1, 1)
        theta_x = theta_x.repeat(1, 1, resolution, resolution)

        concat_feature = torch.cat((theta_x, phi_y), dim=1)
        f = self.concat_project(concat_feature)
        f = f.view(N, -1)
        f = F.softmax(f, dim=1)

        g_y = g_y.squeeze()
        g_y = g_y.permute(1, 2, 0)
        g_y = g_y.view(resolution*resolution, -1)

        z = self.gamma * torch.mm(f, g_y)

        return z


class CompressPersonFeature(nn.Module):
    def __init__(self, unit_nums, in_channels):
        super(CompressPersonFeature, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = in_channels // unit_nums

        self.fc = Linear(self.in_channels, self.inter_channels)
        nn.init.normal_(self.fc.weight, mean=0, std=0.01)
        nn.init.constant_(self.fc.bias, 0)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, person_probs, person_features):
        person_features = self.fc(person_features)
        result = self.gamma * person_features
        return result


class ConcatenationUnit(nn.Module):
    def __init__(self, unit_nums, in_channels):
        super(ConcatenationUnit, self).__init__()
        self.multi_relation_unit = RelationBetweenMulti(unit_nums, in_channels)
        self.pair_relation_unit = RelationBetweenPair(unit_nums, in_channels)
        self.spatial_relation = RelationOnSpatial(unit_nums, in_channels)
        self.person_prior = CompressPersonFeature(unit_nums, in_channels)

    def forward(self, x, person_probs, person_features):
        origin_feature, local_feature, global_feature = x
        origin_relation = self.multi_relation_unit(origin_feature)
        local_relation = self.pair_relation_unit(origin_feature, local_feature)
        global_relation = self.spatial_relation(origin_feature, global_feature)
        person_prior = self.person_prior(person_probs, person_features)
        z = origin_relation + local_relation + global_relation + person_prior
        return z


@RELATION_MODULE_REGISTER.register()
class RelationModule(nn.Module):
    def __init__(self, cfg, in_channels):
        super(RelationModule, self).__init__()
        self.relation_units = []
        unit_nums = cfg.MODEL.RELATION_HEAD.Relation_Unit_Nums
        for idx in range(unit_nums):
            relation_unit = 'relation_unit{}'.format(idx)
            relation_unit_module = ConcatenationUnit(unit_nums, in_channels)
            self.add_module(relation_unit, relation_unit_module)
            self.relation_units.append(relation_unit)
        self.fc = Linear(in_channels, in_channels)
        nn.init.normal_(self.fc.weight, mean=0, std=0.01)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x, person_probs, person_features):
        z = []
        for x_, person_prob_per_img, person_features_per_img in zip(x, person_probs, person_features):
            result = tuple([getattr(self, relation_uint)(x_, person_prob_per_img, person_features_per_img)
                            for relation_uint in self.relation_units])
            y = torch.cat(result, dim=1)
            y = self.fc(y)
            z.append(F.relu(x_[0] + y))
        return z


def make_relation_module(cfg, in_channels, method):
    relation_module = RELATION_MODULE_REGISTER.get(method)(cfg, in_channels)
    return relation_module
