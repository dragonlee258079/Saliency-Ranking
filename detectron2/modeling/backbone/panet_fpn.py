import torch
from torch import nn
import fvcore.nn.weight_init as weight_init
from detectron2.layers import Conv2d
class PAnetFPN(nn.Module):
    def __init__(self):
        super(PAnetFPN, self).__init__()
        self.backbone_stages = 5

        self.panet_buttomup_conv1_modules = nn.ModuleList()
        self.panet_buttomup_conv2_modules = nn.ModuleList()

        for i in range(self.backbone_stages - 1):
            self.panet_buttomup_conv1_modules.append(
                Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, bias=True, norm=None))

            self.panet_buttomup_conv2_modules.append(
                Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True, norm=None))

        for i in range(len(self.panet_buttomup_conv1_modules)):
            weight_init.c2_xavier_fill(self.panet_buttomup_conv2_modules[i])
            weight_init.c2_xavier_fill(self.panet_buttomup_conv1_modules[i])




    def forward(self, x):
        out_features={"p2": None, "p3": None, "p4": None, "p5": None, "p6": None}
        out_features["p2"] = x["p2"]
        out_features["p3"] = self.panet_buttomup_conv2_modules[0](self.panet_buttomup_conv1_modules[0](out_features["p2"]) + x["p3"])
        out_features["p4"] = self.panet_buttomup_conv2_modules[1](self.panet_buttomup_conv1_modules[1](out_features["p3"]) + x["p4"])
        out_features["p5"] = self.panet_buttomup_conv2_modules[2](self.panet_buttomup_conv1_modules[2](out_features["p4"]) + x["p5"])
        out_features["p6"] = self.panet_buttomup_conv2_modules[3](self.panet_buttomup_conv1_modules[3](out_features["p5"]) + x["p6"])

        return out_features




# a = PAnetFPN()
