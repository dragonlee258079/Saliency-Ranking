# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import logging
import numpy as np
import torch
from fvcore.common.file_io import PathManager
from PIL import Image
from torch.utils.data import Dataset
from . import detection_utils as utils
from . import transforms as T
import os
import pickle
from detectron2.structures import BoxMode
import pycocotools.mask as mask_utils


class soc_val(Dataset):
    def __init__(self, cfg, is_train=False):
        self.soc_dir = "/data/lilong/rank_saliency/dataset/dataset_soc_test.pkl"
        f = open(self.soc_dir, 'rb')
        dataset = pickle.load(f)
        self.img_names = list(dataset["test_imgs"].keys())
        self.imgs = list(dataset["test_imgs"].values())
        self.boxes = list(dataset["test_imgs"].values())
        self.segs = list(dataset["test_segs"].values())
        self.ranks = list(dataset["test_ranks"].values())

    def __getitem__(self, idx):
        image = self.imgs[idx]
        image_shape = image.shape[:2]

        dataset_dict = {}
        dataset_dict["file_name"] = self.img_names[idx]
        dataset_dict["image_shape"] = image_shape
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        boxes = self.boxes[idx]
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]

        bit_masks = self.segs[idx]

        rank = self.ranks[idx]
        rank = [sorted(rank).index(i) for i in rank]

        dataset_dict["gt_boxes"] = boxes
        dataset_dict["gt_masks"] = bit_masks
        dataset_dict["gt_rank"] = rank

        return dataset_dict

    def __len__(self):
        return len(self.img_names)
