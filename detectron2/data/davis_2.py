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
import cv2
from detectron2.structures import BoxMode, Instances, Boxes, PolygonMasks


class davis_val(Dataset):
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    def __init__(self, cfg, is_train=False):
        self.crop_gen = None
        self.tfm_gens = utils.build_transform_gen(cfg, is_train)

        self.img_format = 'BGR'
        self.mask_on = True
        self.mask_format = "bitmask"
        self.keypoint_on = False
        self.load_proposals = False

        self.is_train = is_train

        self.video_dir = "/data1/lilong/rank_saliency/dataset/test/dataset_input_test_add_person2.pkl"
        # self.video_dir = '/home/lilong/project/rank_saliency/detectron2/4_add_eval/tools/dataset_input_test.pkl'
        f = open(self.video_dir, 'rb')
        self.dataset_list = pickle.load(f)

    def polygons_to_bitmask(self, polygons, height, width):
        if isinstance(polygons, list):
            assert len(polygons) > 0, "COCOAPI does not support empty polygons"
            rles = mask_utils.frPyObjects(polygons, height, width)
            rle = mask_utils.merge(rles)
            return mask_utils.decode(rle).astype(np.uint8)
        if isinstance(polygons, dict):
            h, w = polygons['size']
            rle = mask_utils.frPyObjects(polygons, h, w)
            mask = mask_utils.decode(rle)
            return mask

    def __getitem__(self, idx):
        image_path = self.dataset_list[idx]['file_name']
        image = utils.read_image(image_path, format=self.img_format)
        image_shape = image.shape[:2]

        dataset_dict = {}
        dataset_dict["file_name"] = image_path
        dataset_dict["image_shape"] = image_shape
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        annos = self.dataset_list[idx].pop("annotations")
        boxes = [BoxMode.convert(obj["bbox"], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS) for obj in annos]

        image_shape = image.shape[:2]

        segms = [obj["segmentation"] for obj in annos]
        bit_masks = [self.polygons_to_bitmask(segm, image_shape[0], image_shape[1]) for segm in segms]

        instances = Instances(image_shape)
        boxes = instances.gt_boxes = Boxes(boxes)
        boxes.clip(image_shape)
        instances.gt_masks = torch.tensor(bit_masks, dtype=torch.int64)

        classes = [0]*len(annos)
        classes = torch.tensor(classes, dtype=torch.int64)
        instances.gt_classes = classes

        instances.gt_ranks = torch.tensor(self.dataset_list[idx].pop("rank"))
        is_person = [obj["is_person"] for obj in annos]
        is_person = torch.tensor(is_person, dtype=torch.int64)
        instances.is_person = is_person

        dataset_dict["instances"] = instances

        return dataset_dict

    def __len__(self):
        return len(self.dataset_list)
        # return 10
