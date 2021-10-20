# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import time
import os
from contextlib import contextmanager
import torch
from tqdm import tqdm
import numpy as np
import copy
import cv2
from .spearman_correlation import evalu as rank_evalu
from .mae_fmeasure import evalu as mf_evalu
from detectron2.data.soc import soc_val



@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)


def inference(cfg, model):
    res = []

    dataset = soc_val(cfg, False)
    with inference_context(model), torch.no_grad():

        for i in range(len(dataset)):
            inputs = [dataset[i]]
            outputs = model(inputs)

            gt_boxes = inputs[0]["gt_boxes"]
            gt_masks = inputs[0]["gt_masks"]
            gt_ranks = inputs[0]["gt_rank"]

            pred_boxes = outputs["roi_results"][0].pred_boxes
            pred_boxes = pred_boxes.tensor.cpu().data.numpy()
            scores = outputs["roi_results"][0].scores
            scores = scores.cpu().data.numpy()
            pred_masks = outputs["roi_results"][0].pred_masks
            pred_masks = pred_masks.cpu().data.numpy()

            saliency_rank = outputs["rank_result"][0].cpu().data.numpy()

            #only keep the box and masks which have score>0.6
            keep = scores > 0.6
            pred_boxes = pred_boxes[keep, :]
            pred_masks = pred_masks[keep, :, :]
            saliency_rank = saliency_rank[keep]

            image_shape = inputs[0]["image_shape"]
            name = inputs[0]["file_name"].split('/')[-1]

            print(name)

            # if name != 'COCO_val2014_000000000387.jpg':
            #     continue

            segmaps = np.zeros([len(pred_masks), image_shape[0], image_shape[1]])

            for j in range(len(pred_masks)):
                x0 = int(pred_boxes[j, 0])
                y0 = int(pred_boxes[j, 1])
                x1 = int(pred_boxes[j, 2])
                y1 = int(pred_boxes[j, 3])

                segmap = pred_masks[j, 0, :, :]
                segmap = cv2.resize(segmap, (x1-x0, y1-y0),
                                    interpolation=cv2.INTER_NEAREST)

                segmaps[j, y0:y1, x0:x1] = segmap

            res.append({'gt_masks': gt_masks, 'segmaps': segmaps, 'scores': scores, 'gt_ranks': gt_ranks,
                        'rank_scores': saliency_rank, 'img_name': name})

            print(len(res))

        r_corre = rank_evalu(res, 0.5)
        r_f = mf_evalu(res)

        return r_corre, r_f
