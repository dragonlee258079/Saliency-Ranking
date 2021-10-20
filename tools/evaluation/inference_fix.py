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
from detectron2.data.davis import davis_val
import pickle as pkl

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

    dataset = davis_val(cfg, False)

    fix_root_dir = '/data1/lilong/rank_saliency/dataset/Fixation_Result'
    fix_models = ['CASNet_II', 'DeepGaze', 'DSCLRCN', 'DVA', 'GazeGAN', 'MSINet', 'SalGAN', 'SAM', 'UNISAL']
    fix_result = [[], [], [], [], [], [], [], [], []]
    fix_corr = []
    with inference_context(model), torch.no_grad():
        for i in range(len(dataset)):
            input = dataset[i]
            gt_boxes = input["gt_boxes"]
            gt_masks = input["gt_masks"]
            gt_ranks = input["gt_rank"]

            outputs = model([input])

            roi_results = outputs["roi_results"]
            pred_boxes = [r.pred_boxes.tensor.cpu().data.numpy() for r in roi_results][0]
            scores = [r.scores.cpu().data.numpy() for r in roi_results][0]
            pred_masks = [r.pred_masks.cpu().data.numpy() for r in roi_results][0]

            saliency_rank = outputs["rank_result"]
            saliency_rank = [s.cpu().data.numpy() for s in saliency_rank][0]

            image_shape = input["image_shape"]
            name = input["file_name"].split('/')[-1]

            score = scores
            keep = score > 0.6
            pred_box = pred_boxes
            pred_box = pred_box[keep, :]
            pred_mask = pred_masks
            pred_mask = pred_mask[keep, :, :]
            sal_rank = saliency_rank
            sal_rank = sal_rank[keep]

            segmaps = np.zeros([len(pred_mask), image_shape[0], image_shape[1]])

            for j in range(len(pred_mask)):
                x0 = int(pred_box[j, 0])
                y0 = int(pred_box[j, 1])
                x1 = int(pred_box[j, 2])
                y1 = int(pred_box[j, 3])

                segmap = pred_mask[j, 0, :, :]
                segmap = cv2.resize(segmap, (x1-x0, y1-y0),
                                    interpolation=cv2.INTER_LANCZOS4)

                segmaps[j, y0:y1, x0:x1] = segmap

            for k in range(len(fix_models)):
                model = fix_models[k]
                fix_dir = os.path.join(fix_root_dir, model)
                fix_img_dir = os.path.join(fix_dir, name)
                fix_img = cv2.imread(fix_img_dir, 0).astype(np.float)
                fix_img /= 255

                segs = copy.deepcopy(segmaps)
                fix_rank_socres = np.zeros(len(pred_mask))
                for j, seg in enumerate(segs):
                    seg[seg >= 0.5] = 1
                    seg[seg < 0.5] = 0
                    ind = seg != 0
                    fix_rank_socres[j] = np.max(fix_img[ind])
                fix_result[k].append({'gt_masks': gt_masks, 'segmaps': segmaps, 'gt_ranks': gt_ranks,
                        'rank_scores': fix_rank_socres, 'img_name': name})

            print("\r{}/{}".format(len(fix_result[0]), len(dataset)), end="", flush=True)

        for i in range(len(fix_result)):
            r_corre = rank_evalu(fix_result[i], 0.5)
            fix_corr.append(r_corre)

        for i in range(len(fix_models)):
            print("{}: {}".format(fix_models[i], fix_corr[i]))

        return {}, {}
