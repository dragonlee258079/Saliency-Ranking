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
from .mae_fmeasure_2 import evalu as mf_evalu
from detectron2.data.davis import davis_val
from detectron2.data import detection_utils as utils



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

    # dataset = davis_val(cfg, False)
    dataset_root = '/data1/lilong/rank_saliency/dataset/selected_images'
    dataset = os.listdir(dataset_root)
    with inference_context(model), torch.no_grad():

        for i in range(len(dataset)):
            # if i != 200:
            #     continue
            image_path = os.path.join(dataset_root, dataset[i])
            image = utils.read_image(image_path, format='BGR')
            image_shape = image.shape[:2]
            dataset_dict = {}
            dataset_dict["image_shape"] = image_shape
            dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
            inputs = [dataset_dict]

            try:
                outputs = model(inputs)
            except:
                continue

            # gt_boxes = inputs[0]["gt_boxes"]
            # gt_masks = inputs[0]["gt_masks"]
            # gt_ranks = inputs[0]["gt_rank"]

            pred_boxes = outputs["roi_results"][0].pred_boxes
            pred_boxes = pred_boxes.tensor.cpu().data.numpy()
            if pred_boxes.shape[0] == 1:
                continue
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
            # name = inputs[0]["file_name"].split('/')[-1]
            name = dataset[i]

            segmaps = np.zeros([len(pred_masks), image_shape[0], image_shape[1]])

            for j in range(len(pred_masks)):
                x0 = int(pred_boxes[j, 0])
                y0 = int(pred_boxes[j, 1])
                x1 = int(pred_boxes[j, 2])
                y1 = int(pred_boxes[j, 3])

                segmap = pred_masks[j, 0, :, :]
                segmap = cv2.resize(segmap, (x1-x0, y1-y0),
                                    interpolation=cv2.INTER_LANCZOS4)

                segmaps[j, y0:y1, x0:x1] = segmap

            # res.append({'gt_masks': gt_masks, 'segmaps': segmaps, 'scores': scores, 'gt_ranks': gt_ranks,
            #             'rank_scores': saliency_rank, 'img_name': name})

            segmaps1 = copy.deepcopy(segmaps)
            all_segmaps = np.zeros_like(segmaps[0], dtype=np.float)
            if len(pred_masks) != 0:
                color_index = [sorted(saliency_rank).index(a) + 1 for a in saliency_rank]
                color = [255. / len(saliency_rank) * a for a in color_index]
                cover_region = all_segmaps != 0
                for k in range(len(segmaps1), 0, -1):
                    obj_id = color_index.index(k)
                    seg = segmaps1[obj_id]
                    seg[seg >= 0.5] = color[obj_id]
                    seg[seg < 0.5] = 0
                    seg[cover_region] = 0
                    all_segmaps += seg
                    cover_region = all_segmaps != 0
                all_segmaps = all_segmaps.astype(np.int)
            cv2.imwrite('./saliency_maps/{}.png'.format(name[:-4]), all_segmaps)

            # print(len(res))
            print('\r{}/{}'.format(i+1, len(dataset)), end="", flush=True)

        # r_corre = rank_evalu(res, 0.5)
        # r_f = mf_evalu(res)
        #
        return 0, 0
        #return 0, r_f
