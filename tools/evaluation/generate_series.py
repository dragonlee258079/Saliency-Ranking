import pickle as pkl
import numpy as np
from scipy.optimize import linear_sum_assignment
import cv2
import os
from PIL import Image



def list_colors(im):
    colors = []

    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if im[i, j].sum() == 0:
                continue
            have_occurred = False
            for color in colors:
                if np.all(color == im[i, j, :]):
                    have_occurred = True
            if not have_occurred:
                colors.append(im[i, j, :])
    return colors


def caculate_mask_iou_matrix(maskA, maskB):
    """
    :param maskA:[n1, h, w]
    :param maskB: [n2, h, w]
    :return: iou_matrix [n1, n2]
    """
    iou_matrix = np.zeros((maskA.shape[0], maskB.shape[0]))
    for i in range(maskA.shape[0]):
        for j in range(maskB.shape[0]):
            intersection = (maskA[i] + maskB[j] >= 2).astype(np.float32).sum()
            iou = intersection / (maskA[i] + maskB[j] >= 1).astype(np.float32).sum()
            iou_matrix[i, j] = iou

    return iou_matrix


def generate_series(predicted_masks, image_name):
    annotation_root_dir = "/data/zhaowangbo/Video Object Segmentation/DAVIS2017 unsurprised/DAVIS-trainval/Annotations_unsupervised/480p/"
    predicted_masks = predicted_masks.astype(np.int)

    gt = cv2.imread(annotation_root_dir + image_name.split("_")[0] + "/" + image_name.split("_")[1] + ".png")
    colors = list_colors(gt)
    print(colors)
    segs = np.zeros([len(colors), gt.shape[0], gt.shape[1]])
    for i, color in enumerate(colors):
        segs[i] = np.all(gt == color, axis=2)
    gt_masks = segs


    if predicted_masks.shape[0] < gt_masks.shape[0]:
        predicted_masks = np.concatenate((predicted_masks, np.zeros([gt_masks.shape[0]-predicted_masks.shape[0], predicted_masks.shape[1], predicted_masks.shape[2]])), axis=0)

    iou_matrix = caculate_mask_iou_matrix(gt_masks, predicted_masks)
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)


    mask_map = np.zeros([gt.shape[0], gt.shape[1]])
    PALETTE = [0, 0, 0]

    for i, j in zip(row_ind, col_ind):
        color_for_save = colors[i]
        PALETTE.extend(color_for_save)

        predicted_mask_for_save = predicted_masks[j] #{h, w]
        predicted_mask_for_save[predicted_mask_for_save != 0] = i+1
        to_zero = (~(predicted_mask_for_save != 0)).astype(np.float32)

        mask_map = mask_map * to_zero + predicted_mask_for_save

    mask_map = mask_map.astype(np.uint8) #!!!!!!!!!!!!!!!must convert to np.uint8

    if not os.path.exists("series/" + image_name.split("_")[0]):
        os.mkdir("series/" + image_name.split("_")[0])

    # mask_map = cv2.applyColorMap(mask_map, cv2.COLORMAP_JET)
    image = Image.fromarray(mask_map, mode="P")
    image.putpalette(PALETTE)
    image.save("series/" + image_name.split("_")[0] + "/" + image_name.split("_")[1] + ".png")


