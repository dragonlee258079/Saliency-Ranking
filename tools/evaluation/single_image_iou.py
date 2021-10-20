import pickle as pkl
import numpy as np

import pickle as pkl
import numpy as np
from scipy.optimize import linear_sum_assignment

def single_image_iou(predicted_masks, image_name):
    with open("/data1/zhaowangbo/davis/val/" +image_name +"/annotation", "rb") as f:
        ann = pkl.load(f)
    gt_masks = ann["segs"]

    #binary
    predicted_masks = predicted_masks.astype(np.int)

    if predicted_masks.shape[0] < gt_masks.shape[0]:
        predicted_masks = np.concatenate((predicted_masks, np.zeros([gt_masks.shape[0]-predicted_masks.shape[0], predicted_masks.shape[1], predicted_masks.shape[2]])), axis=0)

    iou_matrix = caculate_mask_iou_matrix(gt_masks, predicted_masks)
    # print("iou_matrix", iou_matrix)
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    average_iou = sum(iou_matrix[row_ind, col_ind]) / len(iou_matrix[row_ind, col_ind])
    # print(image_name, average_iou)


    return average_iou


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
