import numpy as np
import pickle


def calc_iou(mask_a, mask_b):
    intersection = (mask_a + mask_b >= 2).astype(np.float32).sum()
    iou = intersection / (mask_a + mask_b >= 1).astype(np.float32).sum()
    return iou


def match(matrix, iou_thread, img_name):
    # print(img_name)
    matched_gts = np.arange(matrix.shape[0])
    matched_ranks = matrix.argsort()[:, -1]
    for i, j in zip(matched_gts, matched_ranks):
        if matrix[i][j] < iou_thread:
            matched_ranks[i] = -1
    if len(set(matched_ranks[matched_ranks != -1])) < len(matched_ranks[matched_ranks != -1]):
        print(img_name)
        raise KeyError
    if len(matched_ranks) < matrix.shape[1]:
        for i in range(matrix.shape[1]):
            if i not in matched_ranks:
                matched_ranks = np.append(matched_ranks, i)
    return matched_ranks


def match_rank_scores(gt_masks, segmaps, rank_scores, name):
    segmaps[segmaps > 0.5] = 1
    segmaps[segmaps <= 0.5] = 0

    ious = np.zeros([len(gt_masks), len(segmaps)])
    for i in range(len(gt_masks)):
        for j in range(len(segmaps)):
            ious[i][j] = calc_iou(gt_masks[i], segmaps[j])
    matched_ranks = match(ious, 0.5, name)
    unmatched_index = np.argwhere(matched_ranks == -1).squeeze(1)
    matched_ranks = matched_ranks[matched_ranks >= 0]
    rank_scores = rank_scores[matched_ranks].squeeze(1)
    for i in range(len(unmatched_index)):
        rank_scores = np.insert(rank_scores, unmatched_index[i], 0)
    rank_scores = rank_scores[:len(gt_masks)]
    return rank_scores


def calu_mae(gt_ranks, rank_scores, gt_masks, segmaps, names):
    num = len(gt_ranks)
    mae = 0
    no_seg = []
    for i in range(len(gt_ranks)):
        if len(segmaps[i]) == 0:
            num -= 1
            no_seg.append(names[i])
            continue
        gt_rank = np.asarray(gt_ranks[i]).astype(np.float)
        gt_rank = (gt_rank + 1) / len(gt_rank)
        rank_score = match_rank_scores(gt_masks[i], segmaps[i], rank_scores[i], names[i])
        m = np.mean(np.abs(gt_rank - rank_score))
        mae += m
    print(no_seg)
    return mae / num


def calu_fmeasure(gt, pred):
    betaSqr = 0.3
    positive_set = gt
    P = np.sum(positive_set)
    positive_samples = pred
    TPmat = positive_set * positive_samples
    PS = np.sum(positive_samples)
    TP = np.sum(TPmat)
    TPR = TP / P
    Precision = TP/PS
    if PS == 0:
        F = 0
        Precision = 0
        TPR = 0
    elif TPR == 0:
        F = 0
    else:
        F = (1 + betaSqr) * TPR * Precision / (TPR + betaSqr * Precision)
    return F


def make_map(gt_rank, rank_score, gt_masks, segmaps):
    image_shape = segmaps.shape[1:]
    gt_map = np.zeros(image_shape)
    rank_map = np.zeros(image_shape)
    gt_index = (np.asarray(gt_rank) + 1).astype(np.float)/len(gt_rank)
    rank_index = [sorted(rank_score).index(a) for a in rank_score]
    rank_index = (np.asarray(rank_index) + 1).astype(np.float)/len(rank_index)
    for i in range(len(segmaps)):
        rank_map[segmaps[i] > 0.5] = rank_index[i]
    for i in range(len(gt_masks)):
        gt_map[gt_masks[i] != 0] = gt_index[i]
    return gt_map, rank_map


def f_measure(gt_ranks, rank_scores, gt_masks, segmaps):
    num = len(gt_ranks)
    f_final = 0
    for i in range(len(gt_ranks)):
        gt_map, rank_map = make_map(gt_ranks[i], rank_scores[i], gt_masks[i], segmaps[i])
        instances = np.unique(gt_map)
        f = 0
        for j in range(len(instances)-1):
            thr_low, thr_high = instances[j], instances[j+1]
            gt_mask = (gt_map > thr_low) & (gt_map <= thr_high)
            rank_mask = (rank_map > thr_low) & (rank_map <= thr_high)
            f_level_j = calu_fmeasure(gt_mask, rank_mask)
            f += f_level_j
        f = f / (len(instances)-1)
        f_final += f
    return f_final / num


def evalu(results):
    gt_ranks = [r.pop("gt_ranks") for r in results]
    rank_scores = [r.pop("rank_scores") for r in results]
    gt_masks = [r.pop("gt_masks") for r in results]
    segmaps = [r.pop("segmaps") for r in results]
    names = [r.pop("img_name") for r in results]
    mae = calu_mae(gt_ranks, rank_scores, gt_masks, segmaps, names)
    f_m = f_measure(gt_ranks, rank_scores, gt_masks, segmaps)
    return {"mae": mae, "f_measure": f_m}
    # print(f_m)
    # return f_m


if __name__ == '__main__':
    f = open('../res.pkl', 'rb')
    res = pickle.load(f)
    results = evalu(res)
    print(results)
    pass

