import copy
import numpy as np
import torch
from . import detection_utils as utils
from detectron2.structures import BoxMode, Instances, Boxes, PolygonMasks
from . import transforms as T
from instaboostfast import get_new_data, InstaBoostConfig
import cv2
import pycocotools.mask as mask_utils


class DatasetMapper(object):

    def __init__(self, cfg, is_train=True):
        self.tfm_gens = utils.build_transform_gen(cfg, is_train)

        self.img_format = cfg.INPUT.FORMAT
        self.mask_on = cfg.MODEL.MASK_ON

        self.is_train = is_train

        self.instaBoostcfg = InstaBoostConfig(action_candidate=("normal", "horizontal", "skip",),
                                              action_prob=(0.4, 0.35, 0.25,), scale=(0.8, 1.2),
                                              dx=15, dy=15, theta=(-5, 5), color_prob=0.0, heatmap_flag=True)

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        # print(dataset_dict['file_name'].split('/')[-1])
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        annos = dataset_dict.pop("annotations")
        # len_annos = len(annos)
        # for anno in annos:
        #     anno['category_id'] = 1

        # annos, image = get_new_data(annos, image, self.instaBoostcfg, background=None)

        # assert len_annos == len(annos)

        boxes = [BoxMode.convert(obj["bbox"], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS) for obj in annos]

        # image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        image_shape = image.shape[:2]
        #
        # boxes = [transforms.apply_box([box])[0] for box in boxes]

        # check the box is nonempty
        # boxes_check = np.array(boxes)
        # check_w = boxes_check[:, 2] > boxes_check[:, 0]
        # check_h = boxes_check[:, 3] > boxes_check[:, 1]
        # assert np.unique(check_h) and np.unique(check_w)
        # assert boxes_check[:, 0].all() in range(0, image_shape[1])
        # assert boxes_check[:, 1].all() in range(0, image_shape[0])
        # assert boxes_check[:, 2].all() in range(0, image_shape[1])
        # assert boxes_check[:, 3].all() in range(0, image_shape[0])

        segms = [obj["segmentation"] for obj in annos]
        # segms_tr = []
        # for segm in segms:
        #     polygons = [np.asarray(p).reshape(-1, 2) for p in segm]
        #     segms_tr.append([
        #         p.reshape(-1) for p in transforms.apply_polygons(polygons)
        #     ])
        # masks = PolygonMasks(segms_tr)
        # masks = PolygonMasks(segms)

        def rle_to_polygon(segm):
            if isinstance(segm, list):
                return segm

            if isinstance(segm, dict):
                h, w = segm['size']
                rle = mask_utils.frPyObjects(segm, h, w)
                mask = mask_utils.decode(rle)
                contour, hierarchy = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1
                )
                reshaped_contour = []
                for entity in contour:
                    assert len(entity.shape) == 3
                    assert (
                            entity.shape[1] == 1
                    ), "Hierarchical contours are not allowed"
                    if entity.shape[0] >= 3:
                        reshaped_contour.append(entity.reshape(-1).tolist())
                assert len(reshaped_contour)
                return reshaped_contour

        segms = [rle_to_polygon(segm) for segm in segms]
        masks = PolygonMasks(segms)

        instances = Instances(image_shape)
        boxes = instances.gt_boxes = Boxes(boxes)
        boxes.clip(image_shape)
        instances.gt_masks = masks

        classes = [0]*len(annos)
        classes = torch.tensor(classes, dtype=torch.int64)
        instances.gt_classes = classes

        instances.gt_ranks = torch.tensor(dataset_dict.pop("rank"))

        is_person = [obj["is_person"] for obj in annos]
        is_person = torch.tensor(is_person, dtype=torch.int64)
        instances.is_person = is_person

        dataset_dict["instances"] = instances

        return dataset_dict

