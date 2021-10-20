import pickle
import logging
import operator
import numpy as np
import torch.utils.data
from .common import (
    AspectRatioGroupedDataset,
    DatasetFromList,
    MapDataset,
)
from . import samplers
from .my_dataset_mapper import DatasetMapper
from detectron2.utils.env import seed_all_rng


def build_rank_saliency_train_loader(cfg):
    images_per_batch = cfg.SOLVER.IMS_PER_BATCH
    dataset_dir = cfg.DATASETS.TRAIN
    f = open(dataset_dir[0], 'rb')
    dataset_dicts = pickle.load(f)
    dataset = DatasetFromList(dataset_dicts, copy=False)
    mapper = DatasetMapper(cfg, True)
    dataset = MapDataset(dataset, mapper)

    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    logger = logging.getLogger(__name__)
    logger.info("Using training sampler {}".format(sampler_name))
    sampler = samplers.TrainingSampler(len(dataset))
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        num_workers=1,
        batch_sampler=None,
        collate_fn=operator.itemgetter(0),
        worker_init_fn=worker_init_reset_seed,
    )
    data_loader = AspectRatioGroupedDataset(data_loader, images_per_batch)
    return data_loader


def worker_init_reset_seed(worker_id):
    seed_all_rng(np.random.randint(2 ** 31) + worker_id)
