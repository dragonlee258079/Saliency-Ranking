#!/usr/bin/env python
import sys
import os

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, path)

import logging
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data.my_train_loader_build import build_rank_saliency_train_loader
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)
from evaluation.inference import inference

# from apex import amp
# import setproctitle
#setproctitle.setproctitle("lxtx")

logger = logging.getLogger("detectron2")


def do_test(cfg, model, iteration):
    logger.info('testing {}.model'.format(iteration))
    r_corre, m_f = inference(cfg, model)
    mae, fm = m_f['mae'], m_f['f_measure']
    with open('corre.txt', 'a') as corre_file:
        corre_file.write('{:.4f}\n'.format(r_corre))
        corre_file.close()
    with open('fm.txt', 'a') as fm_file:
        fm_file.write('{:.4f}\n'.format(fm))
        fm_file.close()
    with open('mae.txt', 'a') as mae_file:
        mae_file.write('{:.4f}\n'.format(mae))
        mae_file.close()

def do_train(cfg, model, optimizer, scheduler, out_dir, resume=False):
    model.train()

    checkpointer = DetectionCheckpointer(
        model, out_dir, optimizer=optimizer, scheduler=scheduler
    )

    checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume)

    start_iter = 0
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = (
        [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(out_dir, "metrics.json")),
            TensorboardXWriter(out_dir),
        ]
        if comm.is_main_process()
        else []
    )

    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement
    data_loader = build_rank_saliency_train_loader(cfg)
    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            iteration = iteration + 1
            storage.step()

            loss_dict = model(data)
            if not loss_dict:
                continue
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            '''
            # with amp.scale_loss(losses, optimizer) as scaled_loss:
            #     scaled_loss.backward()
            '''
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            if (
                cfg.TEST.EVAL_PERIOD > 0
                and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter
            ):
                do_test(cfg, model, iteration)
                # Compared to "train_net.py", the test results are not dumped to EventStorage
                # comm.synchronize()

            if iteration - start_iter > 5 and (iteration % 20 == 0 or iteration == max_iter):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(
        cfg, args
    )  # if you don't like any of the default setup, write your own setup code
    return cfg


def main(args):
    cfg = setup(args)

    model = build_model(cfg)

    logger.info("Model:\n{}".format(model))
    # if args.eval_only:
    #     DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
    #         cfg.MODEL.WEIGHTS, resume=args.resume
    #     )
    #     return do_test(cfg, model)

    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    # model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    distributed = comm.get_world_size() > 1

    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )

    do_train(cfg, model, optimizer, scheduler, args.output_dir)
    # return do_test(cfg, model)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
