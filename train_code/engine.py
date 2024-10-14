# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable

import torch
import torch.distributed as dist

import util.misc as utils
from datasets.acc_eval import AccEvaluator
from datasets.coco_eval import CocoEvaluator
from datasets.code_eval import CodeEvaluator
from datasets.data_prefetcher import data_prefetcher
from datasets.ovcoco_eval import OVCocoEvaluator
from datasets.refcoco_eval import RefExpEvaluator
from util.misc import is_dist_avail_and_initialized


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    name: str = 'deformable'):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()

    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        if name == 'deformable':
            outputs = model(samples)
        elif name == 'ov':
            input_ids = torch.cat([t['input_ids'] for t in targets])
            attention_mask = torch.cat([t['attention_mask'] for t in targets])
            mask_infos = [t['mask_infos'] for t in targets]
            outputs = model(samples, input_ids, attention_mask, mask_infos)
        elif name == 'ov_blip2':
            blip2_samples = {
                'image': torch.cat([t['img'].unsqueeze(0) for t in targets]),
                'text_input': [t['caption'] for t in targets],
            }
            mask_infos = [t['mask_infos'] for t in targets]
            outputs = model(samples, blip2_samples, mask_infos)
        else:
            raise ValueError(f"{name} not implemented yet.")

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        samples, targets = prefetcher.next()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, name):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    if name == 'ov':
        iou_types = ('bbox', 'bbox', 'bbox')
        top_ks = (1, 5, -1)
        use_names = (True, True, False)
        det_evaluator = CodeEvaluator(base_ds, iou_types, top_ks, use_names)
    elif name == 'ov_blip2':
        iou_types = ('bbox', )
        top_ks = (-1, )
        use_names = (False, )
        det_evaluator = CodeEvaluator(base_ds, iou_types, top_ks, use_names)
    elif name == 'deformable':
        iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
        det_evaluator = CocoEvaluator(base_ds, iou_types)
    else:
        raise ValueError(f"{name} not implemented yet.")
    acc_evaluator = None
    if name == 'ov':
        k_list = [1, 5]
        acc_evaluator = AccEvaluator(k_list=k_list)
        if is_dist_avail_and_initialized():
            tokenizer = model.module.llm_decoder.tokenizer
        else:
            tokenizer = model.llm_decoder.tokenizer
    res_all = {}

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        orig_target_sizes = torch.stack([t["orig_size"].to(device) for t in targets], dim=0)
        if name == 'deformable':
            outputs = model(samples)
            results = postprocessors['bbox'](outputs, orig_target_sizes)
        elif name == 'ov':
            input_ids = torch.cat([t['input_ids'] for t in targets])
            attention_mask = torch.cat([t['attention_mask'] for t in targets])
            mask_infos = [t['mask_infos'] for t in targets]
            outputs = model(samples, input_ids, attention_mask, mask_infos)
            pred_names = acc_evaluator.update(outputs['pred_mlm_logits'], mask_infos, tokenizer)
            results = postprocessors['bbox'](outputs, orig_target_sizes, pred_names, mask_infos)
        elif name == 'ov_blip2':
            blip2_samples = {
                'image': torch.cat([t['img'].unsqueeze(0) for t in targets]),
                'text_input': [t['caption'] for t in targets],
            }
            # mask_infos = [t['mask_infos'] for t in targets]
            outputs = model(samples, blip2_samples, mask_infos=[])
            mask_infos = outputs['mask_infos_pred']
            pred_names = [[[t] * 5 for t in mask_info.values()] for mask_info in mask_infos]
            results = postprocessors['bbox'](outputs, orig_target_sizes, pred_names, mask_infos)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        res_all.update(res)
        det_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    det_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    det_evaluator.accumulate()
    det_evaluator.summarize()

    if acc_evaluator is not None:
        acc_evaluator.synchronize_between_processes()
        acc_evaluator.summarize()

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    dump_detection_results(res_all, output_dir)
    return stats, det_evaluator


@torch.no_grad()
def evaluate_referring(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, name):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    ref_evaluator = RefExpEvaluator(base_ds, ("bbox"))

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        orig_target_sizes = torch.stack([t["orig_size"].to(device) for t in targets], dim=0)
        if name == 'deformable':
            outputs = model(samples)
            results = postprocessors['bbox'](outputs, orig_target_sizes)
        elif name == 'ov_blip2':
            blip2_samples = {
                'image': torch.cat([t['img'].unsqueeze(0) for t in targets]),
                'text_input': [t['caption'] for t in targets],
            }
            mask_infos = [t['mask_infos'] for t in targets]
            outputs = model(samples, blip2_samples, mask_infos=mask_infos)
            pred_names = [list(m.values()) for m in mask_infos]
            results = postprocessors['bbox'](outputs, orig_target_sizes, pred_names, mask_infos)
        else:
            raise NotImplementedError

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        ref_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    ref_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    ref_evaluator.accumulate()
    ref_evaluator.summarize()

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return stats, ref_evaluator


@torch.no_grad()
def evaluate_ovdet(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, name):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    det_evaluator = OVCocoEvaluator(base_ds, iou_types)

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        orig_target_sizes = torch.stack([t["orig_size"].to(device) for t in targets], dim=0)
        if name == 'ov_blip2':
            blip2_samples = {
                'image': torch.cat([t['img'].unsqueeze(0) for t in targets]),
                'text_input': [t['caption'] for t in targets],
            }
            mask_infos = [t['mask_infos'] for t in targets]
            outputs = model(samples, blip2_samples, mask_infos=mask_infos)
            pred_names = [list(m.values()) for m in mask_infos]
            results = postprocessors['bbox'](outputs, orig_target_sizes, pred_names, mask_infos)
        else:
            raise NotImplementedError

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        det_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    det_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    det_evaluator.accumulate()
    det_evaluator.summarize()

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return stats, det_evaluator


def dump_detection_results(res_all, output_dir):
    dist.barrier()
    rank = dist.get_rank()
    torch.save(res_all, output_dir + f'/pred_{rank}.pth')

    dist.barrier()
    if rank == 0:
        world_size = dist.get_world_size()
        for ind in range(rank + 1, world_size):
            res_all.update(torch.load(output_dir + f'/pred_{ind}.pth'))
        torch.save(res_all, output_dir + '/pred_all.pth')
    dist.barrier()
    return
