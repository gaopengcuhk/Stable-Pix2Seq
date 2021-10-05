# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
import pdb
from util import box_ops

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, lr_scheduler: list = [0]):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    optimizer.param_groups[0]['lr'] = lr_scheduler[epoch]
    optimizer.param_groups[1]['lr'] = lr_scheduler[epoch] * 0.1
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        #(x1 y1 x2 y2 label) * max_box + EOS
        max_box = max([len(target['boxes']) for target in targets])
        max_seq_length = max_box * 5 + 1
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        bins = 1000
        num_box = max(max_box + 2, 100)
        box_labels = []
        input_seqs = []
        output_seqs = []
        start = 2001
        padding = 2002
        end = 2000
        category_start = 1500
        no_known = 2002 # n/a and padding share the same label to be eliminated from loss calculation
        noise = 1998
        for target in targets:
            box = (target['boxes'] * (bins - 1)).int()
            label = target['labels'].unsqueeze(-1) + category_start
            box_label = torch.cat([box, label], dim=-1)
            idx = torch.randperm(box_label.shape[0])
            box_label = box_label[idx]


            random_box = torch.rand(num_box - box_label.shape[0], 4).to(target['boxes'])
            random_box = (random_box * (bins - 1)).int()
            random_label = torch.randint(0, 91, (num_box - box_label.shape[0], 1)).to(label)
            random_label = random_label + category_start
            random_box_label = torch.cat([random_box, random_label], dim=-1)

            input_seq = torch.cat([box_label, random_box_label], dim=0)
            input_seq = torch.cat([torch.ones(1).to(box_label) * start, input_seq.flatten()])
            input_seqs.append(input_seq.unsqueeze(0))

            output_na = torch.ones(num_box - box_label.shape[0], 3).to(input_seq) * no_known
            output_noise = torch.ones(num_box - box_label.shape[0], 1).to(input_seq) * noise
            output_end = torch.ones(num_box - box_label.shape[0], 1).to(input_seq) * end
            output_seq = torch.cat([output_na, output_noise, output_end], dim=-1)

            output_seq = torch.cat([box_label.flatten(), torch.ones(1).to(box_label) * end, output_seq.flatten()])
            output_seqs.append(output_seq.unsqueeze(0))
        input_seqs = torch.cat(input_seqs, dim=0)
        output_seqs = torch.cat(output_seqs, dim=0)
        box_labels = output_seqs.flatten()
#        with torch.cuda.amp.autocast():
        if True:
           outputs = model(samples, input_seqs)
           outputs = outputs[-1].reshape(-1, 2003)
           loss = criterion(outputs[box_labels!=2002], box_labels[box_labels!=2002])
        loss_dict = {'at':loss}
        weight_dict = {'at':1}
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
         
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )
    for samples, targets in data_loader:
        batch = len(targets)
        targets = targets[: batch // 2]
        samples.mask = samples.mask[: batch // 2, :, :]
        samples.tensors = samples.tensors[: batch // 2, :, :, :]
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        seq = torch.ones(len(targets), 1).to(samples.mask) * 2001
        outputs = model(samples, seq)
        batch_index = 0
        results = []
        outputs, values = outputs
        for output in outputs:
            output = output[1:].reshape(-1, 5)
            box = output[:, :4].clip(0, 999).float() / (1000 - 1)
            box = box_ops.box_cxcywh_to_xyxy(box)
            label = output[:, 4].unsqueeze(-1) - 1500
            orig_size = targets[batch_index]["orig_size"]
            img_h, img_w = orig_size[0], orig_size[1]
            scale_fct = torch.stack([img_w, img_h, img_w, img_h]).unsqueeze(0)
            box = scale_fct * box
            value = values[batch_index].reshape(-1, 5)[:, -1]
            threshold = 0.3
            select = (value > threshold)
            results.append({'scores': value[select], 'labels': label.squeeze(-1)[select], 'boxes': box[select]})
            batch_index = batch_index + 1
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)
        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    return 0, coco_evaluator
