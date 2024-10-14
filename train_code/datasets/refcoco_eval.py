# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Modified from https://github.com/ashkamath/mdetr/blob/main/datasets/refexp.py

import copy

import torch
import torch.utils.data

from util.box_ops import generalized_box_iou
from util.misc import all_gather, is_main_process


class RefExpEvaluator(object):
    def __init__(self, refexp_gt, iou_types, k=(1, 5, 10), thresh_iou=0.5):
        assert isinstance(k, (list, tuple))
        refexp_gt = copy.deepcopy(refexp_gt)
        self.refexp_gt = refexp_gt
        self.iou_types = iou_types
        self.img_ids = self.refexp_gt.imgs.keys()
        self.predictions = {}
        self.k = k
        self.thresh_iou = thresh_iou

    def accumulate(self):
        pass

    def update(self, predictions):
        cpu_predictions = {}
        for dict_key, inner_dict in predictions.items():
            cpu_inner_dict = {}
            for tensor_key, cuda_tensor in inner_dict.items():
                if type(cuda_tensor) == torch.Tensor:
                    cuda_tensor = cuda_tensor.cpu()
                cpu_inner_dict[tensor_key] = cuda_tensor
            cpu_predictions[dict_key] = cpu_inner_dict
        self.predictions.update(cpu_predictions)

    def synchronize_between_processes(self):
        all_predictions = all_gather(self.predictions)
        merged_predictions = {}
        for p in all_predictions:
            merged_predictions.update(p)
        self.predictions = merged_predictions

    def summarize(self):
        if is_main_process():
            dataset2score = {
                "refcoco": {k: 0.0 for k in self.k},
                "refcoco+": {k: 0.0 for k in self.k},
                "refcocog": {k: 0.0 for k in self.k},
            }
            dataset2count = {"refcoco": 0.0, "refcoco+": 0.0, "refcocog": 0.0}
            for image_id in self.img_ids:
                ann_ids = self.refexp_gt.getAnnIds(imgIds=image_id)
                assert len(ann_ids) == 1
                img_info = self.refexp_gt.loadImgs(image_id)[0]
                dataset_name = img_info["dataset_name"]

                target = self.refexp_gt.loadAnns(ann_ids[0])
                prediction = self.predictions[image_id]
                assert prediction is not None
                sorted_scores_boxes = sorted(
                    zip(prediction["scores"].tolist(), prediction["boxes"].tolist()), reverse=True
                )
                sorted_scores, sorted_boxes = zip(*sorted_scores_boxes)
                sorted_boxes = torch.cat([torch.as_tensor(x).view(1, 4) for x in sorted_boxes])
                target_bbox = target[0]["bbox"]
                converted_bbox = [
                    target_bbox[0],
                    target_bbox[1],
                    target_bbox[2] + target_bbox[0],
                    target_bbox[3] + target_bbox[1],
                ]
                giou = generalized_box_iou(sorted_boxes, torch.as_tensor(converted_bbox).view(-1, 4))
                for k in self.k:
                    if max(giou[:k]) >= self.thresh_iou:
                        dataset2score[dataset_name][k] += 1.0
                dataset2count[dataset_name] += 1.0

            for key, value in dataset2score.items():
                for k in self.k:
                    if key in dataset2count.keys() and dataset2count[key] > 0:
                        value[k] /= dataset2count[key]
            results = {}
            for key, value in dataset2score.items():
                results[key] = sorted([v for k, v in value.items()])
                print(f" Dataset: {key} - Precision @ 1, 5, 10: {results[key]} \n")

            return results
        return None
