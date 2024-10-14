# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


def build_cond_matcher(args):
    return CondMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox,
                       cost_giou=args.set_cost_giou)


class CondMatcher(nn.Module):
    def __init__(self,
                 cost_class: float = 1,
                 cost_bbox: float = 1,
                 cost_giou: float = 1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets, select_ids, masks):
        num_patch = len(select_ids[0])
        bs, num_queries = outputs["pred_logits"].shape[:2]
        alpha = 0.25
        gamma = 2.0
        num_queries = num_queries // num_patch
        out_prob_all = outputs["pred_logits"].view(bs, num_patch, num_queries,
                                                   -1).sigmoid()
        out_bbox_all = outputs["pred_boxes"].view(bs, num_patch, num_queries,
                                                  -1)

        ans = []
        for b in range(bs):
            ans_b = [[], []]
            for index, (start, end) in enumerate(select_ids[b]):
                out_prob = out_prob_all[b, index, :, :]
                out_bbox = out_bbox_all[b, index, :, :]

                tgt_ids = targets[b]["labels"][masks[b]]
                tgt_bbox = targets[b]["boxes"][masks[b]]

                if "start_ids" in targets[b].keys():
                    start_ids = targets[b]["start_ids"][masks[b]]
                    end_ids = targets[b]["end_ids"][masks[b]]
                    cat_mask = torch.logical_and(start_ids == start, end_ids == end)
                else:
                    cat_mask = None
                if "instance_start_ids" in targets[b].keys():
                    instance_start_ids = targets[b]["instance_start_ids"][masks[b]]
                    instance_end_ids = targets[b]["instance_end_ids"][masks[b]]
                    instance_mask = torch.logical_and(instance_start_ids == start, instance_end_ids == end)
                else:
                    instance_mask = None
                if cat_mask is not None and instance_mask is not None:
                    mask = torch.logical_or(cat_mask, instance_mask)
                elif cat_mask is not None:
                    mask = cat_mask
                elif instance_mask is not None:
                    mask = instance_mask
                mask = mask.nonzero().squeeze(1)
                tgt_ids = tgt_ids[mask]
                tgt_bbox = tgt_bbox[mask]

                neg_cost_class = (1 - alpha) * (out_prob**gamma) * (
                    -(1 - out_prob + 1e-8).log())
                pos_cost_class = alpha * (
                    (1 - out_prob)**gamma) * (-(out_prob + 1e-8).log())
                cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

                # Compute the L1 cost between boxes
                cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

                # Compute the giou cost betwen boxes
                cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox),
                                                 box_cxcywh_to_xyxy(tgt_bbox))

                # Final cost matrix
                C = self.cost_class * cost_class + self.cost_bbox * cost_bbox \
                    + self.cost_giou * cost_giou
                C = C.view(1, num_queries, -1).cpu()

                sizes = len(tgt_bbox)
                indices = [
                    linear_sum_assignment(c[i])
                    for i, c in enumerate(C.split(sizes, -1))
                ]
                for (x, y) in indices:
                    if len(x) == 0:
                        continue
                    x += index * num_queries
                    y_label = mask.data.cpu().numpy()
                    y_label = y_label[y].tolist()
                    ans_b[0] += x.tolist()
                    ans_b[1] += y_label
            ans.append(ans_b)
        return [(torch.as_tensor(i, dtype=torch.int64),
                 torch.as_tensor(j, dtype=torch.int64)) for i, j in ans]
