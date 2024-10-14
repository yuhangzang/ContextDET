import torch
import torch.nn as nn

from util.box_ops import box_cxcywh_to_xyxy, box_iou

from .deformable_detr.assigner import (Matcher, sample_topk_per_gt,
                                       subsample_labels)


class CondStage2Assigner(nn.Module):
    def __init__(self, num_queries, max_k=4):
        super().__init__()
        self.positive_fraction = 0.25
        self.bg_label = 400  # number > 91 to filter out later
        self.batch_size_per_image = num_queries
        self.proposal_matcher = Matcher(thresholds=[0.6], labels=[0, 1], allow_low_quality_matches=True)
        self.k = max_k
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _sample_proposals(
        self, matched_idxs: torch.Tensor, matched_labels: torch.Tensor, gt_classes: torch.Tensor
    ):
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.bg_label
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.bg_label

        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes, self.batch_size_per_image, self.positive_fraction, self.bg_label
        )

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs]

    def forward(self, outputs, targets, select_ids, masks, return_cost_matrix=False):
        # COCO categories are from 1 to 90. They set num_classes=91 and apply sigmoid.
        num_patch = len(select_ids[0])
        bs, num_queries = outputs["pred_logits"].shape[:2]
        num_queries = num_queries // num_patch
        out_bbox_all = outputs["init_reference"].view(bs, num_patch, num_queries, -1)

        bs = len(targets)
        indices = []
        ious = []
        for b in range(bs):
            ans_b = [[], []]
            ious_b = []
            for index, (start, end) in enumerate(select_ids[b]):
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

                iou, _ = box_iou(
                    box_cxcywh_to_xyxy(tgt_bbox),
                    box_cxcywh_to_xyxy(out_bbox_all[b, index].detach()),
                )
                # proposal_id -> highest_iou_gt_id, proposal_id -> [1 if iou > 0.6, 0 ow]
                matched_idxs, matched_labels = self.proposal_matcher(iou)
                # list of sampled proposal_ids, sampled_id -> [0, num_classes)+[bg_label]
                sampled_idxs, sampled_gt_classes = self._sample_proposals(
                    matched_idxs, matched_labels, tgt_ids
                )
                pos_pr_inds = sampled_idxs[sampled_gt_classes != self.bg_label]
                pos_gt_inds = matched_idxs[pos_pr_inds]
                pos_pr_inds, pos_gt_inds = self.postprocess_indices(pos_pr_inds, pos_gt_inds, iou)
                ious_b.append(iou)
                for (x, y) in [(pos_pr_inds, pos_gt_inds)]:
                    if len(x) == 0:
                        continue
                    x += index * num_queries
                    y_label = mask.data.cpu().numpy()
                    y_label = y_label[y.cpu()].tolist()
                    if type(y_label) == int:
                        y_label = [y_label]
                    ans_b[0] += x.tolist()
                    ans_b[1] += y_label
            ans_b = [torch.tensor(ans_b[0]).to(self.device).long(), torch.tensor(ans_b[1]).to(self.device).long()]
            indices.append(ans_b)
            ious_b = torch.cat(ious_b)
            ious.append(ious_b)
        if return_cost_matrix:
            return indices, ious
        return indices

    def postprocess_indices(self, pr_inds, gt_inds, iou):
        return sample_topk_per_gt(pr_inds, gt_inds, iou, self.k)
