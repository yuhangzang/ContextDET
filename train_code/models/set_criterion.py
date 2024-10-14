import copy

import torch
import torch.nn.functional as F

from util import box_ops
from util.misc import accuracy, get_world_size, is_dist_avail_and_initialized

from .assigner import CondStage2Assigner
from .deformable_detr.deformable_detr import SetCriterion
from .deformable_detr.segmentation import sigmoid_focal_loss


class OVSetCriterion(SetCriterion):
    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25,
                 num_queries=300, assign_first_stage=False, assign_second_stage=False,
                 cond_matcher=None):
        super().__init__(num_classes, matcher, weight_dict, losses, focal_alpha,
                         num_queries, assign_first_stage, assign_second_stage)
        self.cond_matcher = cond_matcher
        if self.assign_second_stage:
            self.cond_stg2_assigner = CondStage2Assigner(num_queries)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def loss_mlm(self, outputs, targets, indices, num_boxes):
        pred = outputs['pred_mlm_logits']
        target = torch.cat([t['tokens_gt'] for t in targets])
        # ignore index
        mask = torch.zeros_like(target)
        for b, t in enumerate(targets):
            for start, end in t['mask_infos'].keys():
                mask[b, start: end + 1] = 1
        target = target * mask
        target[target == 0] = -100
        loss_mlm = F.cross_entropy(pred.permute(0, 2, 1), target)
        return {'loss_mlm': loss_mlm}

    def loss_start_end(self, outputs, targets, indices, num_boxes):
        loss_dict = {}
        for key in ['start', 'end', 'instance_start', 'instance_end']:
            pred = outputs[f'pred_{key}']
            target = torch.full(pred.shape[:2], 1, dtype=torch.int64, device=pred.device)
            if f'{key}_ids' not in targets[0].keys():
                loss_dict[f'loss_{key}'] = pred.sum() * 0.0
                continue
            for b in range(len(targets)):
                target[b][torch.unique(targets[b][f'{key}_ids'])] = 0
            target_onehot = torch.zeros(
                [pred.shape[0], pred.shape[1], pred.shape[2] + 1],
                dtype=pred.dtype, layout=pred.layout, device=pred.device)
            target_onehot.scatter_(2, target.unsqueeze(-1), 1)
            target_onehot = target_onehot[:, :, :-1]
            loss = sigmoid_focal_loss(pred, target_onehot, num_boxes, alpha=0.25, gamma=2) * pred.shape[1]
            loss_dict[f'loss_{key}'] = loss
        return loss_dict

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        if 'masks' in outputs.keys():
            masks = outputs['masks']
            target_classes_o = torch.cat([
                t["labels"][m][J] for t, m, (_, J) in zip(targets, masks, indices)
            ])
        else:
            target_classes_o = torch.cat([
                t["labels"][J] for t, (_, J) in zip(targets, indices)
            ])
        target_classes = torch.full(src_logits.shape[:2],
                                    self.num_classes,
                                    dtype=torch.int64,
                                    device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([
            src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
            dtype=src_logits.dtype, layout=src_logits.layout,
            device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(
            src_logits, target_classes_onehot, num_boxes, alpha=0.25,
            gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        if 'masks' in outputs.keys():
            masks = outputs['masks']
            tgt_lengths = torch.as_tensor([len(v["labels"][m]) for v, m in zip(targets, masks)], device=device)
        else:
            tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        if 'masks' in outputs.keys():
            masks = outputs['masks']
            target_boxes = torch.cat([
                t['boxes'][m][i] for t, m, (_, i) in zip(targets, masks, indices)
            ], dim=0)
        else:
            target_boxes = torch.cat([
                t['boxes'][i] for t, (_, i) in zip(targets, indices)
            ], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes),
                box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'start_end': self.loss_start_end,
            'mlm': self.loss_mlm,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        select_ids = outputs['select_ids']
        masks = []
        for b, t in enumerate(targets):
            mask = t["labels"] == -2
            if "start_ids" in t.keys():
                for ind, (start, end) in enumerate(zip(t["start_ids"], t["end_ids"])):
                    if (int(start), int(end)) in select_ids[b]:
                        mask[ind] = True
            if "instance_start_ids" in t.keys():
                for ind, (start, end) in enumerate(zip(t["instance_start_ids"], t["instance_end_ids"])):
                    if (int(start), int(end)) in select_ids[b]:
                        mask[ind] = True
            masks.append(mask)

        # Retrieve the matching between the outputs of the last layer and the targets
        if self.assign_second_stage:
            indices = self.cond_stg2_assigner(outputs_without_aux, targets, select_ids, masks)
        else:
            indices = self.cond_matcher(outputs_without_aux, targets, select_ids, masks)
        outputs['masks'] = masks

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"][m]) for t, m in zip(targets, masks))
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=self.device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                if not self.assign_second_stage:
                    indices = self.cond_matcher(aux_outputs, targets, select_ids, masks)
                aux_outputs['masks'] = masks
                for loss in self.losses:
                    if loss in ['masks', 'mlm', 'start_end']:
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            num_boxes = sum([len(t["labels"]) for t in targets])
            num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=self.device)
            if is_dist_avail_and_initialized():
                torch.distributed.all_reduce(num_boxes)
            num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            if self.assign_first_stage:
                indices = self.stg1_assigner(enc_outputs, bin_targets)
            else:
                indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss in ['masks', 'mlm', 'start_end']:
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                l_dict = {k + '_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)
        return losses
