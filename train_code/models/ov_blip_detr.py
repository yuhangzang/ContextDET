import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from util.misc import (NestedTensor, inverse_sigmoid,
                       nested_tensor_from_tensor_list)

from .blip2_decoder import BLIP2Decoder
from .deformable_detr.backbone import build_backbone
from .deformable_detr.deformable_detr import DeformableDETR
from .deformable_detr.matcher import build_matcher
from .deformable_detr.segmentation import (DETRsegm, PostProcessPanoptic,
                                           PostProcessSegm)
from .matcher import build_cond_matcher
from .post_process import (CondNMSPostProcess, CondPostProcess,
                           OVCondNMSPostProcess)
from .segmentation import VisionLanguageFusionModule
from .set_criterion import OVSetCriterion
from .transformer import build_ov_transformer


class OVBLIP2DETR(DeformableDETR):
    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, two_stage=False, llm_decoder=None):
        super().__init__(backbone, transformer, num_classes, num_queries, num_feature_levels,
                         aux_loss, with_box_refine, two_stage)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.llm_decoder = llm_decoder
        hidden_dim = transformer.d_model
        out_size = self.llm_decoder.model.opt_proj.out_features
        self.llm_proj = nn.Linear(out_size, hidden_dim, device=self.device)
        self.start_end_proj = nn.Linear(hidden_dim, 4)
        for layer in [self.llm_proj, self.start_end_proj]:
            nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(layer.bias)

        self.fusion_module = VisionLanguageFusionModule(d_model=hidden_dim, nhead=8, batch_first=True)

    def forward(self, samples, blip2_samples, mask_infos=None, task_button=None, threshold=0.3):
        logits, hidden_states, opt_tokens, output_text = self.llm_decoder.model.forward(
            blip2_samples, task_button=task_button)
        hidden_states = hidden_states.detach()
        hidden_states = self.llm_proj(hidden_states)

        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            src_proj = self.input_proj[l](src)

            b, c, h, w = src_proj.shape
            src_proj = rearrange(src_proj, 'b c h w -> b (h w) c', b=b, c=c, h=h, w=w)
            src_proj = self.fusion_module(tgt=src_proj, memory=hidden_states)
            src_proj = rearrange(src_proj, 'b (h w) c -> b c h w', b=b, c=c, h=h, w=w)

            srcs.append(src_proj)
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)

                b, c, h, w = src.shape
                src = rearrange(src, 'b c h w -> b (h w) c', b=b, c=c, h=h, w=w)
                src = self.fusion_module(tgt=src, memory=hidden_states)
                src = rearrange(src, 'b (h w) c -> b c h w', b=b, c=c, h=h, w=w)

                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)
        out = {}
        start_end_proj = self.start_end_proj(hidden_states)
        out['pred_start'] = start_end_proj[:, :, 0:1]
        out['pred_end'] = start_end_proj[:, :, 1:2]
        out['pred_instance_start'] = start_end_proj[:, :, 2:3]
        out['pred_instance_end'] = start_end_proj[:, :, 3:4]
        out['output_text'] = output_text
        if self.training:
            k = min([len(mask_info) for mask_info in mask_infos])
            k = min(k, 1)
            select_ids = [random.sample(mask_info.keys(), k) for mask_info in mask_infos]
            # select_ids = [random.choices(list(mask_info.keys()), k=4) for mask_info in mask_infos]
            llm_feat = []
            for b in range(len(select_ids)):
                llm_feat_b = []
                hidden_states_b = hidden_states[b, :, :]
                for start, end in select_ids[b]:
                    llm_feat_b.append(hidden_states_b[start: end + 1].mean(dim=0, keepdim=True))
                llm_feat.append(torch.cat(llm_feat_b)[None])
            llm_feat = torch.cat(llm_feat)
            query_embeds = None
            if not self.two_stage:
                query_embeds = self.query_embed.weight
            hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, anchors = (
                self.transformer(srcs, masks, pos, query_embeds, llm_feat, k)
            )
            outputs_classes = []
            outputs_coords = []
            for lvl in range(hs.shape[0]):
                if lvl == 0:
                    reference = init_reference
                else:
                    reference = inter_references[lvl - 1]
                reference = inverse_sigmoid(reference)
                outputs_class = self.class_embed[lvl](hs[lvl])
                tmp = self.bbox_embed[lvl](hs[lvl])
                if reference.shape[-1] == 4:
                    tmp += reference
                else:
                    assert reference.shape[-1] == 2
                    tmp[..., :2] += reference
                outputs_coord = tmp.sigmoid()
                outputs_classes.append(outputs_class)
                outputs_coords.append(outputs_coord)
            outputs_class = torch.stack(outputs_classes)
            outputs_coord = torch.stack(outputs_coords)

            out.update({'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1],
                        'init_reference': init_reference})
            out['select_ids'] = select_ids

            if self.aux_loss:
                out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
                for temp in out["aux_outputs"]:
                    temp["select_ids"] = select_ids

            if self.two_stage:
                enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
                out['enc_outputs'] = {
                    'pred_logits': enc_outputs_class,
                    'pred_boxes': enc_outputs_coord,
                    'anchors': anchors,
                }
        else:
            bs = len(samples.tensors)
            mask_infos_pred = [{} for _ in range(bs)]
            llm_feat = []
            tokenizer = self.llm_decoder.model.opt_tokenizer
            if mask_infos is None:
                if task_button == 'Grounding':
                    mask_infos = [{(1, hidden_states.shape[1] - 1): ''}]
                # elif task_button == 'Detection':
                elif True:
                    mask_infos = []
                    for b in range(bs):
                        starts = (out['pred_start'][b, :, 0].sigmoid() > threshold).nonzero().squeeze(1)
                        ends = (out['pred_end'][b, :, 0].sigmoid() > threshold).nonzero().squeeze(1)
                        if len(starts) == 0:
                            starts = out['pred_start'][b, :].argmax(0)
                        if len(ends) == 0:
                            ends = out['pred_end'][b, :].argmax(0)
                        mask_infos_b = {}
                        for start, end in zip(starts, ends):
                            mask_infos_b[(int(start), int(end))] = ''
                        mask_infos.append(mask_infos_b)
                else:
                    mask_infos = []
                    for b in range(bs):
                        starts = (out['pred_instance_start'][b, :, 0].sigmoid() > threshold).nonzero().squeeze(1)
                        ends = (out['pred_instance_end'][b, :, 0].sigmoid() > threshold).nonzero().squeeze(1)
                        if len(starts) == 0:
                            starts = out['pred_instance_start'][b, :].argmax(0)
                        if len(ends) == 0:
                            ends = out['pred_instance_end'][b, :].argmax(0)
                        mask_infos_b = {}
                        for start, end in zip(starts, ends):
                            mask_infos_b[(int(start), int(end))] = ''
                        mask_infos.append(mask_infos_b)
            for b in range(bs):
                llm_feat_b = []
                hidden_states_b = hidden_states[b, :, :]
                for start, end in mask_infos[b].keys():
                    llm_feat_b.append(hidden_states_b[start: end + 1].mean(dim=0, keepdim=True))
                    pred_name = tokenizer.decode(opt_tokens.input_ids[b, start: end + 1]).strip()
                    mask_infos_pred[b][(int(start), int(end))] = pred_name
                llm_feat.append(torch.cat(llm_feat_b)[None])
            out['mask_infos_pred'] = mask_infos_pred

            query_embeds = None
            if not self.two_stage:
                query_embeds = self.query_embed.weight

            outputs_classes_list = []
            outputs_coords_list = []
            for b in range(bs):
                srcs_b = [i[b: b + 1] for i in srcs]
                masks_b = [i[b: b + 1] for i in masks]
                pos_b = [i[b: b + 1] for i in pos]
                k = len(mask_infos[b])
                if k == 0:
                    outputs_classes_list.append(torch.zeros(0, 2).to(self.device))
                    outputs_coords_list.append(torch.zeros(0, 4).to(self.device))
                    continue
                num_repeat = math.ceil(k / 4)
                outputs_classes = []
                outputs_coords = []
                for ind in range(num_repeat):
                    llm_feat_b = llm_feat[b][:, ind * 4: (ind + 1) * 4]
                    hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, anchors = (
                        self.transformer(srcs_b, masks_b, pos_b, query_embeds, llm_feat_b, llm_feat_b.shape[1])
                    )
                    lvl = hs.shape[0] - 1
                    reference = inter_references[lvl - 1]
                    reference = inverse_sigmoid(reference)
                    outputs_class = self.class_embed[lvl](hs[lvl])
                    tmp = self.bbox_embed[lvl](hs[lvl])
                    if reference.shape[-1] == 4:
                        tmp += reference
                    else:
                        assert reference.shape[-1] == 2
                        tmp[..., :2] += reference
                    outputs_coord = tmp.sigmoid()
                    outputs_classes.append(outputs_class.flatten(0, 1))
                    outputs_coords.append(outputs_coord.flatten(0, 1))
                outputs_classes = torch.cat(outputs_classes)[None]
                outputs_coords = torch.cat(outputs_coords)[None]
                outputs_classes_list.append(outputs_classes)
                outputs_coords_list.append(outputs_coords)

            out.update({'pred_logits': outputs_classes_list,
                        'pred_boxes': outputs_coords_list})
        return out


def build(args):
    num_classes = 2
    device = torch.device(args.device)

    backbone = build_backbone(args)
    transformer = build_ov_transformer(args)
    llm_decoder = BLIP2Decoder(args.llm_name)

    model = OVBLIP2DETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        llm_decoder=llm_decoder,
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))

    matcher = build_matcher(args)
    cond_matcher = build_cond_matcher(args)
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + '_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    for key in ['loss_mlm', 'loss_start', 'loss_end', 'loss_instance_start', 'loss_instance_end']:
        weight_dict[key] = 1.0

    losses = ['labels', 'boxes', 'cardinality', 'start_end']
    if args.masks:
        losses += ["masks"]
    # num_classes, matcher, weight_dict, losses, focal_alpha=0.25
    criterion = OVSetCriterion(num_classes, matcher, weight_dict, losses, focal_alpha=args.focal_alpha,
                               num_queries=args.num_queries,
                               assign_first_stage=args.assign_first_stage,
                               assign_second_stage=args.assign_second_stage,
                               cond_matcher=cond_matcher)
    criterion.to(device)
    if args.assign_second_stage:
        if args.dataset_file == 'ovcoco':
            postprocessors = {'bbox': OVCondNMSPostProcess(args.num_queries)}
        else:
            postprocessors = {'bbox': CondNMSPostProcess(args.num_queries)}
    else:
        postprocessors = {'bbox': CondPostProcess(args.num_queries)}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors
