import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from util.misc import (NestedTensor, inverse_sigmoid,
                       nested_tensor_from_tensor_list)

from .deformable_detr.backbone import build_backbone
from .deformable_detr.deformable_detr import DeformableDETR
from .deformable_detr.matcher import build_matcher
from .deformable_detr.segmentation import (DETRsegm, PostProcessPanoptic,
                                           PostProcessSegm)
from .llm_decoder import LLMDecoder
from .matcher import build_cond_matcher
from .post_process import CondNMSPostProcess, CondPostProcess
from .set_criterion import OVSetCriterion
from .transformer import build_ov_transformer


class OVDETR(DeformableDETR):
    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, two_stage=False, llm_decoder=None,
                 vis_len=4, ):
        super().__init__(backbone, transformer, num_classes, num_queries, num_feature_levels,
                         aux_loss, with_box_refine, two_stage)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.llm_decoder = llm_decoder
        self.vis_len = vis_len
        out_size = self.llm_decoder.model.bert.config.hidden_size
        self.image_proj = nn.Linear(backbone.num_channels[-1], out_size).to(self.device)
        hidden_dim = transformer.d_model
        self.llm_proj = nn.Linear(out_size, hidden_dim).to(self.device)
        for layer in [self.image_proj, self.llm_proj]:
            nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(layer.bias)

    def demo_inference(self, samples, text):
        tokens_input = self.llm_decoder.tokenizer(text, padding=True, return_tensors='pt')
        tokens_input = tokens_input.to(self.device)
        input_ids = tokens_input.input_ids
        attention_mask = tokens_input.attention_mask

        b = 0
        mask_info = {}
        for ind, token in enumerate(tokens_input.input_ids[b]):
            if token == self.llm_decoder.tokenizer.mask_token_id:
                mask_info[(ind, ind)] = ''

        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
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
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        bert_embeddings = self.llm_decoder.model.bert.embeddings
        inputs_embeds = bert_embeddings.word_embeddings(input_ids)
        img_emb = F.adaptive_avg_pool2d(features[-1].tensors, (2, 2))
        img_emb = img_emb.flatten(1).reshape(len(img_emb), self.vis_len, -1)
        img_emb = self.image_proj(img_emb)
        img_mask = torch.ones(attention_mask.shape[0], self.vis_len).to(self.device)

        inputs_embeds = torch.cat([img_emb, inputs_embeds], dim=1)
        attention_mask = torch.cat([img_mask, attention_mask], dim=1)

        pred_mlm_logits, hidden_states = self.llm_decoder(inputs_embeds, attention_mask)
        pred_mlm_logits = pred_mlm_logits[:, self.vis_len:]
        hidden_states = hidden_states[:, self.vis_len:]
        out = {}
        out['pred_text'] = self.llm_decoder.tokenizer.decode(pred_mlm_logits[0, 1:-1].argmax(1))
        out['mask_infos'] = [mask_info]

        pred_tokens_b = []
        for (start, end) in mask_info.keys():
            mask_token_logits = pred_mlm_logits[b, start:end + 1, :]
            top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices.T.tolist()
            pred_tokens = [self.llm_decoder.tokenizer.decode(token).strip() for token in top_5_tokens]
            pred_tokens_b.append(pred_tokens)
        out['pred_names'] = [pred_tokens_b]

        llm_feat = []
        llm_feat_b = []
        hidden_states_b = hidden_states[b, :, :]
        for start, end in list(mask_info.keys()):
            llm_feat_b.append(hidden_states_b[start: end + 1].mean(dim=0, keepdim=True))
        llm_feat.append(torch.cat(llm_feat_b)[None])
        llm_feat = [self.llm_proj(i.detach()) for i in llm_feat]

        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight

        srcs_b = [i[b: b + 1] for i in srcs]
        masks_b = [i[b: b + 1] for i in masks]
        pos_b = [i[b: b + 1] for i in pos]
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, anchors = (
            self.transformer(srcs_b, masks_b, pos_b, query_embeds, llm_feat[b])
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

        out.update({'pred_logits': [outputs_class[-1]],
                    'pred_boxes': [outputs_coord[-1]]})
        return out

    def forward(self, samples: NestedTensor, input_ids, attention_mask, mask_infos):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
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
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        bert_embeddings = self.llm_decoder.model.bert.embeddings
        inputs_embeds = bert_embeddings.word_embeddings(input_ids)
        img_emb = F.adaptive_avg_pool2d(features[-1].tensors, (2, 2))
        img_emb = img_emb.flatten(1).reshape(len(img_emb), self.vis_len, -1)
        img_emb = self.image_proj(img_emb)
        img_mask = torch.ones(attention_mask.shape[0], self.vis_len).to(self.device)

        inputs_embeds = torch.cat([img_emb, inputs_embeds], dim=1)
        attention_mask = torch.cat([img_mask, attention_mask], dim=1)

        pred_mlm_logits, hidden_states = self.llm_decoder(inputs_embeds, attention_mask)
        pred_mlm_logits = pred_mlm_logits[:, self.vis_len:]
        hidden_states = hidden_states[:, self.vis_len:]
        out = {}
        out['pred_mlm_logits'] = pred_mlm_logits

        if self.training:
            # k = min([len(mask_info) for mask_info in mask_infos])
            # k = min(k, 3)
            # select_ids = [random.sample(mask_info.keys(), k) for mask_info in mask_infos]
            select_ids = [random.choices(list(mask_info.keys()), k=4) for mask_info in mask_infos]
            llm_feat = []
            for b in range(len(select_ids)):
                llm_feat_b = []
                hidden_states_b = hidden_states[b, :, :]
                for start, end in select_ids[b]:
                    llm_feat_b.append(hidden_states_b[start: end + 1].mean(dim=0, keepdim=True))
                llm_feat.append(torch.cat(llm_feat_b)[None])
            llm_feat = torch.cat(llm_feat)
            llm_feat = self.llm_proj(llm_feat.detach())

            query_embeds = None
            if not self.two_stage:
                query_embeds = self.query_embed.weight
            hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, anchors = (
                self.transformer(srcs, masks, pos, query_embeds, llm_feat)
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
            select_ids = [list(mask_info.keys()) for mask_info in mask_infos]
            llm_feat = []
            for b in range(len(select_ids)):
                llm_feat_b = []
                hidden_states_b = hidden_states[b, :, :]
                for start, end in select_ids[b]:
                    llm_feat_b.append(hidden_states_b[start: end + 1].mean(dim=0, keepdim=True))
                llm_feat.append(torch.cat(llm_feat_b)[None])
            llm_feat = [self.llm_proj(i.detach()) for i in llm_feat]

            query_embeds = None
            if not self.two_stage:
                query_embeds = self.query_embed.weight

            outputs_classes_list = []
            outputs_coords_list = []
            for b in range(len(img_emb)):
                srcs_b = [i[b: b + 1] for i in srcs]
                masks_b = [i[b: b + 1] for i in masks]
                pos_b = [i[b: b + 1] for i in pos]
                hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, anchors = (
                    self.transformer(srcs_b, masks_b, pos_b, query_embeds, llm_feat[b])
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
                outputs_classes_list.append(outputs_class[-1])
                outputs_coords_list.append(outputs_coord[-1])

            out.update({'pred_logits': outputs_classes_list,
                        'pred_boxes': outputs_coords_list})
        return out


def build(args):
    num_classes = 2
    device = torch.device(args.device)

    backbone = build_backbone(args)
    transformer = build_ov_transformer(args)
    llm_decoder = LLMDecoder(args.llm_name)
    model = OVDETR(
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
    weight_dict['loss_mlm'] = 1.0

    losses = ['labels', 'boxes', 'cardinality', 'mlm']
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
        postprocessors = {'bbox': CondNMSPostProcess(args.num_queries)}
    else:
        postprocessors = {'bbox': CondPostProcess(args.num_queries)}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors
