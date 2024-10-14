import torch
from torchvision.ops.boxes import batched_nms

from util.box_ops import box_cxcywh_to_xyxy

from .deformable_detr.deformable_transformer import DeformableTransformer
from .segmentation import QueryFusionModule


class OVTransformer(DeformableTransformer):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4, enc_n_points=4,
                 two_stage=False, two_stage_num_proposals=300,
                 assign_first_stage=False):
        super().__init__(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout,
                         activation, return_intermediate_dec, num_feature_levels, dec_n_points, enc_n_points,
                         two_stage, two_stage_num_proposals, assign_first_stage)

    def forward(self, srcs, masks, pos_embeds, query_embed=None, llm_feat=None, num_patch=1):
        assert self.two_stage or query_embed is not None

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios,
                              lvl_pos_embed_flatten, mask_flatten)

        # prepare input for decoder
        bs, _, c = memory.shape
        if self.two_stage:
            output_memory, output_proposals, level_ids = \
                self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)

            # hack implementation for two-stage Deformable DETR
            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
            enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals

            topk = self.two_stage_num_proposals
            proposal_logit = enc_outputs_class[..., 0]

            if self.assign_first_stage:
                proposal_boxes = box_cxcywh_to_xyxy(enc_outputs_coord_unact.sigmoid().float()).clamp(0, 1)
                topk_proposals = []
                for b in range(bs):
                    prop_boxes_b = proposal_boxes[b]
                    prop_logits_b = proposal_logit[b]

                    # pre-nms per-level topk
                    pre_nms_topk = 1000
                    pre_nms_inds = []
                    for lvl in range(len(spatial_shapes)):
                        lvl_mask = level_ids == lvl
                        pre_nms_inds.append(torch.topk(prop_logits_b.sigmoid() * lvl_mask, pre_nms_topk)[1])
                    pre_nms_inds = torch.cat(pre_nms_inds)

                    # nms on topk indices
                    post_nms_inds = batched_nms(prop_boxes_b[pre_nms_inds],
                                                prop_logits_b[pre_nms_inds],
                                                level_ids[pre_nms_inds], 0.9)
                    keep_inds = pre_nms_inds[post_nms_inds]

                    if len(keep_inds) < self.two_stage_num_proposals:
                        print(f'[WARNING] nms proposals ({len(keep_inds)}) < {self.two_stage_num_proposals}')
                        keep_inds = torch.topk(proposal_logit[b], topk)[1]

                    # keep top Q/L indices for L levels
                    q_per_l = topk // len(spatial_shapes)
                    level_shapes = torch.arange(len(spatial_shapes), device=level_ids.device)[:, None]
                    is_level_ordered = level_ids[keep_inds][None] == level_shapes
                    keep_inds_mask = is_level_ordered & (is_level_ordered.cumsum(1) <= q_per_l)  # LS
                    keep_inds_mask = keep_inds_mask.any(0)  # S

                    # pad to Q indices (might let ones filtered from pre-nms sneak by...
                    # unlikely because we pick high conf anyways)
                    if keep_inds_mask.sum() < topk:
                        num_to_add = topk - keep_inds_mask.sum()
                        pad_inds = (~keep_inds_mask).nonzero()[:num_to_add]
                        keep_inds_mask[pad_inds] = True

                    # index
                    keep_inds_topk = keep_inds[keep_inds_mask]
                    topk_proposals.append(keep_inds_topk)
                topk_proposals = torch.stack(topk_proposals)
            else:
                topk_proposals = torch.topk(proposal_logit, topk, dim=1)[1]

            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points
            pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
            query_embed, tgt = torch.split(pos_trans_out, c, dim=2)

            num_queries = query_embed.shape[1]
            query_embed = query_embed.repeat(1, num_patch, 1)
            tgt = tgt.repeat(1, num_patch, 1)
            topk_feats = torch.stack([output_memory[b][topk_proposals[b]] for b in range(bs)]).detach()
            topk_feats = topk_feats.repeat(1, num_patch, 1)
            tgt = tgt + self.pix_trans_norm(self.pix_trans(topk_feats))
            reference_points = reference_points.repeat(1, num_patch, 1)
            init_reference_out = init_reference_out.repeat(1, num_patch, 1)

            llm_feat = llm_feat.repeat_interleave(num_queries, 1)
            tgt = tgt + llm_feat
        else:
            raise NotImplementedError
            query_embed, tgt = torch.split(query_embed, c, dim=1)
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            reference_points = self.reference_points(query_embed).sigmoid()
            init_reference_out = reference_points
        # decoder mask
        decoder_mask = (
            torch.ones(
                num_queries * num_patch,
                num_queries * num_patch,
                device=query_embed.device,
            ) * float("-inf")
        )
        for i in range(num_patch):
            decoder_mask[
                i * num_queries : (i + 1) * num_queries,
                i * num_queries : (i + 1) * num_queries,
            ] = 0

        # decoder
        hs, inter_references = self.decoder(tgt, reference_points, memory,
                                            spatial_shapes, level_start_index, valid_ratios,
                                            query_embed, mask_flatten, tgt_mask=decoder_mask)

        inter_references_out = inter_references
        if self.two_stage:
            return (hs,
                    init_reference_out,
                    inter_references_out,
                    enc_outputs_class,
                    enc_outputs_coord_unact,
                    output_proposals.sigmoid())
        return hs, init_reference_out, inter_references_out, None, None, None


def build_ov_transformer(args):
    return OVTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        two_stage=args.two_stage,
        two_stage_num_proposals=args.num_queries,
        assign_first_stage=args.assign_first_stage)
