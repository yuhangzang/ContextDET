# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

from .deformable_detr.deformable_detr import build as build_deformable
from .ov_blip_detr import build as build_ov_blip
from .ov_detr import build as build_ov


def build_model(args):
    if args.name == 'deformable':
        return build_deformable(args)
    elif args.name == 'ov':
        return build_ov(args)
    elif args.name == 'ov_blip2':
        return build_ov_blip(args)
    else:
        raise ValueError(f'Model {args.name} not supported')
