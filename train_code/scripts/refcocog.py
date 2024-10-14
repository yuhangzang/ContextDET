import json
from pathlib import Path

import torch

from lavis.models import load_model_and_preprocess

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
model, _, _ = load_model_and_preprocess(name="blip2_opt", model_type="pretrain_opt2.7b",
                                        is_eval=True, device=device)
tokenizer = model.opt_tokenizer
tokenizer.padding_side = "right"

image_id, box_id = 158122, 708864

root = Path("data/coco/referring/")
for split in ['train', 'val', 'test', 'testA', 'testB']:
    for prefix in ['refcoco', 'refcoco+', 'refcocog']:
        json_path = root / f'finetune_{prefix}_{split}.json'
        if not json_path.exists():
            continue
        with open(json_path, 'r') as f:
            data = json.load(f)

        data_new = {}
        data_new['images'] = []
        data_new['annotations'] = []

        for anno, box_anno in zip(data['images'], data['annotations']):
            tokens = tokenizer.encode(anno['caption'], return_tensors='pt', padding="longest")[0]
            instance_name = anno['caption']
            instance_start_id = 1
            instance_end_id = len(tokens) - 1

            ids_set = {}
            ids_set[(instance_start_id, instance_end_id)] = instance_name

            new_box_anno = {
                'area': box_anno['area'],
                'iscrowd': box_anno['iscrowd'],
                'image_id': image_id,
                'id': box_id,
                'bbox': box_anno['bbox'],
                'instance_start_id': instance_start_id,
                'instance_end_id': instance_end_id,
                'instance_name': instance_name,
            }
            data_new['annotations'].append(new_box_anno)
            box_id += 1

            new_anno = {
                'file_name': "coco/train2014/" + anno['file_name'],
                'height': anno['height'],
                'width': anno['width'],
                'id': image_id,
                'caption': anno['caption'],
                'mask_ids': list(ids_set.keys()),
                'mask_names': list(ids_set.values()),
                'dataset_name': prefix,
                'split': split,
            }
            data_new['images'].append(new_anno)
            image_id += 1

        out_prefix = 'data/annos_v2/'
        with open(out_prefix + f'{prefix}_opt_{split}.json', 'w') as f:
            json.dump(data_new, f)
