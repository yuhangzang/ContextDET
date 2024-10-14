import json
from collections import defaultdict
from pathlib import Path

import torch
from lavis.models import load_model_and_preprocess
from tqdm import tqdm

from util.dataset_mapping import coco_categories as categories


def find_token_id(caption, start_word_index, end_word_index, tokens_ori, tokenizer):
    caption_list = caption.split(" ")
    caption_mask = " ".join(caption_list[:start_word_index]) + tokenizer.bos_token
    caption_mask = caption_mask + " " + " ".join(caption_list[end_word_index + 1:])
    tokens_mask = tokenizer.encode(caption_mask, return_tensors='pt', padding="longest")
    tokens_mask = tokens_mask.squeeze(0)
    N = len(tokens_ori)
    left = 0
    while left < N:
        if tokens_ori[left] == tokens_mask[left]:
            left += 1
        else:
            break
    right = N - 1
    right_mask = len(tokens_mask) - 1
    while right >= 0:
        if tokens_ori[right] == tokens_mask[right_mask]:
            right -= 1
            right_mask -= 1
        else:
            break
    return left, right


device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
model, _, _ = load_model_and_preprocess(name="blip2_opt", model_type="pretrain_opt2.7b",
                                        is_eval=True, device=device)
tokenizer = model.opt_tokenizer
tokenizer.padding_side = "right"

root = Path('data/coco/annotations')
root_out = Path('data/annos_v2')

name_dict = {c['id']: c['name'] for c in categories}

for path, split in zip(['instances_train2017_seen.json', 'instances_val2017_all.json'],
                       ['train', 'val']):
    with open(root / path, 'r') as f:
        data = json.load(f)

    data_new = {}
    data_new['categories'] = data['categories']
    data_new['images'] = []
    data_new['annotations'] = []

    imgid2annos = defaultdict(list)
    for box_anno in data['annotations']:
        imgid2annos[box_anno['image_id']].append(box_anno)

    for img_anno in tqdm(data['images']):
        image_id = img_anno['id']
        box_annos = imgid2annos[image_id]
        category_ids = [i['category_id'] for i in box_annos]
        category_ids = sorted(list(set(category_ids)))

        category_names = [name_dict[i] for i in category_ids]
        caption = ' '.join(category_names)
        tokens = tokenizer.encode(caption, return_tensors='pt', padding="longest")[0]
        name_mapping = {}
        start = 0
        for name in category_names:
            start_id, end_id = find_token_id(caption, start, start + len(name.split(' ')) - 1, tokens, tokenizer)
            name_mapping[name] = (start_id, end_id)
            start += len(name.split(' '))

        new_anno = {
            'file_name': f"coco/{split}2017/" + img_anno['file_name'],
            'height': img_anno['height'],
            'width': img_anno['width'],
            'id': img_anno['id'],
            'caption': caption,
            'mask_ids': list(name_mapping.values()),
            'mask_names': list(name_mapping.keys()),
            'dataset_name': 'ovcoco',
            'split': split,
        }
        data_new['images'].append(new_anno)

        for box_anno in box_annos:
            name = name_dict[box_anno['category_id']]
            start_id, end_id = name_mapping[name]
            new_box_anno = {
                'area': box_anno['area'],
                'iscrowd': box_anno['iscrowd'],
                'image_id': img_anno['id'],
                'id': box_anno['id'],
                'bbox': box_anno['bbox'],
                'category_id': box_anno['category_id'],
                'start_id': start_id,
                'end_id': end_id,
                'name': name,
            }
            data_new['annotations'].append(new_box_anno)
    with open(root_out / f'ovcoco_opt_{path}', 'w') as f:
        json.dump(data_new, f)
