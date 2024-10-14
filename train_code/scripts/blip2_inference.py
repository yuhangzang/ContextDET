import json
from pathlib import Path

import torch
from lavis.models import load_model_and_preprocess
from PIL import Image
from tqdm import tqdm


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
model, vis_processors, _ = load_model_and_preprocess(name="blip2_opt", model_type="caption_coco_opt2.7b",
                                                     is_eval=True, device=device)
tokenizer = model.opt_tokenizer
tokenizer.padding_side = "right"

root = Path('data/annos_v2')
path = 'ovcoco_opt_instances_val2017_all.json'
with open(root / path, 'r') as f:
    data = json.load(f)

names = [i['name'] for i in data['categories']]
categories = [i['supercategory'] for i in data['categories']]
"""
caption_list = []
mask_ids_list = []
mask_names_pred = []
for index in range(5):
    caption_pred = names[index * 13: index * 13 + 13]
    caption = ' '.join(caption_pred)
    tokens = tokenizer.encode(caption, return_tensors='pt', padding="longest")[0]
    name_mapping = {}
    start = 0
    for name in caption_pred:
        start_id, end_id = find_token_id(caption, start, start + len(name.split(' ')) - 1, tokens, tokenizer)
        name_mapping[name] = (start_id, end_id)
        start += len(name.split(' '))
    caption_list.append(caption)
    mask_ids_list.append(list(name_mapping.values()))
    mask_names_pred.append(list(name_mapping.keys()))

new_images = []
for img_anno in tqdm(data['images']):
    img_anno['caption'] = caption_list[0]
    img_anno['mask_ids'] = mask_ids_list[0]
    img_anno['mask_names'] = mask_names_pred[0]
    new_images.append(img_anno)
"""

new_images = []
for img_anno in tqdm(data['images'][1200 * 0:1200 * 1]):
    file_name = img_anno['file_name']
    raw_image = Image.open(f"data/{file_name}").convert("RGB")
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    caption_pred = []
    for name, cat in zip(names, categories):
        ans = model.generate({"image": image, "prompt": f"Question: Is {cat} {name} in this image? Answer:"})
        if ans[0][:3] == 'Yes' or ans[0][:3] == 'yes':
            caption_pred.append(name)

    caption = ' '.join(caption_pred)
    tokens = tokenizer.encode(caption, return_tensors='pt', padding="longest")[0]
    name_mapping = {}
    start = 0
    for name in caption_pred:
        start_id, end_id = find_token_id(caption, start, start + len(name.split(' ')) - 1, tokens, tokenizer)
        name_mapping[name] = (start_id, end_id)
        start += len(name.split(' '))

    img_anno['caption'] = caption
    img_anno['mask_ids'] = list(name_mapping.values())
    img_anno['mask_names'] = list(name_mapping.keys())
    new_images.append(img_anno)

data['images'] = new_images
with open(root / 'ovcoco_opt_instances_val2017_all_pred_0.json', 'w') as f:
    json.dump(data, f)
