import json
from pathlib import Path

import torch
from tqdm import tqdm

from flickr30k_entities_utils import get_annotations, get_sentence_data
from lavis.models import load_model_and_preprocess


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


def convert_flickr30k_to_coco(root, split, tokenizer, image_id, box_id):
    assert split in ['train', 'val', 'test']
    with open(root / f"{split}.txt") as f:
        file_list = [line.rstrip() for line in f]

    data = {}
    data['images'] = []
    data['annotations'] = []

    for file_name in tqdm(file_list):
        anno_path = root / "Annotations" / f"{file_name}.xml"
        sentence_path = root / "Sentences" / f"{file_name}.txt"

        anno_info = get_annotations(anno_path)
        sentence_info = get_sentence_data(sentence_path)

        for sent_id, sentence in enumerate(sentence_info):
            phrases = [phrase for phrase in sentence["phrases"] if phrase["phrase_id"] in anno_info['boxes'].keys()]
            if len(phrases) > 0:
                caption = sentence['sentence']
                tokens_ori = tokenizer.encode(caption, return_tensors='pt', padding="longest")
                tokens_ori = tokens_ori.squeeze(0)
                caption_list = caption.split(" ")
                ids_set = {}
                for phrase in phrases:
                    phrase_id = phrase['phrase_id']
                    phrase_boxes = anno_info['boxes'][phrase_id]

                    for phrase_box in phrase_boxes:
                        xmin, ymin, xmax, ymax = phrase_box
                        box = [xmin, ymin, xmax - xmin, ymax - ymin]
                        area = box[2] * box[3]

                        first_word_index = phrase['first_word_index']
                        end_word_index = first_word_index + len(phrase['phrase'].split(' ')) - 1
                        name = caption_list[end_word_index].strip()
                        start_id, end_id = find_token_id(caption, end_word_index, end_word_index, tokens_ori, tokenizer)
                        token_name = tokenizer.decode(tokens_ori[start_id:end_id + 1]).strip()

                        if (start_id, end_id) not in ids_set.keys():
                            ids_set[(start_id, end_id)] = name

                        instance_name = " ".join(caption_list[first_word_index:end_word_index + 1]).strip()
                        instance_start_id, instance_end_id = find_token_id(caption, first_word_index, end_word_index, tokens_ori, tokenizer)
                        instance_token_name = tokenizer.decode(tokens_ori[instance_start_id:instance_end_id + 1]).strip()

                        if (instance_start_id, instance_end_id) not in ids_set.keys():
                            ids_set[(instance_start_id, instance_end_id)] = instance_name

                        # if name != token_name:
                        #     print(name, token_name)
                        # if instance_name != instance_token_name:
                        #     print(instance_name, instance_token_name)

                        box_anno = {
                            'area': area,
                            'iscrowd': 0,
                            'image_id': image_id,
                            'id': box_id,
                            'bbox': box,
                            'start_id': start_id,
                            'end_id': end_id,
                            'name': name,
                            'instance_start_id': instance_start_id,
                            'instance_end_id': instance_end_id,
                            'instance_name': instance_name
                        }
                        data['annotations'].append(box_anno)
                        box_id += 1
                images_anno = {
                    'file_name': "flickr30k/flickr30k_images/" + file_name + '.jpg',
                    'height': anno_info['height'],
                    'width': anno_info['width'],
                    'id': image_id,
                    'caption': caption,
                    'mask_ids': list(ids_set.keys()),
                    'mask_names': list(ids_set.values()),
                    'dataset_name': 'flickr30k',
                    'split': split,
                }
                data['images'].append(images_anno)
                image_id += 1

    out_prefix = '/mnt/lustre/share/zangyuhang/OVSC_v2/'
    with open(out_prefix + f'flickr30k_opt_{split}.json', 'w') as f:
        json.dump(data, f)
    return image_id, box_id


if __name__ == '__main__':
    root = Path("/mnt/lustre/share/zangyuhang/Flick30k")
    image_id = 0
    box_id = 0

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    model, _, _ = load_model_and_preprocess(name="blip2_opt", model_type="pretrain_opt2.7b",
                                            is_eval=True, device=device)
    tokenizer = model.opt_tokenizer
    tokenizer.padding_side = "right"

    image_id, box_id = convert_flickr30k_to_coco(root, 'train', tokenizer, image_id, box_id)
    image_id, box_id = convert_flickr30k_to_coco(root, 'val', tokenizer, image_id, box_id)
    image_id, box_id = convert_flickr30k_to_coco(root, 'test', tokenizer, image_id, box_id)
    print(image_id, box_id)

# 158121 708863
