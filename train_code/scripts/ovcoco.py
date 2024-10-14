import json
from pathlib import Path

root = Path('data/coco/annotations')

with open(root / 'instances_train2017.json', 'r') as fin:
    coco_train_anno_seen = json.load(fin)
with open(root / 'instances_val2017.json', 'r') as fin:
    coco_val_anno_all = json.load(fin)

labels_seen = ["toilet", "bicycle", "apple", "train", "laptop", "carrot", "motorcycle", "oven",
               "chair", "mouse", "boat", "kite", "sheep", "horse", "sandwich", "clock", "tv",
               "backpack", "toaster", "bowl", "microwave", "bench", "book", "orange", "bird",
               "pizza", "fork", "frisbee", "bear", "vase", "toothbrush", "spoon", "giraffe",
               "handbag", "broccoli", "refrigerator", "remote", "surfboard", "car", "bed",
               "banana", "donut", "skis", "person", "truck", "bottle", "suitcase", "zebra"]
labels_unseen = ["umbrella", "cow", "cup", "bus", "keyboard", "skateboard", "dog", "couch",
                 "tie", "snowboard", "sink", "elephant", "cake", "scissors", "airplane", "cat", "knife"]

print(len(labels_seen), len(labels_unseen))

labels_all = [item['name'] for item in coco_val_anno_all['categories']]
class_id_to_split = {}
class_name_to_split = {}
for item in coco_val_anno_all['categories']:
    if item['name'] in labels_seen:
        class_id_to_split[item['id']] = 'seen'
        class_name_to_split[item['name']] = 'seen'
    elif item['name'] in labels_unseen:
        class_id_to_split[item['id']] = 'unseen'
        class_name_to_split[item['name']] = 'unseen'
class_list = list(class_name_to_split.keys())


def filter_annotation(anno_dict, split_name_list):
    filtered_categories = []
    for item in anno_dict['categories']:
        if class_id_to_split.get(item['id']) in split_name_list:
            item['split'] = class_id_to_split.get(item['id'])
            filtered_categories.append(item)
    anno_dict['categories'] = filtered_categories

    filtered_images = []
    filtered_annotations = []
    useful_image_ids = set()
    for item in anno_dict['annotations']:
        if class_id_to_split.get(item['category_id']) in split_name_list:
            filtered_annotations.append(item)
            useful_image_ids.add(item['image_id'])
    for item in anno_dict['images']:
        if item['id'] in useful_image_ids:
            filtered_images.append(item)
    anno_dict['annotations'] = filtered_annotations
    anno_dict['images'] = filtered_images


filter_annotation(coco_train_anno_seen, ['seen'])
filter_annotation(coco_val_anno_all, ['seen', 'unseen'])

with open(root / 'instances_train2017_seen.json', 'w') as fout:
    json.dump(coco_train_anno_seen, fout)

with open(root / 'instances_val2017_all.json', 'w') as fout:
    json.dump(coco_val_anno_all, fout)
