import json
from pathlib import Path

merged_data = {'images': [], 'annotations': []}
root = Path("data/annos_v2/")
for split in ['train']:
    for prefix in ['refcoco', 'refcoco+', 'refcocog']:
        json_path = root / f'{prefix}_opt_{split}.json'
        with open(json_path, 'r') as f:
            data = json.load(f)
        merged_data['images'] += data['images']
        merged_data['annotations'] += data['annotations']
        print(len(merged_data['images']))

with open(root / f'refcocomerge_opt_{split}.json', 'w') as f:
    json.dump(merged_data, f)
