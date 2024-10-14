import json
from pathlib import Path

merged_data = {'images': [], 'annotations': []}
root = Path("data/annos_v2/")

path_1 = root / "refcocog_opt_train.json"
path_2 = root / "flickr30k_opt_train.json"
# path_3 = root / "ovcoco_opt_instances_train2017_seen.json"

for json_path in [path_1, path_2]:
    with open(json_path, 'r') as f:
        data = json.load(f)
    merged_data['images'] += data['images']
    merged_data['annotations'] += data['annotations']
    print(len(merged_data['images']))

with open(root / 'refcocog+flickr30k_opt_train.json', 'w') as f:
    json.dump(merged_data, f)
