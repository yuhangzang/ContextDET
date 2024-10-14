import json
from pathlib import Path

root = Path('data/annos_v2')
new_images = []
for ind in range(4):
    path = f'ovcoco_opt_instances_val2017_all_pred_{ind}.json'
    with open(root / path, 'r') as f:
        data = json.load(f)
    print(len(data['images']))
    new_images += data['images']
print(len(new_images))
data['images'] = new_images
with open(root / 'ovcoco_opt_instances_val2017_all_pred.json', 'w') as f:
    json.dump(data, f)
