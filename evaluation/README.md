# CODE Dataset Evaluation

## Leaderboard

## Setting Up Evaluation

1. Download CODE Dataset

Download the CODE annotations from [this Google Drive Link](https://drive.google.com/drive/folders/1Ge4QXBYZgHYCSew6_vnDv86yX0ZcZ_ut?usp=sharing).

Example Structure:

```json
{
    "file_name": "flickr30k/flickr30k_images/1016887272.jpg",
    "height": 500,
    "width": 333,
    "id": 153152,
    "caption": "Several climbers in a row are climbing the rock while the man in red watches and holds the line .",
    "mask_ids": [[2, 2], [9, 9], [12, 12], [14, 14], [19, 19]],
    "mask_names": ["climbers", "rock", "man", "red", "line"],
    "dataset_name": "flickr30k",
    "split": "test"
}
```

2. Evaluation Tools

We provide the official `CodeEvaluator` API modified from [COCO API](https://github.com/cocodataset/cocoapi) to support the evaluation. You can find the code on [here](Eval/coco_eval.py).

You are also required to install the [pycocotools](https://github.com/cocodataset/cocoapi) for evaluation.

3. Prepare Your Predictions

Your model needs to generate predictions in the following format:

```json
{
  "image_id": 148166,
  "bbox": [52, 44, 57, 158],
  "score": 0.97,
  "token_ids": [1, 1],
  "name": "man"
}
```

4. Running Evaluation

Once you have your prepared predictions, follow the following instructions to run the evaluation:

```python
# path for gt annotations
gt_path = 'xxx.json'

code_eval = CodeEvaluator(gt_path, 'bbox', top_ks=5)
#
code_eval.update(predictions)
code_eval.evaluate()
code_eval.accumulate()
code_eval.summarize()
```

The script will then calculate various metrics based on how well your predictions match the ground truth annotations.
