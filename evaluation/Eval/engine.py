import torch
from coco_eval import CodeEvaluator
from code import CODE


annFile = './debug_gt.json'

coco_val = CODE(annFile)
iou_types = ('bbox', 'bbox')
top_ks = (1, 5)

coco_evaluator = CodeEvaluator(coco_val, iou_types, top_ks)

res = {148166: {
       'token_ids': [[1, 1] for _ in range(2)],
       'names': [['woman', 'man'], ['woman', 'men']],
       'boxes': {
        'boxes': torch.tensor([[52.0, 44.0, 52+57, 44+158],
                                [0.0, 0.0, 1.0, 1.0]],),
        'scores': torch.tensor([0.2, 0.3]),
       },
}
}

coco_evaluator.update(res)
coco_evaluator.synchronize_between_processes()
coco_evaluator.accumulate()
coco_evaluator.summarize()