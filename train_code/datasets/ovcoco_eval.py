import numpy as np

from .coco_eval import CocoEvaluator


class OVCocoEvaluator(CocoEvaluator):
    def __init__(self, coco_gt, iou_types):
        super().__init__(coco_gt, iou_types)
        self.unseen_list = [4, 5, 11, 12, 15, 16, 21, 23, 27, 29, 32, 34, 45, 47, 54, 58, 63]

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print("IoU metric: {}".format(iou_type))
            coco_eval.summarize()

            precisions = self.coco_eval[iou_type].eval["precision"]

            results_seen = []
            results_unseen = []
            for idx in range(precisions.shape[-3]):
                # area range index 0: all area ranges
                # max dets index -1: typically 100 per image
                precision = precisions[0, :, idx, 0, -1]
                precision = precision[precision > -1]
                if precision.size:
                    ap = np.mean(precision)
                    if idx not in self.unseen_list:
                        results_seen.append(float(ap * 100))
                    else:
                        results_unseen.append(float(ap * 100))
            print(f"{len(results_seen)} {iou_type} AP seen: {np.mean(results_seen)}")
            print(f"{len(results_unseen)} {iou_type} AP unseen: {np.mean(results_unseen)}")
