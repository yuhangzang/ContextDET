import contextlib
import copy
import os

import numpy as np
import torch

from util.misc import all_gather

from .codeeval import CODE, CODEeval


class CodeEvaluator(object):
    def __init__(self, coco_gt, iou_types, top_ks, use_names):
        assert isinstance(iou_types, (list, tuple))
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt

        self.iou_types = iou_types
        self.top_ks = top_ks
        self.use_names = use_names
        self.coco_eval = {}
        for iou_type, top_k, use_name in zip(iou_types, top_ks, use_names):
            self.coco_eval[(iou_type, top_k, use_name)] = CODEeval(coco_gt, iouType=iou_type,
                                                                   topK=top_k, useName=use_name)

        self.img_ids = []
        self.eval_imgs = {(k, top_k, use_name): [] for k, top_k, use_name in zip(iou_types, top_ks, use_names)}

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type, top_k, use_name in zip(self.iou_types, self.top_ks, self.use_names):
            results = self.prepare(predictions, iou_type)

            # suppress pycocotools prints
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    coco_dt = CODE.loadRes(self.coco_gt, results) if results else CODE()
            coco_eval = self.coco_eval[(iou_type, top_k, use_name)]

            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = list(img_ids)
            img_ids, eval_imgs = evaluate(coco_eval)

            self.eval_imgs[(iou_type, top_k, use_name)].append(eval_imgs)

    def synchronize_between_processes(self):
        for iou_type, top_k, use_name in zip(self.iou_types, self.top_ks, self.use_names):
            self.eval_imgs[(iou_type, top_k, use_name)] = np.concatenate(self.eval_imgs[(iou_type, top_k, use_name)], 2)
            create_common_coco_eval(self.coco_eval[(iou_type, top_k, use_name)],
                                    self.img_ids, self.eval_imgs[(iou_type, top_k, use_name)])

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        for (iou_type, top_k, use_name), coco_eval in self.coco_eval.items():
            print(f"IoU metric: {iou_type}: Top@{top_k}, Use Name@{use_name}")
            coco_eval.summarize()

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        else:
            raise ValueError("Unknown iou type {}".format(iou_type))

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction['boxes']) == 0:
                continue

            names = prediction['names']
            start_id = prediction['start_id']
            end_id = prediction['end_id']
            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "bbox": box,
                        "score": scores[k],
                        "names": n,
                        "start_id": s,
                        "end_id": e,
                    }
                    for k, (box, n, s, e) in enumerate(zip(boxes, names, start_id, end_id))
                ]
            )
        return coco_results


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def merge(img_ids, eval_imgs):
    all_img_ids = all_gather(img_ids)
    all_eval_imgs = all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.append(p)

    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx]

    return merged_img_ids, merged_eval_imgs


def create_common_coco_eval(coco_eval, img_ids, eval_imgs):
    img_ids, eval_imgs = merge(img_ids, eval_imgs)
    img_ids = list(img_ids)
    eval_imgs = list(eval_imgs.flatten())

    coco_eval.evalImgs = eval_imgs
    coco_eval.params.imgIds = img_ids
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)


#################################################################
# From pycocotools, just removed the prints and fixed
# a Python3 bug about unicode not defined
#################################################################


def evaluate(self):
    '''
    Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
    :return: None
    '''
    # tic = time.time()
    # print('Running per image evaluation...')
    p = self.params
    # add backward compatibility if useSegm is specified in params
    if p.useSegm is not None:
        p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
        print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
    # print('Evaluate annotation type *{}*'.format(p.iouType))
    p.imgIds = list(np.unique(p.imgIds))
    if p.useCats:
        p.catIds = list(np.unique(p.catIds))
    p.maxDets = sorted(p.maxDets)
    self.params = p

    self._prepare()
    # loop through images, area range, max detection number
    catIds = p.catIds if p.useCats else [-1]

    if p.iouType == 'segm' or p.iouType == 'bbox':
        computeIoU = self.computeIoU
    elif p.iouType == 'keypoints':
        computeIoU = self.computeOks
    self.ious = {
        (imgId, catId): computeIoU(imgId, catId)
        for imgId in p.imgIds
        for catId in catIds}

    evaluateImg = self.evaluateImg
    maxDet = p.maxDets[-1]
    evalImgs = [
        evaluateImg(imgId, catId, areaRng, maxDet)
        for catId in catIds
        for areaRng in p.areaRng
        for imgId in p.imgIds
    ]
    # this is NOT in the pycocotools code, but could be done outside
    evalImgs = np.asarray(evalImgs).reshape(len(catIds), len(p.areaRng), len(p.imgIds))
    self._paramsEval = copy.deepcopy(self.params)
    # toc = time.time()
    # print('DONE (t={:0.2f}s).'.format(toc-tic))
    return p.imgIds, evalImgs
