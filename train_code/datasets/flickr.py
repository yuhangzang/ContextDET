import contextlib
import os
from pathlib import Path

import torch
import torchvision

import datasets.transforms as T
from util.misc import nested_tensor_from_tensor_list


class Flickr(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, processor):
        # suppress pycocotools prints
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                super(Flickr, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self._processor = processor
        self.prepare = flickr_prepare()

    def __getitem__(self, idx):
        img, target = super(Flickr, self).__getitem__(idx)

        image_id = self.ids[idx]
        img_annos = self.coco.loadImgs(image_id)[0]
        caption = img_annos['caption']
        mask_ids = img_annos['mask_ids']
        mask_names = img_annos['mask_names']

        target = {
            'image_id': image_id,
            'annotations': target,
            'caption': caption,
            'mask_ids': mask_ids,
            'mask_names': mask_names,
        }
        if self._processor is not None:
            img_vis = self._processor(img)
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        if self._processor is not None:
            target['img'] = img_vis
        return img, target


class flickr_prepare(object):
    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]

        caption = target['caption']
        mask_ids = target['mask_ids']
        mask_names = target['mask_names']

        target = {}
        target["boxes"] = boxes
        target["image_id"] = image_id

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]
        target["labels"] = torch.ones_like(target["iscrowd"]).long()

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        target["caption"] = caption
        target["mask_infos"] = {tuple(a): b for a, b in zip(mask_ids, mask_names)}
        if anno and 'start_id' in anno[0].keys():
            start_ids = [ann['start_id'] for ann in anno]
            start_ids = torch.tensor(start_ids)
            start_ids = start_ids[keep]
            end_ids = [ann['end_id'] for ann in anno]
            end_ids = torch.tensor(end_ids)
            end_ids = end_ids[keep]
            target['start_ids'] = start_ids
            target['end_ids'] = end_ids

        if anno and 'instance_start_id' in anno[0].keys():
            start_ids = [ann['instance_start_id'] for ann in anno]
            start_ids = torch.tensor(start_ids)
            start_ids = start_ids[keep]
            end_ids = [ann['instance_end_id'] for ann in anno]
            end_ids = torch.tensor(end_ids)
            end_ids = end_ids[keep]
            target['instance_start_ids'] = start_ids
            target['instance_end_ids'] = end_ids
        return image, target


def collate_fn_flickr(batch, tokenizer):
    batch = list(zip(*batch))
    img = nested_tensor_from_tensor_list(batch[0])
    targets = batch[1]

    captions = [i['caption'] for i in batch[1]]
    tokens_input = tokenizer(captions, padding=True, return_tensors='pt')
    tokend_gt = tokenizer(captions, padding=True, return_tensors='pt')
    mask_infos = [i['mask_infos'] for i in batch[1]]
    for b in range(len(mask_infos)):
        for start, end in mask_infos[b].keys():
            tokens_input.input_ids[b][start:end + 1] = tokenizer.mask_token_id
        targets[b]['tokens_gt'] = tokend_gt.input_ids[b: b + 1]
        targets[b]['input_ids'] = tokens_input.input_ids[b: b + 1]
        targets[b]['attention_mask'] = tokens_input.attention_mask[b: b + 1]
    return tuple([img, targets])


def make_transforms(image_set):
    min_size = 320
    max_size = 640

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.48145466, 0.4578275, 0.40821073],
                    [0.26862954, 0.26130258, 0.27577711])
    ])

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomResize([min_size], max_size=max_size),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([min_size], max_size=max_size),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args, processor):
    root = Path(args.coco_path)
    img_folder = root
    if 'bert' in args.llm_name:
        ann_file = root / "annos" / f'flickr30k_bert_{image_set}.json'
    elif 'opt' in args.llm_name:
        # ann_file = root / "annos" / f'flickr30k_opt_{image_set}.json'
        # ann_file = root / "annos" / f'flickr30k_opt_{image_set}_v2.json'
        ann_file = root / "annos" / f'refcoco_opt_{image_set}_v2.json'
    dataset = Flickr(img_folder, ann_file, transforms=make_transforms(image_set), processor=processor)
    return dataset
