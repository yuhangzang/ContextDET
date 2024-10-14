from pathlib import Path

import datasets.transforms as T

from .flickr import Flickr


def make_coco_transforms(image_set, bigger):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if 'train' in image_set:
        scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    if image_set in ['val', 'test', 'testA', 'testB']:
        scales = [800]

    max_size = 1333

    if bigger:
        scales = [int(1.5 * s) for s in scales]
        max_size = 2000

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=max_size),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=max_size),
                ])
            ),
            normalize,
        ])

    elif image_set in ['val', 'test', 'testA', 'testB']:
        return T.Compose([
            T.RandomResize(scales, max_size=max_size),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args, processor):
    root = Path(args.coco_path)
    assert root.exists(), f'provided Referring COCO path {root} does not exist'
    dataset = args.dataset_file
    assert dataset in ['refcoco', 'refcoco+', 'refcocog']
    if 'opt' in args.llm_name:
        tokenizer = 'opt'
    else:
        raise ValueError(f"{args.llm_name} is not implemented yet.")
    img_folder = root
    if image_set == 'train':
        ann_file = root / "annos_v2" / 'refcocomerge_opt_train.json'
    else:
        ann_file = root / "annos_v2" / f'{dataset}_{tokenizer}_{image_set}.json'

    dataset = Flickr(img_folder, ann_file,
                     transforms=make_coco_transforms(image_set, args.bigger),
                     processor=processor)
    return dataset
