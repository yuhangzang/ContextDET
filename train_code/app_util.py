import argparse
import io

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from models.blip2_decoder import BLIP2Decoder
from models.deformable_detr.backbone import build_backbone
from models.ov_blip_detr import OVBLIP2DETR
from models.post_process import CondNMSPostProcess, CondPostProcess
from models.transformer import build_ov_transformer
from util.misc import nested_tensor_from_tensor_list


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')

    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)

    parser.add_argument('--with_box_refine', default=True, action='store_false')
    parser.add_argument('--two_stage', default=True, action='store_false')

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=5, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.0, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=900, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    parser.add_argument('--assign_first_stage', default=True, action='store_false')
    parser.add_argument('--assign_second_stage', default=True, action='store_false')

    parser.add_argument('--name', default='ov')
    parser.add_argument('--llm_name', default='bert-base-cased')

    parser.add_argument('--resume', default='', type=str)
    return parser.parse_args()


COLORS = [
    [0.000, 0.447, 0.741],
    [0.850, 0.325, 0.098],
    [0.929, 0.694, 0.125],
    [0.494, 0.184, 0.556],
    [0.466, 0.674, 0.188],
    [0.301, 0.745, 0.933]
]


def fig2img(fig):
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


def visualize_prediction(pil_img, output_dict, threshold=0.7):
    keep = output_dict["scores"] > threshold
    boxes = output_dict["boxes"][keep].tolist()
    scores = output_dict["scores"][keep].tolist()
    keep_list = keep.nonzero().squeeze(1).numpy().tolist()
    labels = [output_dict["names"][i] for i in keep_list]

    plt.figure(figsize=(12.8, 8))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for score, (xmin, ymin, xmax, ymax), label, color in zip(scores, boxes, labels, colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=color, linewidth=3))
        ax.text(xmin, ymin, f"{label}: {score:0.2f}", fontsize=15, bbox=dict(facecolor="yellow", alpha=0.5))
    plt.axis("off")
    return fig2img(plt.gcf())


class ChatDetDemo():
    def __init__(self, resume):
        self.transform = T.Compose([
            T.Resize(480),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        args = parse_args()

        args.llm_name = 'caption_coco_opt2.7b'
        args.resume = resume
        # args.resume = 'exps/public/ovcoco/checkpoint.pth'
        # args.resume = 'exps/public/refcocog/checkpoint.pth'
        # args.resume = 'exps/public/flickr/checkpoint.pth'

        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        num_classes = 2
        device = torch.device(args.device)

        backbone = build_backbone(args)
        transformer = build_ov_transformer(args)
        llm_decoder = BLIP2Decoder(args.llm_name)
        model = OVBLIP2DETR(
            backbone,
            transformer,
            num_classes=num_classes,
            num_queries=args.num_queries,
            num_feature_levels=args.num_feature_levels,
            aux_loss=False,
            with_box_refine=args.with_box_refine,
            two_stage=args.two_stage,
            llm_decoder=llm_decoder,
        )
        model = model.to(device)

        checkpoint = torch.load(args.resume, map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))

        if args.assign_second_stage:
            postprocessor = CondNMSPostProcess(args.num_queries)
        else:
            postprocessor = CondPostProcess(args.num_queries)

        self.model = model
        self.model.eval()
        self.postprocessor = postprocessor

    def forward(self, image, text, task_button, history, threshold=0.33):
        samples = self.transform(image).unsqueeze(0)
        samples = nested_tensor_from_tensor_list(samples)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        samples = samples.to(device)
        vis = self.model.llm_decoder.vis_processors

        if task_button == "Question Answering":
            text = f"{text} Answer:"
            history.append(text)
            prompt = " ".join(history)
        elif task_button == "Captioning":
            prompt = "A photo of"
        else:
            prompt = text

        blip2_samples = {
            'image': vis['eval'](image)[None, :].to(device),
            'prompt': [prompt],
        }
        outputs = self.model(samples, blip2_samples, mask_infos=None, task_button=task_button)

        mask_infos = outputs['mask_infos_pred']
        pred_names = [list(mask_info.values()) for mask_info in mask_infos]
        orig_target_sizes = torch.tensor([tuple(reversed(image.size))]).to(device)
        results = self.postprocessor(outputs, orig_target_sizes, pred_names, mask_infos)[0]
        image_vis = visualize_prediction(image, results, threshold)

        out_text = outputs['output_text'][0]
        if task_button == "Question Answering":
            history += [out_text]
            chat = [
                (history[i], history[i + 1]) for i in range(0, len(history) - 1, 2)
            ]
        elif task_button == "Captioning":
            history = []
            chat = [
                ("please describe the image", out_text),
            ]
        else:
            history = []
            chat = []
        return image_vis, chat, history
