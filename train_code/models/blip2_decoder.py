import contextlib
import logging

import torch
import torch.nn as nn
from lavis.common.registry import registry
from lavis.models import Blip2OPT, load_preprocess
from omegaconf import OmegaConf


@registry.register_model("blip2_opt_det")
class Blip2OPTDet(Blip2OPT):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    @torch.no_grad()
    def forward(self, samples,
                use_nucleus_sampling=False,
                num_beams=5,
                max_length=30,
                min_length=1,
                top_p=0.9,
                repetition_penalty=1.0,
                length_penalty=1.0,
                num_captions=1,
                temperature=1,
                task_button=None):
        image = samples["image"]
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        inputs_opt = self.opt_proj(query_output.last_hidden_state)
        atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(image.device)

        self.opt_tokenizer.padding_side = "right"

        if "text_input" in samples.keys():
            text = [t + "\n" for t in samples["text_input"]]

            opt_tokens = self.opt_tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
            ).to(image.device)
            output_text = text
        else:
            assert "prompt" in samples.keys()
            prompt = samples["prompt"]
            assert len(prompt) == image.size(0)

            opt_tokens = self.opt_tokenizer(prompt, return_tensors="pt", padding=True).to(
                image.device
            )
            input_ids = opt_tokens.input_ids
            attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)

            if use_nucleus_sampling:
                query_embeds = inputs_opt.repeat_interleave(num_captions, dim=0)
                num_beams = 1
            else:
                query_embeds = inputs_opt.repeat_interleave(num_beams, dim=0)

            with self.maybe_autocast():
                outputs = self.opt_model.generate(
                    input_ids=input_ids,
                    query_embeds=query_embeds,
                    attention_mask=attention_mask,
                    do_sample=use_nucleus_sampling,
                    top_p=top_p,
                    temperature=temperature,
                    num_beams=num_beams,
                    max_new_tokens=max_length,
                    min_length=min_length,
                    eos_token_id=self.eos_token_id,
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty,
                    num_return_sequences=num_captions,
                )

            prompt_length = opt_tokens.input_ids.shape[1]
            output_text = self.opt_tokenizer.batch_decode(
                outputs[:, prompt_length:], skip_special_tokens=True
            )
            output_text = [text.strip() for text in output_text]
            if task_button == 'Question Answering' or task_button == "Captioning":
                output_text_input = [prompt[0] + ' ' + output_text[0]]
                opt_tokens = self.opt_tokenizer(
                    output_text_input,
                    return_tensors="pt",
                    padding="longest",
                ).to(image.device)

        inputs_embeds = self.opt_model.model.decoder.embed_tokens(opt_tokens.input_ids)
        inputs_embeds = torch.cat([inputs_opt, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)

        with self.maybe_autocast():
            outputs = self.opt_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True
            )
        n_queries = query_tokens.shape[1]
        out_logits = outputs['logits'][:, n_queries:]
        out_hidden = outputs['hidden_states'][-1][:, n_queries:]
        return out_logits, out_hidden, opt_tokens, output_text


def load_model_and_preprocess(name, model_type, is_eval=False, device="cpu"):
    model_cls = registry.get_model_class(name)

    # load model
    model = model_cls.from_pretrained(model_type=model_type)

    if is_eval:
        model.eval()

    # load preprocess
    cfg = OmegaConf.load(model_cls.default_config_path(model_type))
    if cfg is not None:
        preprocess_cfg = cfg.preprocess

        vis_processors, txt_processors = load_preprocess(preprocess_cfg)
    else:
        vis_processors, txt_processors = None, None
        logging.info(
            f"""No default preprocess for model {name} ({model_type}).
                This can happen if the model is not finetuned on downstream datasets,
                or it is not intended for direct use without finetuning.
            """
        )

    if device == "cpu" or device == torch.device("cpu"):
        model = model.float()

    return model.to(device), vis_processors, txt_processors


class BLIP2Decoder(nn.Module):
    def __init__(self, llm_name):
        super(BLIP2Decoder, self).__init__()

        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        if llm_name not in ['pretrain_opt2.7b', 'caption_coco_opt2.7b',
                            'pretrain_opt6.7b', 'caption_coco_opt6.7b']:
            raise ValueError(f"{llm_name} is not support yet")
        model_type = llm_name
        model, vis, _ = load_model_and_preprocess(name="blip2_opt_det",
                                                  model_type=model_type,
                                                  is_eval=True, device=self.device)
        self.model = model
        self.vis_processors = vis
        self.freeze_layers()

    def freeze_layers(self):
        for p in self.model.parameters():
            p.requires_grad = False
