import torch.nn as nn
from transformers import AutoModelForMaskedLM, AutoTokenizer


class LLMDecoder(nn.Module):
    def __init__(self, llm_name):
        super(LLMDecoder, self).__init__()

        prefix = llm_name.split('-')[0]
        if prefix in ['bert', 'roberta']:
            tokenizer = AutoTokenizer.from_pretrained(llm_name)
            llm_model = AutoModelForMaskedLM.from_pretrained(llm_name)
        else:
            raise NotImplementedError(f"{llm_name} has not been implemented.")

        self.model = llm_model
        self.tokenizer = tokenizer
        self.freeze_layers()

    def forward(self, inputs_embeds, attention_mask=None, output_hidden_states=True):
        text_features = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                   output_hidden_states=output_hidden_states)
        if not output_hidden_states:
            return text_features.logits
        else:
            return text_features.logits, text_features.hidden_states[-1]

    def freeze_layers(self):
        for p in self.model.parameters():
            p.requires_grad = False
