import torch.nn as nn
import torch
from transformers import AutoModel

import os
import sys
from importlib import import_module
config = sys.argv[1]
config_dir = os.path.dirname(config)
config_bname = os.path.splitext(os.path.basename(config))[0]
sys.path.append(config_dir)
config = import_module(config_bname)

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = PretrainedLanguageModel()
        self.decoder = nn.Sequential(
            nn.Dropout(p=config.dropout_rate),
            nn.Linear(768, config.output_class_num)
        )

    def forward(self, ids, mask):
        h = self.text_encoder(ids, mask)
        output = self.decoder(h)
        return output

class PretrainedLanguageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = AutoModel.from_pretrained(config.language_model)
        self.reinit_n_layers = config.reinit_n_layers
        if self.reinit_n_layers > 0:
            self._do_reinit()

    def _do_reinit(self):
        # Re-init last n layers.
        for layer in self.language_model.encoder.layer[-1*self.reinit_n_layers:]:
            for module in layer.modules():
                if isinstance(module, nn.Linear):
                    module.weight.data.normal_(mean=0.0, std=self.language_model.config.initializer_range)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.Embedding):
                    module.weight.data.normal_(mean=0.0, std=self.language_model.config.initializer_range)
                    if module.padding_idx is not None:
                        module.weight.data[module.padding_idx].zero_()
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)

    def forward(self, ids, mask):
        output = self.language_model(ids, attention_mask=mask)
        return output[0][:,0,:]
