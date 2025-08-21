import math
import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.utils import transpose
from ...context import current_p_emb


class PLoraLayer:
    def __init__(self, in_features: int, out_features: int):
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_P = nn.ModuleDict({})
        # Mark the weight as unmerged
        self.merged = False
        self.disable_adapters = False
        self.in_features = in_features
        self.out_features = out_features

    def update_layer(
        self,
        adapter_name,
        r,
        lora_alpha,
        lora_dropout,
        init_lora_weights,
    ):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))

        # Actual trainable parameters
        if r > 0:
            self.lora_P.update(
                nn.ModuleDict(
                    {adapter_name: nn.Linear(r, self.out_features, bias=False)}
                )
            )
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)
        self.to(self.weight.device)

    def reset_lora_parameters(self, adapter_name, init_lora_weights):
        if init_lora_weights is False:
            return

        if adapter_name in self.lora_P.keys():
            nn.init.zeros_(self.lora_P[adapter_name].weight)


class Linear(nn.Linear, PLoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)
        nn.Linear.__init__(self, in_features, out_features)
        PLoraLayer.__init__(self, in_features, out_features)
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False
        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T
        nn.Linear.reset_parameters(self)
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.active_adapter = adapter_name

    def merge(self):
        raise NotImplementedError

    def unmerge(self):
        raise NotImplementedError

    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype

        p_emb = current_p_emb.get()
        p_emb = p_emb.reshape(p_emb.shape[0], -1, self.r[self.active_adapter])

        if self.active_adapter not in self.lora_P.keys():
            return F.linear(
                x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias
            )
        if self.disable_adapters:
            if self.r[self.active_adapter] > 0 and self.merged:
                self.unmerge()
            result = F.linear(
                x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias
            )
        elif self.r[self.active_adapter] > 0 and not self.merged:
            result = F.linear(
                x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias
            )
            x = x.to(self.lora_P[self.active_adapter].weight.dtype)

            Ax = torch.bmm(self.lora_dropout[self.active_adapter](x), p_emb)
            BAx = self.lora_P[self.active_adapter](Ax)

            result += BAx * self.scaling[self.active_adapter]

        else:
            result = F.linear(
                x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias
            )
        result = result.to(previous_dtype)
        return result
