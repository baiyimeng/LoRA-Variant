import math
import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.utils import transpose
from ...context import current_p_emb


class MoeLoraLayer:
    def __init__(self, in_features: int, out_features: int, user_features: int):
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})
        # Mark the weight as unmerged
        self.merged = False
        self.disable_adapters = False
        self.in_features = in_features
        self.out_features = out_features

        self.user_features = user_features
        self.num_experts = {}
        self.lora_gating = nn.ModuleDict({})

    def update_layer(
        self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, num_experts
    ):
        self.r[adapter_name] = r
        self.num_experts[adapter_name] = num_experts

        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))

        self.lora_gating.update(
            nn.ModuleDict(
                {adapter_name: nn.Linear(self.user_features, num_experts, bias=False)}
            )
        )

        # Actual trainable parameters
        if r > 0:
            self.lora_A.update(
                nn.ModuleDict(
                    {
                        adapter_name: nn.Linear(
                            self.in_features, r * num_experts, bias=False
                        )
                    }
                )
            )
            self.lora_B.update(
                nn.ModuleDict(
                    {
                        adapter_name: nn.Linear(
                            r * num_experts, self.out_features, bias=False
                        )
                    }
                )
            )
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)
        self.to(self.weight.device)

    def reset_lora_parameters(self, adapter_name, init_lora_weights):
        if init_lora_weights is False:
            return

        if adapter_name in self.lora_A.keys():
            if init_lora_weights is True:
                # initialize A the same way as the default for nn.Linear and B to zero
                # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
                nn.init.kaiming_uniform_(
                    self.lora_A[adapter_name].weight, a=math.sqrt(5)
                )
            elif init_lora_weights.lower() == "gaussian":
                nn.init.normal_(
                    self.lora_A[adapter_name].weight, std=1 / self.r[adapter_name]
                )
            else:
                raise ValueError(f"Unknown initialization {init_lora_weights=}")
            nn.init.zeros_(self.lora_B[adapter_name].weight)


class Linear(nn.Linear, MoeLoraLayer):
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
        user_features: int = 768,
        num_experts: int = 1,
        **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        MoeLoraLayer.__init__(self, in_features, out_features, user_features)
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False
        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T
        nn.Linear.reset_parameters(self)
        self.update_layer(
            adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, num_experts
        )
        self.active_adapter = adapter_name

    def merge(self):
        raise NotImplementedError

    def unmerge(self):
        raise NotImplementedError

    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype
        batch_size, seq_len, _ = x.size()

        p_emb = current_p_emb.get()

        if self.active_adapter not in self.lora_A.keys():
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
            x = x.to(self.lora_A[self.active_adapter].weight.dtype)

            Ax = self.lora_A[self.active_adapter](
                self.lora_dropout[self.active_adapter](x)
            )
            Ax = Ax.reshape(
                batch_size,
                seq_len,
                self.num_experts[self.active_adapter],
                self.r[self.active_adapter],
            )
            B = self.lora_B[self.active_adapter].weight.t()
            B = B.reshape(
                self.num_experts[self.active_adapter],
                self.r[self.active_adapter],
                -1,
            )
            BAx = torch.einsum("e r o, b s e r -> b s e o", B, Ax)
            G = self.lora_gating[self.active_adapter](p_emb).unsqueeze(1).unsqueeze(-1)

            result += (BAx * G).sum(dim=2) * self.scaling[self.active_adapter]

        else:
            result = F.linear(
                x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias
            )
        result = result.to(previous_dtype)
        return result
