import warnings
from enum import Enum
from dataclasses import asdict

import torch.nn as nn

from peft.tuners.tuners_utils import check_target_module_exists
from peft.utils import (
    ModulesToSaveWrapper,
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    _freeze_adapter,
    _get_submodules,
)

from .layer import PLoraLayer, Linear
from ...context import current_p_emb


class PLoraModel(nn.Module):
    def __init__(self, model, peft_config, adapter_name):
        super().__init__()
        self.model = model
        self.peft_config = peft_config

        self.lora_fc = nn.ModuleDict({})
        self.lora_fc[adapter_name] = nn.Linear(
            peft_config[adapter_name].user_features,
            peft_config[adapter_name].r * model.config.hidden_size,
            bias=False,
        )

        self.add_adapter(adapter_name, self.peft_config[adapter_name])

    def forward(self, **kwargs):
        assert "p_emb" in kwargs and kwargs["p_emb"] is not None
        current_p_emb.set(kwargs["p_emb"])
        del kwargs["p_emb"]
        return self.model(**kwargs)

    def generate(self, **kwargs):
        assert "p_emb" in kwargs and kwargs["p_emb"] is not None
        current_p_emb.set(kwargs["p_emb"])
        del kwargs["p_emb"]
        return self.model.generate(**kwargs)

    def add_adapter(self, adapter_name, config=None):
        if config is not None:
            config = self._prepare_lora_config(config, self.model.config.to_dict())
            self.peft_config[adapter_name] = config
        self._find_and_replace(adapter_name)
        if len(self.peft_config) > 1 and self.peft_config[adapter_name].bias != "none":
            raise ValueError(
                "MoeLoraModel supports only 1 adapter with bias. When using multiple adapters, set bias to 'none' for all adapters."
            )
        mark_only_lora_as_trainable(self.model, self.peft_config[adapter_name].bias)
        if self.peft_config[adapter_name].inference_mode:
            _freeze_adapter(self.model, adapter_name)

    @staticmethod
    def _prepare_lora_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if (
                model_config["model_type"]
                not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
            ):
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = (
                TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[
                    model_config["model_type"]
                ]
            )
        return peft_config

    def _find_and_replace(self, adapter_name):
        lora_config = self.peft_config[adapter_name]
        self._check_quantization_dependency()
        is_target_modules_in_base_model = False
        key_list = [key for key, _ in self.model.named_modules()]

        for key in key_list:
            if not self._check_target_module_exists(lora_config, key):
                continue
            is_target_modules_in_base_model = True
            parent, target, target_name = _get_submodules(self.model, key)

            if isinstance(target, PLoraLayer):
                target.update_layer(
                    adapter_name,
                    lora_config.r,
                    lora_config.lora_alpha,
                    lora_config.lora_dropout,
                    lora_config.init_lora_weights,
                )
            else:
                new_module = self._create_new_module(lora_config, adapter_name, target)
                self._replace_module(parent, target_name, new_module, target)

        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {lora_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _check_quantization_dependency(self):
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if loaded_in_4bit or loaded_in_8bit:
            raise ImportError("8-bit or 4-bit quantization is not supported")

    def _check_target_module_exists(self, lora_config, key):
        return check_target_module_exists(lora_config, key)

    def _create_new_module(self, lora_config, adapter_name, target, **kwargs):
        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
        }

        if isinstance(target, nn.Linear):
            in_features, out_features = target.in_features, target.out_features
            if kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                    "Setting fan_in_fan_out to False."
                )
                kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
            new_module = Linear(
                adapter_name, in_features, out_features, bias=bias, **kwargs
            )
        else:
            raise ValueError(
                f"Target module {target} is not supported. "
                f"Currently, only `torch.nn.Linear` is supported."
            )

        return new_module

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight
        if hasattr(old_module, "bias"):
            if old_module.bias is not None:
                new_module.bias = old_module.bias

        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "lora_" in name:
                module.to(old_module.weight.device)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    def get_peft_config_as_dict(self, inference: bool = False):
        config_dict = {}
        for key, value in self.peft_config.items():
            config = {
                k: v.value if isinstance(v, Enum) else v
                for k, v in asdict(value).items()
            }
            if inference:
                config["inference_mode"] = True
        config_dict[key] = config
        return config

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, PLoraLayer):
                module.disable_adapters = False if enabled else True

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)

    def set_adapter(self, adapter_name):
        for module in self.model.modules():
            if isinstance(module, PLoraLayer):
                if module.merged:
                    warnings.warn(
                        "Adapter cannot be set when the model is merged. Unmerging the model first."
                    )
                    module.unmerge()
                module.active_adapter = adapter_name

    def merge_adapter(self):
        raise NotImplementedError

    def unmerge_adapter(self):
        raise NotImplementedError

    def merge_and_unload(self):
        raise NotImplementedError


def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, PLoraLayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError
