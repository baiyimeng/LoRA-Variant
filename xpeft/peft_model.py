from __future__ import annotations

import inspect
import os
import warnings
from contextlib import contextmanager
from typing import Any, Dict, Optional, Union

import torch
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.hooks import (
    AlignDevicesHook,
    add_hook_to_module,
    remove_hook_from_submodules,
)
from accelerate.utils import get_balanced_memory
from safetensors.torch import load_file as safe_load_file
from safetensors.torch import save_file as safe_save_file
from transformers import PreTrainedModel
from transformers.utils import PushToHubMixin

from peft import PeftModel
from peft.config import PeftConfig, PromptLearningConfig
from peft.utils import (
    SAFETENSORS_WEIGHTS_NAME,
    CONFIG_NAME,
    WEIGHTS_NAME,
    _set_adapter,
    _set_trainable,
)

from .utils import PeftType, get_peft_model_state_dict, set_peft_model_state_dict
from .tuners import PLoraModel, MoeLoraModel

PEFT_TYPE_TO_TUNER_MAPPING = {
    PeftType.PLORA: PLoraModel,
    PeftType.MOELORA: MoeLoraModel,
}


class XPeftModel(PeftModel):
    def __init__(
        self,
        model: PreTrainedModel,
        peft_config: PeftConfig,
        adapter_name: str = "default",
    ):
        torch.nn.Module.__init__(self)
        PushToHubMixin.__init__(self)
        self.base_model = model
        self.config = self.base_model.config

        self._is_prompt_learning = False

        self.peft_config = {}
        self.active_adapter = adapter_name
        self.peft_type = peft_config.peft_type

        assert not isinstance(peft_config, PromptLearningConfig)
        self.peft_config[adapter_name] = peft_config
        self.base_model = PEFT_TYPE_TO_TUNER_MAPPING[peft_config.peft_type](
            self.base_model, self.peft_config, adapter_name
        )

        if getattr(model, "is_gradient_checkpointing", True):
            model = self.prepare_model_for_gradient_checkpointing(model)

    def save_pretrained(
        self,
        save_directory: str,
        safe_serialization: bool = False,
        is_main_process: bool = True,
        **kwargs: Any,
    ):
        r"""
        This function saves the adapter model and the adapter configuration files to a directory, so that it can be
        reloaded using the [`LoraModel.from_pretrained`] class method, and also used by the [`LoraModel.push_to_hub`]
        method.

        Args:
            save_directory (`str`):
                Directory where the adapter model and configuration files will be saved (will be created if it does not
                exist).
            kwargs (additional keyword arguments, *optional*):
                Additional keyword arguments passed along to the `push_to_hub` method.
        """
        if os.path.isfile(save_directory):
            raise ValueError(
                f"Provided path ({save_directory}) should be a directory, not a file"
            )
        if is_main_process:
            os.makedirs(save_directory, exist_ok=True)

        for adapter_name, peft_config in self.peft_config.items():
            # save only the trainable weights
            output_state_dict = get_peft_model_state_dict(
                self,
                state_dict=kwargs.get("state_dict", None),
                adapter_name=adapter_name,
            )
            output_dir = (
                os.path.join(save_directory, adapter_name)
                if adapter_name != "default"
                else save_directory
            )
            os.makedirs(output_dir, exist_ok=True)

            if is_main_process and safe_serialization:
                safe_save_file(
                    output_state_dict,
                    os.path.join(output_dir, SAFETENSORS_WEIGHTS_NAME),
                    metadata={"format": "pt"},
                )
            elif is_main_process:
                torch.save(output_state_dict, os.path.join(output_dir, WEIGHTS_NAME))

            # save the config and change the inference mode to `True`
            if peft_config.base_model_name_or_path is None:
                peft_config.base_model_name_or_path = (
                    self.base_model.__dict__.get("name_or_path", None)
                    if isinstance(peft_config, PromptLearningConfig)
                    else self.base_model.model.__dict__.get("name_or_path", None)
                )
            inference_mode = peft_config.inference_mode
            peft_config.inference_mode = True
            peft_config.save_pretrained(output_dir)
            peft_config.inference_mode = inference_mode

    @classmethod
    def from_pretrained(
        cls,
        model: PreTrainedModel,
        model_id: Union[str, os.PathLike],
        adapter_name: str = "default",
        is_trainable: bool = False,
        config: Optional[PeftConfig] = None,
        **kwargs: Any,
    ):
        r"""
        Instantiate a [`LoraModel`] from a pretrained Lora configuration and weights.

        Args:
            model ([`~transformers.PreTrainedModel`]):
                The model to be adapted. The model should be initialized with the
                [`~transformers.PreTrainedModel.from_pretrained`] method from the 🤗 Transformers library.
            model_id (`str` or `os.PathLike`):
                The name of the Lora configuration to use. Can be either:
                    - A string, the `model id` of a Lora configuration hosted inside a model repo on the Hugging Face
                      Hub.
                    - A path to a directory containing a Lora configuration file saved using the `save_pretrained`
                      method (`./my_lora_config_directory/`).
            adapter_name (`str`, *optional*, defaults to `"default"`):
                The name of the adapter to be loaded. This is useful for loading multiple adapters.
            is_trainable (`bool`, *optional*, defaults to `False`):
                Whether the adapter should be trainable or not. If `False`, the adapter will be frozen and use for
                inference
            config ([`~peft.PeftConfig`], *optional*):
                The configuration object to use instead of an automatically loaded configuation. This configuration
                object is mutually exclusive with `model_id` and `kwargs`. This is useful when configuration is already
                loaded before calling `from_pretrained`.
            kwargs: (`optional`):
                Additional keyword arguments passed along to the specific Lora configuration class.
        """
        from .mapping import (
            MODEL_TYPE_TO_PEFT_MODEL_MAPPING,
            PEFT_TYPE_TO_CONFIG_MAPPING,
        )
        import json

        # load the config
        if config is None:
            subfolder = kwargs.get("subfolder", None)
            path = (
                os.path.join(model_id, subfolder) if subfolder is not None else model_id
            )
            config_file = os.path.join(path, CONFIG_NAME)
            with open(config_file) as file:
                json_object = json.load(file)
            config_cls = config = PEFT_TYPE_TO_CONFIG_MAPPING[json_object["peft_type"]]
            config = config_cls(**json_object)
        elif isinstance(config, PeftConfig):
            config.inference_mode = not is_trainable
        else:
            raise ValueError(
                f"The input config must be a PeftConfig, got {config.__class__}"
            )

        if (getattr(model, "hf_device_map", None) is not None) and len(
            set(model.hf_device_map.values()).intersection({"cpu", "disk"})
        ) > 0:
            remove_hook_from_submodules(model)

        if isinstance(config, PromptLearningConfig) and is_trainable:
            raise ValueError(
                "Cannot set a prompt learning adapter to trainable when loading pretrained adapter."
            )
        else:
            config.inference_mode = not is_trainable

        if config.task_type not in MODEL_TYPE_TO_PEFT_MODEL_MAPPING.keys():
            model = cls(model, config, adapter_name)
        else:
            model = MODEL_TYPE_TO_PEFT_MODEL_MAPPING[config.task_type](
                model, config, adapter_name
            )
        model.load_adapter(model_id, adapter_name, is_trainable=is_trainable, **kwargs)

        model = model.to(model.device)

        return model

    def prepare_model_for_gradient_checkpointing(self, model: PreTrainedModel):
        r"""
        Prepares the model for gradient checkpointing if necessary
        """
        self._prepare_model_for_gradient_checkpointing(model)

    def _prepare_model_for_gradient_checkpointing(self, model: PreTrainedModel):
        r"""
        Prepares the model for gradient checkpointing if necessary
        """
        if not (
            getattr(model, "is_loaded_in_8bit", False)
            or getattr(model, "is_loaded_in_4bit", False)
        ):
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(
                    make_inputs_require_grad
                )
        return model

    def print_trainable_parameters(self):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        print(
            f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"
        )

    @contextmanager
    def _enable_peft_forward_hooks(self, *args, **kwargs):
        # If the base model has a method called _enable_peft_forward_hooks, it is invoked as a context. Otherwise, this
        # runs without any changes
        if hasattr(self.base_model, "_enable_peft_forward_hooks"):
            with self.base_model._enable_peft_forward_hooks(*args, **kwargs):
                yield
            return
        else:
            # nothing to enable
            yield
            return

    def forward(self, *args: Any, **kwargs: Any):
        """
        Forward pass of the model.
        """
        with self._enable_peft_forward_hooks(*args, **kwargs):
            kwargs = {
                k: v
                for k, v in kwargs.items()
                if k not in self.special_peft_forward_args
            }
            return self.get_base_model()(*args, **kwargs)

    def generate(self, *args, **kwargs):
        with self._enable_peft_forward_hooks(*args, **kwargs):
            kwargs = {
                k: v
                for k, v in kwargs.items()
                if k not in self.special_peft_forward_args
            }
            return self.get_base_model().generate(*args, **kwargs)

    @contextmanager
    def disable_adapter(self):
        """
        Disables the adapter module.
        """
        assert not isinstance(
            self.peft_config[self.active_adapter], PromptLearningConfig
        )
        self.base_model.disable_adapter_layers()
        yield
        self.base_model.enable_adapter_layers()

    def get_base_model(self):
        """
        Returns the base model.
        """
        assert not isinstance(
            self.peft_config[self.active_adapter], PromptLearningConfig
        )
        return self.base_model.model

    def add_adapter(self, adapter_name: str, peft_config: PeftConfig):
        if peft_config.peft_type != self.peft_type:
            raise ValueError(
                f"Cannot combine adapters with different peft types. "
                f"Found {self.peft_type} and {peft_config.peft_type}."
            )
        self.peft_config[adapter_name] = peft_config
        assert not isinstance(
            self.peft_config[self.active_adapter], PromptLearningConfig
        )
        self.base_model.add_adapter(adapter_name, peft_config)
        if getattr(peft_config, "modules_to_save", None) is not None:
            if self.modules_to_save is None:
                self.modules_to_save = set(peft_config.modules_to_save)
            else:
                self.modules_to_save.update(peft_config.modules_to_save)
            _set_trainable(self, adapter_name)

    def load_adapter(
        self,
        model_id: str,
        adapter_name: str,
        is_trainable: bool = False,
        **kwargs: Any,
    ):
        from .mapping import PEFT_TYPE_TO_CONFIG_MAPPING

        if adapter_name not in self.peft_config:
            # load the config
            peft_config = PEFT_TYPE_TO_CONFIG_MAPPING[
                PeftConfig._get_peft_type(
                    model_id,
                    subfolder=kwargs.get("subfolder", None),
                    revision=kwargs.get("revision", None),
                    cache_dir=kwargs.get("cache_dir", None),
                )
            ].from_pretrained(
                model_id,
                subfolder=kwargs.get("subfolder", None),
                revision=kwargs.get("revision", None),
                cache_dir=kwargs.get("cache_dir", None),
            )
            if isinstance(peft_config, PromptLearningConfig) and is_trainable:
                raise ValueError(
                    "Cannot set a prompt learning adapter to trainable when loading pretrained adapter."
                )
            else:
                peft_config.inference_mode = not is_trainable
            self.add_adapter(adapter_name, peft_config)

        # load weights if any
        path = (
            os.path.join(model_id, kwargs["subfolder"])
            if kwargs.get("subfolder", None) is not None
            else model_id
        )

        if os.path.exists(os.path.join(path, SAFETENSORS_WEIGHTS_NAME)):
            filename = os.path.join(path, SAFETENSORS_WEIGHTS_NAME)
            use_safetensors = True
        elif os.path.exists(os.path.join(path, WEIGHTS_NAME)):
            filename = os.path.join(path, WEIGHTS_NAME)
            use_safetensors = False
        else:
            raise NotImplementedError

        if use_safetensors:
            adapters_weights = safe_load_file(
                filename, device="cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            adapters_weights = torch.load(
                filename,
                map_location=torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                ),
            )

        # load the weights into the model
        load_result = set_peft_model_state_dict(
            self, adapters_weights, adapter_name=adapter_name
        )
        if (
            (getattr(self, "hf_device_map", None) is not None)
            and (
                len(set(self.hf_device_map.values()).intersection({"cpu", "disk"})) > 0
            )
            and len(self.peft_config) == 1
        ):
            device_map = kwargs.get("device_map", "auto")
            max_memory = kwargs.get("max_memory", None)
            offload_dir = kwargs.get("offload_folder", None)
            offload_index = kwargs.get("offload_index", None)

            dispatch_model_kwargs = {}
            # Safety checker for previous `accelerate` versions
            # `offload_index` was introduced in https://github.com/huggingface/accelerate/pull/873/
            if "offload_index" in inspect.signature(dispatch_model).parameters:
                dispatch_model_kwargs["offload_index"] = offload_index

            no_split_module_classes = self._no_split_modules

            if device_map != "sequential":
                max_memory = get_balanced_memory(
                    self,
                    max_memory=max_memory,
                    no_split_module_classes=no_split_module_classes,
                    low_zero=(device_map == "balanced_low_0"),
                )
            if isinstance(device_map, str):
                device_map = infer_auto_device_map(
                    self,
                    max_memory=max_memory,
                    no_split_module_classes=no_split_module_classes,
                )
            dispatch_model(
                self,
                device_map=device_map,
                offload_dir=offload_dir,
                **dispatch_model_kwargs,
            )
            hook = AlignDevicesHook(io_same_device=True)
            assert not isinstance(self.peft_config[adapter_name], PromptLearningConfig)
            add_hook_to_module(self.get_base_model(), hook)

        # Set model in evaluation mode to deactivate Dropout modules by default
        if not is_trainable:
            self.eval()
        return load_result

    def set_adapter(self, adapter_name: str):
        """
        Sets the active adapter.
        """
        if adapter_name not in self.peft_config:
            raise ValueError(f"Adapter {adapter_name} not found.")
        self.active_adapter = adapter_name
        if not isinstance(self.peft_config[adapter_name], PromptLearningConfig):
            self.base_model.set_adapter(adapter_name)
        _set_adapter(self, adapter_name)

    @property
    def active_peft_config(self):
        return self.peft_config[self.active_adapter]

    @property
    def modules_to_save(self):
        return self.peft_config[self.active_adapter].modules_to_save

    @property
    def base_model_torch_dtype(self):
        return getattr(self.base_model, "dtype", None)


class XPeftModelForCausalLM(XPeftModel):
    def __init__(self, model, peft_config: PeftConfig, adapter_name="default"):
        super().__init__(model, peft_config, adapter_name)
        self.base_model_prepare_inputs_for_generation = (
            self.base_model.prepare_inputs_for_generation
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        p_emb=None,
        **kwargs,
    ):
        peft_config = self.active_peft_config
        assert not isinstance(peft_config, PromptLearningConfig)
        if self.base_model.config.model_type == "mpt":
            if inputs_embeds is not None:
                raise AssertionError(
                    "forward in MPTForCausalLM does not support inputs_embeds"
                )
            if hasattr(self.base_model, "lora_fc"):
                p_emb = self.base_model.lora_fc[self.active_adapter](p_emb)

            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                p_emb=p_emb,
                **kwargs,
            )

        if hasattr(self.base_model, "lora_fc"):
            p_emb = self.base_model.lora_fc[self.active_adapter](p_emb)

        return self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            p_emb=p_emb,
            **kwargs,
        )

    def generate(self, **kwargs):
        peft_config = self.active_peft_config
        self.base_model.prepare_inputs_for_generation = (
            self.prepare_inputs_for_generation
        )

        if hasattr(self.base_model, "lora_fc"):
            assert "p_emb" in kwargs and kwargs["p_emb"] is not None
            p_emb = kwargs["p_emb"]
            p_emb = self.base_model.lora_fc[self.active_adapter](p_emb)
            kwargs["p_emb"] = p_emb

        if hasattr(self.base_model, "model"):
            self.base_model.model.generation_config = self.generation_config
        else:
            self.base_model.generation_config = self.generation_config
        try:
            assert not isinstance(peft_config, PromptLearningConfig)
            outputs = self.base_model.generate(**kwargs)
        except:
            self.base_model.prepare_inputs_for_generation = (
                self.base_model_prepare_inputs_for_generation
            )
            raise
        else:
            self.base_model.prepare_inputs_for_generation = (
                self.base_model_prepare_inputs_for_generation
            )
            return outputs

    def prepare_inputs_for_generation(self, *args, **kwargs):
        peft_config = self.active_peft_config
        model_kwargs = self.base_model_prepare_inputs_for_generation(*args, **kwargs)
        assert not isinstance(peft_config, PromptLearningConfig)
        return model_kwargs
