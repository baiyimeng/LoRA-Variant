from .peft_types import PeftType


def get_peft_model_state_dict(model, state_dict=None, adapter_name="default"):
    """
    Get the state dict of the Peft model.

    Args:
        model ([`PeftModel`]): The Peft model. When using torch.nn.DistributedDataParallel, DeepSpeed or FSDP,
        the model should be the underlying model/unwrapped model (i.e. model.module).
        state_dict (`dict`, *optional*, defaults to `None`):
            The state dict of the model. If not provided, the state dict of the model
        will be used.
    """
    config = model.peft_config[adapter_name]
    if state_dict is None:
        state_dict = model.state_dict()
    if config.peft_type in (PeftType.MOELORA, PeftType.PLORA):
        # to_return = lora_state_dict(model, bias=model.peft_config.bias)
        # adapted from `https://github.com/microsoft/LoRA/blob/main/loralib/utils.py`
        # to be used directly with the state dict which is necessary when using DeepSpeed or FSDP
        bias = config.bias
        if bias == "none":
            to_return = {k: state_dict[k] for k in state_dict if "lora_" in k}
        elif bias == "all":
            to_return = {
                k: state_dict[k] for k in state_dict if "lora_" in k or "bias" in k
            }
        elif bias == "lora_only":
            to_return = {}
            for k in state_dict:
                if "lora_" in k:
                    to_return[k] = state_dict[k]
                    bias_name = k.split("lora_")[0] + "bias"
                    if bias_name in state_dict:
                        to_return[bias_name] = state_dict[bias_name]
        else:
            raise NotImplementedError
        to_return = {
            k: v
            for k, v in to_return.items()
            if (("lora_" in k and adapter_name in k) or ("bias" in k))
        }
    else:
        raise NotImplementedError
    if model.modules_to_save is not None:
        for key, value in state_dict.items():
            if any(
                f"{module_name}.modules_to_save.{adapter_name}" in key
                for module_name in model.modules_to_save
            ):
                to_return[key.replace("modules_to_save.", "")] = value

    to_return = {k.replace(f".{adapter_name}", ""): v for k, v in to_return.items()}
    return to_return


def set_peft_model_state_dict(model, peft_model_state_dict, adapter_name="default"):
    """
    Set the state dict of the Peft model.

    Args:
        model ([`PeftModel`]): The Peft model.
        peft_model_state_dict (`dict`): The state dict of the Peft model.
    """
    config = model.peft_config[adapter_name]
    state_dict = {}
    if model.modules_to_save is not None:
        for key, value in peft_model_state_dict.items():
            if any(module_name in key for module_name in model.modules_to_save):
                for module_name in model.modules_to_save:
                    if module_name in key:
                        key = key.replace(
                            module_name, f"{module_name}.modules_to_save.{adapter_name}"
                        )
                        break
            state_dict[key] = value
    else:
        state_dict = peft_model_state_dict

    if config.peft_type in (PeftType.MOELORA, PeftType.PLORA):
        peft_model_state_dict = {}
        for k, v in state_dict.items():
            if "lora_A" in k:
                k = k.replace("lora_A", f"lora_A.{adapter_name}")
                peft_model_state_dict[k] = v
            elif "lora_B" in k:
                k = k.replace("lora_B", f"lora_B.{adapter_name}")
                peft_model_state_dict[k] = v
            elif "lora_gating" in k:
                k = k.replace("lora_gating", f"lora_gating.{adapter_name}")
                peft_model_state_dict[k] = v
            elif "lora_P" in k:
                k = k.replace("lora_P", f"lora_P.{adapter_name}")
                peft_model_state_dict[k] = v
            elif "lora_fc" in k:
                k = k.replace("lora_fc", f"lora_fc.{adapter_name}")
                peft_model_state_dict[k] = v
            else:
                peft_model_state_dict[k] = v
    else:
        raise NotImplementedError
    load_result = model.load_state_dict(peft_model_state_dict, strict=False)
    return load_result
