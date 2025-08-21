from .tuners import PLoraConfig, MoeLoraConfig
from .peft_model import XPeftModelForCausalLM
from peft.config import PromptLearningConfig

PEFT_TYPE_TO_CONFIG_MAPPING = {
    "PLORA": PLoraConfig,
    "MOELORA": MoeLoraConfig,
}
MODEL_TYPE_TO_PEFT_MODEL_MAPPING = {
    "CAUSAL_LM": XPeftModelForCausalLM,
}


def get_peft_model(model, peft_config):
    model_config = (
        model.config.to_dict() if hasattr(model.config, "to_dict") else model.config
    )
    peft_config.base_model_name_or_path = model.__dict__.get("name_or_path", None)
    assert peft_config.task_type in MODEL_TYPE_TO_PEFT_MODEL_MAPPING.keys()
    assert not isinstance(peft_config, PromptLearningConfig)
    return MODEL_TYPE_TO_PEFT_MODEL_MAPPING[peft_config.task_type](model, peft_config)
