from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal, Optional, Union

from peft.config import PeftConfig
from xpeft.utils.peft_types import PeftType


@dataclass
class PLoraConfig(PeftConfig):

    r: int = field(default=8, metadata={"help": "Lora attention dimension"})

    user_features: int = field(
        default=768, metadata={"help": "Dimension of user embeddings"}
    )

    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace with LoRA. "
                "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'. "
                "This can also be a wildcard 'all-linear' which matches all linear/Conv1D "
                "(if the model is a PreTrainedModel, the output layer excluded). "
                "If not specified, modules will be chosen according to the model architecture, If the architecture is "
                "not known, an error will be raised -- in this case, you should specify the target modules manually. "
                "To avoid targeting any modules (because you want to apply `target_parameters`), set "
                "`target_modules=[]`."
            ),
        },
    )
    lora_alpha: int = field(default=8, metadata={"help": "Lora alpha"})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout"})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={
            "help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"
        },
    )
    bias: Literal["none", "all", "lora_only"] = field(
        default="none",
        metadata={"help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'"},
    )
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )
    init_lora_weights: (
        bool
        | Literal[
            "gaussian",
            "eva",
            "olora",
            "pissa",
            "pissa_niter_[number of iters]",
            "corda",
            "loftq",
            "orthogonal",
        ]
    ) = field(
        default=True,
        metadata={
            "help": (
                "How to initialize the weights of the LoRA layers. "
                "Passing True (default) results in the default initialization from the reference implementation from "
                "Microsoft, with the LoRA B weight being set to 0. This means that without further training, the LoRA "
                "adapter will be a no-op. "
                "Setting the initialization to False leads to random initialization of LoRA A and B, meaning that LoRA "
                "is not a no-op before training; this setting is intended for debugging purposes. "
                "Passing `'gaussian'` results in Gaussian initialization scaled by the LoRA rank for linear and layers. "
                "Passing `'eva'` results in a data-driven initialization of Explained Variance Adaptation. "
                "Passing `'olora'` results in OLoRA initialization. "
                "Passing `'pissa'` results in PiSSA initialization. "
                "Passing `'pissa_niter_[number of iters]'` initiates Fast-SVD-based PiSSA initialization, where "
                "[number of iters] indicates the number of subspace iterations to perform fsvd, and must be a "
                "nonnegative integer. "
                "Passing `'corda'` results in CorDA initialization. "
                "Pass `'loftq'` to use LoftQ initialization. "
                "Pass `'orthogonal'` for orthogonal initialization of LoRA A and B."
            ),
        },
    )
    layers_to_transform: Optional[Union[list[int], int]] = field(
        default=None,
        metadata={
            "help": "The layer indexes to transform, is this argument is specified, PEFT will transform only the layers indexes that are specified inside this list. If a single integer is passed, PEFT will transform only the layer at this index. "
            "This only works when target_modules is a list of str."
        },
    )
    layers_pattern: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": "The layer pattern name, used only if `layers_to_transform` is different to None and if the layer pattern is not in the common layers pattern."
            "This only works when target_modules is a list of str. This should target the `nn.ModuleList` of the "
            "model, which is often called `'layers'` or `'h'`."
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.PLORA
