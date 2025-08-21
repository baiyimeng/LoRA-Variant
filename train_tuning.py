import argparse
import os
import sys

import torch
import torch.distributed as dist
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from dataset import AmazonReviewDatasetV2
from peft import (
    IA3Config,
    LoraConfig,
    PrefixTuningConfig,
    PromptTuningConfig,
    TaskType,
    get_peft_model,
)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="/root/autodl-tmp/model/Qwen")
    parser.add_argument("--model_name", type=str, default="Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset_dir", type=str, default="/root/autodl-tmp/dataset")
    parser.add_argument("--dataset_name", type=str, default="Books")
    parser.add_argument("--sample_ratio", type=float, default=1.0)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--peft_type",
        type=str,
        choices=["lora", "prefix", "prompt", "ia3"],
        default="lora",
    )
    return parser.parse_args()


def train(args):
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.model_dir, args.model_name)
    )
    model = AutoModelForCausalLM.from_pretrained(
        os.path.join(args.model_dir, args.model_name),
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    if args.peft_type == "lora":
        peft_config = LoraConfig(
            r=8,
            target_modules=["q_proj", "v_proj"],
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
    elif args.peft_type == "ia3":
        peft_config = IA3Config(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "v_proj"],
        )
    elif args.peft_type == "prefix":
        peft_config = PrefixTuningConfig(
            num_virtual_tokens=32,
            encoder_hidden_size=256,
            prefix_projection=True,
            task_type=TaskType.CAUSAL_LM,
        )
    elif args.peft_type == "prompt":
        peft_config = PromptTuningConfig(
            num_virtual_tokens=32,
            task_type=TaskType.CAUSAL_LM,
            tokenizer_name_or_path=os.path.join(args.model_dir, args.model_name),
        )
    else:
        raise ValueError(f"Unsupported PEFT type: {args.peft_type}")
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    main_dataset = load_from_disk(
        os.path.join(args.dataset_dir, f"review_{args.dataset_name}")
    )
    train_main_dataset = main_dataset["train"]
    val_main_dataset = main_dataset["val"]
    meta_dataset = load_from_disk(
        os.path.join(args.dataset_dir, f"meta_{args.dataset_name}/full")
    )
    train_dataset = AmazonReviewDatasetV2(
        train_main_dataset,
        meta_dataset,
        tokenizer,
        4096,
        True,
        args.sample_ratio,
        cache_path=os.path.join(
            args.dataset_dir, f"review_{args.dataset_name}_cache", "train.pkl"
        ),
    )
    val_dataset = AmazonReviewDatasetV2(
        val_main_dataset,
        meta_dataset,
        tokenizer,
        4096,
        True,
        cache_path=os.path.join(
            args.dataset_dir, f"review_{args.dataset_name}_cache", "val.pkl"
        ),
    )

    output_dir = os.path.join(
        "/root/autodl-tmp/output",
        args.model_name,
        args.peft_type + f"_{args.sample_ratio}",
    )
    os.makedirs(output_dir, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        eval_strategy="epoch",
        logging_steps=100,
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        num_train_epochs=1,
        optim="adamw_torch",
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        weight_decay=1e-4,
        warmup_ratio=0.01,
        deepspeed="./deepspeed/ds_z1_config.json",
        report_to="none",
        bf16=True,
        label_names=["labels"],
        remove_unused_columns=True,
    )

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()

    if dist.is_initialized():
        dist.destroy_process_group()
    sys.exit(0)


if __name__ == "__main__":
    set_seed(42)
    args = parse_arguments()
    print(args)
    train(args)
