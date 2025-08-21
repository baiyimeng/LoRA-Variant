import os
import sys
import json
import math
import argparse

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DataParallel
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    set_seed,
    default_data_collator,
)

from datasets import load_from_disk
import evaluate
import nltk

from dataset import AmazonReviewDatasetV2
from accelerate import Accelerator
from accelerate.utils import gather_object


if "/root/autodl-tmp/nltk_data" not in nltk.data.path:
    nltk.data.path.append("/root/autodl-tmp/nltk_data")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="0,1,2,3")
    parser.add_argument("--model_dir", type=str, default="/root/autodl-tmp/model/Qwen")
    parser.add_argument("--model_name", type=str, default="Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset_dir", type=str, default="/root/autodl-tmp/dataset")
    parser.add_argument("--dataset_name", type=str, default="Books")
    parser.add_argument("--batch_size", type=int, default=1)
    return parser.parse_args()


def evaluation(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    accelerator = Accelerator()

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.model_dir, args.model_name)
    )
    model = AutoModelForCausalLM.from_pretrained(
        os.path.join(args.model_dir, args.model_name),
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    main_dataset = load_from_disk(
        os.path.join(args.dataset_dir, f"review_{args.dataset_name}/test")
    )
    meta_dataset = load_from_disk(
        os.path.join(args.dataset_dir, f"meta_{args.dataset_name}/full")
    )
    test_dataset = AmazonReviewDatasetV2(
        main_dataset,
        meta_dataset,
        tokenizer,
        4096,
        True,
        cache_path=os.path.join(
            args.dataset_dir, f"review_{args.dataset_name}_cache", "test_.pkl"
        ),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=default_data_collator,
    )

    model, test_dataloader = accelerator.prepare(model, test_dataloader)

    total_loss = []
    total_tokens = []

    print("Calculating PPL...")
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            input_ids = batch["input_ids"].long()
            attention_mask = batch["attention_mask"].long()
            labels = batch["labels"].long()
            outputs = model(
                input_ids=input_ids, labels=labels, attention_mask=attention_mask
            )

            loss = outputs.loss
            num_tokens = (labels != -100).sum().item()

            total_loss.append(loss.item() * num_tokens)
            total_tokens.append(num_tokens)

    accelerator.wait_for_everyone()
    total_loss = gather_object(total_loss)
    total_tokens = gather_object(total_tokens)

    if accelerator.is_main_process:
        avg_loss = sum(total_loss) / sum(total_tokens)
        ppl = math.exp(avg_loss)

        result = {
            "model": args.model_name,
            "loss": avg_loss,
            "ppl": ppl,
        }
        print(result)

    if dist.is_initialized():
        dist.destroy_process_group()
    sys.exit(0)


if __name__ == "__main__":
    set_seed(42)
    args = parse_arguments()
    print(args)
    evaluation(args)
