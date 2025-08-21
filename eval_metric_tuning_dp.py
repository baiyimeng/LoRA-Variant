import argparse
import json
import math
import os
import sys

import nltk
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
from tqdm import tqdm

import evaluate
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    set_seed,
    default_data_collator,
)

from peft import PeftModel
from dataset import AmazonReviewDatasetV2
from accelerate import Accelerator
from accelerate.utils import gather_object

from datetime import timedelta
from accelerate.utils import InitProcessGroupKwargs


if "/root/autodl-tmp/nltk_data" not in nltk.data.path:
    nltk.data.path.append("/root/autodl-tmp/nltk_data")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="0,1,2,3,4,5")
    parser.add_argument("--model_dir", type=str, default="/root/autodl-tmp/model/Qwen")
    parser.add_argument("--model_name", type=str, default="Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset_dir", type=str, default="/root/autodl-tmp/dataset")
    parser.add_argument("--dataset_name", type=str, default="Books")
    parser.add_argument(
        "--peft_model_id",
        type=str,
        default="/root/autodl-tmp/output/Qwen2.5-7B-Instruct/lora_1.0/checkpoint-1617",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    return parser.parse_args()


def evaluation(args):
    if "lora" in args.peft_model_id or "prompt" in args.peft_model_id:
        raise ValueError("Please use eval_tuning_vllm.py")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=3600))
    accelerator = Accelerator(kwargs_handlers=[kwargs])

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.model_dir, args.model_name)
    )
    model = AutoModelForCausalLM.from_pretrained(
        os.path.join(args.model_dir, args.model_name),
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model = PeftModel.from_pretrained(model, args.peft_model_id)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    model.print_trainable_parameters()

    generation_config = GenerationConfig(
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        repetition_penalty=1.2,
        max_new_tokens=2048,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
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
        False,
        cache_path=os.path.join(
            args.dataset_dir, f"review_{args.dataset_name}_cache", "test.pkl"
        ),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=default_data_collator,
    )

    model, test_dataloader = accelerator.prepare(model, test_dataloader)

    references = test_dataset.output_str
    predictions = []
    for sample in tqdm(test_dataloader, desc="Generating data"):
        batch = sample
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=batch["input_ids"], generation_config=generation_config
            )
        generated_ids = generated_ids[:, len(batch["input_ids"][0]) :]
        texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        local_batch_output = list(zip(batch["index"].tolist(), texts))
        predictions.extend(local_batch_output)

    accelerator.wait_for_everyone()
    predictions = gather_object(predictions)
    if accelerator.is_main_process:
        unique_predictions = {}
        for idx, text in predictions:
            if idx not in unique_predictions:
                unique_predictions[idx] = text
        predictions = [
            unique_predictions[idx] for idx in sorted(unique_predictions.keys())
        ]

        output_path = args.peft_model_id
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, "pred.json"), "w", encoding="utf-8") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)

        bleu_metric = evaluate.load("./metrics/sacrebleu")
        rouge_metric = evaluate.load("./metrics/rouge")
        meteor_metric = evaluate.load("./metrics/meteor")
        result_bleu = bleu_metric.compute(
            predictions=predictions, references=references
        )
        result_rouge = rouge_metric.compute(
            predictions=predictions, references=references
        )
        result_meteor = meteor_metric.compute(
            predictions=predictions, references=references
        )

        result = {
            "model": args.model_name,
            "peft": args.peft_model_id,
            "rouge-1": result_rouge["rouge1"],
            "rouge-L": result_rouge["rougeL"],
            "meteor": result_meteor["meteor"],
            "bleu": result_bleu["score"],
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
