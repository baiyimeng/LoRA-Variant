import os
import sys
import argparse
import torch
import math
import json
import evaluate

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    set_seed,
    default_data_collator,
)
from datasets import load_from_disk
from torch.utils.data import DataLoader, DistributedSampler
from dataset import AmazonReviewDatasetV2
from tqdm import tqdm
import torch.distributed as dist
import nltk

if "/root/autodl-tmp/nltk_data" not in nltk.data.path:
    nltk.data.path.append("/root/autodl-tmp/nltk_data")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="/root/autodl-tmp/model/Qwen")
    parser.add_argument("--model_name", type=str, default="Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset_dir", type=str, default="/root/autodl-tmp/dataset")
    parser.add_argument("--dataset_name", type=str, default="Books")
    parser.add_argument("--batch_size", type=int, default=64)
    return parser.parse_args()


def evaluation(args):
    raise ValueError("Please use eval_base_vllm.py")
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.model_dir, args.model_name)
    )
    model = AutoModelForCausalLM.from_pretrained(
        os.path.join(args.model_dir, args.model_name),
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cuda",
    )
    model.eval()
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

    references = test_dataset.output_str
    predictions = []
    for sample in tqdm(test_dataloader, desc="Generating data"):
        batch = {k: v.to(model.device) for k, v in sample.items()}
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=batch["input_ids"], generation_config=generation_config
            )
        generated_ids = generated_ids[:, len(batch["input_ids"][0]) :]
        texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        predictions.extend(texts)

    output_path = f"/root/autodl-tmp/output/{args.model_name}/base"
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "pred.json"), "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    bleu_metric = evaluate.load("./metrics/sacrebleu")
    rouge_metric = evaluate.load("./metrics/rouge")
    meteor_metric = evaluate.load("./metrics/meteor")
    result_bleu = bleu_metric.compute(predictions=predictions, references=references)
    result_rouge = rouge_metric.compute(predictions=predictions, references=references)
    result_meteor = meteor_metric.compute(
        predictions=predictions, references=references
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

    total_loss = 0.0
    total_tokens = 0
    device = next(model.parameters()).device

    print("Calculating PPL...")
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            input_ids = batch["input_ids"].long()
            attention_mask = batch["attention_mask"].long()
            labels = batch["labels"].long()
            outputs = model(
                input_ids=input_ids, labels=labels, attention_mask=attention_mask
            )

            loss = outputs.loss
            num_tokens = (labels != -100).sum().item()

            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)

    result = {
        "model": args.model_name,
        "rouge-1": result_rouge["rouge1"],
        "rouge-L": result_rouge["rougeL"],
        "meteor": result_meteor["meteor"],
        "bleu": result_bleu["score"],
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
