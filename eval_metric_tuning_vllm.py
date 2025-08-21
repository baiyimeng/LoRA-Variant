import os
import sys
import json
import math
import argparse

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    set_seed,
    default_data_collator,
)

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.prompt_adapter.request import PromptAdapterRequest

from datasets import load_from_disk
import evaluate
import nltk
from tqdm import tqdm

from dataset import AmazonReviewDatasetV2


if "/root/autodl-tmp/nltk_data" not in nltk.data.path:
    nltk.data.path.append("/root/autodl-tmp/nltk_data")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="0,1,2,3,4,5")
    parser.add_argument("--model_dir", type=str, default="/root/autodl-tmp/model/Qwen")
    parser.add_argument("--model_name", type=str, default="Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset_dir", type=str, default="/root/autodl-tmp/dataset")
    parser.add_argument("--dataset_name", type=str, default="Books")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--peft_model_id",
        type=str,
        default="/root/autodl-tmp/output/Qwen2.5-7B-Instruct/lora_1.0/checkpoint-1617",
    )
    return parser.parse_args()


def evaluation(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.model_dir, args.model_name)
    )
    # model = AutoModelForCausalLM.from_pretrained(
    #     os.path.join(args.model_dir, args.model_name),
    #     torch_dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2",
    # )
    model = LLM(
        model=os.path.join(args.model_dir, args.model_name),
        tokenizer=os.path.join(args.model_dir, args.model_name),
        tensor_parallel_size=len(args.gpu.split(",")),
        gpu_memory_utilization=0.9,
        enable_lora=True,
    )
    sampling_params = SamplingParams(
        repetition_penalty=1.2,
        skip_special_tokens=True,
        temperature=0.8,
        top_p=0.95,
        max_tokens=2048,
    )
    if "lora" in args.peft_model_id:
        lora_request = LoRARequest(
            lora_name="default", lora_int_id=1, lora_path=args.peft_model_id
        )
        prompt_adapter_request = None
    elif "prompt" in args.peft_model_id:
        lora_request = None
        prompt_adapter_request = PromptAdapterRequest(
            prompt_adapter_name="default",
            prompt_adapter_id=1,
            prompt_adapter_local_path=args.peft_model_id,
            prompt_adapter_num_virtual_tokens=32,
        )
    else:
        raise NotImplementedError

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

    references = test_dataset.output_str
    prompts = test_dataset.input_str

    predictions = model.generate(
        prompts,
        sampling_params,
        lora_request=lora_request,
        prompt_adapter_request=prompt_adapter_request,
    )
    predictions = [p.outputs[0].text for p in predictions]

    output_path = args.peft_model_id
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
