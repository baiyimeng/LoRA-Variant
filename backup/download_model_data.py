from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModel,
    GenerationConfig,
)
import torch
from datasets import load_dataset

# models = [
#     "Qwen/Qwen2.5-0.5B-Instruct",
#     "Qwen/Qwen2.5-1.5B-Instruct",
#     "Qwen/Qwen2.5-3B-Instruct",
#     "Qwen/Qwen2.5-7B-Instruct",
#     "Qwen/Qwen2.5-14B-Instruct",
#     "Qwen/Qwen2.5-32B-Instruct",
# ]

# model_name = models[5]
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.save_pretrained("/root/autodl-tmp/model/" + model_name)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype=torch.bfloat16,
# )
# model.save_pretrained("/root/autodl-tmp/model/" + model_name)

# model_name = "BAAI/bge-base-en-v1.5"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModel.from_pretrained(model_name)
# tokenizer.save_pretrained("/root/autodl-tmp/model/" + model_name)
# model.save_pretrained("/root/autodl-tmp/model/" + model_name)

# dataset_name = "baiyimeng/Amazon2023_Books_5ReviewsProfile_ReviewGeneration_Main"
# main_dataset = load_dataset(dataset_name)
# main_dataset.save_to_disk("/root/autodl-tmp/dataset/review_Books")
# dataset_name = "baiyimeng/Amazon2023_Books_5ReviewsProfile_ReviewGeneration_Meta"
# meta_dataset = load_dataset(dataset_name)
# meta_dataset.save_to_disk("/root/autodl-tmp/dataset/meta_Books")
