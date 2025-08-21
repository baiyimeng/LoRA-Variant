import os
import argparse


from datasets import load_from_disk, DatasetDict
import numpy as np


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="/root/autodl-tmp/dataset")
    parser.add_argument("--dataset_name", type=str, default="Books")

    return parser.parse_args()


def run(args):
    args = parse_arguments()
    main_dataset = load_from_disk(
        os.path.join(args.dataset_dir, f"review_{args.dataset_name}")
    )
    meta_dataset = load_from_disk(
        os.path.join(args.dataset_dir, f"meta_{args.dataset_name}")
    )

    os.environ["HF_ENDPOINT"] = "https://huggingface.co"
    hf_token = os.getenv("HF_TOKEN")
    main_dataset.push_to_hub(
        "baiyimeng/Amazon2023_Books_5ReviewsProfile_ReviewGeneration_Main",
        private=False,
        token=hf_token,
    )
    meta_dataset.push_to_hub(
        "baiyimeng/Amazon2023_Books_5ReviewsProfile_ReviewGeneration_Meta",
        private=False,
        token=hf_token,
    )


if __name__ == "__main__":
    args = parse_arguments()
    run(args)
