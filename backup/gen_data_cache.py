import argparse
import os
from datasets import load_from_disk, DatasetDict
from transformers import AutoModel, AutoTokenizer, set_seed

from dataset import AmazonReviewDatasetV2
import pickle


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--model_dir", type=str, default="/root/autodl-tmp/model/Qwen")
    parser.add_argument("--model_name", type=str, default="Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset_dir", type=str, default="/root/autodl-tmp/dataset")
    parser.add_argument("--dataset_name", type=str, default="Books")
    return parser.parse_args()


if __name__ == "__main__":
    set_seed(42)
    args = parse_arguments()
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.model_dir, args.model_name)
    )
    main_dataset = load_from_disk(
        os.path.join(args.dataset_dir, f"review_{args.dataset_name}")
    )
    train_main_dataset = main_dataset["train"]
    val_main_dataset = main_dataset["val"]
    test_main_dataset = main_dataset["test"]

    meta_dataset = load_from_disk(
        os.path.join(args.dataset_dir, f"meta_{args.dataset_name}/full")
    )
    # train_dataset = AmazonReviewDatasetV2(
    #     train_main_dataset, meta_dataset, tokenizer, 4096, True
    # )
    # val_dataset = AmazonReviewDatasetV2(
    #     val_main_dataset, meta_dataset, tokenizer, 4096, True
    # )
    test_dataset = AmazonReviewDatasetV2(
        test_main_dataset, meta_dataset, tokenizer, 4096, False
    )
    # test_dataset_ = AmazonReviewDatasetV2(
    #     test_main_dataset, meta_dataset, tokenizer, 4096, True
    # )

    def save_all(save_path, dataset):
        save_dict = {
            "processed_data": dataset.processed_data,
            "input_str": dataset.input_str,
            "output_str": dataset.output_str,
        }
        with open(save_path, "wb") as f:
            pickle.dump(save_dict, f)

    # save_all("/root/autodl-tmp/dataset/review_Books_cache/train.pkl", train_dataset)
    # save_all("/root/autodl-tmp/dataset/review_Books_cache/val.pkl", val_dataset)
    save_all("/root/autodl-tmp/dataset/review_Books_cache/test.pkl", test_dataset)
    # save_all("/root/autodl-tmp/dataset/review_Books_cache/test_.pkl", test_dataset_)
