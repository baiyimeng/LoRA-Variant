from transformers import AutoModel, AutoTokenizer, set_seed
import argparse
import os
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataset import AmazonReviewDatasetV1
from datasets import load_from_disk, DatasetDict

from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--model_dir", type=str, default="/root/autodl-tmp/model")
    parser.add_argument("--model_name", type=str, default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--dataset_dir", type=str, default="/root/autodl-tmp/dataset")
    parser.add_argument("--dataset_name", type=str, default="Books")
    return parser.parse_args()


def run(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.model_dir, args.model_name)
    )
    model = AutoModel.from_pretrained(
        os.path.join(args.model_dir, args.model_name),
        torch_dtype=torch.float32,
        device_map="auto",
    )
    model.eval()

    main_dataset = load_from_disk(
        os.path.join(args.dataset_dir, f"review_{args.dataset_name}")
    )
    train_main_dataset = main_dataset["train"]
    val_main_dataset = main_dataset["val"]
    test_main_dataset = main_dataset["test"]

    meta_dataset = load_from_disk(
        os.path.join(args.dataset_dir, f"meta_{args.dataset_name}/full")
    )
    train_dataset = AmazonReviewDatasetV1(train_main_dataset, meta_dataset)
    val_dataset = AmazonReviewDatasetV1(val_main_dataset, meta_dataset)
    test_dataset = AmazonReviewDatasetV1(test_main_dataset, meta_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=512, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    def compute_emb(dataloader):
        p_emb_avg = []
        with torch.no_grad():
            for batch in tqdm(dataloader):
                his, tgt = batch["history_str"], batch["target_str"]

                his_emb = []
                for i, h in enumerate(his):
                    inputs = tokenizer(
                        h, padding=True, truncation=True, return_tensors="pt"
                    )
                    for key in inputs:
                        inputs[key] = inputs[key].to(model.device)
                    outputs = model(**inputs)
                    his_emb.append(F.normalize(outputs[0][:, 0], dim=-1))
                his_emb = torch.stack(his_emb, dim=1)

                p_emb_avg.append(his_emb.mean(dim=1))

        p_emb_avg = torch.concat(p_emb_avg, dim=0)

        return p_emb_avg.cpu().numpy().tolist()

    train_p_emb_avg = compute_emb(train_dataloader)
    val_p_emb_avg = compute_emb(val_dataloader)
    test_p_emb_avg = compute_emb(test_dataloader)

    train_main_dataset = train_main_dataset.add_column("p_emb_avg", train_p_emb_avg)

    val_main_dataset = val_main_dataset.add_column("p_emb_avg", val_p_emb_avg)

    test_main_dataset = test_main_dataset.add_column("p_emb_avg", test_p_emb_avg)

    main_dataset_update = DatasetDict(
        {
            "train": train_main_dataset,
            "val": val_main_dataset,
            "test": test_main_dataset,
        }
    )

    main_dataset_update.save_to_disk("/root/autodl-tmp/dataset/review_Books")


if __name__ == "__main__":
    set_seed(42)
    args = parse_arguments()
    run(args)
