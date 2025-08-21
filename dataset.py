import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
import random
import pickle


class QwenPromptTemplate:
    def __init__(self, system_prompt=None):
        self.system_prompt = system_prompt

    def build_prompt(self, user_message):
        if self.system_prompt is not None:
            SYS = f"<|im_start|>system\n{self.system_prompt}<|im_end|>"
        else:
            SYS = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
        CONVO = f"<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
        return SYS + CONVO


class AmazonReviewDatasetV0(Dataset):
    def __init__(
        self,
        main_dataset,
        meta_dataset,
        tokenizer,
        max_length=4096,
        is_training=True,
    ):
        self.main_dataset = main_dataset
        self.meta_dataset = meta_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_training = is_training

        input_max_length = max_length // 2
        output_max_length = max_length // 2

        system_prompt = (
            f"Given the title and description of an item, along with the current user's past reviews, and the output review rating and review title, "
            f"generate a personalized item review for the current user.\n"
            f"The review should be formatted as follows:\n"
            f"[Review]: <review>"
        )
        self.pt = QwenPromptTemplate(system_prompt)

        self.processed_data = []

        meta_map = {
            asin: (title, desc)
            for asin, title, desc in zip(
                meta_dataset["asin"], meta_dataset["title"], meta_dataset["description"]
            )
        }

        self.input_str = []
        self.output_str = []

        for idx in tqdm(range(len(main_dataset)), desc=f"Pre-Processing data"):
            history = main_dataset[idx]["history"]
            for p in history:
                p_item_title, p_item_desc = meta_map[p["asin"]]
                p["item_title"] = p_item_title
                p["item_desc"] = p_item_desc
            history = sorted(history, key=lambda x: x["timestamp"], reverse=True)

            target = self.main_dataset[idx]["target"]
            target_item_title, target_item_desc = meta_map[target["asin"]]

            tmp_inp_str = (
                f"[Item Title]: {target_item_title}\n"
                f"[Item Description]: {target_item_desc}\n"
                f"[Output Review Rating]: {target['rating']}\n"
                f"[Output Review Title]: {target['title']}\n"
            )
            tmp_inp_str = self.pt.build_prompt(tmp_inp_str)
            tmp_ids = self.tokenizer(tmp_inp_str, add_special_tokens=False)["input_ids"]
            tmp_len = len(tmp_ids)
            avail_len = input_max_length - tmp_len

            past_reviews = ""
            for tmp_prof_len in range(len(history), 0, -1):
                past_reviews = "".join(
                    [
                        f"[Review {i+1}]:\n"
                        f"- [Item Title]: {history[i]['item_title']}\n"
                        f"- [Item Description]: {history[i]['item_desc']}\n"
                        f"- [Review Rating]: {history[i]['rating']}\n"
                        f"- [Review Title]: {history[i]['title']}\n"
                        f"- [Review Text]: {history[i]['text']}\n"
                        for i in range(tmp_prof_len)
                    ]
                )
                past_reviews = f"[User's Past Reviews]:\n{past_reviews}\n"
                past_ids = self.tokenizer(past_reviews, add_special_tokens=False)[
                    "input_ids"
                ]
                if len(past_ids) <= avail_len:
                    break
            if past_reviews == "":
                continue

            input_str = (
                f"[Item Title]: {target_item_title}\n"
                f"[Item Description]: {target_item_desc}\n"
                f"{past_reviews}"
                f"[Output Review Rating]: {target['rating']}\n"
                f"[Output Review Title]: {target['title']}\n"
            )
            input_str = self.pt.build_prompt(input_str)
            total_max_length = max_length + 1
            output_str = "[Review]: " + target["text"]

            self.input_str.append(input_str)
            self.output_str.append(output_str)

            inputs = self.tokenizer(
                input_str,
                max_length=input_max_length,
                truncation=True,
                add_special_tokens=False,
            )
            targets = self.tokenizer(
                output_str,
                max_length=output_max_length,
                truncation=True,
                add_special_tokens=False,
            )
            sample = {}
            if is_training:
                inputs_id = (
                    inputs["input_ids"]
                    + targets["input_ids"]
                    + [self.tokenizer.eos_token_id]
                )
                attention_mask = (
                    inputs["attention_mask"] + targets["attention_mask"] + [1]
                )
                label_ids = (
                    [-100] * len(inputs["input_ids"])
                    + targets["input_ids"]
                    + [self.tokenizer.eos_token_id]
                )
                if len(inputs_id) < total_max_length:
                    inputs_id = [self.tokenizer.pad_token_id] * (
                        total_max_length - len(inputs_id)
                    ) + inputs_id
                    attention_mask = [0] * (
                        total_max_length - len(attention_mask)
                    ) + attention_mask
                    label_ids = [-100] * (total_max_length - len(label_ids)) + label_ids
                sample["input_ids"] = np.array(inputs_id, dtype=np.int64)
                sample["attention_mask"] = np.array(attention_mask, dtype=np.int64)
                sample["label_ids"] = np.array(label_ids, dtype=np.int64)
            else:
                inputs_id = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]
                if len(inputs_id) < input_max_length:
                    inputs_id = [self.tokenizer.pad_token_id] * (
                        input_max_length - len(inputs_id)
                    ) + inputs_id
                    attention_mask = [0] * (
                        input_max_length - len(attention_mask)
                    ) + attention_mask
                sample["input_ids"] = np.array(inputs_id, dtype=np.int64)
                sample["attention_mask"] = np.array(attention_mask, dtype=np.int64)
            self.processed_data.append(sample)

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        return self.processed_data[idx]


class AmazonReviewDatasetV1(Dataset):
    def __init__(self, main_dataset, meta_dataset):
        self.main_dataset = main_dataset
        self.meta_dataset = meta_dataset

        self.processed_data = []

        meta_map = {
            asin: (title, desc)
            for asin, title, desc in zip(
                meta_dataset["asin"], meta_dataset["title"], meta_dataset["description"]
            )
        }

        self.history_str = []
        self.target_str = []
        for idx in tqdm(range(len(main_dataset)), desc=f"Pre-Processing data"):
            history = main_dataset[idx]["history"]
            for p in history:
                p_item_title, p_item_desc = meta_map[p["asin"]]
                p["item_title"] = p_item_title
                p["item_desc"] = p_item_desc
            history = sorted(history, key=lambda x: x["timestamp"], reverse=True)
            history_str = [
                f"[Item Title]: {history[i]['item_title']}\n"
                f"[Item Description]: {history[i]['item_desc']}\n"
                f"[Review Rating]: {history[i]['rating']}\n"
                f"[Review Title]: {history[i]['title']}\n"
                f"[Review Text]: {history[i]['text']}\n"
                for i in range(len(history))
            ]
            self.history_str.append(history_str)

            target = self.main_dataset[idx]["target"]
            target_item_title, target_item_desc = meta_map[target["asin"]]
            target_str = (
                f"[Item Title]: {target_item_title}\n"
                f"[Item Description]: {target_item_desc}\n"
                f"[Output Review Rating]: {target['rating']}\n"
                f"[Output Review Title]: {target['title']}\n"
            )
            self.target_str.append(target_str)

            sample = {"history_str": history_str, "target_str": target_str}

            self.processed_data.append(sample)

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        return self.processed_data[idx]


class AmazonReviewDatasetV2(Dataset):
    def __init__(
        self,
        main_dataset,
        meta_dataset,
        tokenizer,
        max_length=4096,
        is_training=True,
        sample_ratio=1.0,
        cache_path=None,
    ):
        self.main_dataset = main_dataset
        self.meta_dataset = meta_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_training = is_training

        if cache_path is not None:
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
            self.processed_data = cache["processed_data"]
            self.input_str = cache["input_str"]
            self.output_str = cache["output_str"]
            if sample_ratio < 1.0:
                total_size = len(self.processed_data)
                sample_size = int(total_size * sample_ratio)
                self.processed_data = self.processed_data[:sample_size]
                self.input_str = self.input_str[:sample_size]
                self.output_str = self.output_str[:sample_size]
            return

        input_max_length = max_length // 2
        output_max_length = max_length // 2

        system_prompt = (
            f"Given the title and description of an item, along with the current user's past reviews, and the output review rating and review title, "
            f"generate a personalized item review for the current user.\n"
            f"The review should be formatted as follows:\n"
            f"[Review]: <review>"
        )
        self.pt = QwenPromptTemplate(system_prompt)

        self.processed_data = []

        meta_map = {
            asin: (title, desc)
            for asin, title, desc in zip(
                meta_dataset["asin"], meta_dataset["title"], meta_dataset["description"]
            )
        }

        self.input_str = []
        self.output_str = []

        total_size = len(main_dataset)
        sample_size = int(total_size * sample_ratio)
        if sample_ratio < 1.0:
            main_dataset = main_dataset[:sample_size]

        for idx in tqdm(
            range(len(main_dataset)), desc=f"Pre-Processing data", mininterval=60
        ):
            history = main_dataset[idx]["history"]
            for p in history:
                p_item_title, p_item_desc = meta_map[p["asin"]]
                p["item_title"] = p_item_title
                p["item_desc"] = p_item_desc
            history = sorted(history, key=lambda x: x["timestamp"], reverse=True)

            target = self.main_dataset[idx]["target"]
            target_item_title, target_item_desc = meta_map[target["asin"]]

            tmp_inp_str = (
                f"[Item Title]: {target_item_title}\n"
                f"[Item Description]: {target_item_desc}\n"
                f"[Output Review Rating]: {target['rating']}\n"
                f"[Output Review Title]: {target['title']}\n"
            )
            tmp_inp_str = self.pt.build_prompt(tmp_inp_str)
            tmp_ids = self.tokenizer(tmp_inp_str, add_special_tokens=False)["input_ids"]
            tmp_len = len(tmp_ids)
            avail_len = input_max_length - tmp_len

            past_reviews = ""
            for tmp_prof_len in range(len(history), 0, -1):
                past_reviews = "".join(
                    [
                        f"[Review {i+1}]:\n"
                        f"- [Item Title]: {history[i]['item_title']}\n"
                        f"- [Item Description]: {history[i]['item_desc']}\n"
                        f"- [Review Rating]: {history[i]['rating']}\n"
                        f"- [Review Title]: {history[i]['title']}\n"
                        f"- [Review Text]: {history[i]['text']}\n"
                        for i in range(tmp_prof_len)
                    ]
                )
                past_reviews = f"[User's Past Reviews]:\n{past_reviews}\n"
                past_ids = self.tokenizer(past_reviews, add_special_tokens=False)[
                    "input_ids"
                ]
                if len(past_ids) <= avail_len:
                    break
            if past_reviews == "":
                continue

            input_str = (
                f"[Item Title]: {target_item_title}\n"
                f"[Item Description]: {target_item_desc}\n"
                f"{past_reviews}"
                f"[Output Review Rating]: {target['rating']}\n"
                f"[Output Review Title]: {target['title']}\n"
            )
            input_str = self.pt.build_prompt(input_str)
            total_max_length = max_length + 1
            output_str = "[Review]: " + target["text"]

            self.input_str.append(input_str)
            self.output_str.append(output_str)

            inputs = self.tokenizer(
                input_str,
                max_length=input_max_length,
                truncation=True,
                add_special_tokens=False,
            )
            targets = self.tokenizer(
                output_str,
                max_length=output_max_length,
                truncation=True,
                add_special_tokens=False,
            )
            sample = {}
            if is_training:
                inputs_id = (
                    inputs["input_ids"]
                    + targets["input_ids"]
                    + [self.tokenizer.eos_token_id]
                )
                attention_mask = (
                    inputs["attention_mask"] + targets["attention_mask"] + [1]
                )
                label_ids = (
                    [-100] * len(inputs["input_ids"])
                    + targets["input_ids"]
                    + [self.tokenizer.eos_token_id]
                )
                if len(inputs_id) < total_max_length:
                    inputs_id = [self.tokenizer.pad_token_id] * (
                        total_max_length - len(inputs_id)
                    ) + inputs_id
                    attention_mask = [0] * (
                        total_max_length - len(attention_mask)
                    ) + attention_mask
                    label_ids = [-100] * (total_max_length - len(label_ids)) + label_ids
                sample["input_ids"] = np.array(inputs_id, dtype=np.int64)
                sample["attention_mask"] = np.array(attention_mask, dtype=np.int64)
                sample["label_ids"] = np.array(label_ids, dtype=np.int64)
            else:
                inputs_id = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]
                if len(inputs_id) < input_max_length:
                    inputs_id = [self.tokenizer.pad_token_id] * (
                        input_max_length - len(inputs_id)
                    ) + inputs_id
                    attention_mask = [0] * (
                        input_max_length - len(attention_mask)
                    ) + attention_mask
                sample["input_ids"] = np.array(inputs_id, dtype=np.int64)
                sample["attention_mask"] = np.array(attention_mask, dtype=np.int64)
                sample["index"] = np.array(idx, dtype=np.int64)

            p_emb = main_dataset[idx]["p_emb_avg"]
            sample["p_emb"] = np.array(p_emb, dtype=np.float32)

            self.processed_data.append(sample)

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        return self.processed_data[idx]
