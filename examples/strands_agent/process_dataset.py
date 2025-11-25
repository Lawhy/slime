import os

from datasets import load_dataset

DATA_DIR = "/root/data"


def transform_dapo_math(example):
    return {
        "prompt": example["prompt"][0]["content"] if example["prompt"] else None,
        "label": example["reward_model"]["ground_truth"],
    }

def transform_aime(example):
    return {
        "prompt": example["prompt"][0]["content"] if example["prompt"] else None,
        "label": example["label"],
    }


if __name__ == "__main__":

    os.makedirs(DATA_DIR, exist_ok=True)

    # training dataset is from DAPO-Math-17k
    dataset = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")
    dataset = dataset.map(transform_dapo_math, remove_columns=dataset.column_names)
    dataset.to_json(os.path.join(DATA_DIR, "dapo_math_17k.jsonl"), orient="records", lines=True)

    # evaluation dataset is from AIME-2024
    dataset = load_dataset("zhuzilin/aime-2024", split="train")
    dataset = dataset.map(transform_aime, remove_columns=dataset.column_names)
    dataset.to_json(os.path.join(DATA_DIR, "aime_2024.jsonl"), orient="records", lines=True)
