import os

from datasets import load_dataset

root_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(root_dir, "data")


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

    # training dataset is from DAPO-Math-17k
    dataset = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")
    dataset = dataset.map(transform_dapo_math, remove_columns=dataset.column_names)
    os.makedirs(data_dir, exist_ok=True)
    dataset.to_json(os.path.join(data_dir, "dapo_math_17k.jsonl"), orient="records", lines=True)

    # evaluation dataset is from AIME-2024
    dataset = load_dataset("zhuzilin/aime-2024", split="train")
    dataset = dataset.map(transform_aime, remove_columns=dataset.column_names)
    os.makedirs(data_dir, exist_ok=True)
    dataset.to_json(os.path.join(data_dir, "aime_2024.jsonl"), orient="records", lines=True)
