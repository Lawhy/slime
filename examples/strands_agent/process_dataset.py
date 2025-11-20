import os

from datasets import load_dataset

root_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(root_dir, "data")


def transform(example):
    return {
        "prompt": example["prompt"][0]["content"] if example["prompt"] else None,
        "label": example["reward_model"]["ground_truth"],
    }


if __name__ == "__main__":
    dataset = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")
    dataset = dataset.map(transform, remove_columns=dataset.column_names)
    os.makedirs(data_dir, exist_ok=True)
    dataset.to_json(os.path.join(data_dir, "dapo_math_17k_cleaned.jsonl"), orient="records", lines=True)
