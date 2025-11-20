from datasets import load_dataset


def transform(example):
    return {
        "prompt": example["prompt"][0]["content"] if example["prompt"] else None,
        "label": example["reward_model"]["ground_truth"],
    }


if __name__ == "__main__":

    dataset = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")
    dataset = dataset.map(transform, remove_columns=dataset.column_names)
    dataset.to_json("dapo_math_17k_cleaned.jsonl", orient="records", lines=True)
