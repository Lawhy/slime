"""
Simple test script for generate function with Strands Agent

Usage: python quick_test.py
"""

import asyncio
import json
from argparse import Namespace

from generate_with_strands import generate, reward_func
from slime.utils.types import Sample


async def main():
    # Load one sample
    with open("data/dapo_math_17k_cleaned.jsonl") as f:
        data = json.loads(f.readline())
    
    sample = Sample(prompt=data["prompt"], label=data["label"])
    
    # Simple args for your Qwen3-8B server
    args = Namespace(
        model_name="Qwen/Qwen3-8B",
        hf_checkpoint="Qwen/Qwen3-8B",
        sglang_router_ip="localhost",
        sglang_router_port=8000,
        partial_rollout=False,
        sglang_server_concurrency=32,
        rollout_num_gpus=8,
        rollout_num_gpus_per_engine=8,
        rollout_temperature=1.0,
        rollout_top_p=1.0,
        rollout_top_k=-1,
        rollout_max_response_len=20480, # same as DAPO paper
        rollout_stop=None,
        rollout_stop_token_ids=None,
        rollout_skip_special_tokens=True,
        sglang_enable_deterministic_inference=False,
        rollout_seed=42,
        n_samples_per_prompt=1,
    )
    
    sampling_params = {"max_new_tokens": 20480, "temperature": 1.0, "top_p": 1.0}
    
    print("Testing generate function with Qwen3-8B...")
    print(f"Ground Truth: {sample.label}\n")
    
    # Generate
    result = await generate(args, sample, sampling_params)
    
    print(f"Status: {result.status}")
    print(f"Tool Calls: {getattr(result, 'tool_call_count', 0)}")
    print(f"\nResponse:\n{result.response}\n")
    
    # Compute reward
    reward = await reward_func(args, result)
    
    print(f"Score: {reward['score']}")
    print(f"Predicted: {reward['pred']}")
    print(f"Correct: {'✓' if reward['score'] > 0 else '✗'}")


if __name__ == "__main__":
    asyncio.run(main())

