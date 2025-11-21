"""
Debug script to see what's actually in the agent response
"""
import asyncio
import json
from argparse import Namespace

from generate_with_strands import generate
from slime.utils.types import Sample


async def main():
    # Load one sample
    with open("data/dapo_math_17k_cleaned.jsonl") as f:
        data = json.loads(f.readline())
    
    # Use a simpler sample
    sample = Sample(
        prompt="Calculate 15 * 23 and then add 100 to the result.",
        label="445"
    )
    
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
        rollout_top_p=0.7,
        rollout_top_k=-1,
        rollout_max_response_len=4096,
        rollout_stop=None,
        rollout_stop_token_ids=None,
        rollout_skip_special_tokens=True,
        sglang_enable_deterministic_inference=False,
        rollout_seed=42,
        n_samples_per_prompt=1,
    )
    
    sampling_params = {"max_new_tokens": 4096, "temperature": 1.0, "top_p": 0.7}
    
    print("=" * 80)
    print("Testing with simple math problem")
    print("=" * 80)
    print(f"Prompt: {sample.prompt}")
    print(f"Expected: {sample.label}\n")
    
    # Generate
    result = await generate(args, sample, sampling_params)
    
    print("\n" + "=" * 80)
    print("RESPONSE CONTENT:")
    print("=" * 80)
    print(result.response)
    print("\n" + "=" * 80)
    print(f"Status: {result.status}")
    print(f"Tool Calls: {getattr(result, 'tool_call_count', 0)}")
    print(f"Response Length (tokens): {result.response_length}")
    print(f"Response Length (chars): {len(result.response)}")


if __name__ == "__main__":
    asyncio.run(main())

