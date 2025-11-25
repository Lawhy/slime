# Slime x Strands-Agents

This is a running example that connects the [Strands-Agents](https://github.com/strands-agents/sdk-python) agent scaffolding framework with Slime for RL training.

## Install Dependencies

1. Pull the `slimerl/slime:latest` image and enter it
2. Goes to slime folder: `cd /root/slime`
3. Install Slime: `pip install -e .`
4. Goes to the example folder: `cd /root/slime/examples/strands_agent`
5. Install other dependencies: `pip install -r requirements.txt`
6. Set up docker daemon in the docker (for code-sandbox): `bash install_docker.sh`

## Prepare Qwen3-8B Model

```bash
# hf checkpoint
huggingface-cli download Qwen/Qwen3-8B --local-dir /root/Qwen3-8B

# mcore checkpoint
cd /root/slime
source scripts/models/qwen3-8B.sh
PYTHONPATH=/root/Megatron-LM:/root/slime python /root/slime/tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/Qwen3-8B \
    --save /root/Qwen3-8B_torch_dist
```

## Prepare Dataset

We follow the same training-eval setup as in `examples/retool` using `dapo_math_17k` as training data and `aime_2024` as eval data.

Run `process_dataset.py` such that 