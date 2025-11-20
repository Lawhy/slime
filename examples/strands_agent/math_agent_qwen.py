"""
Minimal Strands-Agents-based math agent that uses the `OpenAIModel` to calculate the square root of 16.

To start the vLLM server:

docker run --gpus all \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --model Qwen/Qwen3-0.6B
"""

from strands import Agent
from strands.models.openai import OpenAIModel
from strands_tools import calculator

# Configure OpenAIModel to use vLLM-hosted model
# IMPORTANT: model_id MUST match exactly what vLLM is serving!
model = OpenAIModel(
    client_args={
        "api_key": "EMPTY",
        "base_url": "http://240.10.0.8:8000/v1",
    },
    model_id="Qwen/Qwen3-0.6B",  # Must match the --model flag in vLLM command
    params={
        "max_tokens": 2048,
        "temperature": 1.0,
    },
)

if __name__ == "__main__":
    agent = Agent(model=model, tools=[calculator])
    agent("What is the square root of 16?")
